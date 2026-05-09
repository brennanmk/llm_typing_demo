#!/usr/bin/env python3
"""
Belief node. Gives us a centralized place to track objects in the world.

Subscribes to /detection_batch published by object_detection.py and
maintains a belief state about objects in the scene.
"""

import threading
import uuid
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import SetBool

from llm_typing_demo.msg import DetectionBatch, ObjectDetection
from llm_typing_demo.srv import ClearBeliefs, GetBeliefs, RemoveBelief


@dataclass
class BeliefEntry:
    """belief about a single object in the scene."""

    belief_id: str
    class_name: str
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0
    size_y: float = 0.0
    size_z: float = 0.0
    observation_count: int = 0
    best_conf: float = 0.0

    # History buffers (stores last 15 observations)
    hist_x: deque = field(default_factory=lambda: deque(maxlen=15))
    hist_y: deque = field(default_factory=lambda: deque(maxlen=15))
    hist_z: deque = field(default_factory=lambda: deque(maxlen=15))


class BeliefNode(Node):
    """Maintains a persistent belief state from DetectionBatch messages."""

    def __init__(self) -> None:
        super().__init__("belief_node")

        # amount we allow observations to shift by
        self.declare_parameter("belief_merge_distance", 0.3)

        # how many times we need to observe the object before counting it
        self.declare_parameter("min_spatial_samples", 3)

        self._beliefs: dict[str, BeliefEntry] = {}
        self._beliefs_lock = (
            threading.Lock()
        )  # make sure we don't run into fun concurrency bugs
        self._frozen = False

        self._callback_group = ReentrantCallbackGroup()

        self.create_subscription(
            DetectionBatch,
            "/detection_batch",
            self._batch_callback,
            10,
            callback_group=self._callback_group,
        )

        self.create_service(
            GetBeliefs,
            "get_beliefs",
            self._handle_get_beliefs,
            callback_group=self._callback_group,
        )
        self.create_service(
            RemoveBelief,
            "remove_belief",
            self._handle_remove_belief,
            callback_group=self._callback_group,
        )
        self.create_service(
            ClearBeliefs,
            "clear_beliefs",
            self._handle_clear_beliefs,
            callback_group=self._callback_group,
        )
        self.create_service(
            SetBool,
            "freeze_beliefs",
            self._handle_freeze_beliefs,
            callback_group=self._callback_group,
        )

        self.get_logger().info("belief_node ready.")

    def _update_beliefs(self, detections: list[ObjectDetection]) -> None:
        """combine a list of new detections into the current belief state.

        :param detections: List of ObjectDetection messages to incorporate.
        """
        merge_dist = (
            self.get_parameter("belief_merge_distance")
            .get_parameter_value()
            .double_value
        )

        for det in detections:
            conf = det.confidence
            matched_id = None
            best_dist = float("inf")

            for belief_id, entry in self._beliefs.items():
                if entry.class_name != det.class_name and conf <= entry.best_conf:
                    continue
                dist = np.linalg.norm(
                    [
                        entry.center_x - det.center_x,
                        entry.center_y - det.center_y,
                        entry.center_z - det.center_z,
                    ]
                )
                if dist < merge_dist and dist < best_dist:
                    best_dist = dist
                    matched_id = belief_id

            if matched_id is not None:  # repeat observation
                entry = self._beliefs[matched_id]
                if conf > entry.best_conf:
                    entry.class_name = det.class_name
                    entry.best_conf = conf

                # clear history on large position shift
                shift_dist = np.linalg.norm(
                    [
                        entry.center_x - det.center_x,
                        entry.center_y - det.center_y,
                        entry.center_z - det.center_z,
                    ]
                )
                if shift_dist > 0.10:
                    entry.hist_x.clear()
                    entry.hist_y.clear()
                    entry.hist_z.clear()

                entry.hist_x.append(det.center_x)
                entry.hist_y.append(det.center_y)
                entry.hist_z.append(det.center_z)

                # find median observation
                entry.center_x = float(np.median(entry.hist_x))
                entry.center_y = float(np.median(entry.hist_y))
                entry.center_z = float(np.median(entry.hist_z))

                # exponential moving average
                alpha = 0.2  # weight of new observations
                entry.size_y = (1 - alpha) * entry.size_y + alpha * det.size_y
                entry.size_z = (1 - alpha) * entry.size_z + alpha * det.size_z

                entry.observation_count += 1
            else:  # first time observing
                new_id = uuid.uuid4().hex[:8]  # get a random-ish id
                entry = BeliefEntry(
                    belief_id=new_id,
                    class_name=det.class_name,
                    size_y=det.size_y,
                    size_z=det.size_z,
                    observation_count=1,
                    best_conf=conf,
                )
                entry.hist_x.append(det.center_x)
                entry.hist_y.append(det.center_y)
                entry.hist_z.append(det.center_z)
                entry.center_x = det.center_x
                entry.center_y = det.center_y
                entry.center_z = det.center_z

                self._beliefs[new_id] = entry

    def _batch_callback(self, msg: DetectionBatch) -> None:
        """Handle an incoming DetectionBatch message and update beliefs if not frozen.

        :param msg: Incoming DetectionBatch from the object detection node.
        """
        if self._frozen:
            return
        with self._beliefs_lock:
            self._update_beliefs(msg.detections)

    def _handle_get_beliefs(
        self, request: GetBeliefs.Request, response: GetBeliefs.Response
    ) -> GetBeliefs.Response:
        """Return the best belief per class with at least min_spatial_samples observations.

        :param request: GetBeliefs.Request (no fields required).
        :param response: GetBeliefs.Response to populate with the filtered belief list.
        :return: Populated GetBeliefs.Response.
        """
        min_samples = self.get_parameter("min_spatial_samples").value

        with self._beliefs_lock:
            best: dict[str, ObjectDetection] = {}
            for entry in self._beliefs.values():
                if len(entry.hist_x) < min_samples:
                    continue

                if (
                    entry.class_name not in best
                    or entry.best_conf > best[entry.class_name].confidence
                ):
                    det = ObjectDetection()
                    det.belief_id, det.class_name, det.confidence = (
                        entry.belief_id,
                        entry.class_name,
                        entry.best_conf,
                    )
                    det.center_x, det.center_y, det.center_z = (
                        entry.center_x,
                        entry.center_y,
                        entry.center_z,
                    )
                    det.size_y, det.size_z = entry.size_y, entry.size_z
                    best[entry.class_name] = det

            response.objects = list(best.values())
        return response

    def _handle_remove_belief(
        self, request: RemoveBelief.Request, response: RemoveBelief.Response
    ) -> RemoveBelief.Response:
        """Remove a single belief entry given an ID.

        :param request: RemoveBelief.Request with the belief_id to remove.
        :param response: RemoveBelief.Response with success indicating whether the ID was found.
        :return: Populated RemoveBelief.Response.
        """
        with self._beliefs_lock:
            if request.belief_id in self._beliefs:
                del self._beliefs[request.belief_id]
                response.success = True
            else:
                response.success = False
        return response

    def _handle_freeze_beliefs(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        """Freeze or unfreeze belief updates from incoming DetectionBatch messages.

        After we turn the body, we might get some odd detections, so
        we allow freezing during motion.

        :param request: SetBool.Request where data=True freezes and data=False unfreezes.
        :param response: SetBool.Response with success and a status message.
        :return: Populated SetBool.Response.
        """
        self._frozen = request.data
        self.get_logger().info(f"Beliefs {'frozen' if self._frozen else 'unfrozen'}.")
        response.success = True
        response.message = "frozen" if self._frozen else "unfrozen"
        return response

    def _handle_clear_beliefs(
        self, request: ClearBeliefs.Request, response: ClearBeliefs.Response
    ) -> ClearBeliefs.Response:
        """Clear all beliefs

        :param request: ClearBeliefs.Request.
        :param response: ClearBeliefs.Response indicates success.
        :return: Populated ClearBeliefs.Response.
        """
        with self._beliefs_lock:
            self._beliefs.clear()
        response.success = True
        return response


if __name__ == "__main__":
    rclpy.init()
    node = BeliefNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
