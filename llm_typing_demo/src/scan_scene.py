#!/usr/bin/env python3
"""
Contains ScanScene action server.

Moves the Stretch head through pan/tilt waypoints, used to sweep head
(across table in the demo)
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from control_msgs.action import FollowJointTrajectory
from rclpy.executors import ExternalShutdownException
from std_srvs.srv import SetBool
from llm_typing_demo.action import ScanScene
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# tuples of (pan, tilt) waypoints
SWEEP_POINTS = [
    (-0.8, -0.6),
    (-0.4, -0.6),
    (0.0, -0.6),
    (0.4, -0.6),
    (0.8, -0.6),
    (0.0, 0.0),
]


class ScanSceneNode(Node):
    def __init__(self) -> None:
        super().__init__("scan_scene_node")

        self.declare_parameter("wait_time", 7.0)  # time to wait at each head waypoint
        self._wait_time = self.get_parameter("wait_time").value

        # we rely on the default callback group, mutually exclusive,
        # to ensure that we are not processing multiple requests.
        self._action_server = ActionServer(
            self, ScanScene, "scan_scene", self._execute_callback
        )

        self._head_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/stretch_controller/follow_joint_trajectory",
        )

        self._pause_detection_client = self.create_client(SetBool, "pause_detection")

        self.get_logger().info("scan_scene_node ready.")

    def _pause_detection(self, pause: bool) -> None:
        """blocking call to pause detection

        :param pause: pause if true, unpause if false
        :return: None, returns after waiting for future
        """

        req = SetBool.Request()
        req.data = pause
        rclpy.spin_until_future_complete(
            node, self._pause_detection_client.call_async(req)
        )

    def _execute_callback(self, goal_handle: ScanScene.Goal) -> ScanScene.Result:
        """
        executes move head, sweeps through all points, pauses for a configurable amount of time.

        :param goal_handle: goal_handle to populate with success
        :return: goal
        """

        self.get_logger().info("Scan Scene Request Recieved")

        for point in SWEEP_POINTS:
            pan, tilt = point
            trajectory_goal = FollowJointTrajectory.Goal()
            trajectory_goal.trajectory.joint_names = [
                "joint_head_pan",
                "joint_head_tilt",
            ]
            trajectory_goal.trajectory.points = [
                JointTrajectoryPoint(
                    positions=[pan, tilt],
                    time_from_start=Duration(sec=1),
                )
            ]
            trajectory_goal.trajectory.header.frame_id = "base_link"

            self._pause_detection(True)
            # move head
            self._head_client.wait_for_server()
            head_future = self._head_client.send_goal_async(trajectory_goal)
            rclpy.spin_until_future_complete(node, head_future)

            time.sleep(0.2)  # give some time for head to settle after motion
            self._pause_detection(False)

            time.sleep(
                self._wait_time
            )  # wait for configurable amount of time (to allow for collection of multiple samples)

        self.get_logger().info("Scan Scene Request Compete")

        goal_handle.succeed()
        result = ScanScene.Result()
        return result


if __name__ == "__main__":
    rclpy.init()
    node = ScanSceneNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
