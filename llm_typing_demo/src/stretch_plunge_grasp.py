#!/usr/bin/env python3
"""
Implements a node that contains an action server which implements a
plunge grasp. This involves moving the arm above a target position and
plunging to pick up the object. In the case of the stretch, this also
involves moving the base.
"""

import time
import rclpy
import tf2_geometry_msgs
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from stretch_alignment_base import StretchAlignmentBaseNode

from llm_typing_demo.action import PlungeGrasp

_GRASP_WRIST_YAW = 1.57
_GRASP_WRIST_PITCH = -1.4
_GRASP_GRIPPER_CLOSE = -0.35
_GRASP_ARM_STANDOFF = 0.06
_LIFT_EFFORT_THRESHOLD = 35.0


class StretchPlungeGraspNode(StretchAlignmentBaseNode):
    """Action server that picks up an object with a top-down plunge grasp on Stretch."""

    def __init__(self) -> None:
        super().__init__("stretch_plunge_grasp_node")

        # the homing sequence has a predictable error, by modifying this param we can account for that error, likely due to belt wearout
        self.declare_parameter("plunge_depth_offset", 0.06)

        self._action_server = ActionServer(
            self,
            PlungeGrasp,
            "stretch_plunge_grasp",
            self.execute_callback,
            callback_group=self._callback_group,
        )

    def execute_callback(self, goal_handle: ServerGoalHandle) -> PlungeGrasp.Result:
        """Execute a plunge grasp at the requested position.

        Moves to a safe travel pose, aligns the base, extends the arm
        over the object, descends until contact is made, closes the
        gripper, and returns to the start pose.

        :param goal_handle: ROS 2 action goal handle carrying the target
        :return: PlungeGrasp.Result returns success on completion.
        """

        feedback = PlungeGrasp.Feedback()
        result = PlungeGrasp.Result()

        def publish_feedback(msg):
            feedback.current_phase = msg
            goal_handle.publish_feedback(feedback)

        if not self._wait_for_joint_states():
            goal_handle.abort()
            return result

        start_pose = self._get_map_pose()

        original_map_pt = tf2_geometry_msgs.PointStamped()
        original_map_pt.header.frame_id = "map"
        original_map_pt.point = goal_handle.request.position

        # move gripper up so it does not hit table
        safe_lift_z = max(self._read_joint("joint_lift") or 0.5, 0.85)
        publish_feedback("Moving to travel pose")
        self.send_trajectory_sync(
            self.build_trajectory(
                safe_lift_z, 0.0, _GRASP_WRIST_YAW, _GRASP_WRIST_PITCH, 0.10, 3.0
            )
        )

        # Align robot to object
        self.align_and_verify(original_map_pt, feedback_cb=publish_feedback)

        # find target extension and grasp height
        grasp_center_tf = self.tf_buffer.lookup_transform(
            "base_link", "link_grasp_center", rclpy.time.Time()
        )
        map_to_base_tf = self.tf_buffer.lookup_transform(
            "base_link", "map", rclpy.time.Time()
        )
        target_in_base_frame = tf2_geometry_msgs.do_transform_point(
            original_map_pt, map_to_base_tf
        )

        arm_extension = max(
            0.0,
            min(
                grasp_center_tf.transform.translation.y
                - target_in_base_frame.point.y
                - _GRASP_ARM_STANDOFF,
                0.50,
            ),
        )
        lift_height_to_grasp = grasp_center_tf.transform.translation.z - (
            self._read_joint("joint_lift") or safe_lift_z
        )

        plunge_depth_offset = self.get_parameter("plunge_depth_offset").value
        ideal_hover = target_in_base_frame.point.z - lift_height_to_grasp + 0.15
        ideal_plunge = (
            target_in_base_frame.point.z - lift_height_to_grasp - plunge_depth_offset
        )

        # clamp to ensure valid poses
        hover_z = max(0.10, ideal_hover)
        plunge_z = max(0.10, ideal_plunge)

        # approach
        publish_feedback("Extending arm high above target")
        self.send_trajectory_sync(
            self.build_trajectory(
                safe_lift_z,
                arm_extension,
                _GRASP_WRIST_YAW,
                _GRASP_WRIST_PITCH,
                0.10,
                3.0,
            )
        )

        publish_feedback("Dropping mast to hover position")
        self.send_trajectory_sync(
            self.build_trajectory(
                hover_z, arm_extension, _GRASP_WRIST_YAW, _GRASP_WRIST_PITCH, 0.10, 2.0
            )
        )

        # plunge
        publish_feedback("Plunging...")
        plunge_goal = self.build_trajectory(
            plunge_z, arm_extension, _GRASP_WRIST_YAW, _GRASP_WRIST_PITCH, 0.10, 2.0
        )

        goal_future = self._trajectory_client.send_goal_async(plunge_goal)
        rclpy.spin_until_future_complete(self, goal_future)
        trajectory_handle = goal_future.result()
        if not trajectory_handle.accepted:
            self.get_logger().error("Plunge goal rejected")
            goal_handle.abort()
            return result

        res_future = trajectory_handle.get_result_async()

        # press down until we hit something
        contact_detected = False
        while not res_future.done():
            effort = self._read_joint_effort("joint_lift")
            if effort and abs(effort) > _LIFT_EFFORT_THRESHOLD:
                self.get_logger().warn(
                    f"Contact detected! Lift effort spiked to {abs(effort):.2f}"
                )
                trajectory_handle.cancel_goal_async()
                contact_detected = True
                break
            time.sleep(0.05)

        final_lift_z = self._read_joint("joint_lift") if contact_detected else plunge_z

        # grasp
        publish_feedback("Closing gripper")
        self.send_trajectory_sync(
            self.build_trajectory(
                final_lift_z,
                arm_extension,
                _GRASP_WRIST_YAW,
                _GRASP_WRIST_PITCH,
                _GRASP_GRIPPER_CLOSE,
                2.0,
            )
        )

        # lift
        publish_feedback("Lifting object")
        self.send_trajectory_sync(
            self.build_trajectory(
                hover_z,
                0.0,
                _GRASP_WRIST_YAW,
                _GRASP_WRIST_PITCH,
                _GRASP_GRIPPER_CLOSE,
                3.0,
            )
        )

        publish_feedback("Returning to start pose")
        self._return_to_pose(start_pose)

        result.success = True
        goal_handle.succeed()
        return result


if __name__ == "__main__":
    rclpy.init()
    executor = MultiThreadedExecutor()
    node = StretchPlungeGraspNode()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
