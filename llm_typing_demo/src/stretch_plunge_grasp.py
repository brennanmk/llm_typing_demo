#!/usr/bin/env python3
"""
Implements a node that consists of an action server. This action
server implements a plunge grasp, which involves moving the arm above
the target position and plunging to pick up the object.

This variant drives the Stretch robot's joint trajectory controller directly.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.action import ActionServer, ActionClient
from rclpy.action.server import ServerGoalHandle
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math

from llm_typing_demo.action import PlungeGrasp


_GRASP_WRIST_YAW = math.pi / 2  # arm pointing forward
_GRASP_WRIST_PITCH = -math.pi / 2  # gripper facing down
_READY_WRIST_PITCH = 0.0  # wrist level, tool-ready position


class StretchPlungeGraspNode(Node):
    """Action server that picks up an object using a plunge grasp. Also pitches tool into ready position."""

    def __init__(self) -> None:
        super().__init__("stretch_plunge_grasp_node")

        self._action_server = ActionServer(
            self, PlungeGrasp, "stretch_plunge_grasp", self.execute_callback
        )

        self._trajectory_client = ActionClient(
            self, FollowJointTrajectory, "/stretch_controller/follow_joint_trajectory"
        )

    async def execute_callback(
        self, goal_handle: ServerGoalHandle
    ) -> PlungeGrasp.Result:
        """Execute a plunge grasp at the requested position.

        :param goal_handle: ROS 2 action goal handle.
        :return: PlungeGrasp.Result.
        """
        self.get_logger().info("Executing Stretch plunge grasp...")
        feedback = PlungeGrasp.Feedback()
        result = PlungeGrasp.Result()

        if not self._trajectory_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Stretch trajectory server not available.")
            goal_handle.abort()
            return result

        x = goal_handle.request.position.x
        y = goal_handle.request.position.y
        z = goal_handle.request.position.z

        arm_extension = math.sqrt(x**2 + y**2)

        # Open Gripper and hover above target
        feedback.current_phase = "Hovering above target"
        goal_handle.publish_feedback(feedback)

        hover_z = z + 0.15

        hover_goal = self.build_trajectory(
            lift_z=hover_z,
            arm_extension=arm_extension,
            wrist_yaw=_GRASP_WRIST_YAW,
            wrist_pitch=_GRASP_WRIST_PITCH,
            gripper_pos=0.10,
            duration_sec=9,
        )
        trajectory_goal_handle = await self._trajectory_client.send_goal_async(
            hover_goal
        )
        await trajectory_goal_handle.get_result_async()

        # plunge
        feedback.current_phase = "Plunging to grasp depth"
        goal_handle.publish_feedback(feedback)

        plunge_z = z - 0.02

        plunge_goal = self.build_trajectory(
            lift_z=plunge_z,
            arm_extension=arm_extension,
            wrist_yaw=_GRASP_WRIST_YAW,
            wrist_pitch=_GRASP_WRIST_PITCH,
            gripper_pos=0.10,
            duration_sec=2,
        )
        trajectory_goal_handle = await self._trajectory_client.send_goal_async(
            plunge_goal
        )
        await trajectory_goal_handle.get_result_async()

        # Close Gripper
        feedback.current_phase = "Closing gripper"
        goal_handle.publish_feedback(feedback)

        grasp_goal = self.build_trajectory(
            lift_z=plunge_z,
            arm_extension=arm_extension,
            wrist_yaw=_GRASP_WRIST_YAW,
            wrist_pitch=_GRASP_WRIST_PITCH,
            gripper_pos=0.0,
            duration_sec=2,
        )
        trajectory_goal_handle = await self._trajectory_client.send_goal_async(
            grasp_goal
        )
        await trajectory_goal_handle.get_result_async()

        # lift object
        feedback.current_phase = "Lifting object"
        goal_handle.publish_feedback(feedback)

        retreat_goal = self.build_trajectory(
            lift_z=hover_z,
            arm_extension=arm_extension,
            wrist_yaw=_GRASP_WRIST_YAW,
            wrist_pitch=_GRASP_WRIST_PITCH,
            gripper_pos=0.0,
            duration_sec=2,
        )
        trajectory_goal_handle = await self._trajectory_client.send_goal_async(
            retreat_goal
        )
        await trajectory_goal_handle.get_result_async()

        # Pitch wrist forward to tool-ready position
        feedback.current_phase = "Readying tool"
        goal_handle.publish_feedback(feedback)

        ready_goal = self.build_trajectory(
            lift_z=hover_z,
            arm_extension=arm_extension,
            wrist_yaw=_GRASP_WRIST_YAW,
            wrist_pitch=_READY_WRIST_PITCH,
            gripper_pos=0.0,
            duration_sec=2,
        )
        trajectory_goal_handle = await self._trajectory_client.send_goal_async(
            ready_goal
        )
        await trajectory_goal_handle.get_result_async()

        result.success = True
        result.current_phase = "Plunge grasp completed successfully."
        goal_handle.succeed()
        return result

    def build_trajectory(
        self,
        lift_z: float,
        arm_extension: float,
        wrist_yaw: float,
        wrist_pitch: float,
        gripper_pos: float,
        duration_sec: int = 4,
    ) -> FollowJointTrajectory.Goal:
        """Build a single-point FollowJointTrajectory goal for the Stretch arm.

        :param lift_z: Target lift height in meters.
        :param arm_extension: Target arm extension.
        :param wrist_yaw: Target wrist yaw in radians.
        :param wrist_pitch: Target wrist pitch in radians.
        :param gripper_pos: Total gripper opening.
        :param duration_sec: Time allowed to reach the target position in seconds.
        :return: FollowJointTrajectory.Goal.
        """
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = [
            "joint_lift",
            "joint_arm",
            "joint_wrist_yaw",
            "joint_wrist_pitch",
            "gripper_aperture",
        ]

        point = JointTrajectoryPoint()
        point.positions = [
            lift_z,
            arm_extension,
            wrist_yaw,
            wrist_pitch,
            gripper_pos,
        ]

        point.time_from_start = Duration(sec=duration_sec, nanosec=0)
        goal_msg.trajectory.points = [point]

        return goal_msg


if __name__ == "__main__":
    rclpy.init()
    node = StretchPlungeGraspNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
