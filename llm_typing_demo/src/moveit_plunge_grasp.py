#!/usr/bin/env python3
"""
Implements a node that consists of an action server. This action
server implements a plunge grasp, which involves moving the arm above
the target position and plunging to pick up the object.

This version uses moveit and has been tested on a kinova gen3 lite.
"""

from threading import Event, Lock

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.action.server import ServerGoalHandle

from control_msgs.action import GripperCommand
from geometry_msgs.msg import PoseStamped
from moveit.planning import MoveItPy

from llm_typing_demo.action import PlungeGrasp


# quat for gripper facing down
_DOWN_QUAT = [0.0, 1.0, 0.0, 0.0]


class MoveItPlungeGraspNode(Node):
    """Action server that picks up an object using moveit."""

    def __init__(self) -> None:
        super().__init__("moveit_plunge_grasp_node")

        # param setup
        self.declare_parameter("planning_group", "arm")
        self.declare_parameter("base_link", "base_link")
        self.declare_parameter("ee_link", "end_effector_link")
        self.declare_parameter(
            "gripper_action", "/gen3_lite_2f_gripper_controller/gripper_cmd"
        )
        self.declare_parameter("hover_offset", 0.15)
        self.declare_parameter("grasp_depth", 0.02)
        self.declare_parameter("gripper_timeout", 15.0)

        self._gripper_timeout = self.get_parameter("gripper_timeout").get_parameter_value().double_value
        self._planning_group = self.get_parameter("planning_group").get_parameter_value().string_value
        self._base_link = self.get_parameter("base_link").get_parameter_value().string_value
        self._ee_link = self.get_parameter("ee_link").get_parameter_value().string_value
        gripper_action = self.get_parameter("gripper_action").get_parameter_value().string_value
        self._hover_offset = self.get_parameter("hover_offset").get_parameter_value().double_value
        self._grasp_depth = self.get_parameter("grasp_depth").get_parameter_value().double_value

        # moveitpy actually creates its own node
        self._moveit_py = MoveItPy(node_name="moveit_py_node")
        self._arm_component = self._moveit_py.get_planning_component(self._planning_group)

        # action server is reentrant, so we need to manually lock execution
        self._execution_lock = Lock()
        self._callback_group = ReentrantCallbackGroup()

        self._gripper_client = ActionClient(
            self,
            GripperCommand,
            gripper_action,
            callback_group=self._callback_group,
        )

        self._action_server = ActionServer(
            self,
            PlungeGrasp,
            "moveit_plunge_grasp",
            self.execute_callback,
            callback_group=self._callback_group,
        )

        self.get_logger().info(
            f"moveit_plunge_grasp ready (group={self._planning_group}, ee={self._ee_link})"
        )

    def _send_gripper(self, position: float, max_effort: float = 50.0) -> bool:
        """Send a GripperCommand goal and block until complete."""
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Gripper action server not available")
            return False

        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = max_effort

        # event that we wait on to return
        done = Event()
        success_ref = [False]

        def result_callback(future):
            success_ref[0] = True
            done.set()

        def goal_callback(future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Gripper goal rejected")
                done.set()
                return
            goal_handle.get_result_async().add_done_callback(result_callback)

        self._gripper_client.send_goal_async(goal).add_done_callback(goal_callback)
        done.wait(timeout=self._gripper_timeout)
        return success_ref[0]

    def _move_to(self, x: float, y: float, z: float) -> bool:
        """Blocking moveit_py move."""
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = self._base_link
        pose_goal.pose.position.x = float(x)
        pose_goal.pose.position.y = float(y)
        pose_goal.pose.position.z = float(z)
        pose_goal.pose.orientation.x = _DOWN_QUAT[0]
        pose_goal.pose.orientation.y = _DOWN_QUAT[1]
        pose_goal.pose.orientation.z = _DOWN_QUAT[2]
        pose_goal.pose.orientation.w = _DOWN_QUAT[3]

        self._arm_component.set_start_state_to_current_state()
        self._arm_component.set_goal_state(pose_stamped_msg=pose_goal, pose_link=self._ee_link)

        plan_result = self._arm_component.plan()

        if plan_result:
            success = self._moveit_py.execute(plan_result.trajectory, controllers=[])
            if not success:
                self.get_logger().error(f"MoveIt execution failed for target ({x:.2f}, {y:.2f}, {z:.2f})")
            return success
        else:
            self.get_logger().error(f"moveit_py failed to find a valid plan for target ({x:.2f}, {y:.2f}, {z:.2f})")
            return False

    def execute_callback(self, goal_handle: ServerGoalHandle) -> PlungeGrasp.Result:
        """Execute a plunge grasp at the requested position.

        :param goal_handle: ROS 2 action goal handle.
        :return: PlungeGrasp.Result.
        """

        with self._execution_lock:
            self.get_logger().info("Executing MoveIt plunge grasp...")
            feedback = PlungeGrasp.Feedback()
            result = PlungeGrasp.Result()

            x = goal_handle.request.position.x
            y = goal_handle.request.position.y
            z = goal_handle.request.position.z

            hover_z = z + self._hover_offset
            grasp_z = z - self._grasp_depth

            try:
                # Open gripper
                feedback.current_phase = "Open-Gripper"
                goal_handle.publish_feedback(feedback)
                self._send_gripper(position=0.0)

                # Move to hover pose above target
                feedback.current_phase = "Approach"
                goal_handle.publish_feedback(feedback)
                if not self._move_to(x, y, hover_z):
                    goal_handle.abort()
                    return result

                # plunge
                feedback.current_phase = "Plunge"
                goal_handle.publish_feedback(feedback)
                if not self._move_to(x, y, grasp_z):
                    goal_handle.abort()
                    return result

                # Close gripper
                feedback.current_phase = "Grasp"
                goal_handle.publish_feedback(feedback)
                self._send_gripper(position=0.8)

                # lift object
                feedback.current_phase = "Lift"
                goal_handle.publish_feedback(feedback)
                if not self._move_to(x, y, hover_z):
                    goal_handle.abort()
                    return result

                result.success = True
                result.current_phase = "MoveIt plunge grasp completed successfully."
                goal_handle.succeed()
                return result

if __name__ == "__main__":
    rclpy.init()
    node = MoveItPlungeGraspNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
