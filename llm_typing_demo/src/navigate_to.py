#!/usr/bin/env python3
"""
NavigateTo action server (for stretch).


Implements the navigate_to action. Given a target position on the map
(representing an object), the robot stops in front of it
This ensures the arm can reach the object.

The direction matters because the Stretch arm extends to the side.

Requires an accurate map.
"""

import math

import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time

import tf2_ros
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

from llm_typing_demo.action import NavigateTo


class NavigateToNode(Node):
    """Action server that navigates to an approach pose in front of a target position."""

    def __init__(self) -> None:
        super().__init__("navigate_to_node")

        self._callback_group = ReentrantCallbackGroup()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._action_server = ActionServer(
            self,
            NavigateTo,
            "navigate_to",
            self.execute_callback,
            callback_group=self._callback_group,
        )
        self._nav_client = ActionClient(
            self,
            NavigateToPose,
            "navigate_to_pose",
            callback_group=self._callback_group,
        )

        self.get_logger().info("Waiting for Nav2 navigate_to_pose action server...")
        self._nav_client.wait_for_server()
        self.get_logger().info("Nav2 ready.")

    def execute_callback(self, goal_handle: ServerGoalHandle) -> NavigateTo.Result:
        """Compute an approach pose and send it to Nav2 NavigateToPose.

        :param goal_handle: target_pose and approach_distance.
        :return: NavigateTo.Result with success flag and status message.
        """
        result = NavigateTo.Result()
        target_pose = goal_handle.request.target_pose
        approach_dist = goal_handle.request.approach_distance

        try:
            robot_tf = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                Time(),
                rclpy.duration.Duration(seconds=1.0),
            )
            robot_x = robot_tf.transform.translation.x
            robot_y = robot_tf.transform.translation.y
        except Exception as e:
            self.get_logger().error(f"TF lookup failed: {e}")
            goal_handle.abort()
            result.success = False
            result.message = f"TF error: {e}"
            return result

        target_x = target_pose.pose.position.x
        target_y = target_pose.pose.position.y

        # find rotation of base needed for arm to be able to extend to grasp
        angle_to_target = math.atan2(target_y - robot_y, target_x - robot_x)
        approach_pose = PoseStamped()
        approach_pose.header.frame_id = "map"
        approach_pose.header.stamp = self.get_clock().now().to_msg()
        approach_pose.pose.position.x = target_x - approach_dist * math.cos(
            angle_to_target
        )
        approach_pose.pose.position.y = target_y - approach_dist * math.sin(
            angle_to_target
        )
        approach_pose.pose.position.z = 0.0
        approach_pose.pose.orientation.x = 0.0
        approach_pose.pose.orientation.y = 0.0
        approach_pose.pose.orientation.z = math.sin(angle_to_target / 2.0)
        approach_pose.pose.orientation.w = math.cos(angle_to_target / 2.0)

        self.get_logger().info(
            f"Navigating to approach pose ({approach_pose.pose.position.x:.2f}, "
            f"{approach_pose.pose.position.y:.2f}) for target ({target_x:.2f}, {target_y:.2f})"
        )

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = approach_pose

        nav_goal_future = self._nav_client.send_goal_async(nav_goal)
        rclpy.spin_until_future_complete(self, nav_goal_future)
        nav_goal_handle = nav_goal_future.result()
        if not nav_goal_handle.accepted:
            self.get_logger().error("Nav2 rejected goal")
            goal_handle.abort()
            result.success = False
            result.message = "Nav2 rejected goal"
            return result

        # Wait for result with timeout
        result_future = nav_goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=120.0)
        if not result_future.done():
            self.get_logger().warn("Nav2 timed out — cancelling goal")
            cancel_future = nav_goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future)
            goal_handle.abort()
            result.success = False
            result.message = "Navigation timed out"
            return result
        nav_result = result_future.result()

        if nav_result.status == GoalStatus.STATUS_SUCCEEDED:
            goal_handle.succeed()
            result.success = True
            result.message = "Navigation succeeded"
        else:
            goal_handle.abort()
            result.success = False
            result.message = f"Navigation failed (status {nav_result.status})"

        return result


if __name__ == "__main__":
    rclpy.init()
    node = NavigateToNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
