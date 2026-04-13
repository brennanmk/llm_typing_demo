#!/usr/bin/env python3
"""
NavigateTo action server.

Implements the navigate_to action. Given a target position on the map
(representing an object), the robot stops in front of it
This ensures the arm can reach the object.
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

    async def execute_callback(
        self, goal_handle: ServerGoalHandle
    ) -> NavigateTo.Result:
        """Compute an approach pose and send it to Nav2 NavigateToPose.

        Claude 4.6 Sonnet was used to help with this function, notably the transform math.

        :param goal_handle: ROS 2 action goal handle carrying target_pose and approach_distance.
        :return: NavigateTo.Result with success flag and status message.
        """
        result = NavigateTo.Result()
        target = goal_handle.request.target_pose
        approach_dist = goal_handle.request.approach_distance
        if approach_dist <= 0.0:
            approach_dist = 0.7

        # Get current robot pose in map frame to compute approach direction
        try:
            tf = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                Time(),
                rclpy.duration.Duration(seconds=1.0),
            )
            robot_x = tf.transform.translation.x
            robot_y = tf.transform.translation.y
        except Exception as e:
            self.get_logger().error(f"TF lookup failed: {e}")
            goal_handle.abort()
            result.success = False
            result.message = f"TF error: {e}"
            return result

        tx = target.pose.position.x
        ty = target.pose.position.y
        dx, dy = tx - robot_x, ty - robot_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1e-3:
            # Already close enough
            goal_handle.succeed()
            result.success = True
            result.message = "Already at target"
            return result

        angle = math.atan2(dy, dx)
        approach = PoseStamped()
        approach.header.frame_id = "map"
        approach.header.stamp = self.get_clock().now().to_msg()
        approach.pose.position.x = tx - approach_dist * math.cos(angle)
        approach.pose.position.y = ty - approach_dist * math.sin(angle)
        approach.pose.position.z = 0.0
        approach.pose.orientation.x = 0.0
        approach.pose.orientation.y = 0.0
        approach.pose.orientation.z = math.sin(angle / 2.0)
        approach.pose.orientation.w = math.cos(angle / 2.0)

        self.get_logger().info(
            f"Navigating to approach pose ({approach.pose.position.x:.2f}, "
            f"{approach.pose.position.y:.2f}) for target ({tx:.2f}, {ty:.2f})"
        )

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = approach

        nav_goal_handle = await self._nav_client.send_goal_async(nav_goal)
        if not nav_goal_handle.accepted:
            self.get_logger().error("Nav2 rejected goal")
            goal_handle.abort()
            result.success = False
            result.message = "Nav2 rejected goal"
            return result

        nav_result = await nav_goal_handle.get_result_async()

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
