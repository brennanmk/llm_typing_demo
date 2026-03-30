#!/usr/bin/env python3
"""
contains NavigateTo action server implementation. Moves robot within reachibility of an object.

Makes use of Stretch's rtab stack for mapping and nav2 for navigation.
"""

import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose

from llm_typing_demo.action import NavigateTo


class NavigateToNode(Node):
    def __init__(self):
        super().__init__("navigate_to_node")

        self._action_server = ActionServer(
            self, NavigateTo, "navigate_to", self.execute_callback
        )

        # Nav2 action client
        self._nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

    async def execute_callback(self, goal_handle):
        """
        callback to move base to within reachability of object
        """


if __name__ == "__main__":
    try:
        with rclpy.init():
            node = NavigateToNode()
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
