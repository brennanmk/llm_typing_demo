#!/usr/bin/env python3
"""
Implements a node that consists of an action server. This action
server implements a whiteboard wiping action. This action uses the
object held in the gripper, which should be in hand before calling
this action to wipe off a whiteboard.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.action import ActionServer
from llm_typing_demo.action import WipeWhiteboard

class WhiteboardWiperNode(Node):
    def __init__(self):
        super().__init__('whiteboard_wiper')

        self._action_server = ActionServer(self, WipeWhiteboard, 'wipe_whiteboard', self.execute_callback)

    async def execute_callback(self, goal):
        """
        callback to execute whiteboard wiping
        """

if __name__ == "__main__":
    try:
        with rclpy.init():
            node = WhiteboardWiperNode()

            rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass
