#!/usr/bin/env python3
"""
Implements a node that consists of an action server. This action
server implements a plunge grasp, which involves moving the arm above
the target position and plunging to pick up the object.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.action import ActionServer, ActionClient
from control_msgs.action import FollowJointTrajectory

from llm_typing_demo.action import PlungeGrasp


class PlungeGraspNode(Node):
    def __init__(self):
        super().__init__("plunge_grasp_node")

        self._action_server = ActionServer(
            self, PlungeGrasp, "plunge_grasp", self.execute_callback
        )

        # Stretch trajectory controller
        self._trajectory_client = ActionClient(
            self, FollowJointTrajectory, "/stretch_controller/follow_joint_trajectory"
        )

    async def execute_callback(self, goal):
        """
        execute actual grasp.
        """

if __name__ == "__main__":
    try:
        with rclpy.init():
            node = PlungeGraspNode()

            rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass
