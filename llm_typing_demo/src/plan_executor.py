#!/usr/bin/env python3
"""
Orchestrator for demonstration
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException

from llm_typing_demo.action import NavigateTo, PlungeGrasp, WipeWhiteboard


class PlanExecutorNode(Node):
    def __init__(self, registry: dict):
        super().__init__("plan_executor")

        self._nav_client = ActionClient(self, NavigateTo, "navigate_to")
        self._grasp_client = ActionClient(self, PlungeGrasp, "plunge_grasp")
        self._wipe_client = ActionClient(self, WipeWhiteboard, "wipe_whiteboard")

    def orchestrate(self):
        """
        orchestrate actual plan, scan scene, execute actions. This
        might be called by main, or triggered via a service or action.
        """


if __name__ == "__main__":
    try:
        with rclpy.init():
            node = PlanExecutorNode()

            rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass
