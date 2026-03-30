#!/usr/bin/env python3
"""
Implements a node that consists of an action server. This action
server implements a scan scene action, which pans the camera and uses
YOLO26 to detect all objects.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.action import ActionServer, ActionClient
import message_filters
from sensor_msgs.msg import Image
from control_msgs.action import FollowJointTrajectory
import asyncio
from ultralytics import YOLO
from llm_typing_demo.action import ScanScene

class SceneScannerNode(Node):
    def __init__(self):
        super().__init__("scene_scanner_node")

        self.declare_parameter("yolo_model_path", "yolo26.pt")
        self.yolo_model = YOLO(
            self.get_parameter("yolo_model_path").get_parameter_value().string_value
        )

        self.latest_color = None
        self.latest_depth = None

        self._action_server = ActionServer(
            self, ScanScene, "scan_scene", self.execute_callback
        )

        # action client to manipulate head motion of stretch
        self._head_client = ActionClient(
            self, FollowJointTrajectory, "/stretch_controller/follow_joint_trajectory"
        )

        # sync color to depth image
        self.synced_images = message_filters.ApproximateTimeSynchronizer(
            [
                message_filters.Subscriber(self, Image, "/camera/color/image_raw"),
                message_filters.Subscriber(
                    self, Image, "/camera/aligned_depth_to_color/image_raw"
                ),
            ],
            queue_size=10,
            slop=0.05,
        )
        self.synced_images.registerCallback(self.sync_callback)

    def sync_callback(self, color_msg, depth_msg):
        """
        update latest class variables with synced data, questionably should be done with mutex.
        """
        self.latest_color, self.latest_depth = color_msg, depth_msg

    async def execute_callback(self, goal_handle):
        """
        callback to scan scene
        """

if __name__ == "__main__":
    try:
        with rclpy.init():
            node = SceneScannerNode()

            rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass
