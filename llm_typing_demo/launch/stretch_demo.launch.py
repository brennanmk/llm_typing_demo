#!/usr/bin/env python3

"""
Stretch Demo Launch File

Starts the action server nodes needed for the LLM typing demo on the
Stretch RE1.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "yolo_model_path",
                default_value="yolov8n.pt",
                description="Path to the YOLO model weights file",
            ),
            Node(
                package="llm_typing_demo",
                executable="object_detection.py",
                name="scene_scanner_node",
                output="screen",
                parameters=[
                    {"yolo_model_path": LaunchConfiguration("yolo_model_path")}
                ],
            ),
            Node(
                package="llm_typing_demo",
                executable="stretch_plunge_grasp.py",
                name="stretch_plunge_grasp_node",
                output="screen",
            ),
            Node(
                package="llm_typing_demo",
                executable="navigate_to.py",
                name="navigate_to_node",
                output="screen",
            ),
        ]
    )
