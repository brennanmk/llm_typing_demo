#!/usr/bin/env python3

"""
Stretch RE1 Bringup Launch File, starts bringup, navstack, rtab, lidar
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    stretch_core_path = get_package_share_directory("stretch_core")
    stretch_rtabmap_path = get_package_share_directory("stretch_rtabmap")
    stretch_nav2_path = get_package_share_directory("stretch_nav2")

    stretch_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [stretch_core_path, "/launch/stretch_driver.launch.py"]
        ),
        launch_arguments={"mode": "navigation", "broadcast_odom_tf": "True"}.items(),
    )

    d435i_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [stretch_core_path, "/launch/d435i_high_resolution.launch.py"]
        ),
    )

    rplidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([stretch_core_path, "/launch/rplidar.launch.py"]),
    )

    rtabmap_node = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        arguments=["-d"],
        remappings=[
            ("rgb/image", "/camera/color/image_raw"),
            ("depth/image", "/camera/aligned_depth_to_color/image_raw"),
            ("rgb/camera_info", "/camera/color/camera_info"),
            ("grid_map", "map"),
        ],
        output="screen",
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                get_package_share_directory("nav2_bringup"),
                "/launch/navigation_launch.py",
            ]
        ),
        launch_arguments={
            "params_file": stretch_nav2_path + "/config/nav2_params.yaml",
            "use_sim_time": "false",
        }.items(),
    )

    cmd_vel_relay = Node(
        package="topic_tools",
        executable="relay",
        name="cmd_vel_relay",
        arguments=["/cmd_vel", "/stretch/cmd_vel"],
        output="screen",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        condition=IfCondition(LaunchConfiguration("use_rviz")),
        respawn=True,
        arguments=["-d", stretch_rtabmap_path + "/rviz/rtabmap.rviz"],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_rviz", default_value="false", choices=["true", "false"]
            ),
            stretch_driver_launch,
            d435i_launch,
            rplidar_launch,
            rtabmap_node,
            nav2_launch,
            cmd_vel_relay,
            rviz_node,
        ]
    )
