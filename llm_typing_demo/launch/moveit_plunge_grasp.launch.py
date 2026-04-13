#!/usr/bin/env python3

"""
Kinova Launch file to test moveit plunge grasp
"""

from moveit_configs_utils import MoveItConfigsBuilder
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    robot_ip = LaunchConfiguration("robot_ip").perform(context)
    use_fake_hardware = LaunchConfiguration("use_fake_hardware").perform(context)

    kortex_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("kortex_bringup"),
                        "launch",
                        "gen3_lite.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments={
            "robot_ip": robot_ip,
            "use_fake_hardware": use_fake_hardware,
            "launch_rviz": LaunchConfiguration("rviz").perform(context),
        }.items(),
    )

    moveit_config = (
        MoveItConfigsBuilder(
            "gen3_lite_gen3_lite_2f",
            package_name="kinova_gen3_lite_moveit_config",
        )
        .robot_description(mappings={"use_fake_hardware": use_fake_hardware})
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    moveit_plunge_grasp_node = Node(
        package="llm_typing_demo",
        executable="moveit_plunge_grasp.py",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"planning_pipelines": {"pipeline_names": ["ompl"]}},
            {
                "plan_request_params": {
                    "planning_attempts": 10,
                    "planning_pipeline": "ompl",
                    "planning_time": 5.0,
                    "max_velocity_scaling_factor": 1.0,
                    "max_acceleration_scaling_factor": 1.0,
                }
            },
        ],
    )

    return [kortex_bringup, moveit_plunge_grasp_node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_ip",
                default_value="192.168.1.10",
                description="Robot IP (unused when use_fake_hardware:=true)",
            ),
            DeclareLaunchArgument(
                "use_fake_hardware",
                default_value="true",
                description="Use fake hardware — no physical robot required",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                description="Launch RViz",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
