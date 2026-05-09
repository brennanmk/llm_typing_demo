#!/usr/bin/env python3
"""
Implements a node that contains an action server that executes a wipe of a surface using the Stretch
arm. Inherits from StretchAlignmentBaseNode.
"""

import rclpy
import tf2_geometry_msgs
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from stretch_alignment_base import StretchAlignmentBaseNode

from llm_typing_demo.action import WipeBoard

_WRIST_YAW = 0.0  # rotation for tool
_WRIST_PITCH = -1.57  # pitch for usage
_WIPE_STEP = 0.04  # distance in meters between wipes
_GRIP_FIRM = -0.35  # grip firmness


class BoardWiperNode(StretchAlignmentBaseNode):
    """Action server that wipes a board (really any surface)."""

    def __init__(self) -> None:
        super().__init__("board_wiper")

        self.declare_parameter(
            "sweep_offset_x", 0.07
        )  # allows for some slop by overwiping
        self.declare_parameter(
            "lift_z_offset", 0.045
        )  # how hard to press into table (meters)
        self.set_parameters(
            [
                rclpy.parameter.Parameter(
                    "max_comfortable_reach", rclpy.Parameter.Type.DOUBLE, 0.50
                )
            ]
        )  # how far away from the table the arm can be to reach

        self._action_server = ActionServer(
            self,
            WipeBoard,
            "wipe_board",
            self.execute_callback,
            callback_group=self._callback_group,
        )

    def execute_callback(self, goal_handle: ServerGoalHandle) -> WipeBoard.Result:
        """Execute a board wipe by sweeping arm across the surface.

        Aligns the base to the detected surface, calculates 
        bounds from the board's dimensions, and drives base
        while moving the arm depth between rows.

        :param goal_handle: action goal handle carrying the target board
        :return: WipeBoard.Result returns success on completion.
        """
        result = WipeBoard.Result()
        board_msg = goal_handle.request.board

        if not self._wait_for_joint_states():
            goal_handle.abort()
            return result

        start_pose = self._get_map_pose()

        original_map_pt = tf2_geometry_msgs.PointStamped()
        original_map_pt.header.frame_id = "map"
        original_map_pt.point.x = board_msg.center_x
        original_map_pt.point.y = board_msg.center_y
        original_map_pt.point.z = board_msg.center_z

        # tighten grip first, then pivot wrist
        cur_wrist_yaw = self._read_joint("joint_wrist_yaw") or 0.0
        safe_lift_z = max(self._read_joint("joint_lift") or 0.5, 0.95)

        self.send_trajectory_sync(
            self.build_trajectory(
                safe_lift_z, 0.0, cur_wrist_yaw, _WRIST_PITCH, _GRIP_FIRM, 4.0
            )
        )
        self.send_trajectory_sync(
            self.build_trajectory(
                safe_lift_z, 0.0, _WRIST_YAW, _WRIST_PITCH, _GRIP_FIRM, 4.0
            )
        )
        cur_gripper = _GRIP_FIRM

        # Alignment to surface
        self.align_and_verify(original_map_pt)

        map_to_base_tf = self.tf_buffer.lookup_transform(
            "base_link", "map", rclpy.time.Time()
        )
        target_in_base_frame = tf2_geometry_msgs.do_transform_point(
            original_map_pt, map_to_base_tf
        )

        grasp_center_tf = self.tf_buffer.lookup_transform(
            "base_link", "link_grasp_center", rclpy.time.Time()
        )
        grasp_center_lateral_offset = grasp_center_tf.transform.translation.y

        extension_needed = max(
            0.0, min(grasp_center_lateral_offset - target_in_base_frame.point.y, 0.50)
        )

        lift_z = (
            target_in_base_frame.point.z + self.get_parameter("lift_z_offset").value
        )
        self.send_trajectory_sync(
            self.build_trajectory(
                lift_z, extension_needed, _WRIST_YAW, _WRIST_PITCH, cur_gripper, 4.0
            )
        )

        width_x = max(
            0.20, board_msg.size_y
        )  # The long edge of the board (Base driving axis)
        depth_y = max(
            0.15, board_msg.size_z
        )  # The short edge of the board (Arm extension axis)
        margin = 0.02
        sweep_offset_x = self.get_parameter("sweep_offset_x").value

        # Base sweep limits
        sweep_start_x = (
            target_in_base_frame.point.x - width_x / 2.0 + margin - sweep_offset_x
        )
        sweep_end_x = (
            target_in_base_frame.point.x + width_x / 2.0 - margin + sweep_offset_x
        )

        # Arm step limits
        near_y = target_in_base_frame.point.y + depth_y / 2.0 - margin
        far_y = target_in_base_frame.point.y - depth_y / 2.0 + margin

        # clamp so we dont go beyond reach
        near_ext = max(0.0, min(grasp_center_lateral_offset - near_y, 0.50))
        far_ext = max(0.0, min(grasp_center_lateral_offset - far_y, 0.50))

        step_dist = abs(far_ext - near_ext)
        n_rows = max(1, int(step_dist / _WIPE_STEP))
        actual_step = step_dist / n_rows if n_rows > 0 else 0.0

        # Perform actual Wipe
        moving_forward = True
        transit_lift_z = lift_z + 0.05
        current_x = target_in_base_frame.point.x

        dist_to_start = sweep_start_x - current_x
        self._drive_base(dist_to_start, vel=0.1)
        current_x = sweep_start_x

        for i in range(n_rows + 1):
            current_ext = near_ext + (i * actual_step)

            # Lift clear of surface and reposition arm to this row's depth
            self.send_trajectory_sync(
                self.build_trajectory(
                    transit_lift_z,
                    current_ext,
                    _WRIST_YAW,
                    _WRIST_PITCH,
                    cur_gripper,
                    3.0,
                )
            )

            # Lower arm onto board surface
            self.send_trajectory_sync(
                self.build_trajectory(
                    lift_z, current_ext, _WRIST_YAW, _WRIST_PITCH, cur_gripper, 3.0
                )
            )

            # Drive base across board width to wipe this row
            target_x = sweep_end_x if moving_forward else sweep_start_x
            dist_to_drive = target_x - current_x
            self._drive_base(dist_to_drive, vel=0.06)
            current_x = target_x

            # Lift clear of surface before next row
            self.send_trajectory_sync(
                self.build_trajectory(
                    transit_lift_z,
                    current_ext,
                    _WRIST_YAW,
                    _WRIST_PITCH,
                    cur_gripper,
                    3.0,
                )
            )

            moving_forward = not moving_forward

        # Move base back home
        self._drive_base(target_in_base_frame.point.x - current_x, vel=0.1)

        self.send_trajectory_sync(
            self.build_trajectory(
                safe_lift_z, 0.0, _WRIST_YAW, _WRIST_PITCH, cur_gripper, 3.0
            )
        )
        self._return_to_pose(start_pose)
        goal_handle.succeed()
        result.success = True
        return result


if __name__ == "__main__":
    rclpy.init()
    executor = MultiThreadedExecutor()
    node = BoardWiperNode()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
