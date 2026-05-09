#!/usr/bin/env python3
"""
Stretch alignment base node.

Common code shared by StretchPlungeGraspNode and BoardWiperNode.
Allows the stretch base to align the arm to an object so it can be
interacted with
"""

import math
import time
import rclpy
import tf2_geometry_msgs
import tf2_ros
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Twist, PointStamped
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


class StretchAlignmentBaseNode(Node):
    """Base node providing TF-based alignment and motion helpers for the Stretch robot.

    Subclasses inherit closed-loop base driving and rotation, joint state
    reading, trajectory helpers, and the shared ``align_and_verify`` routine
    that positions the base in front of a map-frame target point.
    """

    def __init__(self, node_name: str) -> None:
        super().__init__(node_name)
        self.declare_parameter("max_comfortable_reach", 0.60)  # how far can we reach?
        self.declare_parameter(
            "max_lateral_alignment_m", 0.25
        )  # clamping value for forward offset
        self.declare_parameter(
            "visual_parallax_offset_x", 0.03
        )  # x-axis correction due to parallax

        self._latest_joint_state: JointState | None = None

        self._callback_group = ReentrantCallbackGroup()

        # tf setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._cmd_vel_pub = self.create_publisher(Twist, "/stretch/cmd_vel", 10)

        self._trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/stretch_controller/follow_joint_trajectory",
            callback_group=self._callback_group,
        )

        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
            callback_group=self._callback_group,
        )

    def _joint_state_callback(self, msg: JointState) -> None:
        """Cache the latest JointState message.

        :param msg: Incoming JointState message.
        """
        self._latest_joint_state = msg

    def _read_joint(self, name: str) -> float | None:
        """Return the current position of a named joint, or None if unavailable.

        :param name: Joint name as it appears in the JointState message.
        :return: Joint position in radians/metres, or None if no data yet.
        """
        if self._latest_joint_state is None:
            return None
        try:
            return self._latest_joint_state.position[
                self._latest_joint_state.name.index(name)
            ]
        except ValueError:
            return None

    def _read_joint_effort(self, name: str) -> float | None:
        """Return the current effort of a named joint, or None if unavailable.

        :param name: Joint name as it appears in the JointState message.
        :return: Joint effort in Nm, or None if no data yet or effort is empty.
        """
        if self._latest_joint_state is None or not self._latest_joint_state.effort:
            return None
        try:
            return self._latest_joint_state.effort[
                self._latest_joint_state.name.index(name)
            ]
        except ValueError:
            return None

    def _wait_for_joint_states(self, timeout_sec: float = 5.0) -> bool:
        """Block until the first JointState message is received.

        :param timeout_sec: Maximum seconds to wait before giving up.
        :return: True if a message arrived within the timeout, False otherwise.
        """
        steps = int(timeout_sec / 0.05)
        for _ in range(steps):
            if self._latest_joint_state is not None:
                return True
            time.sleep(0.05)
        return False

    def build_trajectory(
        self,
        lift_z: float,
        arm_extension: float,
        wrist_yaw: float,
        wrist_pitch: float,
        gripper_pos: float,
        duration_sec: float = 4.0,
    ) -> FollowJointTrajectory.Goal:
        """Build a FollowJointTrajectory goal.

        :param lift_z: Target lift height in meters.
        :param arm_extension: Target arm extension in meters.
        :param wrist_yaw: Target wrist yaw angle in radians.
        :param wrist_pitch: Target wrist pitch angle in radians.
        :param gripper_pos: Target gripper position.
        :param duration_sec: Time allowed for the trajectory in seconds.
        :return: FollowJointTrajectory.Goal with a single waypoint.
        """
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = [
            "joint_lift",
            "joint_arm",
            "joint_wrist_yaw",
            "joint_wrist_pitch",
            "gripper_aperture",
        ]
        point = JointTrajectoryPoint()
        point.positions = [
            float(lift_z),
            float(arm_extension),
            float(wrist_yaw),
            float(wrist_pitch),
            float(gripper_pos),
        ]
        sec = int(duration_sec)
        nanosec = int((duration_sec - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)
        goal_msg.trajectory.points = [point]
        return goal_msg

    def send_trajectory_sync(self, goal_msg: FollowJointTrajectory.Goal) -> bool:
        """Send a trajectory goal and block until execution completes.

        :param goal_msg: FollowJointTrajectory goal to execute.
        :return: True if the trajectory completed with error_code 0, False otherwise.
        """
        future = self._trajectory_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        gh = future.result()
        if not gh or not gh.accepted:
            return False
        res_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        return res_future.result().result.error_code == 0

    def _get_map_pose(self):
        """Return (x, y, yaw) of base_link in the map frame."""
        while True:
            try:
                tf = self.tf_buffer.lookup_transform("map", "base_link", Time())
                x = tf.transform.translation.x
                y = tf.transform.translation.y
                q = tf.transform.rotation
                yaw = math.atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z),
                )
                return x, y, yaw
            except Exception:
                time.sleep(0.05)

    def _return_to_pose(self, start_pose: tuple) -> None:
        """Return to a previously recorded (x, y, yaw) map pose using closed-loop TF feedback."""
        sx, sy, syaw = start_pose
        self.get_logger().info(
            f"Returning to map pose ({sx:.2f}, {sy:.2f}, yaw={syaw:.2f})"
        )

        # rotate to face the start position
        cx, cy, cyaw = self._get_map_pose()
        angle_to_target = math.atan2(sy - cy, sx - cx)
        rotate_by = (angle_to_target - cyaw + math.pi) % (2 * math.pi) - math.pi
        self._rotate_base(rotate_by)

        # drive the distance to the start position
        cx, cy, _ = self._get_map_pose()
        dist = math.sqrt((sx - cx) ** 2 + (sy - cy) ** 2)
        if dist > 0.02:
            self._drive_base(dist)

        # rotate to the original yaw
        _, _, cyaw = self._get_map_pose()
        rotate_by = (syaw - cyaw + math.pi) % (2 * math.pi) - math.pi
        self._rotate_base(rotate_by)

    def _rotate_base(self, rotate_by: float, ang_vel: float = 0.3) -> None:
        """Rotate the base by a angle using odom feedback.

        :param rotate_by: Rotation angle in radians (positive = counterclockwise).
        :param ang_vel: Angular velocity in rad/s.
        """

        def get_yaw():
            tf = self.tf_buffer.lookup_transform("odom", "base_link", Time())
            q = tf.transform.rotation
            return math.atan2(
                2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )

        while True:
            try:
                start_yaw = get_yaw()
                break
            except Exception:
                time.sleep(0.05)

        target = abs(rotate_by)
        direction = 1.0 if rotate_by > 0 else -1.0
        accumulated, prev_yaw = 0.0, start_yaw

        msg = Twist()
        msg.angular.z = direction * ang_vel
        while accumulated < target:
            self._cmd_vel_pub.publish(msg)
            time.sleep(0.05)
            try:
                yaw = get_yaw()
                delta = (yaw - prev_yaw + math.pi) % (2 * math.pi) - math.pi
                accumulated += abs(delta)
                prev_yaw = yaw
            except Exception:
                pass
        self._cmd_vel_pub.publish(Twist())
        time.sleep(0.3)

    def _drive_base(self, distance_m: float, vel: float = 0.08) -> None:
        """Drive the base forward or backward by a distance using odom feedback.

        We use odom here because its faster than map

        :param distance_m: Distance in meters (positive = forward, negative = backward).
        :param vel: velocity in m/s.
        """

        def get_pos():
            tf = self.tf_buffer.lookup_transform("odom", "base_link", Time())
            return tf.transform.translation.x, tf.transform.translation.y

        while True:
            try:
                prev_x, prev_y = get_pos()
                break
            except Exception:
                time.sleep(0.05)

        target = abs(distance_m)
        direction = 1.0 if distance_m > 0 else -1.0
        accumulated = 0.0

        msg = Twist()
        msg.linear.x = direction * vel
        while accumulated < target:
            self._cmd_vel_pub.publish(msg)
            time.sleep(0.05)
            try:
                cx, cy = get_pos()
                accumulated += math.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2)
                prev_x, prev_y = cx, cy
            except Exception:
                pass
        self._cmd_vel_pub.publish(Twist())
        time.sleep(0.3)

    def align_and_verify(self, target_map_pt: PointStamped, feedback_cb=None) -> bool:
        """Align the base in front of a map-frame target point.

        First we approach if the target is beyond reach, then we apply
        base rotation to face the target with the arm side, and
        finally we move laterally to center the arm.

        :param target_map_pt: Target position as a PointStamped in the map frame.
        :param feedback_cb: Optional callable that receives a human-readable
            status string at each alignment step, e.g. for action feedback.
        :return: True if alignment succeeded, False on error.
        """

        max_reach = self.get_parameter("max_comfortable_reach").value
        max_lat_align = self.get_parameter("max_lateral_alignment_m").value

        try:
            map_to_base_tf = self.tf_buffer.lookup_transform(
                "base_link", "map", rclpy.time.Time()
            )
            target_in_base_frame = tf2_geometry_msgs.do_transform_point(
                target_map_pt, map_to_base_tf
            )
            grasp_center_tf = self.tf_buffer.lookup_transform(
                "base_link", "link_grasp_center", rclpy.time.Time()
            )
            grasp_center_x = grasp_center_tf.transform.translation.x
            grasp_center_lateral_offset = abs(grasp_center_tf.transform.translation.y)
        except Exception as e:
            self.get_logger().error(f"TF Error: {e}")
            return False

        dist_to_target = math.sqrt(
            target_in_base_frame.point.x**2 + target_in_base_frame.point.y**2
        )
        estimated_arm_extension = dist_to_target - grasp_center_lateral_offset

        # if too far away to reach the object, drive closer first
        if estimated_arm_extension > max_reach:
            approach_dist = min(estimated_arm_extension - max_reach, 0.5)
            if feedback_cb:
                feedback_cb(f"Pre-approaching by {approach_dist:.2f}m")
            self._rotate_base(
                math.atan2(target_in_base_frame.point.y, target_in_base_frame.point.x)
            )
            self._drive_base(approach_dist)

            try:
                map_to_base_tf = self.tf_buffer.lookup_transform(
                    "base_link", "map", rclpy.time.Time()
                )
                target_in_base_frame = tf2_geometry_msgs.do_transform_point(
                    target_map_pt, map_to_base_tf
                )
                grasp_center_tf = self.tf_buffer.lookup_transform(
                    "base_link", "link_grasp_center", rclpy.time.Time()
                )
                grasp_center_x = grasp_center_tf.transform.translation.x
                dist_to_target = math.sqrt(
                    target_in_base_frame.point.x**2 + target_in_base_frame.point.y**2
                )
            except Exception as e:
                self.get_logger().error(f"TF Error: {e}")
                return False

        # Rotate base so arm side faces the target
        if feedback_cb:
            feedback_cb("Rotating base toward target")
        grasp_center_angle_offset = (
            math.asin(grasp_center_x / dist_to_target)
            if dist_to_target > abs(grasp_center_x)
            else 0.0
        )
        rotation_to_face_target = (
            math.atan2(target_in_base_frame.point.y, target_in_base_frame.point.x)
            + (math.pi / 2.0)
            - grasp_center_angle_offset
        )
        self._rotate_base(rotation_to_face_target)

        # Drive laterally to center grasp frame on target x-axis
        if feedback_cb:
            feedback_cb("Aligning base laterally")
        try:
            map_to_base_post_rotation_tf = self.tf_buffer.lookup_transform(
                "base_link", "map", rclpy.time.Time()
            )
            target_in_base_frame_post_rotation = tf2_geometry_msgs.do_transform_point(
                target_map_pt, map_to_base_post_rotation_tf
            )
        except Exception:
            return False

        parallax_shift = self.get_parameter("visual_parallax_offset_x").value
        lateral_correction = max(
            -max_lat_align,
            min(
                max_lat_align,
                target_in_base_frame_post_rotation.point.x
                - grasp_center_x
                + parallax_shift,
            ),
        )
        if abs(lateral_correction) > 0.02:
            self._drive_base(lateral_correction)

        return True
