#!/usr/bin/env python3
"""
Implements a node that consists of an action server. This action
server implements a scan scene action, which pans the camera and uses
YOLO to detect all objects.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionServer, ActionClient
from rclpy.action.server import ServerGoalHandle
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PointStamped
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import cv2
import tf2_ros
import tf2_geometry_msgs
from cv_bridge import CvBridge
import image_geometry
import numpy as np
import time
from ultralytics import YOLO
from llm_typing_demo.action import ScanScene
from llm_typing_demo.msg import ObjectDetection


class SceneScannerNode(Node):
    """Action server that pans the camera and runs YOLO to detect all objects in the scene."""

    def __init__(self) -> None:
        super().__init__("scene_scanner_node")

        self.declare_parameter("yolo_model_path", "yolov8n.pt")
        self.declare_parameter(
            "camera_info_topic", "/camera/aligned_depth_to_color/camera_info"
        )
        self.declare_parameter("color_topic", "/camera/color/image_raw/compressed")
        self.declare_parameter(
            "depth_topic", "/camera/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter("sync_slop", 0.5)
        self.yolo_model = YOLO(
            self.get_parameter("yolo_model_path").get_parameter_value().string_value
        )

        self.cv_bridge = CvBridge()
        self.camera_model = image_geometry.PinholeCameraModel()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.camera_info_received = False
        self.latest_color = None
        self.latest_depth = None

        self._callback_group = ReentrantCallbackGroup()
        self._sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Camera subscriptions are created during scan goal,
        # constantly subbing to these topics saturates the network.
        self._camera_info_sub = None
        self._color_sub = None
        self._depth_sub = None
        self._time_synchronizer = None

        self._action_server = ActionServer(
            self,
            ScanScene,
            "scan_scene",
            self.execute_callback,
            callback_group=self._callback_group,
        )
        self._head_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/stretch_controller/follow_joint_trajectory",
            callback_group=self._callback_group,
        )

    def _subscribe_to_camera(self) -> None:
        """Create synced camera subscriptions."""
        camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        color_topic = (
            self.get_parameter("color_topic").get_parameter_value().string_value
        )
        depth_topic = (
            self.get_parameter("depth_topic").get_parameter_value().string_value
        )
        sync_slop = self.get_parameter("sync_slop").get_parameter_value().double_value

        if not self.camera_info_received:
            self._camera_info_sub = self.create_subscription(
                CameraInfo,
                camera_info_topic,
                self.info_callback,
                self._sensor_qos,
                callback_group=self._callback_group,
            )

        color_sub = message_filters.Subscriber(
            self,
            CompressedImage,
            color_topic,
            qos_profile=self._sensor_qos,
            callback_group=self._callback_group,
        )

        depth_sub = message_filters.Subscriber(
            self,
            Image,
            depth_topic,
            qos_profile=self._sensor_qos,
            callback_group=self._callback_group,
        )

        self._time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=10,
            slop=sync_slop,
        )

        self._time_synchronizer.registerCallback(self.sync_callback)

        # track subs so we can destroy
        self._color_sub = color_sub
        self._depth_sub = depth_sub

    def _unsubscribe_from_camera(self) -> None:
        """Destroy camera subscriptions."""
        if self._color_sub:
            self.destroy_subscription(self._color_sub.sub)
            self._color_sub = None
        if self._depth_sub:
            self.destroy_subscription(self._depth_sub.sub)
            self._depth_sub = None
        self._time_synchronizer = None

    def info_callback(self, msg: CameraInfo) -> None:
        """Initialise the pinhole camera model from the first CameraInfo message received.
        Destroys itself after first received message

        :param msg: ROS CameraInfo message from the depth-aligned colour camera.
        """
        if not self.camera_info_received:
            self.get_logger().debug("image info received")
            self.camera_model.fromCameraInfo(msg)
            self.camera_info_received = True
            self.destroy_subscription(self._camera_info_sub)
            self._camera_info_sub = None

    def sync_callback(self, color_msg: CompressedImage, depth_msg: Image) -> None:
        """Cache the latest synchronised color/depth image pair.

        :param color_msg: Compressed BGR colour image.
        :param depth_msg: Depth image aligned to the colour frame (uint16, mm).
        """
        self.get_logger().debug("Sync image received")
        self.latest_color, self.latest_depth = color_msg, depth_msg

    async def execute_callback(self, goal_handle: ServerGoalHandle) -> ScanScene.Result:
        """Pan the head through three waypoints, run YOLO on each frame, and tracks detections.

        Claude Sonnet 4.6 was used to help, namely with the camera linear algebra.
        
        :param goal_handle: ROS 2 action goal handle for the ScanScene action.
        :return: ScanScene.Result containing all detected ObjectDetection messages.
        """
        result = ScanScene.Result()
        waypoints = [(-0.5, -0.2), (0.0, -0.2), (0.5, -0.2)]
        home = (0.0, -0.2)  # home is slightly looking down

        self._subscribe_to_camera()
        try:
            for idx, (pan, tilt) in enumerate(waypoints):
                self.latest_color, self.latest_depth = None, None

                # Move Head
                self.get_logger().info(
                    f"Waypoint {idx}: moving head to pan={pan}, tilt={tilt}"
                )
                goal_msg = FollowJointTrajectory.Goal()
                goal_msg.trajectory.joint_names = ["joint_head_pan", "joint_head_tilt"]
                goal_msg.trajectory.points = [
                    JointTrajectoryPoint(
                        positions=[pan, tilt], time_from_start=Duration(sec=1)
                    )
                ]
                waypoint_goal_handle = await self._head_client.send_goal_async(goal_msg)
                await waypoint_goal_handle.get_result_async()
                self.get_logger().info(
                    f"Waypoint {idx}: head move sent, waiting for image..."
                )

                # Wait for sync
                wait_count = 0
                while self.latest_color is None:
                    time.sleep(0.1)
                    wait_count += 1
                    if wait_count % 20 == 0:
                        self.get_logger().warn(
                            f"Waypoint {idx}: still waiting for synced image ({wait_count * 0.1:.1f}s)..."
                        )

                self.get_logger().info(
                    f"Waypoint {idx}: image received, running YOLO..."
                )

                # decode compressed image, turn to cv2
                cv_img = cv2.imdecode(
                    np.frombuffer(self.latest_color.data, np.uint8), cv2.IMREAD_COLOR
                )
                depth_cv = self.cv_bridge.imgmsg_to_cv2(
                    self.latest_depth, "passthrough"
                )

                # yolo inference
                yolo_results = self.yolo_model(cv_img, conf=0.5, verbose=False)
                self.get_logger().info(
                    f"Waypoint {idx}: YOLO found {len(yolo_results[0].boxes) if yolo_results else 0} detections"
                )
                if not yolo_results:
                    continue

                for box in yolo_results[0].boxes:
                    class_name = self.yolo_model.names[int(box.cls[0].item())]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    centroid_u, centroid_v = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    # find depth at centroid
                    centroid_depth_m = (
                        np.median(
                            depth_cv[
                                max(0, centroid_v - 2) : centroid_v + 3,
                                max(0, centroid_u - 2) : centroid_u + 3,
                            ]
                        )
                        / 1000.0  # div by 1000 to convert to meters
                    )
                    if centroid_depth_m <= 0.1 or centroid_depth_m > 3.0:
                        continue

                    # Transform Centroid to map frame
                    ray_center = self.camera_model.projectPixelTo3dRay(
                        (centroid_u, centroid_v)
                    )
                    point_in_camera = PointStamped()
                    point_in_camera.header = self.latest_depth.header
                    (
                        point_in_camera.point.x,
                        point_in_camera.point.y,
                        point_in_camera.point.z,
                    ) = (
                        ray_center[0] * centroid_depth_m,
                        ray_center[1] * centroid_depth_m,
                        ray_center[2] * centroid_depth_m,
                    )

                    try:
                        transform = self.tf_buffer.lookup_transform(
                            "map",
                            point_in_camera.header.frame_id,
                            point_in_camera.header.stamp,
                            rclpy.duration.Duration(seconds=0.5),
                        )
                        point_in_map = tf2_geometry_msgs.do_transform_point(
                            point_in_camera, transform
                        )
                    except Exception:
                        continue

                    # find the corners in 3d space to get size
                    ray_top_left = self.camera_model.projectPixelTo3dRay((x1, y1))
                    ray_bottom_right = self.camera_model.projectPixelTo3dRay((x2, y2))

                    width_m = abs(
                        (ray_bottom_right[0] * centroid_depth_m)
                        - (ray_top_left[0] * centroid_depth_m)
                    )
                    height_m = abs(
                        (ray_bottom_right[1] * centroid_depth_m)
                        - (ray_top_left[1] * centroid_depth_m)
                    )

                    # create object detection message
                    obj = ObjectDetection()
                    obj.class_name = class_name
                    obj.center_x, obj.center_y, obj.center_z = (
                        point_in_map.point.x,
                        point_in_map.point.y,
                        point_in_map.point.z,
                    )
                    obj.size_y = width_m
                    obj.size_z = height_m
                    result.detected_objects.append(obj)

        finally:
            self._unsubscribe_from_camera()

        # Return head to home
        self.get_logger().info("Returning head to home position")
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ["joint_head_pan", "joint_head_tilt"]
        goal_msg.trajectory.points = [
            JointTrajectoryPoint(positions=list(home), time_from_start=Duration(sec=1))
        ]
        home_goal_handle = await self._head_client.send_goal_async(goal_msg)
        await home_goal_handle.get_result_async()

        goal_handle.succeed()
        return result


if __name__ == "__main__":
    rclpy.init()
    node = SceneScannerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
