#!/usr/bin/env python3
"""
Object detection node.

Runs DINO and SAM2 on synchronised color and depth frames, publishes
DetectionBatch messages.
"""

import time

import cv2
import image_geometry
import numpy as np
import rclpy
import tf2_geometry_msgs
import tf2_ros
import torch
from geometry_msgs.msg import PointStamped
from PIL import Image as PILImage
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_srvs.srv import SetBool
from std_msgs.msg import Header
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)

from llm_typing_demo.msg import DetectionBatch, ObjectDetection

# map of int rotations to cv2 enums
_ROTATION_TABLE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

# map rotation degrees to inverse coordinate transform: (u, v, unrotated_h, unrotated_w) -> (u_unrotated, v_unrotated)
_INVERSE_ROTATION_TABLE = {
    0: lambda u, v, h, w: (int(u), int(v)),
    90: lambda u, v, h, w: (int(v), int(h - 1 - u)),
    180: lambda u, v, h, w: (int(w - 1 - u), int(h - 1 - v)),
    270: lambda u, v, h, w: (int(w - 1 - v), int(u)),
}


class SceneScannerNode(Node):
    """Detects objects using Grounding DINO + SAM2 and publishes DetectionBatch.

    Grounding DINO provides bounding boxes from a text prompt
    SAM2 produces segmentation from those boxes. Publishes raw
    DetectionBatch messages to /detection_batch on each frame capture.
    """

    def __init__(self) -> None:
        """
        Declares the following ROS parameters:

        * camera_info_topic - CameraInfo topic.
        * color_topic - compressed color image topic.
        * depth_topic - compressedDepth image topic.
        * belief_sample_period - seconds between detections.
        * image_rotation - clockwise rotation of image in degrees (0 / 90 / 180 / 270)
        * gdino_model - Hugging Face model ID for DINO.
        * sam_model - Hugging Face model ID for SAM2.
        * detection_classes - comma-separated list of classes to detect.
        * box_threshold - minimum box confidence for DINO.
        * text_threshold - minimum text confidence for DINO.
        """
        super().__init__("scene_scanner_node")

        self.declare_parameter(
            "camera_info_topic", "/camera/aligned_depth_to_color/camera_info"
        )
        self.declare_parameter("color_topic", "/camera/color/image_raw/compressed")
        self.declare_parameter(
            "depth_topic", "/camera/aligned_depth_to_color/image_raw/compressedDepth"
        )
        self.declare_parameter("belief_sample_period", 2.0)
        self.declare_parameter("image_rotation", 0)
        self.declare_parameter("gdino_model", "IDEA-Research/grounding-dino-base")
        self.declare_parameter("sam_model", "facebook/sam2.1-hiera-large")
        self.declare_parameter(
            "detection_classes",
            "eraser,whiteboard,keyboard,cup,cloth",
        )
        self.declare_parameter("box_threshold", 0.35)
        self.declare_parameter("text_threshold", 0.25)

        # use a gpu if we have one
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self._device}")

        self.camera_model = image_geometry.PinholeCameraModel()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # cached image information initialized to empty
        self.camera_info_received = False
        self.latest_color: CompressedImage | None = None
        self.latest_depth: CompressedImage | None = None

        self._latest_depth_recv_time: float = 0.0
        self._sampling_active = False
        self._paused = False

        self._callback_group = ReentrantCallbackGroup()
        self._reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._batch_pub = self.create_publisher(DetectionBatch, "/detection_batch", 10)
        self.create_service(
            SetBool,
            "pause_detection",
            self._handle_pause_detection,
            callback_group=self._callback_group,
        )

        sample_period = (
            self.get_parameter("belief_sample_period")
            .get_parameter_value()
            .double_value
        )

        camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        color_topic = (
            self.get_parameter("color_topic").get_parameter_value().string_value
        )
        depth_topic = (
            self.get_parameter("depth_topic").get_parameter_value().string_value
        )

        self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.info_callback,
            self._reliable_qos,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            CompressedImage,
            color_topic,
            self._color_callback,
            self._image_qos,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            CompressedImage,
            depth_topic,
            self._depth_callback,
            self._image_qos,
            callback_group=self._callback_group,
        )

        self.create_timer(sample_period, self._sample_timer_callback)

        # Load models after subscriptions are established so we get frames while waiting for load.
        gdino_model_id = (
            self.get_parameter("gdino_model").get_parameter_value().string_value
        )
        self.get_logger().info(f"Loading Grounding DINO: {gdino_model_id}")
        self._gdino_processor = AutoProcessor.from_pretrained(gdino_model_id)
        self._gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            gdino_model_id
        ).to(self._device)

        sam_model_id = (
            self.get_parameter("sam_model").get_parameter_value().string_value
        )
        self.get_logger().info(f"Loading SAM2: {sam_model_id}")
        self._sam2_processor = Sam2Processor.from_pretrained(sam_model_id)
        self._sam2_model = Sam2Model.from_pretrained(sam_model_id).to(self._device)
        self.get_logger().info("object_detection_node ready.")

    def _handle_pause_detection(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        """Pause or resume the periodic detection timer.

        :param request: SetBool.Request where data=True pauses and data=False resumes.
        :param response: SetBool.Response with success and a status message.
        :return: Populated SetBool.Response.
        """
        self._paused = request.data
        self.get_logger().info(f"Detection {'paused' if self._paused else 'resumed'}.")
        response.success = True
        response.message = "paused" if self._paused else "resumed"
        return response

    def info_callback(self, msg: CameraInfo) -> None:
        """Initialise the camera model from the first CameraInfo message received.

        :param msg: Incoming CameraInfo message.
        """
        if not self.camera_info_received:
            self.get_logger().info("Camera info received")
            self.camera_model.fromCameraInfo(msg)
            self.camera_info_received = True

    def _color_callback(self, msg: CompressedImage) -> None:
        """Cache the latest compressed color frame.

        :param msg: Incoming compressed color image.
        """
        self.latest_color = msg

    def _depth_callback(self, msg: CompressedImage) -> None:
        """Cache the latest compressedDepth frame.

        :param msg: Incoming compressedDepth image.
        """
        self.latest_depth = msg
        self._latest_depth_recv_time = time.monotonic()

    @staticmethod
    def _decode_compressed_depth(msg: CompressedImage) -> np.ndarray:
        """Decode a compressedDepth message to a uint16 numpy array.

        :param msg: Incoming compressedDepth message.
        :return: Decoded depth image as a uint16 numpy array (values in mm).
        """
        buf = np.frombuffer(msg.data[12:], dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

    def _build_text_prompt(self) -> str:
        """Build the period-separated text prompt expected by Grounding DINO.

        :return: Prompt string of the form "class_a . class_b . class_c .".
        """
        classes = (
            self.get_parameter("detection_classes").get_parameter_value().string_value
        )
        return " . ".join(c.strip() for c in classes.split(",")) + " ."

    def _run_gdino(
        self, rgb_img: np.ndarray
    ) -> tuple[list[list[float]], list[str], list[float]]:
        """Run Grounding DINO on an RGB image.

        :return: Tuple of (boxes_xyxy, labels, scores).
        """
        box_thresh = (
            self.get_parameter("box_threshold").get_parameter_value().double_value
        )
        text_thresh = (
            self.get_parameter("text_threshold").get_parameter_value().double_value
        )
        pil_img = PILImage.fromarray(rgb_img)
        inputs = self._gdino_processor(
            images=pil_img, text=self._build_text_prompt(), return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self._gdino_model(**inputs)

        img_h, img_w = rgb_img.shape[:2]
        results = self._gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_thresh,
            text_threshold=text_thresh,
            target_sizes=[(img_h, img_w)],
        )

        boxes = [box.tolist() for box in results[0]["boxes"]]
        labels = results[0]["text_labels"]
        scores = [float(score) for score in results[0]["scores"]]
        return boxes, labels, scores

    def _run_sam2(
        self, rgb_img: np.ndarray, boxes: list[list[float]]
    ) -> list[np.ndarray]:
        """Run SAM2 with bounding boxes and return one mask per box.

        :param rgb_img: RGB image as a numpy array.
        :param boxes: List of [x1, y1, x2, y2] boxes from Grounding DINO.
        :return: List of boolean numpy arrays of shape (H, W), one per box.
            An all-False mask is returned for any box SAM2 could not segment.
        """
        pil_img = PILImage.fromarray(rgb_img)
        inputs = self._sam2_processor(
            images=pil_img,
            input_boxes=[boxes],
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._sam2_model(**inputs, multimask_output=False)

        masks_tensor = self._sam2_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]

        return [masks_tensor[i, 0].numpy() for i in range(len(boxes))]

    def _process_frame(
        self,
        cv_img: np.ndarray,
        depth_cv: np.ndarray,
        depth_header: Header,
        rotation: int = 0,
    ) -> list[tuple[ObjectDetection, float]]:
        """Run Grounding DINO + SAM2 on one frame.

        For each detection the median depth within the SAM2 mask is
        used as the representative depth.  The pixel centroid is taken
        over mask pixels near that median depth and back-projected via
        the pinhole camera model before being transformed into the
        odom frame via TF.

        :param cv_img: BGR image as a numpy array.
        :param depth_cv: Aligned depth image.
        :param depth_header: ROS header from the depth image, used for the TF
            timestamp when transforming points to odom.
        :param rotation: Clockwise rotation in degrees applied to the images so
            pixel coordinates can be unrotated back to the original camera frame.
        :return: List of (ObjectDetection in odom frame, confidence) tuples
            for all successfully projected detections.
        """
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        boxes, labels, scores = self._run_gdino(rgb_img)

        if not self.camera_info_received:
            self.get_logger().warn(
                "Waiting for camera info — skipping 3D projection",
                throttle_duration_sec=5.0,
            )
            return []

        if not boxes:
            return []

        mask_list = self._run_sam2(rgb_img, boxes)

        img_h, img_w = cv_img.shape[:2]
        if rotation in (90, 270):
            unrotated_h, unrotated_w = img_w, img_h
        else:
            unrotated_h, unrotated_w = img_h, img_w

        inverse_rotate = _INVERSE_ROTATION_TABLE.get(
            rotation, _INVERSE_ROTATION_TABLE[0]
        )

        results = []

        for box, label, score, mask in zip(boxes, labels, scores, mask_list):
            if not mask.any():
                continue

            x1, y1, x2, y2 = box

            depth_vals = depth_cv[mask].astype(np.float32)
            valid_depths = depth_vals[(depth_vals > 100) & (depth_vals < 3000)]

            if len(valid_depths) == 0:
                self.get_logger().warn(
                    f"{label}: no valid depth in mask", throttle_duration_sec=2.0
                )
                continue

            # get depth from the object's center
            median_depth_mm = np.percentile(valid_depths, 50)
            centroid_depth_m = float(median_depth_mm) / 1000.0

            # find centroid of pixels at the median depth (with 15 mm slop)
            depth_matched_mask = (
                mask
                & (depth_cv > 0)
                & (np.abs(depth_cv.astype(np.float32) - median_depth_mm) < 15.0)
            )
            depth_matched_coords = np.argwhere(depth_matched_mask)

            if len(depth_matched_coords) > 0:
                centroid_v = int(np.mean(depth_matched_coords[:, 0]))
                centroid_u = int(np.mean(depth_matched_coords[:, 1]))
            else:
                # Fallback to full mask centroid if depth filtering removes everything.
                mask_coords = np.argwhere(mask)
                centroid_v = int(np.mean(mask_coords[:, 0]))
                centroid_u = int(np.mean(mask_coords[:, 1]))

            centroid_u = max(0, min(img_w - 1, centroid_u))
            centroid_v = max(0, min(img_h - 1, centroid_v))

            centroid_u_unrotated, centroid_v_unrotated = inverse_rotate(
                centroid_u, centroid_v, unrotated_h, unrotated_w
            )
            ray_center = self.camera_model.projectPixelTo3dRay(
                (centroid_u_unrotated, centroid_v_unrotated)
            )

            # projectPixelTo3dRay returns a unit vector. We need to scale this.
            scale = centroid_depth_m / ray_center[2]
            point_in_camera = PointStamped()
            point_in_camera.header = depth_header
            point_in_camera.point.x = ray_center[0] * scale
            point_in_camera.point.y = ray_center[1] * scale
            point_in_camera.point.z = centroid_depth_m

            try:
                transform = self.tf_buffer.lookup_transform(
                    "map",
                    point_in_camera.header.frame_id,
                    point_in_camera.header.stamp,
                    rclpy.duration.Duration(seconds=0.5),
                )
                point_in_odom = tf2_geometry_msgs.do_transform_point(
                    point_in_camera, transform
                )
            except Exception as e:
                self.get_logger().warn(
                    f"{label}: TF lookup failed: {e}", throttle_duration_sec=2.0
                )
                continue

            # u = column (x-axis), v = row (y-axis) in image coordinates
            top_left_u_unrotated, top_left_v_unrotated = inverse_rotate(
                x1, y1, unrotated_h, unrotated_w
            )
            bottom_right_u_unrotated, bottom_right_v_unrotated = inverse_rotate(
                x2, y2, unrotated_h, unrotated_w
            )
            ray_top_left = self.camera_model.projectPixelTo3dRay(
                (top_left_u_unrotated, top_left_v_unrotated)
            )
            ray_bottom_right = self.camera_model.projectPixelTo3dRay(
                (bottom_right_u_unrotated, bottom_right_v_unrotated)
            )
            width_m = abs((ray_bottom_right[0] - ray_top_left[0]) * scale)
            height_m = abs((ray_bottom_right[1] - ray_top_left[1]) * scale)

            detection = ObjectDetection()
            detection.class_name = label
            detection.confidence = score
            detection.center_x = point_in_odom.point.x
            detection.center_y = point_in_odom.point.y
            detection.center_z = point_in_odom.point.z
            detection.size_y = width_m
            detection.size_z = height_m
            results.append((detection, score))

        return results

    def _publish_batch(
        self, detections: list[tuple[ObjectDetection, float]], depth_header: Header
    ) -> None:
        """Publish a DetectionBatch message to ``/detection_batch``.

        :param detections: List of ``(ObjectDetection, confidence)`` tuples.
        :param depth_header: Header from the source depth image, forwarded as
            the batch header so subscribers can correlate timestamps.
        """
        batch = DetectionBatch()
        batch.header = depth_header
        batch.detections = [det for det, _ in detections]
        self._batch_pub.publish(batch)

    def _sample_timer_callback(self) -> None:
        """Periodically capture one color & depth frame pair and run detection.

        Skips the current tick if a previous detection run is still in progress
        """
        if self._paused or self._sampling_active:
            return

        self._sampling_active = True
        try:
            self.get_logger().debug("Sampler: capturing frame...")

            if self.latest_color is None or self.latest_depth is None:
                self.get_logger().warn(
                    f"Sampler: no frame yet - "
                    f"color={'ok' if self.latest_color else 'missing'} "
                    f"depth={'ok' if self.latest_depth else 'missing'}"
                )
                return

            depth_age = time.monotonic() - self._latest_depth_recv_time
            if depth_age > 2.0:
                self.get_logger().warn(
                    f"Sampler: depth image is {depth_age:.1f}s old - skipping (camera not publishing?)"
                )
                return

            cv_img = cv2.imdecode(
                np.frombuffer(self.latest_color.data, np.uint8), cv2.IMREAD_COLOR
            )
            depth_cv = self._decode_compressed_depth(self.latest_depth)
            depth_header = self.latest_depth.header

            rotation = (
                self.get_parameter("image_rotation").get_parameter_value().integer_value
            )
            rotation_code = _ROTATION_TABLE.get(rotation)
            if rotation_code is not None:
                cv_img = cv2.rotate(cv_img, rotation_code)
                depth_cv = cv2.rotate(depth_cv, rotation_code)

            self.get_logger().debug("Sampler: running inference...")
            detections = self._process_frame(cv_img, depth_cv, depth_header, rotation)
            self.get_logger().info(
                f"Sampler: inference done, {len(detections)} detections"
            )
            self._publish_batch(detections, depth_header)
        finally:
            self._sampling_active = False


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
