import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped

import cv2
from cv_bridge import CvBridge
import numpy as np
import torch
from Perception.model.unet import UNet
from Perception.evaluate import evaluate

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from .rotation_utils import transform_pose

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

device = "cpu"
 
class Perception(Node):
    """Perception Package: Eyes of the autonomous vehicle
    
    Responsibilities:
    - Subscribe to camera feed (/oakd/rgb/image_raw)
    - Use U-Net to segment drivable lane areas
    - Compute lane center and generate dynamic target waypoint
    - Publish target to /pose_msg for controller
    """
    
    def __init__(self, trained_model="/home/xin/Final_Project/src/perception_project/Perception/model.pt"):
        super().__init__('Perception')

        # Debug visualization parameters
        self.declare_parameter('show_debug_viz', True)
        self.declare_parameter('overlay_alpha', 0.65)
        self.declare_parameter('watch_pixel_row', 160)
        self.declare_parameter('line_spread_px', 100)  # Reduced spread to avoid bias
        
        # DISABLE adaptive line shifting - it causes early turn-in during left turns
        self.declare_parameter('enable_adaptive_lines', False)
        self.declare_parameter('line_shift_gain', 0.0)  # Disabled
        
        # Increased lookahead to reduce perception bias in turns
        self.declare_parameter('lookahead_distance', 5.0)  # meters ahead

        # Smoothing for lateral offset
        self.smoothed_target_y = 0.0
        self.smoothing_alpha = 0.35  # Lower = smoother (prevents jitter in turns)
        
        self.image_subscription = self.create_subscription(
            Image,
            '/oakd/rgb/image_raw',
            self.image_callback,
            10)
        
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            '/pose_msg', 
            10)

        # tf buffer for coordinate transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.br = CvBridge()

        # Load trained U-Net model
        save_dict = torch.load(trained_model, map_location=device)
        self.model = UNet()
        self.model.load_state_dict(save_dict["model"])
        self.model.eval()
        self.pose_msg = PoseStamped()
        
        # Track recent lane centers for curvature estimation
        self.recent_centers = []
        self.max_history = 5

        self.get_logger().info('👁️  Perception node started - providing dynamic targets')

        self.turn_bias = 0.0
        self.turn_bias_decay = 0.94   # slow decay

    def calculate_path(self, current_frame, pred):
        """Compute lane center with adaptive detection for curves
        
        Process:
        1. Find left and right lane boundaries in segmentation mask
        2. Calculate lane center
        3. Use curvature history to shift detection lines adaptively
        4. Publish target waypoint at lookahead distance
        """
        
        pred = (pred > 0).astype(np.uint8)
        h, w = pred.shape
        current_frame = cv2.resize(current_frame, (w, h), cv2.INTER_AREA)

        # --- Estimate lane curvature from recent centers ---
        enable_adaptive = bool(self.get_parameter('enable_adaptive_lines').value)
        line_shift_gain = float(self.get_parameter('line_shift_gain').value)
        
        # Calculate left/right lane positions
        min_row = 50  # Ignore top portion of image
        left_mask = pred[min_row:, :w // 2]
        right_mask = pred[min_row:, w // 2:]

        left_xs = np.where(left_mask > 0)[1]
        left_middle = float(np.mean(left_xs)) if left_xs.size > 0 else 0.0

        right_xs = np.where(right_mask > 0)[1]
        right_middle = float(w // 2 + np.mean(right_xs)) if right_xs.size > 0 else float(w - 1)

        left_count = left_xs.size
        right_count = right_xs.size

        dominance_ratio = 0.65  # tune 0.6–0.7

        dominance_ratio = 0.65
        bias_strength = 0.65 * (w // 2)

        if left_count > dominance_ratio * (left_count + right_count):
            # Left turn → bias right
            self.turn_bias = +bias_strength
        elif right_count > dominance_ratio * (left_count + right_count):
            # Right turn → bias left
            self.turn_bias = -bias_strength
        else:
            # No clear dominance → decay bias slowly
            self.turn_bias *= self.turn_bias_decay

        middle = 0.5 * (left_middle + right_middle) + self.turn_bias

        
        # Track center history for curvature estimation
        self.recent_centers.append(middle)
        if len(self.recent_centers) > self.max_history:
            self.recent_centers.pop(0)
        
        # Estimate curvature: rate of change in lane center
        curvature_shift = 0.0
        if enable_adaptive and len(self.recent_centers) >= 3:
            # Positive = curving right, negative = curving left
            curvature_shift = self.recent_centers[-1] - self.recent_centers[-2]
            curvature_shift = float(np.clip(curvature_shift, -w * 0.15, w * 0.15))
        
        # --- Apply adaptive shift to detection lines ---
        # This helps anticipate curves by shifting detection in direction of turn
        adaptive_offset = int(curvature_shift * line_shift_gain)
        
        spread = int(self.get_parameter('line_spread_px').value)

        # Position detection lines
        left_line  = int(left_middle  + adaptive_offset - spread)
        right_line = int(right_middle + adaptive_offset + spread)
        center_line = int(middle + adaptive_offset)

        # Clamp to image bounds
        left_line   = int(np.clip(left_line,   0, w - 1))
        center_line = int(np.clip(center_line, 0, w - 1))
        right_line  = int(np.clip(right_line,  0, w - 1))

        # Calculate lateral offset with smoothing
        raw_y = (w // 2 - center_line) / 10.0
        max_step = 0.45  # meters per frame
        delta = raw_y - self.smoothed_target_y
        delta = np.clip(delta, -max_step, max_step)
        self.smoothed_target_y += delta


        # --- Debug visualization ---
        show_debug = bool(self.get_parameter('show_debug_viz').value)
        if show_debug:
            overlay_alpha = float(self.get_parameter('overlay_alpha').value)
            overlay_alpha = float(np.clip(overlay_alpha, 0.0, 1.0))

            # Blue mask overlay
            overlay = current_frame.copy()
            mask = pred.astype(bool)
            overlay[mask] = (255, 0, 0)
            current_frame = cv2.addWeighted(overlay, overlay_alpha, current_frame, 1.0 - overlay_alpha, 0.0)

            # Crosshair at image center
            cx = w // 2
            cy = h // 2
            cv2.line(current_frame, (cx, 0), (cx, h - 1), (255, 255, 255), 1)
            cv2.line(current_frame, (0, cy), (w - 1, cy), (255, 255, 255), 1)

            # Draw adaptive detection lines
            cv2.line(current_frame, (left_line, 0), (left_line, h - 1), (255, 0, 0), 2)    # blue
            cv2.line(current_frame, (center_line, 0), (center_line, h - 1), (0, 0, 255), 2)  # red
            cv2.line(current_frame, (right_line, 0), (right_line, h - 1), (0, 255, 0), 2)    # green

            cv2.imshow("Perception: Lane Detection", current_frame)
            cv2.waitKey(1)

        # --- Publish dynamic target waypoint ---
        lookahead = float(self.get_parameter('lookahead_distance').value)
        
        self.pose_msg.header.frame_id = "base_link"
        self.pose_msg.header.stamp = self.get_clock().now().to_msg()
        self.pose_msg.pose.position.x = lookahead
        self.pose_msg.pose.position.y = self.smoothed_target_y
        self.pose_msg.pose.position.z = 0.0
        self.pose_msg.pose.orientation.w = 1.0

        self.transform_and_publish_pose(self.pose_msg)

    def image_callback(self, msg: Image):
        """Process incoming camera frame"""
        current_frame = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        segmentation_mask, _ = evaluate(self.model, current_frame)
        self.calculate_path(current_frame, segmentation_mask)
    
    def transform_and_publish_pose(self, pose_msg: PoseStamped):
        """Transform target from base_link to world frame and publish
        
        If transform fails, publish in base_link frame - controller will handle it
        """
        try:
            t = self.tf_buffer.lookup_transform(
                "world",
                pose_msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
            pose_msg.pose = transform_pose(pose_msg.pose, t)
            pose_msg.header.frame_id = "world"
        except TransformException as ex:
            # Transform not available - publish in base_link frame instead
            # Controller will handle body frame transformation
            self.get_logger().info(
                f"Transform unavailable, publishing in base_link frame: {ex}",
                throttle_duration_sec=5.0
            )
        
        # Always publish, even if transform failed
        self.pose_publisher.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    perception = Perception()
    rclpy.spin(perception)
    perception.destroy_node()
    rclpy.shutdown()
 
if __name__ == '__main__':
    main()