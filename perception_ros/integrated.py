#!/usr/bin/env python3
"""Integrated Controller: Pure Pursuit + PID + Perception
Following project architecture:
- Perception: Provides dynamic target waypoints from lane detection
- Pure Pursuit: Calculates steering angle to track dynamic target
- PID: Calculates velocity based on error and steering angle (reduces speed in turns)
"""

import math
import csv
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray, PoseStamped

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class IntegratedController(Node):
    def __init__(self):
        super().__init__('integrated_controller')

        # Parameters
        self.declare_parameter('pose_index', 1)

        # PID parameters - velocity control based on error AND steering
        self.declare_parameter('kp', 0.65)  # Proportional gain
        self.declare_parameter('ki', 0.00)  # Integral gain
        self.declare_parameter('kd', 0.45)  # Derivative gain
        self.declare_parameter('max_velocity', 4.0)  # Maximum velocity
        self.declare_parameter('min_velocity', 0.8)  # Minimum velocity
        
        # Steering-based velocity reduction parameters
        self.declare_parameter('velocity_reduction_factor', 0.75) 
        self.declare_parameter('velocity_reduction_threshold', 0.08) 

        # Pure Pursuit parameters
        self.declare_parameter('wheelbase', 1.86)
        self.declare_parameter('max_steering_angle', 0.61)

        self.declare_parameter('perception_in_body_frame', True)

        # Read parameters
        self.pose_index = int(self.get_parameter('pose_index').value)
        self.kp = float(self.get_parameter('kp').value)
        self.ki = float(self.get_parameter('ki').value)
        self.kd = float(self.get_parameter('kd').value)
        self.max_velocity = float(self.get_parameter('max_velocity').value)
        self.min_velocity = float(self.get_parameter('min_velocity').value)
        self.velocity_reduction_factor = float(self.get_parameter('velocity_reduction_factor').value)
        self.velocity_reduction_threshold = float(self.get_parameter('velocity_reduction_threshold').value)
        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.max_steering_angle = float(self.get_parameter('max_steering_angle').value)
        self.perception_in_body_frame = bool(self.get_parameter('perception_in_body_frame').value)

        # Turn commitment state
        self.turn_sign = 0              # -1 left, +1 right, 0 none
        self.turn_commit_until = 0.0    # time until which we keep turn direction
        self.turn_commit_time = 0.6     # seconds to "finish" a hard turn
        self.turn_enter_deg = 8.0       # start commit when |steer| > this
        self.turn_release_deg = 4.0     # allow release when |steer| < this

        self.prev_steer = 0.0
        self.max_steer_rate = math.radians(120.0)  # rad/s, tune 90-180

        # State
        self.current_pose = None
        self.target_msg: Optional[PoseStamped] = None
        self.prev_error: Optional[float] = None
        self.prev_time: Optional[float] = None
        self.integral = 0.0
        self.start_time = None
        self.startup_duration = 3.0  # seconds
        

        # Logging
        self.time_stamps = []
        self.positions_x = []
        self.positions_y = []
        self.velocities = []
        self.steering_angles = []
        self.errors = []
        self.target_x_log = []
        self.target_y_log = []

        # ROS interfaces
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_sub = self.create_subscription(PoseArray, '/pose_info', self.pose_callback, 10)
        self.target_sub = self.create_subscription(PoseStamped, '/pose_msg', self.target_callback, 10)
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info('=' * 60)
        self.get_logger().info('🚗 Integrated Controller: Pure Pursuit + PID + Perception')
        self.get_logger().info('Architecture: Perception → Dynamic Target → Pure Pursuit (steering) + PID (velocity)')
        self.get_logger().info(f'PID: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}')
        self.get_logger().info(f'Velocity: [{self.min_velocity}, {self.max_velocity}] m/s | Aggressive reduction: {self.velocity_reduction_factor}')
        self.get_logger().info(f'Reduction threshold: {math.degrees(self.velocity_reduction_threshold):.1f}°')
        self.get_logger().info(f'Wheelbase: {self.wheelbase}m | Max steering: {math.degrees(self.max_steering_angle):.1f}°')
        self.get_logger().info('=' * 60)

    def pose_callback(self, msg: PoseArray):
        if not msg.poses:
            return
        idx = self.pose_index if self.pose_index < len(msg.poses) else 0
        self.current_pose = msg.poses[idx]
        if self.start_time is None:
            self.start_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info('✓ Vehicle pose received - starting control')

    def target_callback(self, msg: PoseStamped):
        """Receive dynamic target from Perception package"""
        self.target_msg = msg

    def get_current_position(self) -> Tuple[Optional[float], Optional[float]]:
        if self.current_pose is None:
            return None, None
        return float(self.current_pose.position.x), float(self.current_pose.position.y)

    def get_current_yaw(self) -> float:
        if self.current_pose is None:
            return 0.0
        q = self.current_pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return float(math.atan2(siny_cosp, cosy_cosp))

    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def transform_body_to_world(car_x: float, car_y: float, car_yaw: float,
                               x_body: float, y_body: float) -> Tuple[float, float]:
        c = math.cos(car_yaw)
        s = math.sin(car_yaw)
        x_w = car_x + x_body * c - y_body * s
        y_w = car_y + x_body * s + y_body * c
        return x_w, y_w

    def calculate_steering_angle(self, car_x: float, car_y: float, car_yaw: float,
                                 target_x: float, target_y: float) -> Tuple[float, float]:
        """Pure Pursuit: Calculate steering angle to track dynamic target from Perception"""
        dx = target_x - car_x
        dy = target_y - car_y
        target_angle = math.atan2(dy, dx)
        alpha = self.normalize_angle(target_angle - car_yaw)
        ld = math.hypot(dx, dy)
        
        if ld < 0.1:
            steer = 0.0
        else:
            # Pure Pursuit formula: δ = arctan(2L sin(α) / ld)
            steer = math.atan2(2.0 * self.wheelbase * math.sin(alpha), ld)
        
        steer = float(np.clip(steer, -self.max_steering_angle, self.max_steering_angle))
        return steer, ld

    def calculate_target_velocity(self, error: float, dt: float, steering_angle: float) -> float:
        """PID velocity control with more aggressive steering-based reduction
        
        Key improvements:
        1. Much lower threshold for velocity reduction (starts at 3° instead of 4.6°)
        2. Exponential reduction curve for sharper deceleration
        3. Lower minimum speed multiplier
        """
        v = self.max_velocity

        if self.prev_error is None or dt <= 1e-3:
            return 1.5  # safe startup speed
        
        # --- PID calculation ---
        p = self.kp * error

        if self.ki != 0.0:
            self.integral += error * dt
            self.integral = float(np.clip(self.integral, -10.0, 10.0))
            i = self.ki * self.integral
        else:
            i = 0.0

        if self.prev_error is None or dt <= 1e-3:
            d = 0.0
        else:
            d = self.kd * (error - self.prev_error) / dt
        
        pid_out = p + i + d
        pid_out = float(np.clip(pid_out, -0.3, 0.3))

        # --- Steering-limited base speed (THIS IS KEY) ---
        abs_steer_deg = abs(math.degrees(steering_angle))

        if abs_steer_deg < 6.0:
            base_velocity = 3.6
        elif abs_steer_deg < 10.0:
            base_velocity = 2.6
        elif abs_steer_deg < 15.0:
            base_velocity = 1.9
        else:
            base_velocity = 1.3

        v = base_velocity + pid_out
        v = float(np.clip(v, self.min_velocity, self.max_velocity))
        
        return float(v)


    def control_loop(self):

        now = self.get_clock().now().nanoseconds * 1e-9

        if self.start_time is None:
            return

        elapsed = now - self.start_time
        startup_mode = elapsed < self.startup_duration

        startup_mode = elapsed < self.startup_duration

        if self.current_pose is None:
            self.get_logger().warning('Waiting for vehicle pose...', throttle_duration_sec=2.0)
            return

        if self.target_msg is None:
            self.get_logger().warning('Waiting for perception target...', throttle_duration_sec=2.0)
            return

        car_x, car_y = self.get_current_position()
        car_yaw = self.get_current_yaw()
        if car_x is None or car_y is None:
            return

        # Get dynamic target from Perception
        tx = float(self.target_msg.pose.position.x)
        ty = float(self.target_msg.pose.position.y)
        frame = (self.target_msg.header.frame_id or '').lower().strip()

        now = self.get_clock().now().nanoseconds / 1e9

        if self.start_time is None:
            return

        elapsed = now - self.start_time
        startup_mode = elapsed < self.startup_duration

        target_in_body = self.perception_in_body_frame
        if frame in ('world', 'map', 'odom'):
            target_in_body = False
        
        if startup_mode:
            # Virtual straight-ahead target in BODY frame
            tx = 5.0
            ty = 0.0
            target_in_body = True

        if target_in_body:
            target_x, target_y = self.transform_body_to_world(car_x, car_y, car_yaw, tx, ty)
        else:
            target_x, target_y = tx, ty

        # Calculate error (distance to target)
        error = math.hypot(target_x - car_x, target_y - car_y)

        # Get current time and calculate dt
        now = self.get_clock().now().nanoseconds / 1e9
        if self.prev_time is None:
            self.prev_time = now
            self.prev_error = error
            return

        dt = now - self.prev_time
        if dt <= 1e-4:
            return

        # Pure Pursuit: Calculate steering angle from dynamic target
        steering_angle, ld = self.calculate_steering_angle(car_x, car_y, car_yaw, target_x, target_y)

        # Boost steering when target error is large (curvature commitment)
        if error > 7.0:
            steering_angle *= 1.35
        elif error > 12.0:
            steering_angle *= 1.55

        steering_angle = float(np.clip(
            steering_angle,
            -self.max_steering_angle,
            self.max_steering_angle
        ))

        now = self.get_clock().now().nanoseconds / 1e9
        steer_deg = math.degrees(steering_angle)

        # --- Turn commitment / anti-straightening logic ---
        # Start a commit when we enter a strong turn
        if abs(steer_deg) > self.turn_enter_deg and self.turn_sign == 0:
            self.turn_sign = 1 if steering_angle > 0 else -1
            self.turn_commit_until = now + self.turn_commit_time

        max_delta = self.max_steer_rate * dt
        steering_angle = float(np.clip(
            steering_angle,
            self.prev_steer - max_delta,
            self.prev_steer + max_delta
        ))
        self.prev_steer = steering_angle

        # While committed, prevent the steering from flipping sign too early
        if now < self.turn_commit_until and self.turn_sign != 0:
            # If perception suddenly asks for opposite steering (straightening or reversal),
            # clamp it to stay in the committed direction (but allow it to reduce magnitude).
            if steering_angle * self.turn_sign < 0:
                steering_angle = 0.0  # don't reverse; at worst go neutral

        # Release commit when we're past time AND steering has settled
        if now >= self.turn_commit_until and abs(steer_deg) < self.turn_release_deg:
            self.turn_sign = 0

        # PID: Calculate velocity based on error AND steering angle
        target_velocity = self.calculate_target_velocity(error, dt, steering_angle)

        # Publish combined command
        cmd = Twist()
        cmd.linear.x = float(target_velocity)
        cmd.angular.z = float(steering_angle)
        self.cmd_vel_pub.publish(cmd)

        # Logging
        if self.start_time is not None:
            elapsed = now - self.start_time
            self.time_stamps.append(elapsed)
            self.positions_x.append(car_x)
            self.positions_y.append(car_y)
            self.velocities.append(target_velocity)
            self.steering_angles.append(steering_angle)
            self.errors.append(error)
            self.target_x_log.append(target_x)
            self.target_y_log.append(target_y)

        if self.time_stamps and (len(self.time_stamps) % 40 == 0):
            self.get_logger().info(
                f'Position: ({car_x:.1f}, {car_y:.1f}) | '
                f'Target: ({target_x:.1f}, {target_y:.1f}) | '
                f'Error: {error:.2f}m | v: {target_velocity:.2f} | '
                f'Steering Angle: {math.degrees(steering_angle):.1f}° | '
            )

        self.prev_error = error
        self.prev_time = now

    def save_results(self):
        if len(self.time_stamps) < 10:
            self.get_logger().warning('Not enough data to save')
            return

        csv_file = 'integrated_controller_results.csv'
        try:
            with open(csv_file, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['time', 'x', 'y', 'target_x', 'target_y', 'velocity', 'steering_angle', 'error'])
                for i in range(len(self.time_stamps)):
                    w.writerow([
                        self.time_stamps[i], self.positions_x[i], self.positions_y[i],
                        self.target_x_log[i], self.target_y_log[i], self.velocities[i],
                        self.steering_angles[i], self.errors[i],
                    ])
            self.get_logger().info(f'Saved results to {csv_file}')
        except Exception as e:
            self.get_logger().error(f'Failed to save CSV: {e}')

        # Plots
        try:
            steering_deg = [math.degrees(s) for s in self.steering_angles]

            plt.figure(figsize=(10, 8))
            plt.plot(self.positions_x, self.positions_y, 'b-', linewidth=2, label='Vehicle Path')
            plt.plot(self.target_x_log, self.target_y_log, 'r--', alpha=0.6, label='Target Path')
            plt.xlabel('X (m)', fontsize=12)
            plt.ylabel('Y (m)', fontsize=12)
            plt.title('Trajectory', fontsize=14, fontweight='bold')
            plt.axis('equal')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('integrated_trajectory.png', dpi=200)
            plt.close()

            plt.figure(figsize=(12, 4))
            plt.plot(self.time_stamps, self.errors, 'b-', linewidth=1.5)
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Error (m)', fontsize=12)
            plt.title('Distance Error', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('integrated_error.png', dpi=200)
            plt.close()

            plt.figure(figsize=(12, 4))
            plt.plot(self.time_stamps, self.velocities, 'b-', linewidth=1.5)
            plt.axhline(self.max_velocity, color='r', linestyle='--', alpha=0.5, label=f'Max: {self.max_velocity} m/s')
            plt.axhline(self.min_velocity, color='orange', linestyle='--', alpha=0.5, label=f'Min: {self.min_velocity} m/s')
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Velocity (m/s)', fontsize=12)
            plt.title('Velocity Command (PID + Aggressive Steering Reduction)', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('integrated_velocity.png', dpi=200)
            plt.close()

            plt.figure(figsize=(12, 4))
            plt.plot(self.time_stamps, steering_deg, 'b-', linewidth=1.5)
            plt.axhline(math.degrees(self.max_steering_angle), color='r', linestyle='--', alpha=0.5)
            plt.axhline(-math.degrees(self.max_steering_angle), color='r', linestyle='--', alpha=0.5)
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Steering (deg)', fontsize=12)
            plt.title('Steering Command (Pure Pursuit)', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('integrated_steering.png', dpi=200)
            plt.close()

            self.get_logger().info('=' * 60)
            self.get_logger().info('📊 Integrated Controller Summary')
            self.get_logger().info(f'Mean Error: {float(np.mean(self.errors)):.3f}m')
            self.get_logger().info(f'Max Error: {float(np.max(self.errors)):.3f}m')
            self.get_logger().info(f'Min Error: {float(np.min(self.errors)):.3f}m')
            self.get_logger().info(f'Mean Velocity: {float(np.mean(self.velocities)):.3f} m/s')
            self.get_logger().info(f'Max Velocity: {float(np.max(self.velocities)):.3f} m/s')
            self.get_logger().info(f'Min Velocity: {float(np.min(self.velocities)):.3f} m/s')
            self.get_logger().info(f'Mean |Steering|: {float(np.mean([abs(s) for s in steering_deg])):.2f}°')
            self.get_logger().info(f'Max |Steering|: {float(np.max([abs(s) for s in steering_deg])):.2f}°')
            self.get_logger().info(f'Total Time: {self.time_stamps[-1]:.2f}s')
            self.get_logger().info('=' * 60)

        except Exception as e:
            self.get_logger().error(f'Failed to generate plots: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = IntegratedController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt')
    finally:
        node.get_logger().info('Shutting down...')
        node.save_results()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()