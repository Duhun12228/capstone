#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Obstacle Avoidance Node for WEGO (Gap Finding Algorithm)
- Uses LiDAR to scan front ±30 degrees
- Divides into sectors and finds the best gap
- Steers toward the widest open gap
- Returns to lane tracing when path is clear
"""

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import String, Bool, Float32
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
from dynamic_reconfigure.server import Server
from wego.cfg import ObstacleAvoidConfig


class ObstacleAvoidNode:
    def __init__(self):
        rospy.init_node('obstacle_avoid_node')
        
        # Parameters (from yaml/cfg)
        self.scan_angle = rospy.get_param('~scan_angle', 30.0)
        self.safe_distance = rospy.get_param('~safe_distance', 0.5)
        self.stop_distance = rospy.get_param('~stop_distance', 0.2)
        self.num_sectors = rospy.get_param('~num_sectors', 12)
        
        # Lane speed from lane_detect_node
        self.lane_speed = 0.3  # default, updated from lane_detect_node
        self.avoid_speed = rospy.get_param('~avoid_speed', 0.2)
        self.max_steering = rospy.get_param('~max_steering', 0.5)
        self.steering_gain = rospy.get_param('~steering_gain', 0.02)
        
        # State
        self.avoiding = False
        self.clear_count = 0
        self.clear_threshold = rospy.get_param('~clear_threshold', 20)
        self.last_state = ""
        
        # Sector data
        self.sector_distances = [10.0] * self.num_sectors
        self.sector_angles = []
        
        # Lane steering from lane_detect_node
        self.lane_steering = 0.0
        
        # LiDAR data
        self.ranges = None
        self.angle_increment = 0
        
        # CV Bridge for visualization
        self.bridge = CvBridge()
        self.vis_size = 400
        
        # Dynamic reconfigure
        self.srv = Server(ObstacleAvoidConfig, self.reconfigure_callback)
        
        # Publishers - AckermannDriveStamped for direct motor control
        self.cmd_vel_pub = rospy.Publisher('/low_level/ackermann_cmd_mux/input/navigation', 
                                            AckermannDriveStamped, queue_size=1)
        self.state_pub = rospy.Publisher('/webot/obstacle/state', String, queue_size=1)
        self.avoiding_pub = rospy.Publisher('/webot/obstacle/is_avoiding', Bool, queue_size=1)
        self.pub_debug_image = rospy.Publisher('/webot/obstacle/debug', Image, queue_size=1)
        
        # Subscribers
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        self.sub_steering = rospy.Subscriber('/webot/steering_offset', Float32, self.steering_callback, queue_size=1)
        self.sub_speed = rospy.Subscriber('/webot/lane_speed', Float32, self.speed_callback, queue_size=1)
        
        # Timers
        self.timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)
        self.vis_timer = rospy.Timer(rospy.Duration(0.1), self.visualization_callback)
        
        self._init_sectors()
        
        rospy.loginfo("="*50)
        rospy.loginfo("Obstacle Avoid Node initialized (Gap Finding)")
        rospy.loginfo(f"Scan: ±{self.scan_angle}°, Safe: {self.safe_distance}m")
        rospy.loginfo("View: rqt_image_view /webot/obstacle/debug")
        rospy.loginfo("="*50)
    
    def _init_sectors(self):
        """Initialize sector center angles"""
        total_angle = self.scan_angle * 2
        sector_size = total_angle / self.num_sectors
        
        self.sector_angles = []
        for i in range(self.num_sectors):
            angle = self.scan_angle - sector_size * (i + 0.5)
            self.sector_angles.append(angle)
        
        self.sector_distances = [10.0] * self.num_sectors
    
    def steering_callback(self, msg):
        self.lane_steering = msg.data
    
    def speed_callback(self, msg):
        """Speed callback from lane_detect_node"""
        self.lane_speed = msg.data
    
    def reconfigure_callback(self, config, level):
        self.scan_angle = config.scan_angle
        self.safe_distance = config.safe_distance
        self.stop_distance = config.stop_distance
        self.num_sectors = config.num_sectors
        self.avoid_speed = config.avoid_speed
        self.max_steering = config.max_steering
        self.steering_gain = config.steering_gain
        self.clear_threshold = config.clear_threshold
        self._init_sectors()
        rospy.loginfo(f"[Obstacle] Config updated: scan=±{self.scan_angle}°, safe={self.safe_distance}m, clear={self.clear_threshold}")
        return config
    
    def scan_callback(self, msg):
        self.ranges = np.array(msg.ranges)
        self.angle_increment = msg.angle_increment
        self._calculate_sector_distances()
    
    def _calculate_sector_distances(self):
        if self.ranges is None or self.angle_increment == 0:
            return
        
        total_points = len(self.ranges)
        # laser_link가 180도 회전되어 있으므로 인덱스 0이 앞쪽
        center_idx = 0
        points_per_degree = 1.0 / np.degrees(self.angle_increment)
        sector_size_deg = (self.scan_angle * 2) / self.num_sectors
        
        for i in range(self.num_sectors):
            start_angle = -self.scan_angle + sector_size_deg * i
            end_angle = start_angle + sector_size_deg
            
            start_idx = int(center_idx + start_angle * points_per_degree)
            end_idx = int(center_idx + end_angle * points_per_degree)
            
            start_idx = max(0, min(start_idx, total_points - 1))
            end_idx = max(0, min(end_idx, total_points - 1))
            
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            
            sector_ranges = self.ranges[start_idx:end_idx + 1]
            valid = sector_ranges[(sector_ranges > 0.01) & (sector_ranges < 10.0)]
            
            self.sector_distances[i] = np.min(valid) if len(valid) > 0 else 10.0
    
    def _find_best_gap(self):
        """Find the best gap (widest open space)"""
        is_open = [d >= self.safe_distance for d in self.sector_distances]
        
        gaps = []
        gap_start = None
        
        for i, open_sector in enumerate(is_open):
            if open_sector and gap_start is None:
                gap_start = i
            elif not open_sector and gap_start is not None:
                gaps.append((gap_start, i - 1))
                gap_start = None
        
        if gap_start is not None:
            gaps.append((gap_start, len(is_open) - 1))
        
        if not gaps:
            max_idx = np.argmax(self.sector_distances)
            return self.sector_angles[max_idx], 1
        
        best_gap = max(gaps, key=lambda g: g[1] - g[0])
        gap_start_idx, gap_end_idx = best_gap
        gap_width = gap_end_idx - gap_start_idx + 1
        
        center_idx = (gap_start_idx + gap_end_idx) / 2.0
        if center_idx == int(center_idx):
            gap_center_angle = self.sector_angles[int(center_idx)]
        else:
            low_idx = int(center_idx)
            high_idx = min(low_idx + 1, len(self.sector_angles) - 1)
            ratio = center_idx - low_idx
            gap_center_angle = (self.sector_angles[low_idx] * (1 - ratio) + 
                               self.sector_angles[high_idx] * ratio)
        
        return gap_center_angle, gap_width
    
    def _has_obstacle_in_front(self):
        center_sectors = self.num_sectors // 3
        start = (self.num_sectors - center_sectors) // 2
        end = start + center_sectors
        front_distances = self.sector_distances[start:end]
        return min(front_distances) < self.safe_distance if front_distances else False
    
    def _get_min_front_distance(self):
        center_sectors = self.num_sectors // 3
        start = (self.num_sectors - center_sectors) // 2
        end = start + center_sectors
        front_distances = self.sector_distances[start:end]
        return min(front_distances) if front_distances else 10.0
    
    def publish_debug_image(self):
        """LiDAR sector visualization for rqt_image_view"""
        if self.pub_debug_image.get_num_connections() == 0:
            return
            
        size = self.vis_size
        img = np.zeros((size, size, 3), dtype=np.uint8)
        
        cx, cy = size // 2, size - 50
        
        # Grid circles
        for r_meters in [0.5, 1.0, 1.5, 2.0]:
            r_pixels = int(r_meters * 100)
            cv2.circle(img, (cx, cy), r_pixels, (40, 40, 40), 1)
            cv2.putText(img, f"{r_meters}m", (cx + r_pixels - 20, cy - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
        
        # Safe distance circle
        safe_r = int(self.safe_distance * 100)
        cv2.circle(img, (cx, cy), safe_r, (0, 100, 100), 2)
        
        # Stop distance circle
        stop_r = int(self.stop_distance * 100)
        cv2.circle(img, (cx, cy), stop_r, (0, 0, 100), 2)
        
        # Sectors
        for i, (angle, dist) in enumerate(zip(self.sector_angles, self.sector_distances)):
            rad = np.radians(90 - angle)
            display_dist = min(dist, 2.0)
            r = int(display_dist * 100)
            
            ex = int(cx + r * np.cos(rad))
            ey = int(cy - r * np.sin(rad))
            
            if dist >= self.safe_distance:
                color = (0, 255, 0)  # Green - open
            elif dist >= self.stop_distance:
                color = (0, 165, 255)  # Orange - caution
            else:
                color = (0, 0, 255)  # Red - stop
            
            cv2.line(img, (cx, cy), (ex, ey), color, 2)
            if dist < 2.0:
                cv2.circle(img, (ex, ey), 4, color, -1)
        
        # Best gap arrow
        gap_angle, gap_width = self._find_best_gap()
        gap_rad = np.radians(90 - gap_angle)
        gap_len = 150
        gap_x = int(cx + gap_len * np.cos(gap_rad))
        gap_y = int(cy - gap_len * np.sin(gap_rad))
        cv2.arrowedLine(img, (cx, cy), (gap_x, gap_y), (255, 255, 0), 3, tipLength=0.2)
        
        # Robot marker
        cv2.circle(img, (cx, cy), 8, (255, 255, 255), -1)
        cv2.putText(img, "R", (cx - 4, cy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        # Info panel
        cv2.rectangle(img, (0, 0), (size, 70), (30, 30, 30), -1)
        
        state_text = self.last_state if self.last_state else "IDLE"
        if "AVOIDING" in state_text:
            state_color = (0, 255, 255)  # Cyan
        elif "LANE" in state_text:
            state_color = (0, 255, 0)  # Green
        elif "TOO_CLOSE" in state_text:
            state_color = (0, 0, 255)  # Red
        else:
            state_color = (200, 200, 200)
        
        cv2.putText(img, "OBSTACLE AVOID", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"State: {state_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)
        cv2.putText(img, f"Gap: {gap_angle:.1f}deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        min_dist = self._get_min_front_distance()
        cv2.putText(img, f"Min: {min_dist:.2f}m", (size - 100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"Speed: {self.lane_speed:.2f}", (size - 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        try:
            self.pub_debug_image.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        except Exception as e:
            rospy.logwarn_throttle(5, f"Debug image error: {e}")
    
    def visualization_callback(self, event):
        if self.ranges is not None:
            self.publish_debug_image()
    
    def control_loop(self, event):
        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header.stamp = rospy.Time.now()
        ackermann_msg.header.frame_id = "base_link"
        
        state = "IDLE"
        
        # Too close - stop
        if self._get_min_front_distance() < self.stop_distance:
            ackermann_msg.drive.speed = 0.0
            ackermann_msg.drive.steering_angle = 0.0
            state = "TOO_CLOSE"
            self.avoiding = True
            self.clear_count = 0
        
        # Obstacle in front - start avoiding
        elif self._has_obstacle_in_front():
            self.avoiding = True
            self.clear_count = 0
            
            gap_angle, gap_width = self._find_best_gap()
            steering = -self.steering_gain * gap_angle
            steering = np.clip(steering, -self.max_steering, self.max_steering)
            
            ackermann_msg.drive.speed = self.avoid_speed
            ackermann_msg.drive.steering_angle = steering
            state = f"AVOIDING (gap={gap_angle:.1f})"
        
        # Avoiding mode - continue avoiding while checking LiDAR
        elif self.avoiding:
            # 계속 LiDAR로 장애물 확인하며 회피 진행
            gap_angle, _ = self._find_best_gap()
            steering = -self.steering_gain * gap_angle
            steering = np.clip(steering, -self.max_steering, self.max_steering)
            
            ackermann_msg.drive.speed = self.avoid_speed
            ackermann_msg.drive.steering_angle = steering
            
            # 장애물이 없는 상태가 지속되면 clear_count 증가
            self.clear_count += 1
            
            if self.clear_count >= self.clear_threshold:
                # 충분히 오래 장애물 없음 -> lane tracing 복귀
                self.avoiding = False
                self.clear_count = 0
                ackermann_msg.drive.speed = self.lane_speed
                ackermann_msg.drive.steering_angle = self.lane_steering
                state = "LANE_TRACING"
            else:
                state = f"AVOIDING_CLEAR ({self.clear_count}/{self.clear_threshold})"
        
        # Clear - lane tracing
        else:
            ackermann_msg.drive.speed = self.lane_speed
            ackermann_msg.drive.steering_angle = self.lane_steering
            state = "LANE_TRACING"
        
        if state != self.last_state:
            rospy.loginfo(f"[Obstacle] {state}")
            self.last_state = state
        
        self.cmd_vel_pub.publish(ackermann_msg)
        self.state_pub.publish(String(data=state))
        self.avoiding_pub.publish(Bool(data=self.avoiding))
        
        rospy.loginfo_throttle(1, f"[Obstacle] {state} | Speed: {ackermann_msg.drive.speed:.2f}")
    
    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = ObstacleAvoidNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
