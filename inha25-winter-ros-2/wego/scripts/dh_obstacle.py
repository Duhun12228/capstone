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
from wego.cfg import PedestrianConfig

class ObstacleNode:
    def __init__(self):
        rospy.init_node('pedestrian_node')
        
        # Parameters (from yaml/cfg)
        self.scan_angle = rospy.get_param('~scan_angle', 20.0)
        self.detect_distance = rospy.get_param('~detect_distance', 0.35)
        self.stop_distance = rospy.get_param('~stop_distance', 0.30)
        self.num_sectors = rospy.get_param('~num_sectors', 8)
        
        # Lane speed from lane_detect_node
        self.lane_speed = 0.3  # default, updated from lane_detect_node
        self.lane_steering = 0.0
        
        # State
        self.obstacle_detected = False
        self.last_state = ""
        
        # Sector data
        self.sector_distances = [10.0] * self.num_sectors
        self.sector_angles = []
        
        # LiDAR data
        self.ranges = None
        self.angle_increment = 0
        
        # CV Bridge for visualization
        self.bridge = CvBridge()
        self.vis_size = 400
        
        # Dynamic reconfigure
        self.srv = Server(PedestrianConfig, self.reconfigure_callback)
        
        self.obstacle_state_pub = rospy.Publisher('/obstacle_state', String, queue_size=1)
        self.pub_debug_image = rospy.Publisher('/webot/pedestrian/debug', Image, queue_size=1)
        
        # Subscribers
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        
        
        self._init_sectors()
        
        rospy.loginfo("="*50)
        rospy.loginfo("Pedestrian Node initialized (Stop & Wait)")
        rospy.loginfo(f"Scan: ±{self.scan_angle}°, Detect: {self.detect_distance}m")
        rospy.loginfo("View: rqt_image_view /webot/pedestrian/debug")
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
    

    def reconfigure_callback(self, config, level):
        self.scan_angle = config.scan_angle
        self.detect_distance = 0.50
        self.stop_distance = 0.30
        self.num_sectors = config.num_sectors
        self._init_sectors()
        rospy.loginfo(f"[Pedestrian] Config updated: scan=±{self.scan_angle}°, detect={self.detect_distance}m")
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
            valid = sector_ranges[(sector_ranges > 0.05) & (sector_ranges < 10.0)]
            
            self.sector_distances[i] = np.min(valid) if len(valid) > 0 else 10.0
    
    def _get_min_front_distance(self):
        return min(self.sector_distances) if self.sector_distances else 10.0
    
    def publish_debug_image(self):
        """LiDAR visualization for rqt_image_view"""
        if self.pub_debug_image.get_num_connections() == 0:
            return
            
        size = self.vis_size
        img = np.zeros((size, size, 3), dtype=np.uint8)
        
        cx, cy = size // 2, size - 50
        
        # Grid circles
        for r_meters in [0.5, 1.0, 1.5]:
            r_pixels = int(r_meters * 150)
            cv2.circle(img, (cx, cy), r_pixels, (40, 40, 40), 1)
            cv2.putText(img, f"{r_meters}m", (cx + r_pixels - 20, cy - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
        
        # Detection distance circle
        detect_r = int(self.detect_distance * 150)
        cv2.circle(img, (cx, cy), detect_r, (0, 100, 100), 2)
        
        # Stop distance circle
        stop_r = int(self.stop_distance * 150)
        cv2.circle(img, (cx, cy), stop_r, (0, 0, 100), 2)
        
        # Sectors
        for i, (angle, dist) in enumerate(zip(self.sector_angles, self.sector_distances)):
            rad = np.radians(90 - angle)
            display_dist = min(dist, 1.5)
            r = int(display_dist * 150)
            
            ex = int(cx + r * np.cos(rad))
            ey = int(cy - r * np.sin(rad))
            
            if dist < self.stop_distance:
                color = (0, 0, 255)  # Red - emergency stop
            elif dist < self.detect_distance:
                color = (0, 165, 255)  # Orange - pedestrian detected
            else:
                color = (0, 255, 0)  # Green - clear
            
            cv2.line(img, (cx, cy), (ex, ey), color, 3)
            if dist < 1.5:
                cv2.circle(img, (ex, ey), 5, color, -1)
        
        # Robot marker
        cv2.circle(img, (cx, cy), 10, (255, 255, 255), -1)
        cv2.putText(img, "R", (cx - 5, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Info panel
        cv2.rectangle(img, (0, 0), (size, 70), (30, 30, 30), -1)
        
        state_text = self.last_state if self.last_state else "IDLE"
        if "STOP" in state_text:
            state_color = (0, 0, 255)
        elif "LANE" in state_text:
            state_color = (0, 255, 0)
        else:
            state_color = (200, 200, 200)
        
        cv2.putText(img, "PEDESTRIAN DETECT", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"State: {state_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)
        
        min_dist = self._get_min_front_distance()
        cv2.putText(img, f"Min Dist: {min_dist:.2f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"Speed: {self.lane_speed:.2f}", (size - 100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"Steer: {self.lane_steering:.3f}", (size - 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        try:
            self.pub_debug_image.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        except Exception as e:
            rospy.logwarn_throttle(5, f"Debug image error: {e}")
    
    def main(self):
        if self.ranges is None:
            return
        
        min_distance = self._get_min_front_distance()
        msg = String()
        if min_distance < self.stop_distance:
            # Emergency Stop
            self.obstacle_detected = True
            self.last_state = "EMERGENCY STOP"
            msg.data = "STOP"
            rospy.loginfo("[Pedestrian] EMERGENCY STOP! Distance: %.2f m", min_distance)
            self.obstacle_state_pub.publish(msg)
        elif min_distance < self.detect_distance:
            # Pedestrian Detected - Slow Down
            self.obstacle_detected = True
            msg.data = "SLOW"
            self.last_state = "PEDESTRIAN DETECTED - SLOW DOWN"
            rospy.loginfo("[Pedestrian] PEDESTRIAN DETECTED! Distance: %.2f m", min_distance)
            self.obstacle_state_pub.publish(msg)
        else:
            # Path Clear - Resume Lane Following
            if self.obstacle_detected:
                self.last_state = "PATH CLEAR - RESUME LANE"
            else:
                self.last_state = "CLEAR"
                msg.data = "CLEAR"
            self.obstacle_detected = False
            rospy.loginfo("[Pedestrian] PATH CLEAR. Distance: %.2f m", min_distance)
            self.obstacle_state_pub.publish(msg)
        
        self.publish_debug_image()

if __name__ == '__main__':
    try:
        on = ObstacleNode()
        rate = rospy.Rate(30)  # 30Hz

        while not rospy.is_shutdown():
            on.main()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass

