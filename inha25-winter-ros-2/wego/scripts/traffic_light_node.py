#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traffic Light Detection Node for WEGO
- HSV-based red/green light detection
- Subscribes to steering/speed from lane_detect_node
- Only intervenes when RED light detected (stops)
- GREEN or UNKNOWN: continues lane tracing normally
"""

import rospy
import cv2
import numpy as np
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String, Bool, Float32
from ackermann_msgs.msg import AckermannDriveStamped
from dynamic_reconfigure.server import Server
from wego.cfg import TrafficLightConfig


class TrafficLightNode:
    def __init__(self):
        rospy.init_node('traffic_light_node', anonymous=True)
        
        self.bridge = CvBridge()
        self.current_state = "UNKNOWN"
        
        # Enable flag for sequence control
        self.enabled = True
        self.green_passed = False
        
        # Lane tracing parameters (from lane_detect_node)
        self.lane_steering = 0.0
        self.lane_speed = 0.3  # default, will be updated from lane_detect
        
        # Load fisheye calibration
        self.camera_matrix, self.dist_coeffs = self._load_calibration()
        
        # Dynamic Reconfigure
        self.config = None
        self.srv = Server(TrafficLightConfig, self.reconfigure_callback)
        
        # Publishers - AckermannDriveStamped for direct motor control
        self.cmd_vel_pub = rospy.Publisher('/low_level/ackermann_cmd_mux/input/navigation', 
                                            AckermannDriveStamped, queue_size=1)
        self.state_pub = rospy.Publisher('/webot/traffic_light/state', String, queue_size=1)
        self.stop_pub = rospy.Publisher('/webot/traffic_stop', Bool, queue_size=1)
        self.passed_pub = rospy.Publisher('/webot/traffic_passed', Bool, queue_size=1)
        self.pub_image = rospy.Publisher('/webot/traffic_light/image', Image, queue_size=1)
        self.pub_debug = rospy.Publisher('/webot/traffic_light/debug', Image, queue_size=1)
        
        # Subscriber for steering and speed from lane_detect_node
        self.sub_steering = rospy.Subscriber('/webot/steering_offset', Float32, self.steering_callback, queue_size=1)
        self.sub_speed = rospy.Subscriber('/webot/lane_speed', Float32, self.speed_callback, queue_size=1)
        
        # Enable subscriber for sequence control
        self.sub_enable = rospy.Subscriber('/webot/traffic_enable', Bool, self.enable_callback, queue_size=1)
        
        # Image subscriber
        self.image_sub = rospy.Subscriber(
            '/usb_cam/image_raw/compressed',
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )
        
        # Control loop timer (20Hz)
        self.timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)
        
        rospy.loginfo("="*50)
        rospy.loginfo("Traffic Light Node initialized")
        rospy.loginfo("Subscribing to /webot/steering_offset and /webot/lane_speed")
        rospy.loginfo("RED: Stop, GREEN/UNKNOWN: Lane tracing")
        rospy.loginfo("View: rqt_image_view /webot/traffic_light/image")
        rospy.loginfo("="*50)
    
    def _load_calibration(self):
        """Load fisheye camera calibration"""
        try:
            calib_file = rospy.get_param('~calibration_file',
                '/home/wego/catkin_ws/src/usb_cam/calibration/usb_cam.yaml')
            with open(calib_file, 'r') as f:
                calib = yaml.safe_load(f)
            camera_matrix = np.array(calib['camera_matrix']['data']).reshape(3, 3)
            dist_coeffs = np.array(calib['distortion_coefficients']['data'])
            rospy.loginfo("[TrafficLight] Calibration loaded from %s", calib_file)
            return camera_matrix, dist_coeffs
        except Exception as e:
            rospy.logwarn("[TrafficLight] Calibration load failed: %s", str(e))
            return None, None
    
    def undistort(self, img):
        """Apply fisheye undistortion"""
        if self.camera_matrix is None:
            return img
        h, w = img.shape[:2]
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.camera_matrix, self.dist_coeffs, (w, h), np.eye(3), balance=0.0
        )
        return cv2.fisheye.undistortImage(
            img, self.camera_matrix, self.dist_coeffs, Knew=new_K
        )
    
    def reconfigure_callback(self, config, level):
        self.config = config
        rospy.loginfo("[TrafficLight] Config updated: brightness=%.2f", config.brightness_factor)
        return config
    
    def steering_callback(self, msg):
        self.lane_steering = msg.data
    
    def speed_callback(self, msg):
        self.lane_speed = msg.data
    
    def enable_callback(self, msg):
        self.enabled = msg.data
        rospy.loginfo(f"[TrafficLight] Enabled: {self.enabled}")
    
    def detect_color(self, hsv_image, lower1, upper1, lower2=None, upper2=None):
        """Detect color in HSV image, optionally with two ranges (for red)"""
        mask = cv2.inRange(hsv_image, lower1, upper1)
        if lower2 is not None and upper2 is not None:
            mask2 = cv2.inRange(hsv_image, lower2, upper2)
            mask = cv2.bitwise_or(mask, mask2)
        return mask
    
    def check_rectangularity(self, contour):
        """Check if contour is rectangular (traffic light shape)"""
        area = cv2.contourArea(contour)
        if area < self.config.min_area:
            return False
            
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
            
        circularity = 4 * np.pi * area / (perimeter ** 2)
        return circularity > self.config.circularity_thresh
    
    def control_loop(self, event):
        """Control loop - publish motor commands based on state"""
        if not self.enabled:
            return
        
        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header.stamp = rospy.Time.now()
        ackermann_msg.header.frame_id = "base_link"
        
        if self.current_state == "RED":
            # Stop on red light
            ackermann_msg.drive.speed = 0.0
            ackermann_msg.drive.steering_angle = 0.0
            self.stop_pub.publish(Bool(True))
        else:
            # GREEN or UNKNOWN: continue lane tracing normally
            ackermann_msg.drive.speed = self.lane_speed
            ackermann_msg.drive.steering_angle = self.lane_steering
            self.stop_pub.publish(Bool(False))
            
            # Signal that we passed green
            if self.current_state == "GREEN" and not self.green_passed:
                self.green_passed = True
                self.passed_pub.publish(Bool(True))
        
        self.cmd_vel_pub.publish(ackermann_msg)
        self.state_pub.publish(String(self.current_state))
        
        # Throttled log
        rospy.loginfo_throttle(1, f"[Traffic] State: {self.current_state} | Speed: {ackermann_msg.drive.speed:.2f} | Steer: {ackermann_msg.drive.steering_angle:.3f}")
    
    def image_callback(self, msg):
        if self.config is None or not self.enabled:
            return
            
        try:
            # Decompress image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                return
            
            # Apply fisheye undistortion
            cv_image = self.undistort(cv_image)
            
            # Apply brightness adjustment
            adjusted = cv2.convertScaleAbs(cv_image, alpha=self.config.brightness_factor, beta=0)
            
            # Extract ROI
            roi = adjusted[self.config.roi_top:self.config.roi_bottom, 
                          self.config.roi_left:self.config.roi_right]
            
            # Convert to HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Detect red (two ranges for hue wrap-around)
            red_lower1 = np.array([self.config.red_h_low1, self.config.red_s_low, self.config.red_v_low])
            red_upper1 = np.array([self.config.red_h_high1, self.config.red_s_high, self.config.red_v_high])
            red_lower2 = np.array([self.config.red_h_low2, self.config.red_s_low, self.config.red_v_low])
            red_upper2 = np.array([self.config.red_h_high2, self.config.red_s_high, self.config.red_v_high])
            red_mask = self.detect_color(hsv, red_lower1, red_upper1, red_lower2, red_upper2)
            
            # Detect green
            green_lower = np.array([self.config.green_h_low, self.config.green_s_low, self.config.green_v_low])
            green_upper = np.array([self.config.green_h_high, self.config.green_s_high, self.config.green_v_high])
            green_mask = self.detect_color(hsv, green_lower, green_upper)
            
            # Find contours
            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check for valid detections
            red_detected = any(self.check_rectangularity(c) for c in red_contours)
            green_detected = any(self.check_rectangularity(c) for c in green_contours)
            
            # State machine
            prev_state = self.current_state
            
            if red_detected and not green_detected:
                self.current_state = "RED"
            elif green_detected and not red_detected:
                self.current_state = "GREEN"
            else:
                self.current_state = "UNKNOWN"
            
            # Log state changes
            if self.current_state != prev_state:
                rospy.loginfo(f"Traffic Light: {prev_state} -> {self.current_state}")
            
            # Publish debug mask image
            if self.pub_debug.get_num_connections() > 0:
                mask_image = cv2.merge([np.zeros_like(red_mask), green_mask, red_mask])
                self.pub_debug.publish(self.bridge.cv2_to_imgmsg(mask_image, "bgr8"))
            
            # Visualization image
            if self.pub_image.get_num_connections() > 0:
                debug_image = cv_image.copy()
                
                # Draw ROI
                cv2.rectangle(debug_image, 
                              (self.config.roi_left, self.config.roi_top),
                              (self.config.roi_right, self.config.roi_bottom),
                              (255, 255, 0), 2)
                
                # Draw detections
                offset_x, offset_y = self.config.roi_left, self.config.roi_top
                for cnt in red_contours:
                    if self.check_rectangularity(cnt):
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(debug_image, (x+offset_x, y+offset_y), 
                                      (x+w+offset_x, y+h+offset_y), (0, 0, 255), 3)
                
                for cnt in green_contours:
                    if self.check_rectangularity(cnt):
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(debug_image, (x+offset_x, y+offset_y), 
                                      (x+w+offset_x, y+h+offset_y), (0, 255, 0), 3)
                
                # State text
                color = (0, 0, 255) if self.current_state == "RED" else \
                        (0, 255, 0) if self.current_state == "GREEN" else (128, 128, 128)
                cv2.putText(debug_image, f"State: {self.current_state}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(debug_image, f"Steering: {self.lane_steering:.3f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(debug_image, f"Speed: {self.lane_speed:.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                self.pub_image.publish(self.bridge.cv2_to_imgmsg(debug_image, "bgr8"))
                
        except Exception as e:
            rospy.logerr("[TrafficLight] Error: %s", str(e))
    
    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = TrafficLightNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
