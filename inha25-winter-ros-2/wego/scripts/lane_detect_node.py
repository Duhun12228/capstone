#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lane Detection Node for WEGO
- HSV-based white lane detection
- Histogram-based lane center finding
- PID control for steering
- Publishes steering/speed for other nodes to use
- Only controls motor when publish_cmd_vel=True
"""

import rospy
import cv2
import numpy as np
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32, Int32, Bool, String
from ackermann_msgs.msg import AckermannDriveStamped
from dynamic_reconfigure.server import Server
from wego.cfg import LaneDetectConfig


class LaneDetectNode:
    def __init__(self):
        rospy.init_node('lane_detect_node', anonymous=True)
        
        self.bridge = CvBridge()
        
        # PID state
        self.integral = 0
        self.prev_error = 0
        
        # Parameters from launch file
        self.publish_cmd_vel = rospy.get_param("~publish_cmd_vel", False)
        self.debug_view = rospy.get_param("~debug_view", True)
        
        # Stop flag (from traffic light or other nodes)
        self.stop_flag = False
        
        # Load fisheye calibration
        self.camera_matrix, self.dist_coeffs = self._load_calibration()
        
        # Dynamic Reconfigure
        self.config = None
        self.srv = Server(LaneDetectConfig, self.reconfigure_callback)
        
        # Image dimensions (will be updated from first image)
        self.img_width = 640
        self.img_height = 480
        
        # Publishers - steering/speed for other nodes
        self.pub_steering = rospy.Publisher('/webot/steering_offset', Float32, queue_size=1)
        self.pub_speed = rospy.Publisher('/webot/lane_speed', Float32, queue_size=1)
        self.pub_center_x = rospy.Publisher('/webot/lane_center_x', Int32, queue_size=1)
        
        # Image publishers
        self.pub_image = rospy.Publisher('/webot/lane_detect/image', Image, queue_size=1)
        self.pub_mask = rospy.Publisher('/webot/lane_detect/mask', Image, queue_size=1)
        
        # cmd_vel publisher (only when publish_cmd_vel is True)
        if self.publish_cmd_vel:
            self.cmd_vel_pub = rospy.Publisher('/low_level/ackermann_cmd_mux/input/navigation', 
                                                AckermannDriveStamped, queue_size=1)
            # Subscribe to stop flag from traffic light
            self.sub_stop = rospy.Subscriber('/webot/traffic_stop', Bool, self.stop_callback, queue_size=1)
        
        # Subscriber - wego uses usb_cam
        self.image_sub = rospy.Subscriber(
            '/usb_cam/image_raw/compressed',
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )
        
        rospy.loginfo("="*50)
        rospy.loginfo("Lane Detect Node initialized")
        rospy.loginfo(f"publish_cmd_vel: {self.publish_cmd_vel}")
        rospy.loginfo("Steering topic: /webot/steering_offset")
        rospy.loginfo("Speed topic: /webot/lane_speed")
        rospy.loginfo("View: rqt_image_view /webot/lane_detect/image")
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
            rospy.loginfo("[LaneDetect] Calibration loaded from %s", calib_file)
            return camera_matrix, dist_coeffs
        except Exception as e:
            rospy.logwarn("[LaneDetect] Calibration load failed: %s", str(e))
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
        rospy.loginfo(f"[LaneDetect] Config updated: speed={config.base_speed}, kp={config.kp}, kd={config.kd}")
        return config
    
    def stop_callback(self, msg):
        """Callback for stop flag from traffic light"""
        self.stop_flag = msg.data
    
    def find_lane_center(self, mask):
        """Find lane center using histogram method"""
        h, w = mask.shape
        
        # Histogram of bottom half
        histogram = np.sum(mask[h//2:, :], axis=0)
        
        # Find peaks in left and right regions
        midpoint = w // 2
        left_peak = np.argmax(histogram[:midpoint])
        right_peak = np.argmax(histogram[midpoint:]) + midpoint
        
        left_val = histogram[left_peak]
        right_val = histogram[right_peak]
        
        min_pixels = self.config.min_pixels if self.config else 500
        
        left_detected = left_val > min_pixels
        right_detected = right_val > min_pixels
        
        if left_detected and right_detected:
            # Both lanes detected - use center
            center_x = (left_peak + right_peak) // 2
            detected = True
        elif left_detected:
            # Only left lane - offset to right
            center_x = left_peak + self.config.lane_offset
            detected = True
        elif right_detected:
            # Only right lane - offset to left
            center_x = right_peak - self.config.lane_offset
            detected = True
        else:
            # No lane detected -> keep center
            center_x = w // 2
            detected = False
        
        return center_x, detected, left_peak if left_detected else -1, right_peak if right_detected else -1
    
    def pid_control(self, error):
        """PID steering control"""
        # P term
        p_term = self.config.kp * error
        
        # I term with anti-windup
        self.integral += error
        self.integral = np.clip(self.integral, -1000, 1000)
        i_term = self.config.ki * self.integral
        
        # D term
        d_term = self.config.kd * (error - self.prev_error)
        self.prev_error = error
        
        # Combined steering
        steering = p_term + i_term + d_term
        steering = np.clip(steering, -self.config.max_steering, self.config.max_steering)
        
        return steering
    
    def image_callback(self, msg):
        if self.config is None:
            return
            
        try:
            # Decompress image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                return
            
            # Apply fisheye undistortion
            cv_image = self.undistort(cv_image)
            
            height, width = cv_image.shape[:2]
            self.img_width = width
            self.img_height = height
            
            # Extract ROI (ratio-based)
            roi_top = int(height * self.config.roi_top_ratio)
            roi_bottom = int(height * self.config.roi_bottom_ratio)
            roi = cv_image[roi_top:roi_bottom, :]
            
            # Convert to HSV and create mask
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower = np.array([self.config.hsv_h_low, self.config.hsv_s_low, self.config.hsv_v_low])
            upper = np.array([self.config.hsv_h_high, self.config.hsv_s_high, self.config.hsv_v_high])
            mask = cv2.inRange(hsv, lower, upper)
            
            # Apply morphology to clean up
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find lane center using histogram
            center_x, lane_detected, left_x, right_x = self.find_lane_center(mask)
            
            # Calculate error (from image center)
            image_center = width // 2
            error = center_x - image_center if lane_detected else 0
            
            # PID steering control
            steering = self.pid_control(error)
            
            # Publish steering and speed for other nodes
            self.pub_steering.publish(Float32(steering))
            self.pub_speed.publish(Float32(self.config.base_speed))
            self.pub_center_x.publish(Int32(center_x))
            
            # Publish AckermannDriveStamped if publish_cmd_vel is True
            if self.publish_cmd_vel:
                ackermann_msg = AckermannDriveStamped()
                ackermann_msg.header.stamp = rospy.Time.now()
                ackermann_msg.header.frame_id = "base_link"
                
                if self.stop_flag:
                    ackermann_msg.drive.speed = 0.0
                    ackermann_msg.drive.steering_angle = 0.0
                else:
                    ackermann_msg.drive.speed = self.config.base_speed
                    ackermann_msg.drive.steering_angle = steering
                
                self.cmd_vel_pub.publish(ackermann_msg)
            
            # Debug visualization
            if self.debug_view and self.pub_image.get_num_connections() > 0:
                debug_image = cv_image.copy()
                
                # Draw ROI region
                cv2.rectangle(debug_image, (0, roi_top), (width, roi_bottom), (255, 255, 0), 2)
                
                # Draw lane center line
                roi_mid_y = (roi_top + roi_bottom) // 2
                
                # Draw detected lane points
                if left_x > 0:
                    cv2.circle(debug_image, (left_x, roi_mid_y), 10, (255, 0, 0), -1)  # Blue for left
                if right_x > 0:
                    cv2.circle(debug_image, (right_x, roi_mid_y), 10, (0, 0, 255), -1)  # Red for right
                
                # Draw lane center
                cv2.circle(debug_image, (center_x, roi_mid_y), 10, (0, 255, 0), -1)  # Green for center
                
                # Draw image center line
                cv2.line(debug_image, (image_center, roi_top), (image_center, roi_bottom), (0, 255, 255), 2)
                
                # Draw arrow from center to lane center
                cv2.arrowedLine(debug_image, (image_center, roi_mid_y), (center_x, roi_mid_y), (0, 255, 0), 3)
                
                # Status text
                cv2.putText(debug_image, f"Offset: {error}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(debug_image, f"Steering: {steering:.3f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(debug_image, f"L:{left_x} R:{right_x}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                self.pub_image.publish(self.bridge.cv2_to_imgmsg(debug_image, "bgr8"))
            
            # Publish mask image if anyone is listening
            if self.pub_mask.get_num_connections() > 0:
                self.pub_mask.publish(self.bridge.cv2_to_imgmsg(mask, "mono8"))
            
            # Throttled log
            rospy.loginfo_throttle(1, 
                f"Center: {center_x} | Error: {error:.0f} | Steer: {np.degrees(steering):.1f}° | Speed: {self.config.base_speed:.2f}")
                
        except Exception as e:
            rospy.logerr("[LaneDetect] Error: %s", str(e))
    
    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = LaneDetectNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
