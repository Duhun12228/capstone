#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roundabout Navigation Node for WEGO (Simplified)
- Trigger 받으면 장애물 확인 후 라인트레이싱
- lane_detect_node의 steering 사용 (별도 튜닝 가능)
- cmd_vel 직접 발행 (AckermannDriveStamped)
"""

import rospy
import numpy as np
import math
import cv2
from sensor_msgs.msg import LaserScan, Imu, CompressedImage, Image
from std_msgs.msg import String, Bool, Float32
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion
from dynamic_reconfigure.server import Server
from dynamic_reconfigure.client import Client as DynClient
from wego.cfg import RoundaboutConfig


class RoundaboutState:
    IDLE = "IDLE"                    # 대기 (트리거 대기)
    WAITING = "WAITING"              # 장애물 대기
    DRIVING = "DRIVING"              # 라인트레이싱 주행 중


class RoundaboutNode:
    def __init__(self):
        rospy.init_node('roundabout_node')
        
        # === LiDAR 장애물 감지 파라미터 ===
        self.detect_x_min = rospy.get_param('~detect_x_min', 0.2)
        self.detect_x_max = rospy.get_param('~detect_x_max', 0.8)
        self.detect_y_min = rospy.get_param('~detect_y_min', -0.3)
        self.detect_y_max = rospy.get_param('~detect_y_max', 0.3)
        self.obstacle_threshold = rospy.get_param('~obstacle_threshold', 3)
        
        # === 속도 파라미터 (lane_detect_node에서 수신) ===
        self.lane_speed = 0.3       # default, updated from lane_detect_node
        self.steering_gain = rospy.get_param('~steering_gain', 1.0)
        
        # === Roundabout 전용 lane_detect 튜닝 값 ===
        self.roundabout_lane_params = {
            'kp': 0.015,
            'kd': 0.008,
            'base_speed': 0.25,
            'lane_offset': 160,
        }
        self.original_lane_params = None
        self.lane_client = None
        
        # === 상태 ===
        self.state = RoundaboutState.DRIVING
        self.active = True
        
        # === 장애물 감지 ===
        self.obstacle_detected = False
        self.obstacle_points = []
        self.scan_data = None
        
        # === IMU (표시만) ===
        self.current_yaw = 0.0
        
        # === 차선 상태 (lane_detect_node에서 수신) ===
        self.lane_steering = 0.0
        
        # === 이미지 처리 ===
        self.bridge = CvBridge()
        self.current_image = None
        self.img_width = 640
        self.img_height = 480
        
        # Dynamic reconfigure
        self.srv = Server(RoundaboutConfig, self.reconfigure_callback)
        
        # Publishers - AckermannDriveStamped for direct motor control
        self.cmd_vel_pub = rospy.Publisher('/low_level/ackermann_cmd_mux/input/navigation',
                                           AckermannDriveStamped, queue_size=1)
        self.state_pub = rospy.Publisher('/webot/roundabout/state', String, queue_size=1)
        self.active_pub = rospy.Publisher('/webot/roundabout/active', Bool, queue_size=1)
        self.done_pub = rospy.Publisher('/webot/roundabout/done', Bool, queue_size=1)
        self.pub_debug_image = rospy.Publisher('/webot/roundabout/debug', Image, queue_size=1)
        
        # Subscribers
        self.sub_image = rospy.Subscriber('/usb_cam/image_raw/compressed', CompressedImage, 
                                          self.image_callback, queue_size=1)
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        self.sub_imu = rospy.Subscriber('/imu', Imu, self.imu_callback, queue_size=1)
        self.sub_steering = rospy.Subscriber('/webot/steering_offset', Float32, 
                                              self.steering_callback, queue_size=1)
        self.sub_speed = rospy.Subscriber('/webot/lane_speed', Float32, 
                                           self.speed_callback, queue_size=1)
        # lane_detect dynamic_reconfigure client 연결 시도
        self.setup_lane_client()
        
        # 시작 시 roundabout 파라미터 적용
        self.apply_roundabout_params()
        
        # Timer for control loop (20Hz)
        self.timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)
        self.vis_timer = rospy.Timer(rospy.Duration(0.1), self.visualization_callback)
        
        rospy.loginfo("=" * 50)
        rospy.loginfo("Roundabout Node (Auto Start)")
        rospy.loginfo("View: rqt_image_view /webot/roundabout/debug")
        rospy.loginfo("=" * 50)
    
    def reconfigure_callback(self, config, level):
        """Dynamic reconfigure"""
        self.detect_x_min = config.detect_x_min
        self.detect_x_max = config.detect_x_max
        self.obstacle_threshold = config.obstacle_threshold
        self.steering_gain = config.steering_gain
        rospy.loginfo(f"[Roundabout] Config updated: steering_gain={self.steering_gain:.2f}")
        return config
    
    def setup_lane_client(self):
        """lane_detect_node의 dynamic_reconfigure client 설정"""
        try:
            self.lane_client = DynClient('/lane_detect_node', timeout=2.0)
            rospy.loginfo("[Roundabout] Connected to lane_detect_node dynamic_reconfigure")
        except Exception as e:
            rospy.logwarn(f"[Roundabout] Could not connect to lane_detect_node: {e}")
            self.lane_client = None
    
    def apply_roundabout_params(self):
        """Roundabout용 lane_detect 파라미터 적용"""
        if self.lane_client is None:
            self.setup_lane_client()
        
        if self.lane_client is not None:
            try:
                current = self.lane_client.get_configuration()
                self.original_lane_params = {
                    'kp': current['kp'],
                    'kd': current['kd'],
                    'base_speed': current['base_speed'],
                    'lane_offset': current['lane_offset'],
                }
                rospy.loginfo(f"[Roundabout] Saved original params: {self.original_lane_params}")
                
                self.lane_client.update_configuration(self.roundabout_lane_params)
                rospy.loginfo(f"[Roundabout] Applied roundabout params: {self.roundabout_lane_params}")
            except Exception as e:
                rospy.logwarn(f"[Roundabout] Failed to apply params: {e}")
    
    def restore_lane_params(self):
        """원래 lane_detect 파라미터 복구"""
        if self.lane_client is not None and self.original_lane_params is not None:
            try:
                self.lane_client.update_configuration(self.original_lane_params)
                rospy.loginfo(f"[Roundabout] Restored original params: {self.original_lane_params}")
            except Exception as e:
                rospy.logwarn(f"[Roundabout] Failed to restore params: {e}")
    
    # ==================== Callbacks ====================
    
    def scan_callback(self, msg):
        """LiDAR 스캔 콜백 - laser_link 180도 회전 보정"""
        self.scan_data = msg
        self.process_lidar()
    
    def imu_callback(self, msg):
        """IMU 콜백 - YAW 표시용"""
        orientation = msg.orientation
        _, _, yaw = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        self.current_yaw = math.degrees(yaw)
    
    def steering_callback(self, msg):
        """lane_detect_node에서 조향값 수신"""
        self.lane_steering = msg.data
    
    def speed_callback(self, msg):
        """lane_detect_node에서 속도값 수신"""
        self.lane_speed = msg.data
    
    def image_callback(self, msg):
        """카메라 이미지 수신"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.current_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if self.current_image is not None:
                self.img_height, self.img_width = self.current_image.shape[:2]
        except Exception as e:
            rospy.logwarn_throttle(5, f"[Roundabout] Image decode error: {e}")
    
    # ==================== LiDAR Processing ====================
    
    def process_lidar(self):
        """LiDAR 데이터로 전방 장애물 감지 - laser_link 180도 회전 보정"""
        if self.scan_data is None:
            return
        
        self.obstacle_points = []
        angle = self.scan_data.angle_min
        
        for r in self.scan_data.ranges:
            if self.scan_data.range_min < r < self.scan_data.range_max:
                # laser_link가 180도 회전되어 있으므로 각도 보정
                corrected_angle = angle + math.pi
                # -π ~ π 범위로 정규화
                while corrected_angle > math.pi:
                    corrected_angle -= 2 * math.pi
                while corrected_angle < -math.pi:
                    corrected_angle += 2 * math.pi
                
                # 극좌표 → 직교좌표 (x: 전방, y: 좌측 양수)
                x = r * math.cos(corrected_angle)
                y = r * math.sin(corrected_angle)
                
                # 감지 영역 내 포인트 필터링 (전방 x > 0)
                if (self.detect_x_min <= x <= self.detect_x_max and
                    self.detect_y_min <= y <= self.detect_y_max):
                    self.obstacle_points.append((x, y))
            
            angle += self.scan_data.angle_increment
        
        self.obstacle_detected = len(self.obstacle_points) >= self.obstacle_threshold
        
        if self.obstacle_detected:
            rospy.loginfo_throttle(1, f"[Roundabout] Obstacle: {len(self.obstacle_points)} pts")
    
    # ==================== Control Loop ====================
    
    def control_loop(self, event):
        """메인 제어 루프"""
        # 상태 발행
        self.state_pub.publish(String(data=self.state))
        self.active_pub.publish(Bool(data=self.active))
        
        if not self.active:
            return
        
        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header.stamp = rospy.Time.now()
        ackermann_msg.header.frame_id = "base_link"
        
        # === WAITING: 장애물 대기 ===
        if self.state == RoundaboutState.WAITING:
            ackermann_msg.drive.speed = 0.0
            ackermann_msg.drive.steering_angle = 0.0
            
            if not self.obstacle_detected:
                self.state = RoundaboutState.DRIVING
                rospy.loginfo("[Roundabout] Clear - DRIVING")
        
        # === DRIVING: 라인트레이싱 주행 ===
        elif self.state == RoundaboutState.DRIVING:
            if self.obstacle_detected:
                ackermann_msg.drive.speed = 0.0
                ackermann_msg.drive.steering_angle = 0.0
                self.state = RoundaboutState.WAITING
                rospy.loginfo("[Roundabout] Obstacle - WAITING")
            else:
                ackermann_msg.drive.speed = self.lane_speed
                ackermann_msg.drive.steering_angle = self.lane_steering * self.steering_gain
        
        self.cmd_vel_pub.publish(ackermann_msg)
        
        rospy.loginfo_throttle(1, f"[Roundabout] {self.state} | Speed: {ackermann_msg.drive.speed:.2f}")
    
    # ==================== Visualization ====================
    
    def visualization_callback(self, event):
        """시각화 타이머 콜백"""
        self.publish_debug_image()
    
    def publish_debug_image(self):
        """상태 정보를 오버레이한 디버그 이미지 발행"""
        if self.pub_debug_image.get_num_connections() == 0:
            return
        
        # LiDAR 미니맵 + 카메라 이미지 합성
        vis_width = 640
        vis_height = 480
        
        if self.current_image is not None:
            vis_img = cv2.resize(self.current_image.copy(), (vis_width, vis_height))
        else:
            vis_img = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        
        # 상태별 색상 정의
        state_colors = {
            RoundaboutState.IDLE: (128, 128, 128),
            RoundaboutState.WAITING: (0, 0, 255),
            RoundaboutState.DRIVING: (0, 255, 0),
        }
        color = state_colors.get(self.state, (255, 255, 255))
        
        # 상태 배너 (상단)
        cv2.rectangle(vis_img, (0, 0), (vis_width, 50), (30, 30, 30), -1)
        status_text = f"ROUNDABOUT: {self.state}"
        if not self.active:
            status_text += " (Waiting Trigger)"
        cv2.putText(vis_img, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 정보 패널 (하단)
        info_y = vis_height - 100
        cv2.rectangle(vis_img, (0, info_y), (vis_width, vis_height), (30, 30, 30), -1)
        
        # 장애물 상태
        obs_color = (0, 0, 255) if self.obstacle_detected else (0, 255, 0)
        obs_text = f"Obstacle: {'BLOCKED' if self.obstacle_detected else 'CLEAR'} ({len(self.obstacle_points)}pts)"
        cv2.putText(vis_img, obs_text, (10, info_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, obs_color, 1)
        
        # 조향/속도
        cv2.putText(vis_img, f"Steering: {self.lane_steering:.3f}", (10, info_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(vis_img, f"Speed: {self.lane_speed:.2f}", (10, info_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # IMU YAW
        cv2.putText(vis_img, f"YAW: {self.current_yaw:.1f} deg", (200, info_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # LiDAR 미니맵 (오른쪽)
        minimap_size = 150
        minimap = self.create_lidar_minimap(minimap_size)
        x_offset = vis_width - minimap_size - 10
        y_offset = info_y - minimap_size - 10
        vis_img[y_offset:y_offset + minimap_size, x_offset:x_offset + minimap_size] = minimap
        
        try:
            self.pub_debug_image.publish(self.bridge.cv2_to_imgmsg(vis_img, "bgr8"))
        except Exception as e:
            rospy.logwarn_throttle(5, f"[Roundabout] Debug image error: {e}")
    
    def create_lidar_minimap(self, size):
        """LiDAR 감지 영역 미니맵 생성"""
        minimap = np.zeros((size, size, 3), dtype=np.uint8)
        
        cx = size // 2
        cy = size - 10
        scale = size / 2.0  # 1m = size/2 pixels
        
        # 그리드
        for r_m in [0.5, 1.0]:
            r_px = int(r_m * scale)
            cv2.circle(minimap, (cx, cy), r_px, (40, 40, 40), 1)
        
        # 감지 영역 박스
        box_x1 = int(cx - self.detect_y_max * scale)
        box_x2 = int(cx - self.detect_y_min * scale)
        box_y1 = int(cy - self.detect_x_max * scale)
        box_y2 = int(cy - self.detect_x_min * scale)
        cv2.rectangle(minimap, (box_x1, box_y1), (box_x2, box_y2), (100, 100, 0), 1)
        
        # 장애물 포인트
        for px, py in self.obstacle_points:
            mx = int(cx - py * scale)
            my = int(cy - px * scale)
            if 0 <= mx < size and 0 <= my < size:
                cv2.circle(minimap, (mx, my), 2, (0, 0, 255), -1)
        
        # 로봇 위치
        cv2.circle(minimap, (cx, cy), 4, (0, 255, 0), -1)
        
        return minimap
    
    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = RoundaboutNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
