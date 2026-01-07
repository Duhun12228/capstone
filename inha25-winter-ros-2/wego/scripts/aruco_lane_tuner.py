#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArUco Lane Tuning Node for WEGO (Simple Version)
- ArUco 마커 감지 + lane_detect_node 튜닝 변경
- ID 4 감지 시 튜닝 프리셋 적용
- 단일 파일로 동작 (yaml 불필요)
"""

import rospy
import cv2
import numpy as np
import dynamic_reconfigure.client
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
import threading


class ArucoLaneTuner:
    def __init__(self):
        rospy.init_node('aruco_lane_tuner')
        
        # ========================================
        # ArUco 설정
        # ========================================
        self.target_marker_id = rospy.get_param('~target_marker_id', 4)  # 감지할 마커 ID
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        try:
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
        except AttributeError:
            self.use_new_api = False
        
        # ========================================
        # 튜닝 프리셋 (마커 감지 시 적용할 값만 설정)
        # ========================================
        self.PRESET_MARKER = {
            'kp': 0.015,
            'kd': 0.008,
            'base_speed': 0.1,
            'max_steering': 0.6,
            'lane_offset': 120,
        }
        
        # 원래 값 저장용 (마커 감지 전에 자동 저장)
        self.original_params = None
        
        # ========================================
        # 상태
        # ========================================
        self.current_preset = "DEFAULT"
        self.last_detect_time = None
        self.cooldown = 3.0  # 재감지 쿨다운 (초)
        self.revert_timeout = 10.0  # 복귀 시간 (초)
        self.is_tuning = False
        
        # ========================================
        # Dynamic Reconfigure Client
        # ========================================
        self.dyn_client = None
        self.client_ready = False
        threading.Thread(target=self._connect_client, daemon=True).start()
        
        # ========================================
        # ROS - webot 토픽 프리픽스
        # ========================================
        self.pub_detected = rospy.Publisher('/webot/aruco/marker_detected', Bool, queue_size=1)
        
        # wego 카메라 토픽
        self.sub_image = rospy.Subscriber(
            '/usb_cam/image_raw/compressed',
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )
        
        # 복귀 타이머
        rospy.Timer(rospy.Duration(1.0), self.check_revert)
        
        rospy.loginfo("=" * 50)
        rospy.loginfo("ArUco Lane Tuner (WEGO)")
        rospy.loginfo(f"Target Marker ID: {self.target_marker_id}")
        rospy.loginfo("=" * 50)
    
    def _connect_client(self):
        """Dynamic reconfigure 연결"""
        try:
            rospy.loginfo("[ArUco Tuner] Connecting to lane_detect_node...")
            self.dyn_client = dynamic_reconfigure.client.Client('/lane_detect_node', timeout=30)
            self.client_ready = True
            rospy.loginfo("[ArUco Tuner] Connected!")
        except Exception as e:
            rospy.logerr(f"[ArUco Tuner] Connection failed: {e}")
    
    def image_callback(self, msg):
        """이미지에서 ArUco 감지"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                return
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if self.use_new_api:
                corners, ids, _ = self.detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            
            if ids is not None:
                for marker_id in ids.flatten():
                    if marker_id == self.target_marker_id:
                        self.on_marker_detected()
                        
        except Exception as e:
            rospy.logerr_throttle(5, f"[ArUco Tuner] Error: {e}")
    
    def on_marker_detected(self):
        """마커 감지 시 처리"""
        now = rospy.Time.now()
        
        # 쿨다운 체크
        if self.last_detect_time:
            elapsed = (now - self.last_detect_time).to_sec()
            if elapsed < self.cooldown:
                return
        
        self.last_detect_time = now
        self.pub_detected.publish(Bool(data=True))
        
        rospy.loginfo(f"[ArUco Tuner] Marker {self.target_marker_id} detected!")
        
        # 이미 적용 중이면 스킵
        if self.current_preset == "MARKER":
            return
        
        # 원래 값 저장 후 적용
        threading.Thread(
            target=self.save_and_apply_preset,
            daemon=True
        ).start()
    
    def save_and_apply_preset(self):
        """원래 값 저장 후 마커 프리셋 적용"""
        if not self.client_ready or self.is_tuning:
            return
        
        self.is_tuning = True
        try:
            # 현재 파라미터 저장
            current = self.dyn_client.get_configuration()
            self.original_params = {
                'kp': current['kp'],
                'kd': current['kd'],
                'base_speed': current['base_speed'],
                'max_steering': current['max_steering'],
                'lane_offset': current['lane_offset'],
            }
            rospy.loginfo(f"[ArUco Tuner] Saved original params: {self.original_params}")
            
            # 마커 프리셋 적용
            rospy.loginfo("[ArUco Tuner] Applying MARKER preset...")
            self.dyn_client.update_configuration(self.PRESET_MARKER)
            rospy.sleep(0.5)
            self.current_preset = "MARKER"
            rospy.loginfo("[ArUco Tuner] MARKER preset applied!")
        except Exception as e:
            rospy.logerr(f"[ArUco Tuner] Apply failed: {e}")
        finally:
            self.is_tuning = False
    
    def check_revert(self, event):
        """마커 안 보이면 원래 값으로 복귀"""
        if self.current_preset == "ORIGINAL":
            return
        
        if self.last_detect_time is None:
            return
        
        elapsed = (rospy.Time.now() - self.last_detect_time).to_sec()
        
        if elapsed > self.revert_timeout:
            rospy.loginfo(f"[ArUco Tuner] Marker not seen for {elapsed:.1f}s, reverting to original...")
            threading.Thread(
                target=self.restore_original,
                daemon=True
            ).start()
            self.last_detect_time = None
    
    def restore_original(self):
        """원래 파라미터로 복귀"""
        if not self.client_ready or self.is_tuning:
            return
        
        if self.original_params is None:
            rospy.logwarn("[ArUco Tuner] No original params saved!")
            self.current_preset = "ORIGINAL"
            return
        
        self.is_tuning = True
        try:
            rospy.loginfo(f"[ArUco Tuner] Restoring original params: {self.original_params}")
            self.dyn_client.update_configuration(self.original_params)
            rospy.sleep(0.5)
            self.current_preset = "ORIGINAL"
            rospy.loginfo("[ArUco Tuner] Original params restored!")
        except Exception as e:
            rospy.logerr(f"[ArUco Tuner] Restore failed: {e}")
        finally:
            self.is_tuning = False
    
    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = ArucoLaneTuner()
        node.run()
    except rospy.ROSInterruptException:
        pass
