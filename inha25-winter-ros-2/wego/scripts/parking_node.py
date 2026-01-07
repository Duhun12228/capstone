#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parking Mission Node for WEGO
- 후진 평행주차 수행
- ArUco Marker ID가 일정 거리 이내일 때 자동 시작
- 시간 기반 제어 (AckermannDriveStamped)
- 주차 완료 후 정지 및 lane_detect_node 종료
"""

import rospy
import subprocess
from std_msgs.msg import Bool, String, Float32, Float32MultiArray
from ackermann_msgs.msg import AckermannDriveStamped


class ParkingState:
    IDLE = "IDLE"                      # 대기 (라인트레이싱 중)
    # === 주차 ===
    STOP_BEFORE = "STOP_BEFORE"        # 주차 구역 앞에서 정지
    FORWARD_PASS = "FORWARD_PASS"      # 주차 구역 지나서 전진
    STOP_ALIGN = "STOP_ALIGN"          # 정렬 위해 정지
    REVERSE_RIGHT = "REVERSE_RIGHT"    # 후진 + 우회전
    REVERSE_LEFT = "REVERSE_LEFT"      # 후진 + 좌회전 (정렬)
    FORWARD_STRAIGHT = "FORWARD_STRAIGHT"  # 전진 직진 (진입)
    DONE = "DONE"                      # 주차 완료 (종료)


class ParkingNode:
    def __init__(self):
        rospy.init_node('parking_node')
        
        # ========================================
        # 파라미터 (코드에서 직접 조정 또는 yaml로 로드)
        # ========================================
        self.speed_slow = rospy.get_param('~speed_slow', 0.15)           # 저속
        self.speed_reverse = rospy.get_param('~speed_reverse', -0.13)    # 후진 속도
        self.steering_angle = rospy.get_param('~steering_angle', 0.5)    # 조향 각도 (rad)
        
        self.forward_time = rospy.get_param('~forward_time', 1.5)              # 주차 구역 지나 전진 시간 (초)
        self.reverse_right_time = rospy.get_param('~reverse_right_time', 3.0)  # 후진 우회전 시간 (초)
        self.reverse_left_time = rospy.get_param('~reverse_left_time', 3.0)    # 후진 좌회전 시간 (초)
        self.final_forward_time = rospy.get_param('~final_forward_time', 1.0)  # 마지막 전진 직진 시간 (초)
        
        # 트리거 거리 (미터) - 이 거리 이내일 때 주차 시작
        self.trigger_distance = rospy.get_param('~trigger_distance', 0.5)  # 0.5m
        self.target_marker_id = rospy.get_param('~target_marker_id', 0)    # 마커 ID 0
        
        # IDLE 상태에서 lane_detect 값 사용할지 여부
        self.use_lane_in_idle = rospy.get_param('~use_lane_in_idle', True)
        
        # ========================================
        # 상태
        # ========================================
        self.state = ParkingState.IDLE
        self.phase_start_time = None
        self.parking_triggered = False  # 주차 트리거 (1회만)
        
        # Lane 값 (lane_detect_node에서 수신)
        self.lane_steering = 0.0
        self.lane_speed = 0.3
        
        # ========================================
        # ROS Publishers
        # ========================================
        self.pub_cmd = rospy.Publisher('/low_level/ackermann_cmd_mux/input/navigation', 
                                        AckermannDriveStamped, queue_size=1)
        self.pub_state = rospy.Publisher('/webot/parking/state', String, queue_size=1)
        self.pub_done = rospy.Publisher('/webot/parking/done', Bool, queue_size=1)
        
        # ========================================
        # ROS Subscribers
        # ========================================
        # ArUco 마커 정보 구독 (ID + 거리)
        self.sub_aruco = rospy.Subscriber('/webot/aruco/marker_info', Float32MultiArray, 
                                          self.aruco_callback, queue_size=1)
        
        # Lane detect 값 구독 (IDLE 상태에서 lane tracing 유지용)
        self.sub_steering = rospy.Subscriber('/webot/steering_offset', Float32, 
                                             self.steering_callback, queue_size=1)
        self.sub_speed = rospy.Subscriber('/webot/lane_speed', Float32, 
                                          self.speed_callback, queue_size=1)
        
        # 제어 루프 (20Hz)
        self.timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)
        
        rospy.loginfo("=" * 50)
        rospy.loginfo("Parking Node initialized (WEGO)")
        rospy.loginfo(f"Trigger: ArUco Marker ID {self.target_marker_id}")
        rospy.loginfo(f"Trigger distance: {self.trigger_distance}m")
        rospy.loginfo(f"Motor topic: /low_level/ackermann_cmd_mux/input/navigation")
        rospy.loginfo("=" * 50)
    
    def steering_callback(self, msg):
        self.lane_steering = msg.data
    
    def speed_callback(self, msg):
        self.lane_speed = msg.data
    
    def aruco_callback(self, msg):
        """ArUco 마커 정보 콜백 - ID가 target이고 거리가 임계값 이내면 주차 시작
        
        msg.data 형식: [id1, distance1, id2, distance2, ...]
        """
        if self.state != ParkingState.IDLE or self.parking_triggered:
            return
        
        # 데이터 파싱: [id, dist, id, dist, ...]
        data = msg.data
        if len(data) < 2:
            return
        
        for i in range(0, len(data) - 1, 2):
            marker_id = int(data[i])
            distance = data[i + 1]
            
            if marker_id == self.target_marker_id:
                rospy.loginfo_throttle(0.5, f"ArUco ID:{marker_id} Distance:{distance:.3f}m (trigger: {self.trigger_distance}m)")
                
                if distance <= self.trigger_distance:
                    rospy.loginfo(f"ArUco Marker ID {marker_id} within {self.trigger_distance}m! Starting parking...")
                    self.parking_triggered = True
                    self.state = ParkingState.STOP_BEFORE
                    self.phase_start_time = rospy.Time.now()
                    return
    
    def elapsed_time(self):
        """현재 페이즈 경과 시간"""
        if self.phase_start_time is None:
            return 0
        return (rospy.Time.now() - self.phase_start_time).to_sec()
    
    def next_phase(self, new_state):
        """다음 페이즈로 전환"""
        rospy.loginfo(f"Phase: {self.state} -> {new_state}")
        self.state = new_state
        self.phase_start_time = rospy.Time.now()
    
    def publish_ackermann(self, speed, steering):
        """AckermannDriveStamped 메시지 발행"""
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.drive.speed = speed
        msg.drive.steering_angle = steering
        self.pub_cmd.publish(msg)
    
    def shutdown_lane_detect(self):
        """lane_detect_node 종료"""
        try:
            # rosnode kill 명령으로 lane_detect_node 종료
            subprocess.Popen(['rosnode', 'kill', '/lane_detect_node'], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            rospy.loginfo("lane_detect_node killed")
        except Exception as e:
            rospy.logwarn(f"Failed to kill lane_detect_node: {e}")
    
    def control_loop(self, event):
        """메인 제어 루프"""
        speed = 0.0
        steering = 0.0
        
        # 상태 발행
        self.pub_state.publish(String(data=self.state))
        
        # ========================================
        # IDLE: 대기 (lane tracing 유지)
        # ========================================
        if self.state == ParkingState.IDLE:
            if self.use_lane_in_idle:
                speed = self.lane_speed
                steering = self.lane_steering
            else:
                speed = 0.0
                steering = 0.0
        
        # ========================================
        # STOP_BEFORE: 잠시 정지 (1초)
        # ========================================
        elif self.state == ParkingState.STOP_BEFORE:
            speed = 0.0
            steering = 0.0
            
            if self.elapsed_time() > 1.0:
                self.next_phase(ParkingState.FORWARD_PASS)
        
        # ========================================
        # FORWARD_PASS: 주차 구역 지나서 전진
        # ========================================
        elif self.state == ParkingState.FORWARD_PASS:
            speed = self.speed_slow
            steering = 0.0
            
            if self.elapsed_time() > self.forward_time:
                self.next_phase(ParkingState.STOP_ALIGN)
        
        # ========================================
        # STOP_ALIGN: 정렬 위해 정지 (1초)
        # ========================================
        elif self.state == ParkingState.STOP_ALIGN:
            speed = 0.0
            steering = 0.0
            
            if self.elapsed_time() > 1.0:
                self.next_phase(ParkingState.REVERSE_RIGHT)
        
        # ========================================
        # REVERSE_RIGHT: 후진 + 우회전 (시간 기반)
        # ========================================
        elif self.state == ParkingState.REVERSE_RIGHT:
            speed = self.speed_reverse
            steering = self.steering_angle  # 후진 시 우회전 = 양수 (바퀴 반대)
            
            rospy.loginfo_throttle(0.5, f"REVERSE_RIGHT: {self.elapsed_time():.1f}s / {self.reverse_right_time}s")
            
            if self.elapsed_time() >= self.reverse_right_time:
                self.next_phase(ParkingState.REVERSE_LEFT)
        
        # ========================================
        # REVERSE_LEFT: 후진 + 좌회전 (시간 기반)
        # ========================================
        elif self.state == ParkingState.REVERSE_LEFT:
            speed = self.speed_reverse
            steering = -self.steering_angle  # 후진 시 좌회전 = 음수 (바퀴 반대)
            
            rospy.loginfo_throttle(0.5, f"REVERSE_LEFT: {self.elapsed_time():.1f}s / {self.reverse_left_time}s")
            
            if self.elapsed_time() >= self.reverse_left_time:
                self.next_phase(ParkingState.FORWARD_STRAIGHT)
        
        # ========================================
        # FORWARD_STRAIGHT: 전진 직진 (주차 구역 안으로)
        # ========================================
        elif self.state == ParkingState.FORWARD_STRAIGHT:
            speed = self.speed_slow
            steering = 0.0
            
            if self.elapsed_time() > self.final_forward_time:
                self.next_phase(ParkingState.DONE)
        
        # ========================================
        # DONE: 주차 완료 - 정지 (미션 종료)
        # ========================================
        elif self.state == ParkingState.DONE:
            speed = 0.0
            steering = 0.0
            self.pub_done.publish(Bool(data=True))
            rospy.loginfo_throttle(3, "Parking complete! Mission finished.")
            
            # lane_detect_node 종료 (한 번만)
            if not hasattr(self, '_lane_killed'):
                self._lane_killed = True
                self.shutdown_lane_detect()
        
        # Ackermann 메시지 발행
        self.publish_ackermann(speed, steering)
    
    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = ParkingNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
