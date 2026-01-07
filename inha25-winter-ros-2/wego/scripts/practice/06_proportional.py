#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from ackermann_msgs.msg import AckermannDriveStamped

# ========================================
# 제어 설정
# ========================================
# 비례 게인
# 값이 크면: 빠르게 반응하지만 진동 가능
# 값이 작으면: 부드럽지만 느림
KP = 0.005

# 기본 속도
BASE_SPEED = 0.5

# 조향 제한
MAX_STEERING = 0.34

# ========================================
# 슬라이딩 윈도우 설정 (05_sliding_window에서 복사)
# ========================================
NUM_WINDOWS = 8
WINDOW_WIDTH = 80
MIN_PIXELS = 30

H_MIN, H_MAX = 0, 180
S_MIN, S_MAX = 0, 100
V_MIN, V_MAX = 0, 50

SRC_POINTS = np.float32([
    [100, 400], [200, 300], [440, 300], [540, 400]
])
DST_WIDTH = 400
DST_HEIGHT = 300
DST_POINTS = np.float32([
    [0, DST_HEIGHT], [0, 0], [DST_WIDTH, 0], [DST_WIDTH, DST_HEIGHT]
])

# Publisher (전역)
pub_visual = None
pub_drive = None

# 변환 행렬 (전역)
M = None


def publish_image(publisher, img):
    """이미지를 CompressedImage로 발행"""
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "jpeg"
    msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
    publisher.publish(msg)


def calculate_offset(binary):
    """
    슬라이딩 윈도우로 오프셋 계산
    
    Returns:
        offset: 차선 중심 - 이미지 중심 (픽셀)
        output: 시각화 이미지
    """
    h, w = binary.shape
    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    # 히스토그램으로 시작점 찾기
    histogram = np.sum(binary[h//2:, :], axis=0)
    midpoint = w // 2
    
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    
    window_height = h // NUM_WINDOWS
    left_current = left_base
    right_current = right_base
    
    left_found = False
    right_found = False
    
    for win_idx in range(NUM_WINDOWS):
        win_y_low = h - (win_idx + 1) * window_height
        win_y_high = h - win_idx * window_height
        
        win_left_x_low = max(0, left_current - WINDOW_WIDTH // 2)
        win_left_x_high = min(w, left_current + WINDOW_WIDTH // 2)
        win_right_x_low = max(0, right_current - WINDOW_WIDTH // 2)
        win_right_x_high = min(w, right_current + WINDOW_WIDTH // 2)
        
        cv2.rectangle(output, (win_left_x_low, win_y_low), 
                     (win_left_x_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(output, (win_right_x_low, win_y_low),
                     (win_right_x_high, win_y_high), (0, 0, 255), 2)
        
        left_roi = binary[win_y_low:win_y_high, win_left_x_low:win_left_x_high]
        right_roi = binary[win_y_low:win_y_high, win_right_x_low:win_right_x_high]
        
        left_pixels = np.where(left_roi > 0)
        right_pixels = np.where(right_roi > 0)
        
        if len(left_pixels[1]) > MIN_PIXELS:
            left_current = int(np.mean(left_pixels[1])) + win_left_x_low
            left_found = True
            
        if len(right_pixels[1]) > MIN_PIXELS:
            right_current = int(np.mean(right_pixels[1])) + win_right_x_low
            right_found = True
    
    # 오프셋 계산
    if left_found and right_found:
        lane_center = (left_current + right_current) // 2
    elif left_found:
        # 왼쪽만 발견 → 오른쪽에 있다고 가정
        lane_center = left_current + 100
    elif right_found:
        # 오른쪽만 발견 → 왼쪽에 있다고 가정
        lane_center = right_current - 100
    else:
        # 둘 다 없음 → 중앙 유지
        lane_center = w // 2
    
    image_center = w // 2
    offset = lane_center - image_center
    
    # 시각화
    cv2.line(output, (lane_center, 0), (lane_center, h), (255, 0, 255), 2)
    cv2.line(output, (image_center, 0), (image_center, h), (255, 255, 0), 1)
    
    return offset, output


def image_callback(msg):
    """이미지 콜백 + 제어"""
    global pub_drive
    
    np_arr = np.frombuffer(msg.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        return
    
    # BEV 변환
    bev = cv2.warpPerspective(img, M, (DST_WIDTH, DST_HEIGHT))
    
    # HSV 이진화
    hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
    lower = np.array([H_MIN, S_MIN, V_MIN])
    upper = np.array([H_MAX, S_MAX, V_MAX])
    binary = cv2.inRange(hsv, lower, upper)
    
    # 오프셋 계산
    offset, visual = calculate_offset(binary)
    
    # ========================================
    # 핵심: 비례 제어
    # ========================================
    # offset > 0: 차선 중심이 오른쪽 → 왼쪽 조향 필요
    # steering > 0: 실제 왼쪽으로 움직임
    # 따라서: steering = KP * offset (부호 같음)
    steering = KP * offset
    
    # 조향 제한
    steering = max(-MAX_STEERING, min(MAX_STEERING, steering))
    
    # 제어 명령 발행
    drive_msg = AckermannDriveStamped()
    drive_msg.header.stamp = rospy.Time.now()
    drive_msg.drive.speed = BASE_SPEED
    drive_msg.drive.steering_angle = steering
    pub_drive.publish(drive_msg)
    
    # 시각화에 정보 추가
    cv2.putText(visual, f"Offset: {offset:+d}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(visual, f"Steering: {steering:+.3f}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(visual, f"Kp: {KP}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    publish_image(pub_visual, visual)


def main():
    global pub_visual, pub_drive, M
    
    rospy.init_node('proportional_control')
    
    # 변환 행렬 계산
    M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
    
    # Publisher 설정
    pub_visual = rospy.Publisher('/practice/control/compressed', CompressedImage, queue_size=1)
    pub_drive = rospy.Publisher('/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1)
    
    # Subscriber 설정
    rospy.Subscriber('/usb_cam/image_raw/compressed', CompressedImage, image_callback, queue_size=1)
    
    rospy.loginfo("=" * 50)
    rospy.loginfo("비례 제어 차선 주행 시작!")
    rospy.loginfo("")
    rospy.loginfo("rqt_image_view: /practice/control")
    rospy.loginfo("")
    rospy.loginfo("RB 버튼으로 자율주행 모드 활성화 필요!")
    rospy.loginfo("Kp 값 조절: KP = %.3f" % KP)
    rospy.loginfo("=" * 50)
    
    rospy.spin()


if __name__ == '__main__':
    main()
