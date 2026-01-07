#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage

# ========================================
# 슬라이딩 윈도우 설정 (학생이 수정할 부분)
# ========================================
NUM_WINDOWS = 8        # 윈도우 개수
WINDOW_WIDTH = 80      # 윈도우 폭
MIN_PIXELS = 30        # 윈도우 이동 최소 픽셀

# HSV 임계값 (검은색 차선용)
# 이전 practice 참고하여 수정
H_MIN, H_MAX = 0, 180
S_MIN, S_MAX = 0, 100
V_MIN, V_MAX = 0, 50

# BEV 설정 (04_perspective 참고)
SRC_POINTS = np.float32([
    [100, 400], [200, 300], [440, 300], [540, 400]
])
DST_WIDTH = 400
DST_HEIGHT = 300
DST_POINTS = np.float32([
    [0, DST_HEIGHT], [0, 0], [DST_WIDTH, 0], [DST_WIDTH, DST_HEIGHT]
])

# Publisher (전역)
pub_binary = None
pub_sliding = None

# 변환 행렬 (전역)
M = None


def publish_image(publisher, img):
    """이미지를 CompressedImage로 발행"""
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "jpeg"
    msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
    publisher.publish(msg)


def sliding_window(binary):
    """
    슬라이딩 윈도우로 차선 중심 찾기
    
    Returns:
        lane_x: 각 윈도우의 x 좌표 리스트
        output: 시각화 이미지
    """
    h, w = binary.shape
    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    # ========================================
    # 1단계: 히스토그램으로 시작점 찾기
    # ========================================
    # 하단 절반의 열 합계
    histogram = np.sum(binary[h//2:, :], axis=0)
    
    # 중앙에서 가장 큰 값의 위치 = 차선 시작점
    midpoint = w // 2
    
    # 왼쪽 차선 시작점
    left_base = np.argmax(histogram[:midpoint])
    # 오른쪽 차선 시작점  
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # ========================================
    # 2단계: 슬라이딩 윈도우
    # ========================================
    window_height = h // NUM_WINDOWS
    
    left_current = left_base
    right_current = right_base
    
    left_points = []
    right_points = []
    
    for win_idx in range(NUM_WINDOWS):
        # 윈도우 Y 좌표 (아래에서 위로)
        win_y_low = h - (win_idx + 1) * window_height
        win_y_high = h - win_idx * window_height
        
        # 왼쪽 윈도우 X 좌표
        win_left_x_low = max(0, left_current - WINDOW_WIDTH // 2)
        win_left_x_high = min(w, left_current + WINDOW_WIDTH // 2)
        
        # 오른쪽 윈도우 X 좌표
        win_right_x_low = max(0, right_current - WINDOW_WIDTH // 2)
        win_right_x_high = min(w, right_current + WINDOW_WIDTH // 2)
        
        # 윈도우 그리기
        cv2.rectangle(output, (win_left_x_low, win_y_low), 
                     (win_left_x_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(output, (win_right_x_low, win_y_low),
                     (win_right_x_high, win_y_high), (0, 0, 255), 2)
        
        # 윈도우 내 흰색 픽셀 찾기
        left_roi = binary[win_y_low:win_y_high, win_left_x_low:win_left_x_high]
        right_roi = binary[win_y_low:win_y_high, win_right_x_low:win_right_x_high]
        
        left_pixels = np.where(left_roi > 0)
        right_pixels = np.where(right_roi > 0)
        
        # 충분한 픽셀이 있으면 중심 업데이트
        if len(left_pixels[1]) > MIN_PIXELS:
            new_left = int(np.mean(left_pixels[1])) + win_left_x_low
            left_current = new_left
            left_points.append((left_current, (win_y_low + win_y_high) // 2))
            
        if len(right_pixels[1]) > MIN_PIXELS:
            new_right = int(np.mean(right_pixels[1])) + win_right_x_low
            right_current = new_right
            right_points.append((right_current, (win_y_low + win_y_high) // 2))
    
    # ========================================
    # 3단계: 차선 중심 계산
    # ========================================
    if left_points and right_points:
        lane_center = (left_current + right_current) // 2
        image_center = w // 2
        offset = lane_center - image_center
        
        # 중심선 그리기
        cv2.line(output, (lane_center, 0), (lane_center, h), (255, 0, 255), 2)
        cv2.line(output, (image_center, 0), (image_center, h), (255, 255, 0), 1)
        
        # 오프셋 표시
        cv2.putText(output, f"Offset: {offset}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return output


def image_callback(msg):
    """이미지 콜백"""
    np_arr = np.frombuffer(msg.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        return
    
    # BEV 변환
    bev = cv2.warpPerspective(img, M, (DST_WIDTH, DST_HEIGHT))
    
    # HSV 변환 및 이진화
    hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
    lower = np.array([H_MIN, S_MIN, V_MIN])
    upper = np.array([H_MAX, S_MAX, V_MAX])
    binary = cv2.inRange(hsv, lower, upper)
    
    # 슬라이딩 윈도우
    result = sliding_window(binary)
    
    # 발행
    publish_image(pub_binary, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
    publish_image(pub_sliding, result)


def main():
    global pub_binary, pub_sliding, M
    
    rospy.init_node('sliding_window')
    
    # 변환 행렬 계산
    M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
    
    # Publisher 설정
    pub_binary = rospy.Publisher('/practice/binary/compressed', CompressedImage, queue_size=1)
    pub_sliding = rospy.Publisher('/practice/sliding/compressed', CompressedImage, queue_size=1)
    
    # Subscriber 설정
    rospy.Subscriber('/usb_cam/image_raw/compressed', CompressedImage, image_callback, queue_size=1)
    
    rospy.loginfo("=" * 50)
    rospy.loginfo("슬라이딩 윈도우 차선 검출 시작!")
    rospy.loginfo("rqt_image_view에서 확인:")
    rospy.loginfo("  /practice/binary (이진화)")
    rospy.loginfo("  /practice/sliding (윈도우 + 오프셋)")
    rospy.loginfo("=" * 50)
    
    rospy.spin()


if __name__ == '__main__':
    main()
