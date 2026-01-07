#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage

# ========================================
# 원근 변환 설정 (학생이 수정할 부분)
# ========================================
# 원본 이미지에서 선택할 4개 점 (사다리꼴)
# 좌하단 → 좌상단 → 우상단 → 우하단 순서
SRC_POINTS = np.float32([
    [100, 400],   # 좌하단
    [200, 300],   # 좌상단
    [440, 300],   # 우상단
    [540, 400]    # 우하단
])

# 변환 후 이미지 크기
DST_WIDTH = 400
DST_HEIGHT = 300

# 변환 후 4개 점 (직사각형)
DST_POINTS = np.float32([
    [0, DST_HEIGHT],       # 좌하단
    [0, 0],                # 좌상단
    [DST_WIDTH, 0],        # 우상단
    [DST_WIDTH, DST_HEIGHT] # 우하단
])

# Publisher (전역)
pub_original = None
pub_bev = None

# 변환 행렬 (전역)
M = None


def publish_image(publisher, img):
    """이미지를 CompressedImage로 발행"""
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "jpeg"
    msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
    publisher.publish(msg)


def image_callback(msg):
    """이미지 콜백"""
    np_arr = np.frombuffer(msg.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        return
    
    # 원본에 ROI 표시
    original = img.copy()
    pts = SRC_POINTS.astype(np.int32)
    cv2.polylines(original, [pts], True, (0, 255, 0), 2)
    
    # 점 번호 표시
    for i, pt in enumerate(pts):
        cv2.circle(original, tuple(pt), 5, (0, 0, 255), -1)
        cv2.putText(original, str(i+1), (pt[0]+10, pt[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # ========================================
    # 핵심: 원근 변환
    # ========================================
    bev = cv2.warpPerspective(img, M, (DST_WIDTH, DST_HEIGHT))
    
    # 발행
    publish_image(pub_original, original)
    publish_image(pub_bev, bev)


def main():
    global pub_original, pub_bev, M
    
    rospy.init_node('perspective_transform')
    
    # 변환 행렬 계산
    M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
    
    # Publisher 설정
    pub_original = rospy.Publisher('/practice/original/compressed', CompressedImage, queue_size=1)
    pub_bev = rospy.Publisher('/practice/bev/compressed', CompressedImage, queue_size=1)
    
    # Subscriber 설정
    rospy.Subscriber('/usb_cam/image_raw/compressed', CompressedImage, image_callback, queue_size=1)
    
    rospy.loginfo("=" * 50)
    rospy.loginfo("BEV 원근 변환 시작!")
    rospy.loginfo("rqt_image_view에서 확인:")
    rospy.loginfo("  /practice/original (ROI 표시)")
    rospy.loginfo("  /practice/bev (변환 결과)")
    rospy.loginfo("=" * 50)
    
    rospy.spin()


if __name__ == '__main__':
    main()
