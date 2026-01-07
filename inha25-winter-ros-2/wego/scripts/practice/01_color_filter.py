#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage

# ========================================
# HSV 임계값 설정
# ========================================
B_LOW = (15, 50, 50)
B_HIGH = (35, 255, 255)


G_LOW = (30, 80, 80)
G_HIGH = (90, 255, 255)
R_LOW = (0, 100, 100)
R_HIGH = (30, 255, 255)
R2_LOW = (150, 100, 100)
R2_HIGH = (180, 255, 255)

# Publisher (전역)
pub_original = None
pub_blue = None
pub_green = None
pub_red = None


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
    
    # BGR → HSV 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 각 색상 마스크 생성
    b_mask = cv2.inRange(hsv, B_LOW, B_HIGH)
    g_mask = cv2.inRange(hsv, G_LOW, G_HIGH)
    r_mask1 = cv2.inRange(hsv, R_LOW, R_HIGH)
    r_mask2 = cv2.inRange(hsv, R2_LOW, R2_HIGH)
    r_mask = cv2.bitwise_or(r_mask1, r_mask2)
    
    # 마스크 적용
    blue_img = cv2.bitwise_and(img, img, mask=b_mask)
    green_img = cv2.bitwise_and(img, img, mask=g_mask)
    red_img = cv2.bitwise_and(img, img, mask=r_mask)
    cv2.imshow("blue", blue_img)
    cv2.waitKey(1)

    # 발행
    publish_image(pub_original, img)
    publish_image(pub_blue, blue_img)
    publish_image(pub_green, green_img)
    publish_image(pub_red, red_img)


def main():
    global pub_original, pub_blue, pub_green, pub_red
    
    rospy.init_node('color_filter')
    
    # Publisher 설정
    pub_original = rospy.Publisher('/practice/original/compressed', CompressedImage, queue_size=1)
    pub_blue = rospy.Publisher('/practice/blue/compressed', CompressedImage, queue_size=1)
    pub_green = rospy.Publisher('/practice/green/compressed', CompressedImage, queue_size=1)
    pub_red = rospy.Publisher('/practice/red/compressed', CompressedImage, queue_size=1)
    
    # Subscriber 설정
    rospy.Subscriber('/usb_cam/image_raw/compressed', CompressedImage, image_callback, queue_size=1)
    
    rospy.loginfo("=" * 50)
    rospy.loginfo("HSV 색상 분리 시작!")
    rospy.loginfo("rqt_image_view에서 /practice/xxx/compressed 확인")
    rospy.loginfo("=" * 50)
    
    rospy.spin()


if __name__ == '__main__':
    main()
