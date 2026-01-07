#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32, Int32, Bool, String
import math
import time


class ConeNode:
    def __init__(self):
        rospy.init_node('cone_node')

        # Subscriber
        self.image_sub = rospy.Subscriber(
            '/usb_cam/image_raw/compressed',
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )
        self.state_pub = rospy.Publisher('/cone_state', String, queue_size=1)
        self.bridge = CvBridge()

        # HSV 및 판단 기준 설정
        self.yellow_low = np.array([15, 60, 60]) # 조금 더 강건한 범위로 수정
        self.yellow_high = np.array([35, 255, 255])
        self.threshold = 0.25
        self.state = None

        # 카메라 보정 데이터 로드
        self.camera_matrix, self.dist_coeffs = self._load_calibration()
        self.under = None
        self.bgr = None
        rospy.loginfo("Cone Node Initialized")

    def _load_calibration(self):
        try:
            # 경로가 정확한지 확인 필요
            calib_file = rospy.get_param('~calibration_file', 
                                        '/home/wego/catkin_ws/src/usb_cam/calibration/usb_cam.yaml')
            with open(calib_file, 'r') as f:
                calib = yaml.safe_load(f)
            cam_mat = np.array(calib['camera_matrix']['data']).reshape(3, 3)
            dist_coef = np.array(calib['distortion_coefficients']['data'])
            rospy.loginfo("Calibration Data Loaded Successfully")
            return cam_mat, dist_coef
        except Exception as e:
            rospy.logwarn(f"Calibration Load Failed: {e}. Using raw image.")
            return None, None

    def undistort(self, img):
        if self.camera_matrix is None:
            return img
        h, w = img.shape[:2]
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.camera_matrix, self.dist_coeffs, (w, h), np.eye(3), balance=0.0
        )
        return cv2.fisheye.undistortImage(img, self.camera_matrix, self.dist_coeffs, Knew=new_K)

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def check_cone_detection(self, yellow_binary_img):
        # 사용자가 제공한 최종 좌표 적용
        roi_coords = [
            {'name': 'Left',   'x': 230, 'y': 181, 'w': 71, 'h': 60, 'color': (0,0,255)},
            {'name': 'Center', 'x': 288, 'y': 179, 'w': 76, 'h': 66, 'color': (0,255,0)},
            {'name': 'Right',  'x': 363, 'y': 175, 'w': 70, 'h': 67, 'color': (255,0,0)}
        ]
        
        detection_bools = []
        detection_ratios = []
        viz_img = cv2.cvtColor(yellow_binary_img, cv2.COLOR_GRAY2BGR) # 시각화용

        for roi in roi_coords:
            x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
            
            # ROI 추출 (이미지 범위를 벗어나지 않게 처리)
            roi_segment = yellow_binary_img[y:y+h, x:x+w]
            
            if roi_segment.size == 0:
                ratio = 0
            else:
                yellow_pixel_count = cv2.countNonZero(roi_segment)
                ratio = yellow_pixel_count / float(w * h)
            
            is_detected = ratio > self.threshold
            detection_bools.append(is_detected)
            detection_ratios.append(ratio)

            # 시각화 박스 그리기 (True면 두껍게)
            thickness = 3 if is_detected else 1
            cv2.rectangle(viz_img, (x, y), (x+w, y+h), roi['color'], thickness)
            cv2.putText(viz_img, f"{ratio:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, roi['color'], 1)

        return detection_bools, detection_ratios, viz_img

    def detect_yellow_line(self, img, min_area=500, min_ratio=5.0):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt) # 단순 w*h보다 contourArea가 더 정확함

            if area < min_area: continue
            if h == 0: continue
            
            ratio = float(w) / float(h)
            if ratio >= min_ratio:
                return "line"
        return "none"

    def main(self):
        if self.bgr is None:
            return
        
        # 1. 왜곡 보정 적용
        undistorted_img = self.undistort(self.bgr)
        
        # 2. HSV 변환 및 마스크 생성
        hsv = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, self.yellow_low, self.yellow_high)
        self.under = yellow_mask[320:480, 0:640]


        # 3. 콘 탐지 로직 (ROI 분석)
        detected, ratios, viz_mask = self.check_cone_detection(yellow_mask)
        
        # 4. 라인 탐지 로직
        line_result = self.detect_yellow_line(self.under)
        
        # 결과 출력 (터미널)
        rospy.loginfo(f"Cone: L:{detected[0]} C:{detected[1]} R:{detected[2]} | Line: {line_result}")
        msg = String()
        if self.state == None:
            if detected[1] and line_result == "line":
                self.state = 'cone_detected'
                msg.data = 'cone_detected'
                self.state_pub.publish(msg)

        elif self.state == 'cone_detected':
            if detected[2]:
                self.state = 'left_cone'
                msg.data = 'left_cone'
                self.state_pub.publish(msg)
                
            elif detected[0]:
                self.state = 'right_cone'
                msg.data = 'right_cone'
                self.state_pub.publish(msg)
            
            #한번만 돌아가도록
            self.state = 'none'


        # 상태 유지 로직 추가 가능 (예: 일정 시간 후 초기화 등) 
        # 5. 화면 표시 (모니터링용)
        cv2.imshow("Detected ROI", viz_mask)
        #cv2.imshow("Original (Undistorted)", undistorted_img)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        node = ConeNode()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            node.main()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()