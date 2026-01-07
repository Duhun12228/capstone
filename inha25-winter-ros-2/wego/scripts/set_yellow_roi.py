#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import yaml
from sensor_msgs.msg import CompressedImage

# 강건한 노란색 HSV 범위
Y_LOW = np.array([15, 60, 60])
Y_HIGH = np.array([35, 255, 255])

class ROIManager:
    def __init__(self):
        # 초기 ROI 설정: [x, y, w, h, color, name]
        self.rois = [
            [84, 155, 150, 150, (0, 0, 255), "Left"],
            [239, 156, 150, 150, (0, 255, 0), "Center"],
            [394, 152, 150, 150, (255, 0, 0), "Right"]
        ]
        self.selected_roi_idx = -1
        self.is_resizing = False # 크기 조절 모드인지 이동 모드인지 구분
        self.drag_start = None
        self.camera_matrix, self.dist_coeffs = self._load_calibration()

    def _load_calibration(self):
        try:
            calib_file = rospy.get_param('~calibration_file', 
                                        '/home/wego/catkin_ws/src/usb_cam/calibration/usb_cam.yaml')
            with open(calib_file, 'r') as f:
                calib = yaml.safe_load(f)
            return np.array(calib['camera_matrix']['data']).reshape(3, 3), np.array(calib['distortion_coefficients']['data'])
        except: return None, None

    def undistort(self, img):
        if self.camera_matrix is None: return img
        h, w = img.shape[:2]
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.camera_matrix, self.dist_coeffs, (w, h), np.eye(3), balance=0.0)
        return cv2.fisheye.undistortImage(img, self.camera_matrix, self.dist_coeffs, Knew=new_K)

    def mouse_callback(self, event, x, y, flags, param):
        margin = 15  # 모서리 판정 범위 (픽셀)

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (rx, ry, rw, rh, _, _) in enumerate(self.rois):
                # 1. 우측 하단 모서리 클릭 시 -> 크기 조절 모드
                if (rx + rw - margin < x < rx + rw + margin) and (ry + rh - margin < y < ry + rh + margin):
                    self.selected_roi_idx = i
                    self.is_resizing = True
                    return
                # 2. 사각형 내부 클릭 시 -> 이동 모드
                elif rx < x < rx + rw and ry < y < ry + rh:
                    self.selected_roi_idx = i
                    self.is_resizing = False
                    self.drag_start = (x - rx, y - ry)
                    return

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selected_roi_idx != -1:
                idx = self.selected_roi_idx
                if self.is_resizing:
                    # 크기 조절: 현재 마우스 좌표를 기준으로 w, h 업데이트 (최소 20)
                    self.rois[idx][2] = max(20, x - self.rois[idx][0])
                    self.rois[idx][3] = max(20, y - self.rois[idx][1])
                else:
                    # 위치 이동
                    off_x, off_y = self.drag_start
                    self.rois[idx][0] = x - off_x
                    self.rois[idx][1] = y - off_y

        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_roi_idx = -1

roi_manager = ROIManager()

def image_callback(msg):
    np_arr = np.frombuffer(msg.data, np.uint8)
    raw_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if raw_img is None: return

    img = roi_manager.undistort(raw_img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, Y_LOW, Y_HIGH)
    
    display_img = img.copy()
    h, w = img.shape[:2]

    for r in roi_manager.rois:
        rx, ry, rw, rh, col, name = r
        y_s, y_e, x_s, x_e = max(0, ry), min(ry+rh, h), max(0, rx), min(rx+rw, w)
        
        # 비율 계산
        roi_mask = yellow_mask[y_s:y_e, x_s:x_e]
        ratio = (cv2.countNonZero(roi_mask) / (rw * rh)) if rw*rh > 0 else 0
        
        # 시각화
        thick = 4 if ratio > 0.25 else 1
        cv2.rectangle(display_img, (rx, ry), (rx+rw, ry+rh), col, thick)
        # 우측 하단 크기조절 핸들 표시 (작은 사각형)
        cv2.rectangle(display_img, (rx+rw-5, ry+rh-5), (rx+rw+5, ry+rh+5), col, -1)
        
        cv2.putText(display_img, f"{name}: {ratio*100:.1f}%", (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    cv2.imshow("Control: Move(Center) / Resize(Bottom-Right)", display_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("Quit")

def main():
    rospy.init_node('cone_filter_node')
    win_name = "Control: Move(Center) / Resize(Bottom-Right)"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, roi_manager.mouse_callback)
    rospy.Subscriber('/usb_cam/image_raw/compressed', CompressedImage, image_callback)
    rospy.spin()
    
    print("\n" + "="*20 + " FINAL ROI COORDINATES " + "="*20)
    for r in roi_manager.rois:
        print(f"{r[5]} ROI: x={r[0]}, y={r[1]}, w={r[2]}, h={r[3]}")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()