#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lane Detection Node for WEGO
- HSV-based white lane detection
- Siding_window-based lane center finding
- Stanley control for steering
- Publishes steering/speed for other nodes to use
- Only controls motor when publish_cmd_vel=True
- Only publish debug topics when debug_view=True
"""

import rospy
import cv2 as cv
import numpy as np
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32, Int32, Bool, String
from ackermann_msgs.msg import AckermannDriveStamped
from dynamic_reconfigure.server import Server
from wego.cfg import LaneDetectConfig
import math
import time


class LaneFollow:
    def __init__(self):
        rospy.init_node('lanefollow')
        self.config = None
        self.cv_bridge = CvBridge()

        # Parameters from launch file
        self.publish_cmd_vel = rospy.get_param("~publish_cmd_vel", True)
        self.debug_view = rospy.get_param("~debug_view", True) 

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

        self.obastacle_sub = rospy.Subscriber(
            '/obstacle_state', String, self.obstacle_callback, queue_size=1
        )
        self.cone_sub = rospy.Subscriber(
            '/cone_state', String, self.cone_callback, queue_size=1
        )

        self.debug_publisher1 = rospy.Publisher('/binary_LaneFollow',Image,queue_size = 10)
        self.debug_publisher2 = rospy.Publisher('/sliding_window_debug',Image,queue_size = 10)
        self.debug_publisher3 = rospy.Publisher('/lane_follow_debug',Image,queue_size = 10)

        self.white_lower = np.array(rospy.get_param('~white_lower', [0, 0, 200]), dtype=np.uint8)
        self.white_upper = np.array(rospy.get_param('~white_upper', [180, 40, 255]), dtype=np.uint8)

        self.yellow_lower = np.array(rospy.get_param('~yellow_lower', [20, 40, 100]), dtype=np.uint8)
        self.yellow_upper = np.array(rospy.get_param('~yellow_upper',[38, 110, 255]), dtype=np.uint8)   

        # Stop flag (from traffic light or other nodes)
        self.stop_flag = False

        # Load fisheye calibration
        self.camera_matrix, self.dist_coeffs = self._load_calibration()
 
        # Dynamic Reconfigure
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

        self.src_points= np.float32([
            [0, 310],
            [640, 310],
            [0, 480],
            [640, 480]
        ])
        self.dst_points= np.float32([
            [0,   310],
            [640,   310],
            [225 , 480],
            [415, 480]
        ])

        self.warp_mat = cv.getPerspectiveTransform(self.src_points,self.dst_points)
        self.inv_warp_mat = cv.getPerspectiveTransform(self.dst_points,self.src_points)
        
        # ---------- PARAMETERS ----------
        # Speeds
        self.forward_speed = rospy.get_param("~forward_speed", 0.22)
        self.reverse_speed = rospy.get_param("~reverse_speed", -0.22)

        # Sharp turn steering angle
        self.sharp_left = rospy.get_param("~sharp_left", math.radians(43.0))
        self.straight = 0.0

        # Timings (tune on track)
        self.turn_into_time = rospy.get_param("~turn_into_time", 2.7)       # swing left
        self.return_time = rospy.get_param("~return_time",2.7)             # return back
        self.forward_105_time = rospy.get_param("~forward_105_time", 4.5)   # forward 1.05m

        # publish rate
        self.control_rate = rospy.get_param("~control_rate", 20.0)
        self.bgr = None
        self.warp_img_ori = None
        self.warp_img = None
        self.white_img = None
        self.filtered_img = None
        self.gaussian_sigma = 1
        self.gear = 3 # 3.이 default
        self.yaw = 0
        self.error = 0
        self.steer  = 0
        self.state = None
        self.last_speed = 0.0  # For speed smoothing
        self.cone_left = 0
        self.cone_right = 0
        self.is_gone = False

        rospy.loginfo("="*50)
        rospy.loginfo("lanefollow node initialized")
        rospy.loginfo(f"publish_cmd_vel: {self.publish_cmd_vel}")
        rospy.loginfo("Steering topic: /webot/steering_offset")
        rospy.loginfo("Speed topic: /webot/lane_speed")
        rospy.loginfo("View: rqt_image_view /webot/lane_detect/image")
        rospy.loginfo("="*50)
       
    def obstacle_callback(self,msg):
        self.state = msg.data
        #rospy.loginfo(f'[LaneFollow] Obstacle state updated: {self.state}')

    def cone_callback(self,msg):
        if msg.data == 'left_cone':
            self.cone_left +=1
        elif msg.data == 'right_cone':
            self.cone_right +=1

        #rospy.loginfo(f'[LaneFollow] Cone state updated: {self.state}')

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
        new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.camera_matrix, self.dist_coeffs, (w, h), np.eye(3), balance=0.0
        )
        return cv.fisheye.undistortImage(
            img, self.camera_matrix, self.dist_coeffs, Knew=new_K
        )
    
    def reconfigure_callback(self, config, level):
        self.config = config
        rospy.loginfo(f"[LaneFollow] Config updated: speed={config.base_speed}, k={config.k}, yaw_k={config.yaw_k}")
        return config
    
    def stop_callback(self, msg):
        """Callback for stop flag from traffic light"""
        self.stop_flag = msg.data

    def warpping(self,img):
        h,w = img.shape[:2]
        warp_img = cv.warpPerspective(img,self.warp_mat,(w,h))
        return warp_img
    
    def Gaussian_filter(self,img):
        filtered_img = cv.GaussianBlur(img,(0,0),self.gaussian_sigma)
        return filtered_img
    
    def white_color_filter_hsv(self,img):
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        white_hsv = cv.inRange(hsv,self.white_lower,self.white_upper)
        return white_hsv

    def roi_set(self,img):
        roi_img = img[310:480,0:640]
        return roi_img    
    
    def sliding_window(self,img,n_windows=15,margin = 12,minpix = 3):
        y = img.shape[0]
        x = img.shape[1]
        
        hist_area = np.copy(img[y // 2:, :])
        
        center_x = x // 2
        #기본 30px로 설정
        mask_width = self.config.masked_pixel if self.config else 30

        start_col = center_x - (mask_width // 2) 
        end_col = center_x + (mask_width // 2) + (mask_width % 2) 
        
        hist_area[:, start_col:end_col] = 0 
        
        histogram = np.sum(hist_area, axis=0)
        midpoint = int(histogram.shape[0]/2)
        leftx_current = np.argmax(histogram[:midpoint])
        
        #Fallback 오른쪽 차선 검출 안된 경우
        if sum(histogram[midpoint:]) < 15:
            rightx_current = midpoint*2
        else:
            rightx_current = np.argmax(histogram[midpoint:]) + midpoint
        
        window_height = int(y/n_windows)
        nz = img.nonzero()

        left_lane_inds = []
        right_lane_inds = []
    
        lx, ly, rx, ry = [], [], [], []

        out_img = np.dstack((img,img,img))*255

        for window in range(n_windows):
                
            win_yl = y - (window+1)*window_height
            win_yh = y - window*window_height

            win_xll = leftx_current - margin  
            win_xlh = leftx_current + margin
            win_xrl = rightx_current - margin
            win_xrh = rightx_current + margin

            cv.rectangle(out_img,(win_xll,win_yl),(win_xlh,win_yh),(0,255,0), 2) 
            cv.rectangle(out_img,(win_xrl,win_yl),(win_xrh,win_yh),(0,255,0), 2) 

            # 슬라이딩 윈도우 박스(녹색박스) 하나 안에 있는 흰색 픽셀의 x좌표를 모두 모은다.
            good_left_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xll)&(nz[1] < win_xlh)).nonzero()[0]
            good_right_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xrl)&(nz[1] < win_xrh)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # 구한 x좌표 리스트에서 흰색점이 5개 이상인 경우에 한해 x 좌표의 평균값을 구함. -> 이 값을 슬라이딩 윈도우의 중심점으로 사용
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nz[1][good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = int(np.mean(nz[1][good_right_inds]))
            

            lx.append(leftx_current)
            ly.append((win_yl + win_yh)/2)

            rx.append(rightx_current)
            ry.append((win_yl + win_yh)/2)

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        lfit = np.polyfit(np.array(ly),np.array(lx),1)
        rfit = np.polyfit(np.array(ry),np.array(rx),1)

        out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
        out_img[nz[0][right_lane_inds] , nz[1][right_lane_inds]] = [0, 0, 255]

        #cv.imshow("viewer", out_img)

        if self.debug_view:
            self.debug_publisher2.publish(self.cv_bridge.cv2_to_imgmsg(out_img))
        
        return lfit, rfit

    def cal_center_line(self, lfit, rfit):
        """
        lfit, rfit : np.polyfit으로 구한 왼쪽/오른쪽 차선의 1차 다항식 계수
                     x = a*y + b 형태 (len == 2)
        반환값:
            yaw   : 중앙 차선의 진행 방향 각도 (라디안)
            error : 차량(이미지 중앙) 기준, 차선 중앙의 x 오프셋(px)
        """


        cfit = (lfit + rfit) / 2.0  # [a, b]

        if self.filtered_img is not None:
            h, w = self.filtered_img.shape[:2]
        else:
            h, w = 160, 640

        y_eval = h * 0.75  # 이미지 높이의 3/4 지점을 계산에 사용

        
        a, b = cfit
        x_center = a * (y_eval) + b 

        #기울기 계산: a
        dx_dy = a
        yaw = np.arctan(dx_dy)  # 전방(y 방향) 기준 x의 변화량에 대한 각도

        #차량을 이미지 가로 중앙에 있다고 가정하고, 중앙선과의 오프셋 계산
        img_center_x = w / 2.0
        error =  - x_center + img_center_x  

        self.pub_center_x.publish(Int32(x_center))

        return yaw, error,x_center
    
    def cal_steering(self, yaw, error, gear=3, k=0.005, yaw_k=1.0):

        # --- 1) 상태에 따른 목표 속도(target_speed) ---
        if self.state == 'STOP':
            target_speed = 0.0
        else:
            target_speed = self.config.base_speed
            if self.state == 'SLOW':
                rospy.loginfo("[LaneFollow] SLOW MODE ACTIVATED")
                target_speed = 0.20

        # stop_flag가 최우선
        if self.stop_flag:
            target_speed = 0.0

        # # --- 2) 속도 스무싱(EMA) ---
        # alpha = 0.3  # 0~1 (낮을수록 더 부드러움)
        # smooth_speed = self.last_speed + alpha * (target_speed - self.last_speed)
        # self.last_speed = smooth_speed

        # --- 3) 조향 계산 ---
        k = 0.01
        yaw_k = self.config.yaw_k

        # 속도가 0에 가까우면 atan2 분모가 0되니까 방지
        safe_speed = target_speed
        steering = yaw_k * yaw + np.arctan2(k * error, safe_speed)
        self.steer = steering

        # --- 4) publish ---
        self.pub_steering.publish(Float32(self.steer))
        self.pub_speed.publish(Float32(target_speed))

        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'LaneFollow'

        if self.stop_flag or self.state == 'STOP':
            msg.drive.speed = 0.0
            msg.drive.steering_angle = 0.0
        else:
            msg.drive.speed = float(target_speed)
            msg.drive.steering_angle = float(steering)

        if self.publish_cmd_vel:
            self.cmd_vel_pub.publish(msg)


    def image_callback(self,msg):
        if self.config is None:
            return
        
        try:
            cv_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
            if cv_image is None:
                return
            
            self.bgr = self.undistort(cv_image)

        except Exception as e:
            rospy.logerr("[LaneFollow] Error: %s", str(e))

    def draw_lane(self,image, warp_roi,warp_ori,inv_mat, left_fit, right_fit):
            """
            image    : 원본 BGR 이미지
            warp_roi : ROI만 잘라낸 warp 이미지 (self.warp_img)
            inv_mat  : self.inv_warp_mat
            left_fit, right_fit : ROI 좌표계 기준 polyfit 결과
            """


            full_h, full_w = warp_ori.shape[:2]
            roi_h, roi_w   = warp_roi.shape[:2]

            roi_offset_y = 310

            yMax = roi_h
            ploty = np.linspace(0, yMax - 1, yMax)

            # ROI 기준 x좌표
            left_fitx  = left_fit[0] * ploty + left_fit[1] 
            right_fitx = right_fit[0] * ploty + right_fit[1] 

            ploty_full = ploty + 310  

            pts_left  = np.array([np.transpose(np.vstack([left_fitx,  ploty_full]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty_full])))])

            pts = np.hstack((pts_left, pts_right))

            # 전체 warp 크기의 빈 컬러 이미지 만들고 lane area 채우기
            color_warp = np.zeros_like(warp_ori).astype(np.uint8)  # full_h x full_w

            cv.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))

            # 역원근변환으로 원본 이미지 좌표계로 되돌리고 오버레이
            newwarp = cv.warpPerspective(color_warp, inv_mat, (image.shape[1], image.shape[0]))
            result = cv.addWeighted(image, 1, newwarp, 0.3, 0)
            
            text1 = f"yaw: {self.yaw:.3f} rad ({self.steer:.1f} rad)"
            text2 = f"err: {self.error:.1f} px"

            #디버깅 텍스트 추가
            cv.putText(result, text1, (30, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)
            cv.putText(result, text2, (30, 110),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv.LINE_AA)

            return result

    
    def sliding_window_right(self,img,n_windows=10,margin = 12,minpix = 5):
        y = img.shape[0]
        x =img.shape[1]
        # 1. 히스토그램을 계산할 이미지 영역 (아래쪽 절반) 복사
        hist_area = np.copy(img[y // 2:, :])
        
        # 2. 가운데 15px 영역 정의
        center_x = x // 2
        mask_width = 0
        
        # 제외할 영역의 시작과 끝 인덱스 계산
        # 정수 나눗셈 // 을 사용하여 계산
        start_col = center_x - (mask_width // 2) 
        end_col = center_x + (mask_width // 2) + (mask_width % 2) 
        
        # 3. 해당 영역의 픽셀 값을 0으로 설정 (마스킹)
        # 이미지 아래쪽 절반 (hist_area)에 적용
        hist_area[:, start_col:end_col] = 0 
        
        # 4. 마스킹된 이미지로 히스토그램 계산
        histogram = np.sum(hist_area, axis=0)
        midpoint = int(histogram.shape[0]/2)
        leftx_current = np.argmax(histogram[:midpoint])
        
        if sum(histogram[midpoint:]) < 15:
            rightx_current = midpoint*2
        else:
            rightx_current = np.argmax(histogram[midpoint:]) + midpoint
        
        window_height = int(y/n_windows)
        nz = img.nonzero()

        right_lane_inds = []
        rx, ry = [], []

        out_img = np.dstack((img,img,img))*255

        for window in range(n_windows):
            win_yl = y - (window+1)*window_height
            win_yh = y - window*window_height

            win_xrl = rightx_current - margin
            win_xrh = rightx_current + margin

            cv.rectangle(out_img,(win_xrl,win_yl),(win_xrh,win_yh),(0,255,0), 2) 

            good_right_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&
                               (nz[1] >= win_xrl)&(nz[1] < win_xrh)).nonzero()[0]

            right_lane_inds.append(good_right_inds)

            if len(good_right_inds) > minpix:        
                rightx_current = int(np.mean(nz[1][good_right_inds]))

            rx.append(rightx_current)
            ry.append((win_yl + win_yh)/2)

        right_lane_inds = np.concatenate(right_lane_inds)

        rfit = np.polyfit(np.array(ry),np.array(rx),1)
        
        out_img[nz[0][right_lane_inds] , nz[1][right_lane_inds]] = [0, 0, 255]
        cv.imshow("right_viewer", out_img)

        return rfit

    def sliding_window_left(self,img,n_windows=10,margin = 12,minpix = 5):
        y = img.shape[0]
        x = img.shape[1]

        # 1. 히스토그램을 계산할 이미지 영역 (아래쪽 절반) 복사
        hist_area = np.copy(img[y // 2:, :])
        
        # 2. 가운데 15px 영역 정의
        center_x = x // 2
        mask_width = 0
        
        # 제외할 영역의 시작과 끝 인덱스 계산
        # 정수 나눗셈 // 을 사용하여 계산
        start_col = center_x - (mask_width // 2) 
        end_col = center_x + (mask_width // 2) + (mask_width % 2) 
        
        # 3. 해당 영역의 픽셀 값을 0으로 설정 (마스킹)
        # 이미지 아래쪽 절반 (hist_area)에 적용
        hist_area[:, start_col:end_col] = 0 
        
        # 4. 마스킹된 이미지로 히스토그램 계산
        histogram = np.sum(hist_area, axis=0)
        midpoint = int(histogram.shape[0]/2)
        leftx_current = np.argmax(histogram[:midpoint])
        
        if sum(histogram[midpoint:]) < 15:
            rightx_current = midpoint*2
        else:
            rightx_current = np.argmax(histogram[midpoint:]) + midpoint
            
        
        window_height = int(y/n_windows)
        nz = img.nonzero()

        left_lane_inds = []
        right_lane_inds = []
    
        lx, ly, rx, ry = [], [], [], []

        out_img = np.dstack((img,img,img))*255

        for window in range(n_windows):
                
            win_yl = y - (window+1)*window_height
            win_yh = y - window*window_height

            win_xll = leftx_current - margin  # 녹색사각형 크기 : 가로 24, 세로 26
            win_xlh = leftx_current + margin

            cv.rectangle(out_img,(win_xll,win_yl),(win_xlh,win_yh),(0,255,0), 2) 

            # 슬라이딩 윈도우 박스(녹색박스) 하나 안에 있는 흰색 픽셀의 x좌표를 모두 모은다.
            good_left_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xll)&(nz[1] < win_xlh)).nonzero()[0]

            left_lane_inds.append(good_left_inds)

            # 구한 x좌표 리스트에서 흰색점이 5개 이상인 경우에 한해 x 좌표의 평균값을 구함. -> 이 값을 슬라이딩 윈도우의 중심점으로 사용
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nz[1][good_left_inds]))

            lx.append(leftx_current)
            ly.append((win_yl + win_yh)/2)


        left_lane_inds = np.concatenate(left_lane_inds)

        # 슬라이딩 윈도우의 중심점(x좌표) 9개를 가지고 2차 함수를 만들어낸다.    
        lfit = np.polyfit(np.array(ly),np.array(lx),1)

        # 왼쪽과 오른쪽 각각 파란색과 빨간색으로 색상 변경
        out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
        cv.imshow("viewer", out_img)
        
        return lfit

    def cal_center_line_right(self, rfit):
        a,b = rfit
        cfit = [a,b-130]  # 오른쪽 차선에서 중앙선으로 보정

        h, w = 170, 640

        y_eval = h * 0.75

        a, b = cfit
        x_center = a * (y_eval) + b 

        dx_dy = a
        yaw = np.arctan(dx_dy)

        img_center_x = w / 2.0
        error =  - x_center + img_center_x

        return yaw, error

    def cal_center_line_left(self, lfit):
        a,b = lfit
        cfit = [a+130,b]  # 오른쪽 차선에서 중앙선으로 보정

        h, w = 170, 640

        y_eval = h * 0.75

        a, b = cfit
        x_center = a * (y_eval) + b 

        dx_dy = a
        yaw = np.arctan(dx_dy)

        img_center_x = w / 2.0
        error =  - x_center + img_center_x

        return yaw, error

    # --- Low-level send ---
    def send_cmd(self, speed, steering):
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.ack_pub_1.publish(msg)

    # --- Hold movement for duration ---
    def timed_move(self, speed, steering, duration):
        rate = rospy.Rate(self.control_rate)
        end_time = rospy.Time.now() + rospy.Duration.from_sec(duration)

        while not rospy.is_shutdown() and rospy.Time.now() < end_time:
            self.send_cmd(speed, steering)
            rate.sleep()

        # brief brake
        self.send_cmd(0.0, 0.0)
        rospy.sleep(0.15)



    def main(self):
        if self.bgr is None:
            return
        cv.imshow("original", self.bgr)
        cv.waitKey(1)
        self.warp_img_ori = self.warpping(self.bgr)
        self.warp_img = self.roi_set(self.warp_img_ori)

        g_filltered = self.Gaussian_filter(self.warp_img)
        self.white_image = self.white_color_filter_hsv(g_filltered)

        if self.debug_view:
            self.debug_publisher1.publish(self.cv_bridge.cv2_to_imgmsg(self.white_image))

        # lfit,rfit = self.sliding_window(self.white_image)
        # self.yaw,self.error,x_center = self.cal_center_line(lfit,rfit)

        if self.cone_left > 0:
            if self.is_gone == False:
                hard_code = HardCode()
                hard_code.turn_left()
                self.is_gone = True
            lfit = self.sliding_window_left(self.white_image)
            self.yaw,self.error= self.cal_center_line_left(lfit)
        
        elif self.cone_right > 0:
            if self.is_gone == False:
                hard_code = HardCode()
                hard_code.trun_right()
                self.is_gone = True
            rfit = self.sliding_window_right(self.white_image)
            self.yaw,self.error= self.cal_center_line_right(rfit)
        else:
            rfit = self.sliding_window_right(self.white_image)
            self.yaw,self.error= self.cal_center_line_right(rfit)

        self.cal_steering(yaw=self.yaw,error=self.error)

        #debug_img = self.draw_lane(self.bgr,self.warp_img,self.warp_img_ori,self.inv_warp_mat,lfit,rfit)
        
        if self.debug_view:
            self.debug_publisher1.publish(self.cv_bridge.cv2_to_imgmsg(self.white_image, encoding="mono8"))
            #self.debug_publisher3.publish(self.cv_bridge.cv2_to_imgmsg(debug_img, encoding="bgr8"))
    






class HardCode:

    def __init__(self):
        self.cmd_vel_pub = rospy.Publisher('/low_level/ackermann_cmd_mux/input/navigation', 
                                                AckermannDriveStamped, queue_size=1)

        # ---------- PARAMETERS ----------
        # Speeds
        self.forward_speed = rospy.get_param("~forward_speed", 0.22)
        self.reverse_speed = rospy.get_param("~reverse_speed", -0.22)

        # Sharp turn steering angle
        self.sharp_left = rospy.get_param("~sharp_left", math.radians(20.0))
        self.straight = 0.0

        # Timings (tune on track)
        self.turn_into_time = rospy.get_param("~turn_into_time", 2.7)       # swing left
        self.return_time = rospy.get_param("~return_time",2.7)             # return back
        self.forward_105_time = rospy.get_param("~forward_105_time", 4.5)   # forward 1.05m

        # publish rate
        self.control_rate = rospy.get_param("~control_rate", 20.0)

    # --- Low-level send ---
    def send_cmd(self, speed, steering):
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.ack_pub_1.publish(msg)

    # --- Hold movement for duration ---
    def timed_move(self, speed, steering, duration):
        rate = rospy.Rate(self.control_rate)
        end_time = rospy.Time.now() + rospy.Duration.from_sec(duration)

        while not rospy.is_shutdown() and rospy.Time.now() < end_time:
            self.send_cmd(speed, steering)
            rate.sleep()

        # brief brake
        self.send_cmd(0.0, 0.0)
        rospy.sleep(0.15)

    def turn_left(self):
        rospy.loginfo("=== NEW OFFICIAL PARKING START ===")
        rospy.sleep(3.0)
        # --------------------------
        # STEP 1 → Sharp left turn into the parking area
        # --------------------------
        rospy.loginfo("STEP 1: Sharp LEFT swing into parking zone")
        self.timed_move(self.forward_speed,0,0.4)
        self.timed_move(self.forward_speed, self.sharp_left, self.turn_into_time)

        # --------------------------
        # STEP 2 → STOP & stay inside for 3 seconds
        # --------------------------
        rospy.loginfo("STEP 2: STOPPING inside parking zone for REQUIRED 3 seconds")
        self.send_cmd(0.0, 0.0)   # full stop
        rospy.sleep(2.0)

        # --------------------------
        # STEP 3 → Return back to white line (reverse)
        # --------------------------
        rospy.loginfo("STEP 3: Returning back to white line")
        self.timed_move(self.forward_speed,0.0,0.2)

        # --------------------------
        # STEP 5 → Final stop / mission complete
        # --------------------------
        rospy.loginfo("=== PARKING MISSION COMPLETE ===")

        self.send_cmd(0.0, 0.0)
        rospy.sleep(0.2)


    def turn_right(self):
        pass


if __name__ == '__main__':
    try:
        lf = LaneFollow()
        rate = rospy.Rate(30)  # 30Hz

        while not rospy.is_shutdown():
            lf.main()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass

