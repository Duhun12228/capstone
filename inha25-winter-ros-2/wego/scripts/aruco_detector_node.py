#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArUco Marker Detector Node for WEGO
- Detects ArUco markers using OpenCV
- Applies fisheye calibration (undistort)
- Publishes detected marker IDs and distances
- Publishes visualization image (calibration applied)

Topics:
  - /webot/aruco/markers (Int32MultiArray): 감지된 마커 ID 배열
  - /webot/aruco/marker_info (Float32MultiArray): [id, distance, id, distance, ...]
  - /webot/aruco/debug (Image): 시각화 이미지 (calibration 적용됨)
"""

import rospy
import cv2
import numpy as np
import yaml
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int32MultiArray, Float32MultiArray
from cv_bridge import CvBridge


class ArucoDetectorNode:
    def __init__(self):
        rospy.init_node('aruco_detector_node')
        
        self.bridge = CvBridge()
        
        # ArUco 딕셔너리 선택 (4x4, 5x5, 6x6 등)
        # OpenCV 버전에 따라 다른 API 사용
        try:
            # OpenCV 4.7+
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
        except AttributeError:
            # OpenCV 4.0 ~ 4.6
            try:
                self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
                self.aruco_params = cv2.aruco.DetectorParameters_create()
                self.use_new_api = False
            except AttributeError:
                # OpenCV 3.x 또는 다른 버전
                self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                self.aruco_params = cv2.aruco.DetectorParameters_create()
                self.use_new_api = False
        
        # 마커 실제 크기 (미터 단위)
        self.marker_size = rospy.get_param('~marker_size', 0.05)  # 5cm 기본값
        
        # ========================================
        # Fisheye 카메라 캘리브레이션 로드
        # ========================================
        self.camera_matrix, self.dist_coeffs = self._load_calibration()
        
        # Undistort용 카메라 행렬 (캐싱)
        self.new_K = None
        self.map1 = None
        self.map2 = None
        
        # Publishers - webot 토픽 프리픽스
        self.pub_image = rospy.Publisher('/webot/aruco/debug', Image, queue_size=1)
        self.pub_image_compressed = rospy.Publisher('/webot/aruco/debug/compressed', CompressedImage, queue_size=1)
        self.pub_markers = rospy.Publisher('/webot/aruco/markers', Int32MultiArray, queue_size=1)
        self.pub_marker_info = rospy.Publisher('/webot/aruco/marker_info', Float32MultiArray, queue_size=1)
        
        # Subscriber - wego 카메라 토픽
        self.sub_image = rospy.Subscriber(
            '/usb_cam/image_raw/compressed',
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )
        
        rospy.loginfo("=" * 50)
        rospy.loginfo("ArUco Detector Node initialized (WEGO)")
        rospy.loginfo(f"Dictionary: DICT_4X4_50")
        rospy.loginfo(f"Marker size: {self.marker_size}m")
        rospy.loginfo(f"Calibration: {'Loaded' if self.camera_matrix is not None else 'Not loaded'}")
        rospy.loginfo("Topics:")
        rospy.loginfo("  - /webot/aruco/markers (IDs)")
        rospy.loginfo("  - /webot/aruco/marker_info (ID+Distance)")
        rospy.loginfo("  - /webot/aruco/debug (Image)")
        rospy.loginfo("=" * 50)
    
    def _load_calibration(self):
        """Fisheye 카메라 캘리브레이션 파일 로드"""
        try:
            calib_file = rospy.get_param('~calibration_file',
                '/home/wego/catkin_ws/src/usb_cam/calibration/usb_cam.yaml')
            with open(calib_file, 'r') as f:
                calib = yaml.safe_load(f)
            camera_matrix = np.array(calib['camera_matrix']['data']).reshape(3, 3)
            dist_coeffs = np.array(calib['distortion_coefficients']['data'])
            rospy.loginfo(f"[ArUco] Calibration loaded from {calib_file}")
            return camera_matrix, dist_coeffs
        except Exception as e:
            rospy.logwarn(f"[ArUco] Calibration load failed: {e}")
            # 기본 카메라 파라미터 (fallback)
            camera_matrix = np.array([
                [600.0, 0.0, 320.0],
                [0.0, 600.0, 240.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return camera_matrix, dist_coeffs
    
    def undistort(self, img):
        """Fisheye 왜곡 보정 적용"""
        if self.camera_matrix is None:
            return img
        
        h, w = img.shape[:2]
        
        # 처음 한 번만 맵 계산 (캐싱)
        if self.new_K is None or self.map1 is None:
            try:
                self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    self.camera_matrix, self.dist_coeffs, (w, h), np.eye(3), balance=0.0
                )
                self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                    self.camera_matrix, self.dist_coeffs, np.eye(3), self.new_K, (w, h), cv2.CV_16SC2
                )
            except Exception as e:
                rospy.logwarn_throttle(5, f"[ArUco] Undistort map init failed: {e}")
                return img
        
        try:
            return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
        except Exception as e:
            rospy.logwarn_throttle(5, f"[ArUco] Undistort failed: {e}")
            return img
    
    def detect_markers(self, img):
        """ArUco 마커 감지"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.use_new_api:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        return corners, ids
    
    def estimate_distance(self, corners):
        """마커와의 거리 추정 (미터)"""
        try:
            # Undistort 적용 후에는 new_K 사용, 아니면 원본 camera_matrix 사용
            cam_matrix = self.new_K if self.new_K is not None else self.camera_matrix
            # Undistort된 이미지에서는 dist_coeffs = 0
            dist_coeffs = np.zeros(5)
            
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, cam_matrix, dist_coeffs
            )
            
            distances = []
            for tvec in tvecs:
                distance = tvec[0][2]
                distances.append(distance)
            
            return distances, rvecs, tvecs
        except Exception as e:
            rospy.logwarn_throttle(5, f"Distance estimation failed: {e}")
            return None, None, None
    
    def image_callback(self, msg):
        """이미지 콜백"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is None:
                return
            
            # ========================================
            # Fisheye Undistort 적용
            # ========================================
            img = self.undistort(img)
            
            corners, ids = self.detect_markers(img)
            
            vis_img = img.copy()
            detected_ids = []
            marker_info = []
            
            if ids is not None and len(corners) > 0:
                cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)
                
                distances, rvecs, tvecs = self.estimate_distance(corners)
                
                # Undistort 적용 후 사용할 카메라 행렬
                cam_matrix = self.new_K if self.new_K is not None else self.camera_matrix
                dist_coeffs = np.zeros(5)
                
                for i, marker_id in enumerate(ids.flatten()):
                    detected_ids.append(int(marker_id))
                    
                    c = corners[i][0]
                    cx = int(np.mean(c[:, 0]))
                    cy = int(np.mean(c[:, 1]))
                    
                    if distances is not None and i < len(distances):
                        dist = distances[i]
                        marker_info.extend([float(marker_id), dist])
                        
                        cv2.putText(vis_img, f"ID:{marker_id} D:{dist:.2f}m", (cx - 50, cy - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        if rvecs is not None and tvecs is not None:
                            cv2.drawFrameAxes(vis_img, cam_matrix, dist_coeffs,
                                            rvecs[i], tvecs[i], self.marker_size * 0.5)
                        
                        rospy.loginfo_throttle(1, f"ArUco ID:{marker_id} Distance:{dist:.3f}m")
                    else:
                        cv2.putText(vis_img, f"ID:{marker_id}", (cx - 30, cy - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 상태 표시
            status_text = f"Markers: {len(detected_ids)}"
            if detected_ids:
                status_text += f" | IDs: {detected_ids}"
            cv2.putText(vis_img, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis_img, "Calibrated", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Publish markers
            if detected_ids:
                marker_msg = Int32MultiArray(data=detected_ids)
                self.pub_markers.publish(marker_msg)
            
            # Publish marker info
            if marker_info:
                info_msg = Float32MultiArray(data=marker_info)
                self.pub_marker_info.publish(info_msg)
            
            # Publish image
            if self.pub_image.get_num_connections() > 0:
                self.pub_image.publish(self.bridge.cv2_to_imgmsg(vis_img, "bgr8"))
            
            # Publish compressed
            if self.pub_image_compressed.get_num_connections() > 0:
                _, compressed = cv2.imencode('.jpg', vis_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                compressed_msg = CompressedImage()
                compressed_msg.header.stamp = rospy.Time.now()
                compressed_msg.format = "jpeg"
                compressed_msg.data = compressed.tobytes()
                self.pub_image_compressed.publish(compressed_msg)
            
        except Exception as e:
            rospy.logerr(f"ArUco detect error: {e}")
    
    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = ArucoDetectorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
