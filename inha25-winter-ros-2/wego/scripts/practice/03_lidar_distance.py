#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

TARGET_DISTANCE = 0.3   # 목표 거리 (m)
STOP_DISTANCE = 0.15    # 정지 거리 (m)
NORMAL_SPEED = 0.5      # 정상 속도 (m/s)
SLOW_SPEED = 0.2        # 감속 속도 (m/s)
FRONT_WINDOW = 50       # 전방 감지 범위 (포인트 수)

# Publisher (전역)
drive_pub = None


def scan_callback(msg):
    """LiDAR 콜백"""
    ranges = np.array(msg.ranges)
    total = len(ranges)
    
    # 전방 범위 추출 (인덱스 0 근처가 전방)
    front_left = ranges[:FRONT_WINDOW]
    front_right = ranges[total-FRONT_WINDOW:]
    front_ranges = np.concatenate([front_left, front_right])
    
    # 최소 거리 계산 (0이나 무한대 제외)
    valid = front_ranges[(front_ranges > 0.01) & (front_ranges < 10.0)]
    
    if len(valid) == 0:
        min_dist = 10.0
    else:
        min_dist = np.min(valid)
    
    # 속도 결정
    if min_dist < STOP_DISTANCE:
        speed = 0.0
        status = "STOP"
    elif min_dist < TARGET_DISTANCE:
        speed = SLOW_SPEED
        status = "SLOW"
    else:
        speed = NORMAL_SPEED
        status = "GO"
    
    # 명령 발행
    drive_msg = AckermannDriveStamped()
    drive_msg.header.stamp = rospy.Time.now()
    drive_msg.drive.speed = speed
    drive_msg.drive.steering_angle = 0.0
    drive_pub.publish(drive_msg)
    
    # 터미널 출력
    rospy.loginfo_throttle(0.3, f"전방: {min_dist:.2f}m | {status} | {speed:.1f}m/s")


def main():
    global drive_pub
    
    rospy.init_node('lidar_distance')
    
    # Publisher 설정
    drive_pub = rospy.Publisher('/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1)
    
    # Subscriber 설정
    rospy.Subscriber('/scan', LaserScan, scan_callback, queue_size=1)
    
    rospy.loginfo("=" * 50)
    rospy.loginfo("LiDAR 거리 유지 시작!")
    rospy.loginfo(f"목표 거리: {TARGET_DISTANCE}m, 정지 거리: {STOP_DISTANCE}m")
    rospy.loginfo("=" * 50)
    
    rospy.spin()


if __name__ == '__main__':
    main()