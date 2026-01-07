# WEGO Mission Package

WEGO 자율주행 교육용 미션 패키지

---

## 📦 패키지 구조

```
wego/
├── cfg/                          # Dynamic Reconfigure 설정
│   ├── LaneDetect.cfg            # 차선 감지 파라미터
│   ├── TrafficLight.cfg          # 신호등 감지 파라미터
│   ├── Pedestrian.cfg            # 보행자 감지 파라미터
│   ├── Roundabout.cfg            # 회전교차로 파라미터
│   └── ObstacleAvoid.cfg         # 장애물 회피 파라미터
│
├── config/                       # YAML 파라미터 파일
│   ├── lane_detect.yaml
│   ├── traffic_light.yaml
│   ├── pedestrian.yaml
│   ├── roundabout.yaml
│   ├── obstacle_avoid.yaml
│   ├── parking.yaml
│   └── aruco.yaml
│
├── scripts/                      # Python 노드
│   ├── lane_detect_node.py       # 차선 감지 (기본 노드)
│   ├── traffic_light_node.py     # 신호등 감지
│   ├── pedestrian_node.py        # 보행자 감지
│   ├── roundabout_node.py        # 회전교차로
│   ├── obstacle_avoid_node.py    # 장애물 회피
│   ├── parking_node.py           # 주차
│   └── aruco_detector_node.py    # ArUco 마커 감지
│
├── launch/                       # Launch 파일
│   ├── bringup.launch            # 하드웨어 초기화
│   ├── lane_tracing.launch       # 라인 트레이싱만
│   ├── traffic_light.launch      # 신호등 미션
│   ├── pedestrian.launch         # 보행자 미션
│   ├── roundabout.launch         # 회전교차로 미션
│   ├── obstacle_avoid.launch     # 장애물 회피 미션
│   └── parking.launch            # 주차 미션
│
├── CMakeLists.txt
├── package.xml
└── README.md
```

---

## 🔌 ROS 통신 구조

### 핵심 개념: 센서 → 인지 → 제어

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SENSORS (센서)                               │
├─────────────────────────────────────────────────────────────────────┤
│  /usb_cam/image_raw/compressed         USB 카메라 영상              │
│  /scan                                 RPLiDAR 스캔 데이터          │
│  /imu                                  IMU 센서 데이터              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PERCEPTION (인지)                               │
├─────────────────────────────────────────────────────────────────────┤
│  lane_detect_node       → /webot/steering_offset (조향값)           │
│                         → /webot/lane_speed (속도)                  │
│  traffic_light_node     → /webot/traffic_stop (정지 신호)           │
│  pedestrian_node        → /webot/pedestrian/detected (감지 여부)    │
│  aruco_detector_node    → /webot/aruco/marker_info (ID+거리)        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        CONTROL (제어)                                │
├─────────────────────────────────────────────────────────────────────┤
│  각 미션 노드가 AckermannDriveStamped 발행 → 로봇 구동              │
│  /low_level/ackermann_cmd_mux/input/navigation                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 노드별 ROS 토픽

### 1. lane_detect_node (차선 감지)

모든 미션의 **기본 노드**. 카메라 이미지에서 차선을 감지하고 조향값을 계산합니다.

| 방향 | 토픽 | 타입 | 설명 |
|------|------|------|------|
| **Sub** | `/usb_cam/image_raw/compressed` | CompressedImage | 카메라 이미지 |
| **Sub** | `/webot/traffic_stop` | Bool | 신호등 정지 명령 |
| **Pub** | `/webot/steering_offset` | Float32 | 계산된 조향값 (rad) |
| **Pub** | `/webot/lane_speed` | Float32 | 기본 속도 (m/s) |
| **Pub** | `/webot/lane_center_x` | Int32 | 차선 중심 x좌표 |
| **Pub** | `/webot/lane_detect/image` | Image | 디버그 이미지 |
| **Pub** | `/low_level/.../navigation` | AckermannDriveStamped | 모터 제어 (publish_cmd_vel=true 시) |

```
                    ┌─────────────────────┐
   카메라 ────────▶│  lane_detect_node   │────────▶ /webot/steering_offset
                    │                     │────────▶ /webot/lane_speed
   traffic_stop ──▶│  (차선 감지)        │────────▶ /cmd_vel (옵션)
                    └─────────────────────┘
```

---

### 2. traffic_light_node (신호등 감지)

HSV 색상 기반 신호등 감지. lane_detect_node에서 조향값을 받아 주행합니다.

| 방향 | 토픽 | 타입 | 설명 |
|------|------|------|------|
| **Sub** | `/usb_cam/image_raw/compressed` | CompressedImage | 카메라 이미지 |
| **Sub** | `/webot/steering_offset` | Float32 | 조향값 (lane_detect에서) |
| **Sub** | `/webot/lane_speed` | Float32 | 속도 (lane_detect에서) |
| **Sub** | `/webot/traffic_enable` | Bool | 미션 활성화 |
| **Pub** | `/webot/traffic_light/state` | String | 감지 결과 (RED/GREEN/UNKNOWN) |
| **Pub** | `/webot/traffic_stop` | Bool | 정지 명령 |
| **Pub** | `/webot/traffic_passed` | Bool | 통과 완료 신호 |
| **Pub** | `/webot/traffic_light/image` | Image | 디버그 이미지 |
| **Pub** | `/low_level/.../navigation` | AckermannDriveStamped | 모터 제어 |

```
   lane_detect_node                    traffic_light_node
  ┌──────────────┐                    ┌──────────────────┐
  │              │ steering_offset    │                  │
  │  차선 감지   │──────────────────▶│   신호등 감지    │────▶ AckermannCmd
  │              │ lane_speed         │                  │
  └──────────────┘──────────────────▶│   RED → 정지     │
                                      │   GREEN → 주행   │
                                      └──────────────────┘
```

---

### 3. pedestrian_node (보행자 감지)

LiDAR 기반 전방 장애물(보행자) 감지. 장애물 감지 시 정지, 사라지면 재출발.

| 방향 | 토픽 | 타입 | 설명 |
|------|------|------|------|
| **Sub** | `/scan` | LaserScan | LiDAR 스캔 데이터 |
| **Sub** | `/webot/steering_offset` | Float32 | 조향값 |
| **Sub** | `/webot/lane_speed` | Float32 | 속도 |
| **Pub** | `/webot/pedestrian/state` | String | 상태 (DRIVING/STOPPING/STOPPED) |
| **Pub** | `/webot/pedestrian/detected` | Bool | 감지 여부 |
| **Pub** | `/webot/pedestrian/debug` | Image | 디버그 시각화 |
| **Pub** | `/low_level/.../navigation` | AckermannDriveStamped | 모터 제어 |

```
   lane_detect_node                    pedestrian_node
  ┌──────────────┐                    ┌──────────────────┐
  │              │ steering_offset    │                  │
  │  차선 감지   │──────────────────▶│   보행자 감지    │────▶ AckermannCmd
  │              │                    │                  │
  └──────────────┘                    │   LiDAR ──▶ 정지 │
                                      └──────────────────┘
                                              ▲
                                      /scan ──┘
```

---

### 4. roundabout_node (회전교차로)

LiDAR로 전방 장애물 감지. 장애물 있으면 대기, 없으면 통과.

| 방향 | 토픽 | 타입 | 설명 |
|------|------|------|------|
| **Sub** | `/scan` | LaserScan | LiDAR 스캔 |
| **Sub** | `/imu` | Imu | IMU 데이터 |
| **Sub** | `/usb_cam/image_raw/compressed` | CompressedImage | 카메라 이미지 |
| **Sub** | `/webot/steering_offset` | Float32 | 조향값 |
| **Sub** | `/webot/lane_speed` | Float32 | 속도 |
| **Pub** | `/webot/roundabout/state` | String | 상태 (IDLE/WAITING/DRIVING) |
| **Pub** | `/webot/roundabout/active` | Bool | 활성화 여부 |
| **Pub** | `/webot/roundabout/done` | Bool | 완료 신호 |
| **Pub** | `/webot/roundabout/debug` | Image | 디버그 이미지 |
| **Pub** | `/low_level/.../navigation` | AckermannDriveStamped | 모터 제어 |

---

### 5. obstacle_avoid_node (장애물 회피)

Gap Finding 알고리즘으로 장애물 회피 주행.

| 방향 | 토픽 | 타입 | 설명 |
|------|------|------|------|
| **Sub** | `/scan` | LaserScan | LiDAR 스캔 |
| **Sub** | `/webot/steering_offset` | Float32 | 조향값 |
| **Sub** | `/webot/lane_speed` | Float32 | 속도 |
| **Pub** | `/webot/obstacle/state` | String | 상태 |
| **Pub** | `/webot/obstacle/is_avoiding` | Bool | 회피 중 여부 |
| **Pub** | `/webot/obstacle/debug` | Image | 디버그 시각화 |
| **Pub** | `/low_level/.../navigation` | AckermannDriveStamped | 모터 제어 |

---

### 6. parking_node (주차)

ArUco 마커 기반 트리거 후 후진 평행주차 수행.

| 방향 | 토픽 | 타입 | 설명 |
|------|------|------|------|
| **Sub** | `/webot/aruco/marker_info` | Float32MultiArray | 마커 ID + 거리 |
| **Sub** | `/webot/steering_offset` | Float32 | 조향값 (IDLE 상태) |
| **Sub** | `/webot/lane_speed` | Float32 | 속도 (IDLE 상태) |
| **Pub** | `/webot/parking/state` | String | 상태 |
| **Pub** | `/webot/parking/done` | Bool | 완료 신호 |
| **Pub** | `/low_level/.../navigation` | AckermannDriveStamped | 모터 제어 |

---

### 7. aruco_detector_node (ArUco 감지)

ArUco 마커 감지 및 거리 계산.

| 방향 | 토픽 | 타입 | 설명 |
|------|------|------|------|
| **Sub** | `/usb_cam/image_raw/compressed` | CompressedImage | 카메라 이미지 |
| **Pub** | `/webot/aruco/markers` | Int32MultiArray | 감지된 마커 ID 배열 |
| **Pub** | `/webot/aruco/marker_info` | Float32MultiArray | [id, distance, ...] |
| **Pub** | `/webot/aruco/debug` | Image | 디버그 이미지 |

---

## 🚀 미션별 실행 방법

### 개별 미션 실행

```bash
# 1. 하드웨어 초기화 (모든 미션 전 필수)
roslaunch wego bringup.launch

# 2. 미션 선택 실행 (새 터미널)
roslaunch wego lane_tracing.launch      # 라인 트레이싱
roslaunch wego traffic_light.launch     # 신호등
roslaunch wego pedestrian.launch        # 보행자
roslaunch wego roundabout.launch        # 회전교차로
roslaunch wego obstacle_avoid.launch    # 장애물 회피
roslaunch wego parking.launch           # 주차
```

### 파라미터 튜닝

```bash
# rqt_reconfigure로 실시간 파라미터 조정
rosrun rqt_reconfigure rqt_reconfigure
```

---

## 🔗 미션 연결 설계 가이드

### 통신 기반 미션 연결 원리

각 미션은 **완료 신호(done)**를 발행합니다. 이를 활용해 순차 미션을 구성할 수 있습니다.

```
┌─────────────────┐     done     ┌─────────────────┐     done     ┌─────────────────┐
│   Mission A     │─────────────▶│   Mission B     │─────────────▶│   Mission C     │
│ (traffic_light) │              │  (roundabout)   │              │   (parking)     │
└─────────────────┘              └─────────────────┘              └─────────────────┘
```

### 미션별 트리거/완료 토픽

| 미션 | 트리거 방법 | 완료 토픽 |
|------|-------------|-----------|
| traffic_light | launch 시 자동 시작 | `/webot/traffic_passed` (GREEN 통과 시) |
| pedestrian | launch 시 자동 시작 | 상태 모니터링으로 판단 |
| roundabout | launch 시 자동 시작 | `/webot/roundabout/done` |
| obstacle_avoid | launch 시 자동 시작 | 상태 모니터링으로 판단 |
| parking | ArUco 마커 ID 0 감지 시 | `/webot/parking/done` |

### 예시: 순차 미션 컨트롤러 설계

```python
#!/usr/bin/env python3
"""
Sequential Mission Controller 예시
- 학생들이 직접 구현해볼 수 있는 템플릿
"""
import rospy
from std_msgs.msg import Bool, String

class SequenceController:
    def __init__(self):
        rospy.init_node('sequence_controller')
        
        self.current_mission = "TRAFFIC_LIGHT"
        
        # 미션 완료 구독
        rospy.Subscriber('/webot/traffic_passed', Bool, self.traffic_done)
        rospy.Subscriber('/webot/roundabout/done', Bool, self.roundabout_done)
        rospy.Subscriber('/webot/parking/done', Bool, self.parking_done)
        
        # 미션 트리거 발행 (필요시)
        self.pub_traffic_enable = rospy.Publisher('/webot/traffic_enable', Bool, queue_size=1)
        
    def traffic_done(self, msg):
        if msg.data and self.current_mission == "TRAFFIC_LIGHT":
            rospy.loginfo("Traffic light passed! Starting roundabout...")
            self.current_mission = "ROUNDABOUT"
    
    def roundabout_done(self, msg):
        if msg.data and self.current_mission == "ROUNDABOUT":
            rospy.loginfo("Roundabout done! Next mission...")
            self.current_mission = "PARKING"
            # parking은 ArUco 마커로 자동 트리거
    
    def parking_done(self, msg):
        if msg.data and self.current_mission == "PARKING":
            rospy.loginfo("All missions complete!")
            self.current_mission = "FINISHED"

if __name__ == '__main__':
    try:
        controller = SequenceController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

---

## 📊 토픽 모니터링

### 실시간 토픽 확인

```bash
# 전체 토픽 목록
rostopic list

# 특정 토픽 모니터링
rostopic echo /webot/steering_offset
rostopic echo /webot/traffic_light/state
rostopic echo /webot/parking/state

# 토픽 발행 빈도 확인
rostopic hz /low_level/ackermann_cmd_mux/input/navigation
```

### rqt_graph로 노드 연결 시각화

```bash
rqt_graph
```

---

## 📊 주요 토픽 정리

### 내부 통신 토픽 (`/webot/` prefix)

| 토픽 | 타입 | 발행 노드 | 설명 |
|------|------|----------|------|
| `/webot/steering_offset` | Float32 | lane_detect | 조향각 (rad) |
| `/webot/lane_speed` | Float32 | lane_detect | 기본 속도 (m/s) |
| `/webot/lane_center_x` | Int32 | lane_detect | 차선 중심 x좌표 |
| `/webot/traffic_light/state` | String | traffic_light | RED/GREEN/UNKNOWN |
| `/webot/traffic_stop` | Bool | traffic_light | 정지 신호 |
| `/webot/traffic_passed` | Bool | traffic_light | 통과 완료 |
| `/webot/pedestrian/state` | String | pedestrian | 상태 |
| `/webot/pedestrian/detected` | Bool | pedestrian | 감지 여부 |
| `/webot/roundabout/state` | String | roundabout | 상태 |
| `/webot/roundabout/done` | Bool | roundabout | 완료 신호 |
| `/webot/obstacle/state` | String | obstacle_avoid | 상태 |
| `/webot/obstacle/is_avoiding` | Bool | obstacle_avoid | 회피 중 여부 |
| `/webot/aruco/markers` | Int32MultiArray | aruco_detector | 마커 ID 배열 |
| `/webot/aruco/marker_info` | Float32MultiArray | aruco_detector | [id, dist, ...] |
| `/webot/parking/state` | String | parking | 주차 상태 |
| `/webot/parking/done` | Bool | parking | 주차 완료 |

### 디버그 이미지 토픽

| 토픽 | 설명 |
|------|------|
| `/webot/lane_detect/image` | 차선 검출 시각화 |
| `/webot/traffic_light/image` | 신호등 검출 시각화 |
| `/webot/pedestrian/debug` | 보행자 감지 시각화 |
| `/webot/roundabout/debug` | 회전교차로 시각화 |
| `/webot/obstacle/debug` | LiDAR 섹터 시각화 |
| `/webot/aruco/debug` | ArUco 마커 시각화 |

---

## 🎓 학습 포인트

### 1. Publisher / Subscriber 패턴
- 각 노드가 독립적으로 동작하며 토픽을 통해 통신
- 느슨한 결합(loose coupling)으로 모듈 교체 용이

### 2. 센서 데이터 흐름
- Camera → 이미지 처리 → 조향값 계산
- LiDAR → 거리 측정 → 장애물 판단

### 3. 제어 구조
- 모든 미션 노드가 AckermannDriveStamped를 발행
- 하나의 미션만 활성화되어야 충돌 방지

### 4. 상태 기반 설계
- 각 노드가 상태(State)를 발행
- 외부에서 상태를 모니터링하여 미션 흐름 제어

---

## ⚙️ 파라미터 튜닝

### Dynamic Reconfigure 사용
```bash
rosrun rqt_reconfigure rqt_reconfigure
```

### YAML 파일 수정
`config/` 폴더의 yaml 파일 수정 후 재실행

| 파일 | 설명 |
|------|------|
| `lane_detect.yaml` | 차선 HSV, PID, ROI |
| `traffic_light.yaml` | 신호등 색상 범위 |
| `pedestrian.yaml` | 보행자 감지 거리 |
| `roundabout.yaml` | 회전교차로 파라미터 |
| `obstacle_avoid.yaml` | LiDAR 거리 임계값 |
| `parking.yaml` | 주차 시간/각도 |
| `aruco.yaml` | ArUco 마커 설정 |

---

## 🛠️ 디버깅 팁

### 이미지 확인
```bash
rqt_image_view /webot/lane_detect/image
rqt_image_view /webot/traffic_light/image
rqt_image_view /webot/obstacle/debug
```

### 토픽 모니터링
```bash
rostopic echo /webot/steering_offset
rostopic hz /webot/lane_speed
```

### 노드 그래프 확인
```bash
rqt_graph
```

---

## 🛠️ 확장 과제

1. **새로운 미션 추가**: 위 구조를 참고하여 새로운 미션 노드 개발
2. **순차 컨트롤러 구현**: 여러 미션을 자동으로 연결하는 컨트롤러 작성
3. **파라미터 최적화**: rqt_reconfigure로 각 미션 파라미터 튜닝
4. **시각화 도구 활용**: rqt_graph, rqt_image_view로 시스템 분석

---

## 📝 새 미션 추가 가이드

1. **노드 작성** (`scripts/my_mission_node.py`)
   - `/webot/steering_offset`, `/webot/lane_speed` 구독
   - `/low_level/ackermann_cmd_mux/input/navigation` 발행
   - 완료 시 `/webot/my_mission/done` 발행

2. **설정 파일** (`config/my_mission.yaml`)

3. **Launch 파일** (`launch/my_mission.launch`)
   - lane_detect_node (publish_cmd_vel=false)
   - my_mission_node

4. **Dynamic Reconfigure** (`cfg/MyMission.cfg`) - 선택사항

5. **CMakeLists.txt** 업데이트
   ```cmake
   catkin_install_python(PROGRAMS
     scripts/my_mission_node.py
     ...
   )
   ```

6. **빌드**
   ```bash
   cd ~/catkin_ws && catkin_make
   ```

---

## 📚 참고 자료

- [ROS Wiki - Topics](http://wiki.ros.org/Topics)
- [ROS Wiki - Publishers and Subscribers](http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29)
- [Dynamic Reconfigure](http://wiki.ros.org/dynamic_reconfigure)
- [Ackermann Messages](http://wiki.ros.org/ackermann_msgs)

---
