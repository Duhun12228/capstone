cmake_minimum_required(VERSION 3.0.2)
project(wego)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  dynamic_reconfigure
)

# Dynamic reconfigure
generate_dynamic_reconfigure_options(
  cfg/LaneDetect.cfg
  cfg/TrafficLight.cfg
  cfg/Roundabout.cfg
  cfg/ObstacleAvoid.cfg
  cfg/Pedestrian.cfg
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs cv_bridge dynamic_reconfigure
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/lane_detect_node.py
  scripts/traffic_light_node.py
  scripts/roundabout_node.py
  scripts/obstacle_avoid_node.py
  scripts/pedestrian_node.py
  scripts/aruco_detector_node.py
  scripts/aruco_lane_tuner.py
  scripts/parking_node.py
  scripts/dh_obstacle.py
  scripts/dh_lanefollow.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

