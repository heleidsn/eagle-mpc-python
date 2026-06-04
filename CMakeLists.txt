cmake_minimum_required(VERSION 3.0.2)
project(eagle_mpc_python)

find_package(catkin REQUIRED)

catkin_package()

#############
## Install ##
#############

## Mark executable Python helper nodes for installation so that
## `rosrun eagle_mpc_python <node>.py` and <node pkg=... type=...> resolve.
catkin_install_python(PROGRAMS
  scripts/ros_nodes/groundtruth_pub.py
  scripts/ros_nodes/reference_path_pub.py
  scripts/ros_nodes/pose_to_tf_pub.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install launch / worlds / models / config so the package is usable from the
## install space as well as the devel space.
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

install(DIRECTORY worlds/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/worlds
)

install(DIRECTORY models/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
