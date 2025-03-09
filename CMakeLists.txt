cmake_minimum_required(VERSION 3.8)
project(cv_optical_flow)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_executable(optical_flow_node src/optical_flow_node.cpp)
ament_target_dependencies(optical_flow_node rclcpp sensor_msgs cv_bridge OpenCV)

add_executable(dense_optical_flow_node src/dense_optical_flow_node.cpp)
ament_target_dependencies(dense_optical_flow_node rclcpp sensor_msgs cv_bridge OpenCV)

install(TARGETS
  optical_flow_node
  dense_optical_flow_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()