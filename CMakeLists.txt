cmake_minimum_required(VERSION 3.16)
project(manipulation_grasping)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

# ─── C++ Kinematics & Trajectory Library ─────────────────────────────
add_library(manipulation_kinematics_core SHARED
  src/kinematics_engine.cpp
  src/trajectory_generator.cpp
)
ament_target_dependencies(manipulation_kinematics_core
  rclcpp
  geometry_msgs
  trajectory_msgs
  visualization_msgs
  tf2_ros
)
target_include_directories(manipulation_kinematics_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(manipulation_kinematics_core PUBLIC cxx_std_17)

# ─── C++ Manipulation Controller Node ────────────────────────────────
add_executable(manipulation_controller_node
  src/manipulation_controller_node.cpp
)
target_link_libraries(manipulation_controller_node
  manipulation_kinematics_core
)
ament_target_dependencies(manipulation_controller_node
  rclcpp
  geometry_msgs
  sensor_msgs
  std_msgs
  trajectory_msgs
  visualization_msgs
  tf2_ros
)
target_compile_features(manipulation_controller_node PRIVATE cxx_std_17)

# ─── Installation ────────────────────────────────────────────────────
install(TARGETS
  manipulation_kinematics_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS
  manipulation_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

# ─── Python Package ──────────────────────────────────────────────────
ament_python_install_package(${PROJECT_NAME})

# ─── Python Executable Scripts ───────────────────────────────────────
install(PROGRAMS
  scripts/pick_and_place_node.py
  scripts/perception_node.py
  scripts/benchmark_manipulation.py
  scripts/run_full_demo.py
  DESTINATION lib/${PROJECT_NAME}
)

# ─── Resource Files ──────────────────────────────────────────────────
install(DIRECTORY
  launch config urdf worlds rviz
  DESTINATION share/${PROJECT_NAME}
)

# ─── Pytest Tests ────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_kinematics test/test_kinematics.py)
  ament_add_pytest_test(test_trajectory test/test_trajectory.py)
  ament_add_pytest_test(test_perception test/test_perception.py)
  ament_add_pytest_test(test_grasp_planner test/test_grasp_planner.py)
  ament_add_pytest_test(test_fsm test/test_fsm.py)
  ament_add_pytest_test(test_benchmarks test/test_benchmarks.py)
endif()

ament_export_include_directories(include)
ament_export_libraries(manipulation_kinematics_core)
ament_export_dependencies(rclcpp geometry_msgs trajectory_msgs tf2_ros)
ament_package()
