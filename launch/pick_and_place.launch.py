#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('manipulation_grasping')
    rviz_config = os.path.join(pkg_dir, 'rviz', 'manipulation.rviz')
    xacro_file = os.path.join(pkg_dir, 'urdf', 'manipulation_scene.urdf.xacro')

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    robot_description = Command(['xacro ', xacro_file])

    # 1. Robot State Publisher Node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time
        }]
    )

    # 2. 3D Perception Pipeline Node
    perception_node = Node(
        package='manipulation_grasping',
        executable='perception_node.py',
        name='perception_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # 3. Autonomous Pick-and-Place Sequencer Node
    pick_and_place_node = Node(
        package='manipulation_grasping',
        executable='pick_and_place_node.py',
        name='pick_and_place_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # 4. RViz2 Visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        robot_state_publisher,
        perception_node,
        pick_and_place_node,
        rviz_node,
    ])
