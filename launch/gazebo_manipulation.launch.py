#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('manipulation_grasping')
    ros_gz_sim_dir = get_package_share_directory('ros_gz_sim')

    world_path = os.path.join(pkg_dir, 'worlds', 'tabletop_manipulation.sdf')
    bridge_config = os.path.join(pkg_dir, 'config', 'ros_gz_bridge.yaml')
    rviz_config = os.path.join(pkg_dir, 'rviz', 'manipulation.rviz')
    xacro_file = os.path.join(pkg_dir, 'urdf', 'manipulation_scene.urdf.xacro')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Convert Xacro to URDF robot description
    robot_description = Command(['xacro ', xacro_file])

    # 1. Gazebo Harmonic Simulation Launch
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_dir, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_path}'}.items(),
    )

    # 2. Robot State Publisher Node
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

    # 3. Spawn Robot in Gazebo
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'manipulation_robot',
            '-topic', 'robot_description',
            '-x', '0.0', '-y', '0.0', '-z', '0.0'
        ],
        output='screen',
    )

    # 4. ROS-GZ Bridge Node
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={bridge_config}'],
        output='screen',
    )

    # 5. RViz2 Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        ros_gz_bridge,
        rviz_node,
    ])
