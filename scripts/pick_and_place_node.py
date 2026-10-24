#!/usr/bin/env python3
"""
Autonomous Pick-and-Place Sequencer ROS 2 Node:
Integrates Perception, Antipodal Grasp Synthesis, Trajectory Planning,
and Gripper Sequencing into an autonomous state-machine driven execution loop.
"""

import sys
import rclpy
from rclpy.node import Node
import numpy as np

from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from manipulation_grasping.fsm import PickAndPlaceFSM, FSMState


class PickAndPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_and_place_node')

        self.fsm = PickAndPlaceFSM()
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # Publishers
        self.status_pub = self.create_publisher(String, '/manipulation/fsm_status', 10)
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.traj_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10
        )
        self.grasp_marker_pub = self.create_publisher(MarkerArray, '/manipulation/grasp_markers', 10)

        # Timer to advance FSM step-by-step
        self.fsm_timer = self.create_wall_timer(0.8, self.fsm_step_callback)
        self.get_logger().info('PickAndPlaceNode started. FSM initialized to IDLE.')

    def fsm_step_callback(self):
        old_state = self.fsm.state
        new_state = self.fsm.step()

        # Publish FSM status string
        status_msg = String()
        status_msg.data = f'STATE: {new_state.name} | Cycle: {self.fsm.context.cycle_count}'
        self.status_pub.publish(status_msg)

        if old_state != new_state:
            self.get_logger().info(f'Transition: {old_state.name} -> {new_state.name}')

        # Publish active joint states
        js_msg = JointState()
        js_msg.header.stamp = self.get_clock().now().to_msg()
        js_msg.name = self.joint_names + ['gripper_left_finger_joint', 'gripper_right_finger_joint']
        js_msg.position = list(self.fsm.context.current_q) + [
            self.fsm.context.gripper_state,
            self.fsm.context.gripper_state
        ]
        js_msg.velocity = [0.0] * 8
        js_msg.effort = [0.0] * 8
        self.joint_state_pub.publish(js_msg)

        # If a new trajectory is active, publish JointTrajectory message
        if self.fsm.context.active_trajectory is not None:
            traj_msg = JointTrajectory()
            traj_msg.header.stamp = self.get_clock().now().to_msg()
            traj_msg.joint_names = self.joint_names

            for pt in self.fsm.context.active_trajectory:
                jt_pt = JointTrajectoryPoint()
                jt_pt.time_from_start.sec = int(pt.time)
                jt_pt.time_from_start.nanosec = int((pt.time - int(pt.time)) * 1e9)
                jt_pt.positions = list(pt.positions)
                jt_pt.velocities = list(pt.velocities)
                jt_pt.accelerations = list(pt.accelerations)
                traj_msg.points.append(jt_pt)

            self.traj_pub.publish(traj_msg)

        # Publish grasp visualization markers
        if self.fsm.context.active_grasp is not None:
            self.publish_grasp_markers(self.fsm.context.active_grasp)

        # Auto-restart on success after a brief pause
        if new_state == FSMState.SUCCESS:
            self.get_logger().info('Pick-and-place completed successfully! Resetting to IDLE...')
            self.fsm.set_state(FSMState.IDLE)
        elif new_state == FSMState.FAILED:
            self.get_logger().warn(f'Pick-and-place failed: {self.fsm.context.error_message}. Retrying...')
            self.fsm.set_state(FSMState.IDLE)

    def publish_grasp_markers(self, grasp):
        ma = MarkerArray()

        # Approach Arrow
        arrow = Marker()
        arrow.header.frame_id = 'base_link'
        arrow.header.stamp = self.get_clock().now().to_msg()
        arrow.ns = 'grasp_approach'
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD

        p_start = Point(
            x=float(grasp.pre_grasp_position[0]),
            y=float(grasp.pre_grasp_position[1]),
            z=float(grasp.pre_grasp_position[2])
        )
        p_end = Point(
            x=float(grasp.position[0]),
            y=float(grasp.position[1]),
            z=float(grasp.position[2])
        )
        arrow.points = [p_start, p_end]
        arrow.scale.x = 0.012  # shaft diameter
        arrow.scale.y = 0.02   # head diameter
        arrow.scale.z = 0.03   # head length
        arrow.color.r = 1.0
        arrow.color.g = 0.8
        arrow.color.b = 0.0
        arrow.color.a = 0.95
        ma.markers.append(arrow)

        # Grasp Contact Points (spheres)
        for i, p_contact in enumerate([grasp.contact_p1, grasp.contact_p2]):
            sphere = Marker()
            sphere.header.frame_id = 'base_link'
            sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = 'grasp_contacts'
            sphere.id = i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(p_contact[0])
            sphere.pose.position.y = float(p_contact[1])
            sphere.pose.position.z = float(p_contact[2])
            sphere.scale.x = 0.015
            sphere.scale.y = 0.015
            sphere.scale.z = 0.015
            sphere.color.r = 0.9
            sphere.color.g = 0.1
            sphere.color.b = 0.1
            sphere.color.a = 0.9
            ma.markers.append(sphere)

        self.grasp_marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
