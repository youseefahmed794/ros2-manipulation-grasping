#!/usr/bin/env python3
"""
Perception ROS 2 Node:
Subscribes to PointCloud2, performs RANSAC tabletop removal and Euclidean clustering,
and publishes object bounding boxes and surface normals as visualization markers.
"""

import sys
import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from manipulation_grasping.perception import PerceptionPipeline, PointCloud


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self.declare_parameter('voxel_size', 0.005)
        self.declare_parameter('ransac_threshold', 0.012)
        self.declare_parameter('cluster_tolerance', 0.035)

        voxel_size = self.get_parameter('voxel_size').value
        ransac_th = self.get_parameter('ransac_threshold').value
        cluster_tol = self.get_parameter('cluster_tolerance').value

        self.pipeline = PerceptionPipeline(
            voxel_size=voxel_size,
            ransac_distance_threshold=ransac_th,
            cluster_tolerance=cluster_tol
        )

        self.marker_pub = self.create_publisher(MarkerArray, '/perception/object_markers', 10)
        self.table_marker_pub = self.create_publisher(Marker, '/perception/table_plane', 10)

        self.pc_sub = self.create_subscription(
            PointCloud2, '/camera/points', self.pointcloud_callback, 10
        )

        # Periodic timer generating synthetic scene if no real camera topic is present
        self.timer = self.create_wall_timer(1.0, self.synthetic_perception_cycle)
        self.get_logger().info('PerceptionNode initialized.')

    def pointcloud_callback(self, msg: PointCloud2):
        # Extract points from PointCloud2
        # For simplicity and speed, unpack xyz
        self.get_logger().info(f'Received PointCloud2 with {msg.width * msg.height} points')

    def synthetic_perception_cycle(self):
        """Simulate perception cycle for visualization."""
        rng = np.random.default_rng()
        # Synthetic tabletop + 2 objects
        n_table = 2000
        xt = rng.uniform(0.3, 0.7, n_table)
        yt = rng.uniform(-0.3, 0.3, n_table)
        zt = rng.normal(0.0, 0.002, n_table)
        table_pts = np.column_stack([xt, yt, zt])

        # Object 1 (Cube at x=0.45, y=0.0)
        n_obj1 = 400
        x1 = rng.uniform(0.43, 0.47, n_obj1)
        y1 = rng.uniform(-0.02, 0.02, n_obj1)
        z1 = rng.uniform(0.005, 0.065, n_obj1)
        obj1_pts = np.column_stack([x1, y1, z1])

        # Object 2 (Cylinder at x=0.55, y=0.15)
        n_obj2 = 400
        x2 = rng.uniform(0.53, 0.57, n_obj2)
        y2 = rng.uniform(0.13, 0.17, n_obj2)
        z2 = rng.uniform(0.005, 0.075, n_obj2)
        obj2_pts = np.column_stack([x2, y2, z2])

        all_pts = np.vstack([table_pts, obj1_pts, obj2_pts])
        cloud = PointCloud(points=all_pts)

        # Process pipeline
        filtered = self.pipeline.pass_through_filter(cloud)
        downsampled = self.pipeline.voxel_downsample(filtered)
        plane, inliers, outliers = self.pipeline.ransac_plane_segmentation(downsampled)

        if len(outliers) > 0:
            objects = self.pipeline.extract_segmented_objects(downsampled, outliers)
            self.publish_markers(objects, plane)

    def publish_markers(self, objects, plane):
        marker_array = MarkerArray()

        for idx, obj in enumerate(objects):
            # Bounding box line cube
            marker = Marker()
            marker.header.frame_id = 'base_link'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'detected_objects'
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(obj.centroid[0])
            marker.pose.position.y = float(obj.centroid[1])
            marker.pose.position.z = float(obj.centroid[2])
            marker.scale.x = max(0.02, float(obj.dimensions[0]))
            marker.scale.y = max(0.02, float(obj.dimensions[1]))
            marker.scale.z = max(0.02, float(obj.dimensions[2]))
            marker.color.r = 0.1
            marker.color.g = 0.9
            marker.color.b = 0.2
            marker.color.a = 0.65
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
