import pytest
import numpy as np

from manipulation_grasping.perception import (
    PerceptionPipeline,
    PointCloud,
)


def test_pass_through_and_voxel_filter():
    pipeline = PerceptionPipeline(voxel_size=0.01)

    # Generate points inside and outside bounds
    pts_in = np.array([
        [0.4, 0.0, 0.2],
        [0.5, 0.1, 0.3],
        [0.6, -0.1, 0.4]
    ])
    pts_out = np.array([
        [0.0, 0.0, 0.0],  # x < 0.2
        [1.5, 0.0, 0.2],  # x > 0.85
        [0.4, 0.8, 0.2],  # y > 0.45
    ])
    all_pts = np.vstack([pts_in, pts_out])
    cloud = PointCloud(points=all_pts)

    filtered = pipeline.pass_through_filter(cloud)
    assert len(filtered.points) == 3

    # Voxel downsampling test
    dense_pts = np.random.uniform(0.3, 0.4, size=(200, 3))
    dense_cloud = PointCloud(points=dense_pts)
    downsampled = pipeline.voxel_downsample(dense_cloud)
    assert len(downsampled.points) < len(dense_pts)


def test_ransac_plane_segmentation():
    pipeline = PerceptionPipeline(ransac_distance_threshold=0.01)

    rng = np.random.default_rng(42)
    # 1000 points on a horizontal table z = 0.0 with small Gaussian noise
    x_table = rng.uniform(0.2, 0.8, 1000)
    y_table = rng.uniform(-0.4, 0.4, 1000)
    z_table = rng.normal(0.0, 0.001, 1000)
    table_pts = np.column_stack([x_table, y_table, z_table])

    # 200 points on a cube sitting on top of the table (z between 0.02 and 0.07)
    x_cube = rng.uniform(0.4, 0.5, 200)
    y_cube = rng.uniform(-0.05, 0.05, 200)
    z_cube = rng.uniform(0.02, 0.07, 200)
    cube_pts = np.column_stack([x_cube, y_cube, z_cube])

    cloud = PointCloud(points=np.vstack([table_pts, cube_pts]))

    plane, inliers, outliers = pipeline.ransac_plane_segmentation(cloud)

    # Normal should point predominantly along +Z: [0, 0, 1]
    normal = plane[:3]
    assert normal[2] > 0.95
    assert len(inliers) > 900
    assert len(outliers) > 100


def test_euclidean_clustering_and_obb():
    pipeline = PerceptionPipeline(
        cluster_tolerance=0.04,
        min_cluster_size=20,
        max_cluster_size=1000
    )

    rng = np.random.default_rng(123)
    # Generate 2 separated clusters:
    # Cluster 1: centered at (0.4, -0.15, 0.05)
    c1 = rng.normal(loc=[0.4, -0.15, 0.05], scale=0.01, size=(60, 3))
    # Cluster 2: centered at (0.55, 0.15, 0.05)
    c2 = rng.normal(loc=[0.55, 0.15, 0.05], scale=0.01, size=(70, 3))

    pts = np.vstack([c1, c2])
    cloud = PointCloud(points=pts)
    outlier_indices = np.arange(len(pts))

    objects = pipeline.extract_segmented_objects(cloud, outlier_indices)
    assert len(objects) == 2

    # Check centroids are close to actual means
    centroids = sorted([obj.centroid for obj in objects], key=lambda c: c[0])
    assert np.linalg.norm(centroids[0] - np.array([0.4, -0.15, 0.05])) < 0.03
    assert np.linalg.norm(centroids[1] - np.array([0.55, 0.15, 0.05])) < 0.03

    # Check OBB dimensions are positive
    for obj in objects:
        assert np.all(obj.dimensions > 0.0)
        assert obj.bounding_box_corners.shape == (8, 3)
