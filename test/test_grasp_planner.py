import pytest
import numpy as np

from manipulation_grasping.perception import SegmentedObject
from manipulation_grasping.grasp_planner import GraspPlanner
from manipulation_grasping.kinematics import KinematicsEngine


def test_grasp_planner_sampling_and_filtering():
    kinematics = KinematicsEngine()
    planner = GraspPlanner(kinematics, max_gripper_width=0.085, min_gripper_width=0.015)

    # Synthetic box object
    rng = np.random.default_rng(42)
    centroid = np.array([0.45, 0.0, 0.04])
    dimensions = np.array([0.04, 0.05, 0.06])
    principal_axes = np.eye(3)

    # Box surface points
    pts = rng.uniform(
        centroid - 0.5 * dimensions,
        centroid + 0.5 * dimensions,
        size=(300, 3)
    )
    normals = np.zeros_like(pts)
    for i in range(len(pts)):
        # Normal pointing away from centroid
        diff = pts[i] - centroid
        axis = np.argmax(np.abs(diff))
        normals[i, axis] = np.sign(diff[axis])

    corners = np.zeros((8, 3))

    obj = SegmentedObject(
        cluster_id=0,
        points=pts,
        centroid=centroid,
        dimensions=dimensions,
        principal_axes=principal_axes,
        surface_normals=normals,
        bounding_box_corners=corners,
    )

    grasps = planner.sample_antipodal_grasps(obj, num_samples=40, verify_ik=True)

    assert len(grasps) > 0, "Failed to generate any valid grasp candidate"
    best_grasp = grasps[0]

    # Verify grasp attributes
    assert best_grasp.grasp_width <= planner.max_gripper_width
    assert best_grasp.grasp_width >= planner.min_gripper_width
    assert 0.0 <= best_grasp.quality_score <= 1.0

    # Verify SE(3) orientation matrix is orthonormal with det = 1
    R = best_grasp.orientation_matrix
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)

    # Verify pre-grasp is offset along approach direction
    dist_standoff = np.linalg.norm(best_grasp.position - best_grasp.pre_grasp_position)
    assert np.isclose(dist_standoff, planner.standoff_distance, atol=1e-3)
