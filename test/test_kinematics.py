import pytest
import numpy as np

from manipulation_grasping.kinematics import (
    KinematicsEngine,
    ArmConfig,
    rpy_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
)


def test_rotation_helpers():
    # Test RPY to matrix and roundtrip with quaternion
    roll, pitch, yaw = 0.2, -0.3, 0.5
    R = rpy_to_rotation_matrix(roll, pitch, yaw)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)

    q = rotation_matrix_to_quaternion(R)
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)

    R_rec = quaternion_to_rotation_matrix(q)
    assert np.allclose(R, R_rec, atol=1e-5)


def test_forward_kinematics():
    engine = KinematicsEngine()
    q_zero = np.zeros(6)
    T_tcp, frames = engine.forward_kinematics(q_zero, return_all_frames=True)

    assert T_tcp.shape == (4, 4)
    assert len(frames) == 8  # base + 6 links + tcp
    assert np.allclose(T_tcp[:3, :3] @ T_tcp[:3, :3].T, np.eye(3), atol=1e-6)

    # Position should have positive height above base plane
    pos = T_tcp[:3, 3]
    assert pos[2] > 0.0

    # Nominal home posture reaches full operational elevation
    q_home = np.array([0.0, -np.pi / 4, np.pi / 3, -np.pi / 6, np.pi / 2, 0.0])
    T_home, _ = engine.forward_kinematics(q_home)
    assert T_home[2, 3] > 0.3


def test_jacobian_and_manipulability():
    engine = KinematicsEngine()
    q = np.array([0.0, -np.pi / 4, np.pi / 3, -np.pi / 6, np.pi / 2, 0.0])
    J = engine.compute_jacobian(q)

    assert J.shape == (6, 6)
    assert not np.any(np.isnan(J))

    w = engine.manipulability(q)
    assert w > 0.005, f"Expected non-singular manipulability, got {w}"

    cond = engine.condition_number(q)
    assert cond > 1.0 and not np.isnan(cond)


def test_inverse_kinematics_convergence():
    engine = KinematicsEngine()
    # Choose a target reached by known joint angles
    q_target = np.array([0.3, -0.6, 0.8, -0.4, 0.7, 0.1])
    T_target, _ = engine.forward_kinematics(q_target)
    target_pos = T_target[:3, 3]
    target_rot = T_target[:3, :3]

    # Start from perturbed seed
    q_init = q_target + np.array([0.15, -0.1, 0.1, 0.05, -0.1, 0.05])

    ok, q_sol, iters, err = engine.inverse_kinematics(
        target_pos=target_pos,
        target_rot=target_rot,
        q_init=q_init,
        max_iter=100,
        pos_tol=1e-3,
        rot_tol=2e-3,
    )

    assert ok, f"IK failed to converge! Iterations: {iters}, Error: {err}"
    assert iters < 80
    assert err < 2e-3

    # Check forward kinematics of solution matches target
    T_sol, _ = engine.forward_kinematics(q_sol)
    pos_err = np.linalg.norm(T_sol[:3, 3] - target_pos)
    assert pos_err < 2e-3


def test_singularity_damping():
    engine = KinematicsEngine()
    # Wrist singularity: joint 5 is zero, axes 4 and 6 align
    q_singular = np.array([0.0, -0.2, 0.2, 0.0, 0.0, 0.0])
    w = engine.manipulability(q_singular)
    assert w < 0.01  # very close to singularity

    T_sing, _ = engine.forward_kinematics(q_singular)
    # Attempt IK near singularity with adaptive damping
    ok, q_sol, iters, err = engine.inverse_kinematics(
        target_pos=T_sing[:3, 3],
        target_rot=T_sing[:3, :3],
        q_init=q_singular + 0.05,
        max_iter=60,
        adaptive_damping=True,
    )

    # Must not produce NaN or overflow
    assert not np.any(np.isnan(q_sol))
    assert not np.any(np.isinf(q_sol))
