import pytest
import numpy as np

from manipulation_grasping.trajectory import (
    QuinticPolynomial,
    TrapezoidalProfile,
    slerp,
    TrajectoryGenerator,
)
from manipulation_grasping.kinematics import (
    KinematicsEngine,
    rotation_matrix_to_quaternion,
)


def test_quintic_polynomial_boundary_conditions():
    q0, q1 = -0.5, 1.2
    v0, v1 = 0.0, 0.0
    a0, a1 = 0.0, 0.0
    T = 2.5

    poly = QuinticPolynomial(q0, q1, v0=v0, v1=v1, acc0=a0, acc1=a1, duration=T)

    # Initial state at t = 0
    p_start, v_start, acc_start, _ = poly.eval(0.0)
    assert np.isclose(p_start, q0, atol=1e-6)
    assert np.isclose(v_start, v0, atol=1e-6)
    assert np.isclose(acc_start, a0, atol=1e-6)

    # Final state at t = T
    p_end, v_end, acc_end, _ = poly.eval(T)
    assert np.isclose(p_end, q1, atol=1e-6)
    assert np.isclose(v_end, v1, atol=1e-6)
    assert np.isclose(acc_end, a1, atol=1e-6)


def test_trapezoidal_velocity_profile():
    q0, q1 = 0.0, 1.0
    v_max = 1.0
    a_max = 2.0
    trap = TrapezoidalProfile(q0, q1, v_max=v_max, a_max=a_max)

    assert trap.T > 0.0
    p_start, v_start, _ = trap.eval(0.0)
    assert np.isclose(p_start, q0, atol=1e-5)
    assert np.isclose(v_start, 0.0, atol=1e-5)

    p_end, v_end, _ = trap.eval(trap.T)
    assert np.isclose(p_end, q1, atol=1e-4)

    # Velocity must not exceed v_max
    times = np.linspace(0.0, trap.T, 50)
    for t in times:
        _, v, _ = trap.eval(t)
        assert abs(v) <= v_max + 1e-5


def test_slerp_interpolation():
    q0 = np.array([0.0, 0.0, 0.0, 1.0])  # identity
    q1 = np.array([0.0, 0.7071068, 0.0, 0.7071068])  # 90 deg around Y

    # Test endpoints
    assert np.allclose(slerp(q0, q1, 0.0), q0, atol=1e-5)
    assert np.allclose(slerp(q0, q1, 1.0), q1, atol=1e-5)

    # Test midpoint
    q_mid = slerp(q0, q1, 0.5)
    assert np.isclose(np.linalg.norm(q_mid), 1.0, atol=1e-5)


def test_joint_and_cartesian_trajectory_generation():
    engine = KinematicsEngine()
    gen = TrajectoryGenerator(engine)

    q_start = np.array([0.0, -0.4, 0.5, -0.2, 0.3, 0.0])
    q_goal = np.array([0.2, -0.6, 0.8, -0.4, 0.5, 0.1])

    traj = gen.generate_quintic_joint_trajectory(q_start, q_goal, duration=2.0, dt=0.05)
    assert len(traj) > 30

    metrics = gen.compute_metrics(traj)
    assert metrics.duration == 2.0
    assert metrics.peak_velocity > 0.0
    assert metrics.integrated_squared_jerk > 0.0
    assert metrics.min_manipulability is not None
    assert metrics.min_manipulability > 0.0
    assert metrics.mean_manipulability >= metrics.min_manipulability

    # Test Yoshikawa manipulability profile directly
    w_profile, min_w, mean_w = gen.compute_manipulability_profile(traj)
    assert len(w_profile) == len(traj)
    assert min_w > 0.0
    assert np.isclose(min_w, metrics.min_manipulability)

    # Cartesian straight path test
    T_start, _ = engine.forward_kinematics(q_start)
    p_start = T_start[:3, 3]
    R_start = T_start[:3, :3]

    p_goal = p_start + np.array([0.05, -0.05, 0.08])
    R_goal = R_start

    ok, c_traj = gen.generate_cartesian_linear_path(
        p_start=p_start,
        p_goal=p_goal,
        R_start=R_start,
        R_goal=R_goal,
        q_init=q_start,
        num_waypoints=15,
        duration=1.5,
    )
    assert ok, "Cartesian trajectory generation failed"
    assert len(c_traj) == 15
