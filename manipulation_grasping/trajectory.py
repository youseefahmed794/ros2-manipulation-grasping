"""
Trajectory Generation & Motion Smoothing:
Quintic Polynomials (C2-continuous, jerk-bounded), Trapezoidal Velocity Profiles,
and Cartesian Path Interpolation with SLERP.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from manipulation_grasping.kinematics import (
    KinematicsEngine,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)


@dataclass
class TrajectoryPoint:
    """State point along trajectory."""
    time: float
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    jerks: Optional[np.ndarray] = None


@dataclass
class TrajectoryMetrics:
    """Benchmark metrics for trajectory evaluation."""
    duration: float
    num_points: int
    peak_velocity: float
    peak_acceleration: float
    peak_jerk: float
    integrated_squared_jerk: float


class QuinticPolynomial:
    """
    Quintic (5th order) polynomial segment:
    s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    Provides C2 continuity with continuous acceleration and bounded jerk.
    """

    def __init__(
        self,
        q0: float,
        q1: float,
        v0: float = 0.0,
        v1: float = 0.0,
        acc0: float = 0.0,
        acc1: float = 0.0,
        duration: float = 1.0,
    ):
        assert duration > 0.0, "Trajectory duration must be positive"
        self.T = duration
        self.q0, self.q1 = q0, q1

        T, T2, T3, T4, T5 = (
            self.T,
            self.T ** 2,
            self.T ** 3,
            self.T ** 4,
            self.T ** 5,
        )
        h = q1 - q0

        self.a0 = q0
        self.a1 = v0
        self.a2 = 0.5 * acc0

        # Exact closed-form coefficients:
        self.a3 = (20.0 * h - (8.0 * v1 + 12.0 * v0) * T - (3.0 * acc0 - acc1) * T2) / (2.0 * T3)
        self.a4 = (-30.0 * h + (14.0 * v1 + 16.0 * v0) * T + (3.0 * acc0 - 2.0 * acc1) * T2) / (2.0 * T4)
        self.a5 = (12.0 * h - 6.0 * (v1 + v0) * T - (acc0 - acc1) * T2) / (2.0 * T5)

    def eval(self, t: float) -> Tuple[float, float, float, float]:
        """Evaluate position, velocity, acceleration, and jerk at time t."""
        t_clamped = np.clip(t, 0.0, self.T)
        t2 = t_clamped ** 2
        t3 = t2 * t_clamped
        t4 = t3 * t_clamped
        t5 = t4 * t_clamped

        pos = (
            self.a0
            + self.a1 * t_clamped
            + self.a2 * t2
            + self.a3 * t3
            + self.a4 * t4
            + self.a5 * t5
        )
        vel = (
            self.a1
            + 2.0 * self.a2 * t_clamped
            + 3.0 * self.a3 * t2
            + 4.0 * self.a4 * t3
            + 5.0 * self.a5 * t4
        )
        acc = (
            2.0 * self.a2
            + 6.0 * self.a3 * t_clamped
            + 12.0 * self.a4 * t2
            + 20.0 * self.a5 * t3
        )
        jerk = 6.0 * self.a3 + 24.0 * self.a4 * t_clamped + 60.0 * self.a5 * t2

        return pos, vel, acc, jerk


class TrapezoidalProfile:
    """Standard 3-phase trapezoidal velocity profile."""

    def __init__(
        self,
        q0: float,
        q1: float,
        v_max: float = 1.0,
        a_max: float = 2.0,
    ):
        self.q0, self.q1 = q0, q1
        self.delta = q1 - q0
        self.sign = np.sign(self.delta) if abs(self.delta) > 1e-9 else 1.0
        self.d = abs(self.delta)

        self.a_max = abs(a_max)
        self.v_max = abs(v_max)

        # Acceleration distance
        d_acc = (self.v_max ** 2) / (2.0 * self.a_max)

        if self.d < 2.0 * d_acc:
            # Triangular profile (cannot reach v_max)
            self.v_peak = np.sqrt(self.d * self.a_max)
            self.t_acc = self.v_peak / self.a_max
            self.t_cruise = 0.0
            self.t_dec = self.t_acc
        else:
            # Trapezoidal profile
            self.v_peak = self.v_max
            self.t_acc = self.v_max / self.a_max
            d_cruise = self.d - 2.0 * d_acc
            self.t_cruise = d_cruise / self.v_max
            self.t_dec = self.t_acc

        self.T = self.t_acc + self.t_cruise + self.t_dec

    def eval(self, t: float) -> Tuple[float, float, float]:
        """Evaluate position, velocity, and acceleration at time t."""
        if self.T <= 1e-9:
            return self.q0, 0.0, 0.0

        t = np.clip(t, 0.0, self.T)

        if t <= self.t_acc:
            s = 0.5 * self.a_max * (t ** 2)
            v = self.a_max * t
            a = self.a_max
        elif t <= self.t_acc + self.t_cruise:
            t_rel = t - self.t_acc
            s_acc = 0.5 * self.a_max * (self.t_acc ** 2)
            s = s_acc + self.v_peak * t_rel
            v = self.v_peak
            a = 0.0
        else:
            t_rel = t - (self.t_acc + self.t_cruise)
            s_acc = 0.5 * self.a_max * (self.t_acc ** 2)
            s_cruise = self.v_peak * self.t_cruise
            s = s_acc + s_cruise + self.v_peak * t_rel - 0.5 * self.a_max * (t_rel ** 2)
            v = self.v_peak - self.a_max * t_rel
            a = -self.a_max

        pos = self.q0 + self.sign * s
        vel = self.sign * v
        acc = self.sign * a
        return pos, vel, acc


def slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Spherical Linear Interpolation between two quaternions [x, y, z, w]."""
    u = np.clip(u, 0.0, 1.0)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        # Linear approximation for very close orientations
        result = q0 + u * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * u
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    result = (s0 * q0) + (s1 * q1)
    return result / np.linalg.norm(result)


class TrajectoryGenerator:
    """Multi-DOF trajectory planner and Cartesian path generator."""

    def __init__(self, kinematics: Optional[KinematicsEngine] = None):
        self.kinematics = kinematics or KinematicsEngine()

    def generate_quintic_joint_trajectory(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        duration: float = 3.0,
        dt: float = 0.02,
    ) -> List[TrajectoryPoint]:
        """
        Generate smooth C2-continuous quintic polynomial trajectory between joint configurations.
        """
        assert len(q_start) == len(q_goal)
        num_joints = len(q_start)
        num_steps = int(np.ceil(duration / dt)) + 1
        time_vector = np.linspace(0.0, duration, num_steps)

        polys = [
            QuinticPolynomial(q_start[i], q_goal[i], duration=duration)
            for i in range(num_joints)
        ]

        trajectory = []
        for t in time_vector:
            pos = np.zeros(num_joints)
            vel = np.zeros(num_joints)
            acc = np.zeros(num_joints)
            jerk = np.zeros(num_joints)

            for i in range(num_joints):
                p, v, a, j = polys[i].eval(t)
                pos[i] = p
                vel[i] = v
                acc[i] = a
                jerk[i] = j

            trajectory.append(
                TrajectoryPoint(
                    time=t,
                    positions=pos,
                    velocities=vel,
                    accelerations=acc,
                    jerks=jerk,
                )
            )

        return trajectory

    def generate_cartesian_linear_path(
        self,
        p_start: np.ndarray,
        p_goal: np.ndarray,
        R_start: np.ndarray,
        R_goal: np.ndarray,
        q_init: np.ndarray,
        num_waypoints: int = 30,
        duration: float = 3.0,
    ) -> Tuple[bool, List[TrajectoryPoint]]:
        """
        Generate straight-line Cartesian path with SLERP rotation and resolved IK at each waypoint.
        Essential for approach, lift, and insert manipulation phases.
        """
        q_rot_0 = rotation_matrix_to_quaternion(R_start)
        q_rot_1 = rotation_matrix_to_quaternion(R_goal)

        dt = duration / (num_waypoints - 1)
        trajectory: List[TrajectoryPoint] = []
        q_seed = q_init.copy()

        # Quintic timing scaling parameter s(t) in [0, 1]
        scaling_poly = QuinticPolynomial(0.0, 1.0, duration=duration)

        for step in range(num_waypoints):
            t = step * dt
            s, s_dot, s_ddot, _ = scaling_poly.eval(t)

            # Cartesian position linear interpolation
            p_t = p_start + s * (p_goal - p_start)
            # Cartesian orientation SLERP
            q_rot_t = slerp(q_rot_0, q_rot_1, s)
            R_t = quaternion_to_rotation_matrix(q_rot_t)

            # Solve IK for waypoint
            success, q_sol, _, _ = self.kinematics.inverse_kinematics(
                target_pos=p_t,
                target_rot=R_t,
                q_init=q_seed,
                max_iter=60,
                pos_tol=1e-3,
                rot_tol=3e-3,
            )

            if not success:
                return False, trajectory

            q_seed = q_sol.copy()

            # Cartesian velocity & Jacobian inversion for joint velocities: dq = J^(+) * v
            v_cart = s_dot * (p_goal - p_start)
            # Approximate angular velocity: omega = 0 for pure linear translation
            twist = np.zeros(6)
            twist[:3] = v_cart
            J = self.kinematics.compute_jacobian(q_sol)
            JJT = J @ J.T + 1e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJT, twist)

            trajectory.append(
                TrajectoryPoint(
                    time=t,
                    positions=q_sol,
                    velocities=dq,
                    accelerations=np.zeros_like(q_sol),
                )
            )

        return True, trajectory

    @staticmethod
    def compute_metrics(trajectory: List[TrajectoryPoint]) -> TrajectoryMetrics:
        """Calculate academic benchmark metrics for trajectory smoothness."""
        assert len(trajectory) > 1, "Trajectory must have at least 2 points"
        duration = trajectory[-1].time - trajectory[0].time
        dt = trajectory[1].time - trajectory[0].time

        velocities = np.array([pt.velocities for pt in trajectory])
        accelerations = np.array([pt.accelerations for pt in trajectory])

        peak_v = float(np.max(np.abs(velocities)))
        peak_a = float(np.max(np.abs(accelerations)))

        if trajectory[0].jerks is not None:
            jerks = np.array([pt.jerks for pt in trajectory])
            peak_jerk = float(np.max(np.abs(jerks)))
            # Integral of squared jerk sum over all joints
            isj = float(np.sum(jerks ** 2) * dt)
        else:
            # Numerical derivative of acceleration
            jerks = np.gradient(accelerations, dt, axis=0)
            peak_jerk = float(np.max(np.abs(jerks)))
            isj = float(np.sum(jerks ** 2) * dt)

        return TrajectoryMetrics(
            duration=duration,
            num_points=len(trajectory),
            peak_velocity=peak_v,
            peak_acceleration=peak_a,
            peak_jerk=peak_jerk,
            integrated_squared_jerk=isj,
        )
