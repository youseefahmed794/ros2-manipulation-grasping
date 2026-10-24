"""
Arm Kinematics Engine: Forward Kinematics, Geometric Jacobian,
and Damped Least Squares (Levenberg-Marquardt) Inverse Kinematics.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class DHParameter:
    """Standard Denavit-Hartenberg parameters for a single link."""
    a: float  # link length (m)
    alpha: float  # link twist (rad)
    d: float  # link offset (m)
    theta_offset: float = 0.0  # joint angle offset (rad)


@dataclass
class ArmConfig:
    """Kinematic configuration for a 6-DOF industrial manipulator."""
    # Industrial 6-DOF DH table (standard DH convention)
    dh_table: List[DHParameter] = field(default_factory=lambda: [
        DHParameter(a=0.0, alpha=np.pi / 2, d=0.1625, theta_offset=0.0),    # Joint 1: Base to Shoulder Pan
        DHParameter(a=-0.425, alpha=0.0, d=0.0, theta_offset=0.0),          # Joint 2: Shoulder Lift
        DHParameter(a=-0.3922, alpha=0.0, d=0.0, theta_offset=0.0),         # Joint 3: Elbow
        DHParameter(a=0.0, alpha=np.pi / 2, d=0.1333, theta_offset=0.0),    # Joint 4: Wrist 1
        DHParameter(a=0.0, alpha=-np.pi / 2, d=0.0997, theta_offset=0.0),   # Joint 5: Wrist 2
        DHParameter(a=0.0, alpha=0.0, d=0.0996, theta_offset=0.0),          # Joint 6: Wrist 3
    ])
    tcp_offset_z: float = 0.14  # Tool Center Point offset along tool z-axis (gripper grasp center)
    joint_limits_min: np.ndarray = field(default_factory=lambda: np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]))
    joint_limits_max: np.ndarray = field(default_factory=lambda: np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]))
    joint_velocity_limits: np.ndarray = field(default_factory=lambda: np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14]))


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def trans_z(d: float) -> np.ndarray:
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, d],
        [0.0, 0.0, 0.0, 1.0]
    ])


def trans_x(a: float) -> np.ndarray:
    return np.array([
        [1.0, 0.0, 0.0, a],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def rot_x(alpha: float) -> np.ndarray:
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0,  c, -s, 0.0],
        [0.0,  s,  c, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Standard DH homogeneous transformation matrix: Rot_z * Trans_z * Trans_x * Rot_x."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0, sa,       ca,      d],
        [0.0, 0.0,      0.0,     1.0]
    ])


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler XYZ (roll, pitch, yaw) to 3x3 rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]
    ])


class KinematicsEngine:
    """High-fidelity 6-DOF manipulator kinematics engine."""

    def __init__(self, config: Optional[ArmConfig] = None):
        self.config = config or ArmConfig()
        self.num_joints = len(self.config.dh_table)

    def forward_kinematics(
        self, q: np.ndarray, return_all_frames: bool = False
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        Compute forward kinematics for given joint positions q.
        
        Returns:
            T_tcp: 4x4 homogeneous transformation of Tool Center Point relative to base.
            frames: Optional list of intermediate frame transformations [T_0_0, T_0_1, ..., T_0_6, T_0_tcp].
        """
        assert len(q) == self.num_joints, f"Expected {self.num_joints} joints, got {len(q)}"

        frames = [np.eye(4)]
        T_curr = np.eye(4)

        for i, (dh, q_i) in enumerate(zip(self.config.dh_table, q)):
            theta = q_i + dh.theta_offset
            T_link = dh_transform(theta, dh.d, dh.a, dh.alpha)
            T_curr = T_curr @ T_link
            frames.append(T_curr.copy())

        # TCP offset along z-axis of link 6
        T_tcp_offset = np.eye(4)
        T_tcp_offset[2, 3] = self.config.tcp_offset_z
        T_tcp = T_curr @ T_tcp_offset
        frames.append(T_tcp.copy())

        if return_all_frames:
            return T_tcp, frames
        return T_tcp, None

    def compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Compute 6x6 geometric Jacobian J(q) mapping joint velocities to TCP twist.
        J(q) = [J_v; J_omega]
        """
        T_tcp, frames = self.forward_kinematics(q, return_all_frames=True)
        p_tcp = T_tcp[:3, 3]

        J = np.zeros((6, self.num_joints))

        for i in range(self.num_joints):
            T_i_prev = frames[i]  # Base frame is index 0
            z_i_prev = T_i_prev[:3, 2]  # z-axis of joint i
            p_i_prev = T_i_prev[:3, 3]  # origin of joint i

            # Linear velocity component: z_{i-1} x (p_tcp - p_{i-1})
            J[:3, i] = np.cross(z_i_prev, p_tcp - p_i_prev)
            # Angular velocity component: z_{i-1}
            J[3:, i] = z_i_prev

        return J

    def manipulability(self, q: np.ndarray) -> float:
        """
        Compute Yoshikawa Manipulability measure: w(q) = sqrt(det(J * J^T)).
        Measures distance from kinematic singularities.
        """
        J = self.compute_jacobian(q)
        det_JJT = np.linalg.det(J @ J.T)
        if det_JJT < 0.0:
            return 0.0
        return float(np.sqrt(det_JJT))

    def condition_number(self, q: np.ndarray) -> float:
        """Condition number kappa(J) = sigma_max / sigma_min."""
        J = self.compute_jacobian(q)
        s = np.linalg.svd(J, compute_uv=False)
        if s[-1] < 1e-12:
            return float('inf')
        return float(s[0] / s[-1])

    @staticmethod
    def compute_pose_error(
        T_curr: np.ndarray, T_target: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        Compute 6D task-space error e = [e_pos; e_rot] in base frame.
        Orientation error calculated via vector cross-product of column vectors.
        """
        pos_err = T_target[:3, 3] - T_curr[:3, 3]

        R_c = T_curr[:3, :3]
        R_d = T_target[:3, :3]

        # Orientation error: 0.5 * (n_c x n_d + s_c x s_d + a_c x a_d)
        n_c, s_c, a_c = R_c[:, 0], R_c[:, 1], R_c[:, 2]
        n_d, s_d, a_d = R_d[:, 0], R_d[:, 1], R_d[:, 2]
        rot_err = 0.5 * (
            np.cross(n_c, n_d) + np.cross(s_c, s_d) + np.cross(a_c, a_d)
        )

        pos_norm = float(np.linalg.norm(pos_err))
        rot_norm = float(np.linalg.norm(rot_err))
        err_vector = np.concatenate([pos_err, rot_err])
        return err_vector, pos_norm, rot_norm

    def _solve_dls(
        self,
        T_target: np.ndarray,
        q_start: np.ndarray,
        max_iter: int,
        pos_tol: float,
        rot_tol: float,
        damping_lambda: float,
        adaptive_damping: bool,
        null_space_weight: float,
    ) -> Tuple[bool, np.ndarray, int, float]:
        """Core Damped Least Squares Levenberg-Marquardt numerical iteration."""
        q = np.array(q_start, dtype=float).copy()
        q_mid = 0.5 * (self.config.joint_limits_min + self.config.joint_limits_max)
        q_range = self.config.joint_limits_max - self.config.joint_limits_min

        for it in range(max_iter):
            T_curr, _ = self.forward_kinematics(q)
            e, pos_err_norm, rot_err_norm = self.compute_pose_error(T_curr, T_target)

            if pos_err_norm < pos_tol and rot_err_norm < rot_tol:
                return True, q, it, pos_err_norm

            J = self.compute_jacobian(q)

            # Yoshikawa manipulability for adaptive damping
            w = self.manipulability(q)
            w_threshold = 0.03

            if adaptive_damping:
                if w < w_threshold:
                    lam2 = (damping_lambda ** 2) * ((1.0 - (w / w_threshold)) ** 2)
                else:
                    lam2 = 1e-6
            else:
                lam2 = damping_lambda ** 2

            # Damped Least Squares: dq = J^T * (J * J^T + lam2 * I)^(-1) * e
            JJT = J @ J.T
            dls_mat = JJT + lam2 * np.eye(6)
            dq_primary = J.T @ np.linalg.solve(dls_mat, e)

            # Null-space joint limit avoidance
            if null_space_weight > 0.0:
                J_pinv = J.T @ np.linalg.solve(JJT + 1e-4 * np.eye(6), np.eye(6))
                null_projector = np.eye(self.num_joints) - J_pinv @ J
                grad_H = -2.0 * (q - q_mid) / (q_range ** 2)
                dq_null = null_space_weight * (null_projector @ grad_H)
            else:
                dq_null = np.zeros(self.num_joints)

            dq = dq_primary + dq_null

            # Limit step size per iteration to avoid overshoot
            max_step = 0.25  # rad
            step_norm = np.max(np.abs(dq))
            if step_norm > max_step:
                dq = dq * (max_step / step_norm)

            q += dq
            q = np.clip(q, self.config.joint_limits_min, self.config.joint_limits_max)

        # Final check
        T_curr, _ = self.forward_kinematics(q)
        _, final_pos_err, final_rot_err = self.compute_pose_error(T_curr, T_target)
        success = (final_pos_err < pos_tol * 2.0) and (final_rot_err < rot_tol * 2.0)
        return success, q, max_iter, final_pos_err

    def inverse_kinematics(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        max_iter: int = 150,
        pos_tol: float = 1e-3,
        rot_tol: float = 2e-3,
        damping_lambda: float = 0.04,
        adaptive_damping: bool = True,
        null_space_weight: float = 0.05,
    ) -> Tuple[bool, np.ndarray, int, float]:
        """
        Solve Inverse Kinematics using Damped Least Squares (Levenberg-Marquardt)
        with adaptive singularity damping, multi-seed restart, and null-space joint limit avoidance.

        Args:
            target_pos: 3D position [x, y, z] in meters.
            target_rot: 3x3 rotation matrix or 4-element quaternion [x, y, z, w].
            q_init: Initial joint angles guess (default: multi-seed geometric restarts).
            max_iter: Maximum solver iterations.
            pos_tol: Position tolerance in meters (default: 1 mm).
            rot_tol: Orientation tolerance in radians.
            damping_lambda: Base damping parameter lambda_0.
            adaptive_damping: Adjust lambda dynamically based on manipulability.
            null_space_weight: Weight for null-space projection pushing joints away from limits.

        Returns:
            success: Boolean indicating convergence.
            q: Solved joint configuration (within limits).
            iterations: Number of iterations performed.
            final_err: Final Euclidean norm of task-space error.
        """
        if target_rot.shape == (4,):
            R_target = quaternion_to_rotation_matrix(target_rot)
        else:
            R_target = np.asarray(target_rot, dtype=float)

        T_target = np.eye(4)
        T_target[:3, :3] = R_target
        T_target[:3, 3] = target_pos

        # Attempt 1: Try user-supplied initial seed if given
        best_result = None
        min_error = float("inf")
        total_iters = 0

        if q_init is not None:
            ok, q_sol, it, err = self._solve_dls(
                T_target, q_init, max_iter, pos_tol, rot_tol,
                damping_lambda, adaptive_damping, null_space_weight
            )
            total_iters += it
            if ok:
                return True, q_sol, total_iters, err
            best_result = (ok, q_sol, it, err)
            min_error = err

        # Attempt 2+: Intelligent multi-seed restart with shoulder offset compensation
        r = float(np.linalg.norm(target_pos[:2]))
        phi = float(np.arctan2(target_pos[1], target_pos[0]))
        d4 = float(self.config.dh_table[3].d) if len(self.config.dh_table) > 3 else 0.1333
        ratio = float(np.clip(d4 / max(r, 0.05), -0.99, 0.99))
        alpha = float(np.arcsin(ratio))

        q1_candidates = [
            phi + np.pi - alpha,
            phi + np.pi + alpha,
            phi - alpha,
            phi + alpha,
            phi + np.pi,
            phi,
            0.0,
        ]

        postures = [
            np.array([-1.0, 1.5, 0.5, 1.57, 0.0]),
            np.array([-0.8, 1.8, -1.0, 1.57, 0.0]),
            np.array([-1.2, 1.2, 0.0, 1.57, 0.0]),
            np.array([-np.pi / 4, np.pi / 3, -np.pi / 6, np.pi / 2, 0.0]),
            np.array([-np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0]),
        ]

        iter_per_seed = min(max_iter, 50)
        seen_q1 = []
        for q1_raw in q1_candidates:
            # Wrap into [-pi, pi]
            q1_wrapped = (q1_raw + np.pi) % (2.0 * np.pi) - np.pi
            # Clamp away from hard limits
            q1_clamped = float(np.clip(
                q1_wrapped,
                self.config.joint_limits_min[0] + 0.05,
                self.config.joint_limits_max[0] - 0.05,
            ))
            if any(abs(q1_clamped - prev) < 0.08 for prev in seen_q1):
                continue
            seen_q1.append(q1_clamped)

            for posture in postures:
                seed = np.concatenate([[q1_clamped], posture])
                ok, q_sol, it, err = self._solve_dls(
                    T_target, seed, iter_per_seed, pos_tol, rot_tol,
                    damping_lambda, adaptive_damping, null_space_weight
                )
                total_iters += it
                if ok:
                    return True, q_sol, total_iters, err
                if err < min_error:
                    min_error = err
                    best_result = (ok, q_sol, total_iters, err)

        if best_result is not None:
            return best_result
        return False, np.zeros(self.num_joints), total_iters, min_error
