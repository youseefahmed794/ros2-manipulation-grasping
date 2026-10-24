"""
Antipodal Grasp Pose Synthesis & Quality Evaluation:
Antipodal Contact Sampling, Friction Cone Verification,
SE(3) Grasp Frame Construction, and Kinematic Feasibility Checking.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from manipulation_grasping.kinematics import (
    KinematicsEngine,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from manipulation_grasping.perception import SegmentedObject


@dataclass
class GraspCandidate:
    """Evaluated 6D Grasp Candidate."""
    position: np.ndarray  # (3,) Tool Center Point at grasp
    orientation_matrix: np.ndarray  # (3, 3) End-effector rotation matrix
    quaternion: np.ndarray  # (4,) [x, y, z, w]
    pre_grasp_position: np.ndarray  # (3,) Standoff approach position
    pre_grasp_orientation: np.ndarray  # (3, 3)
    contact_p1: np.ndarray  # (3,)
    contact_p2: np.ndarray  # (3,)
    grasp_width: float  # Distance between fingers (m)
    quality_score: float  # [0, 1] composite quality metric
    ik_joint_solution: Optional[np.ndarray] = None


class GraspPlanner:
    """Antipodal grasp synthesis for parallel-jaw grippers."""

    def __init__(
        self,
        kinematics: Optional[KinematicsEngine] = None,
        max_gripper_width: float = 0.085,
        min_gripper_width: float = 0.015,
        friction_coefficient: float = 0.5,
        standoff_distance: float = 0.10,
    ):
        self.kinematics = kinematics or KinematicsEngine()
        self.max_gripper_width = max_gripper_width
        self.min_gripper_width = min_gripper_width
        self.friction_coeff = friction_coefficient
        self.friction_angle = np.arctan(friction_coefficient)
        self.standoff_distance = standoff_distance

    def sample_antipodal_grasps(
        self,
        obj: SegmentedObject,
        num_samples: int = 60,
        verify_ik: bool = True,
        q_current: Optional[np.ndarray] = None,
    ) -> List[GraspCandidate]:
        """
        Sample and filter antipodal grasps across the segmented object.
        """
        pts = obj.points
        normals = obj.surface_normals
        N = len(pts)
        if N < 4:
            return []

        candidates: List[GraspCandidate] = []
        rng = np.random.default_rng(101)

        # 1. Strategy A: Principal-axis bounding box grasps (highly reliable for geometric primitives)
        obb_grasps = self._generate_obb_grasps(obj)
        candidates.extend(obb_grasps)

        # 2. Strategy B: Surface antipodal point pair sampling
        for _ in range(num_samples):
            idx1 = rng.choice(N)
            p1 = pts[idx1]
            n1 = normals[idx1]

            # Find points roughly antipodal to p1
            # Closing vector from p1 to p2 should align with -n1
            dists = np.linalg.norm(pts - p1, axis=1)
            valid_width_mask = (dists >= self.min_gripper_width) & (dists <= self.max_gripper_width)
            candidate_indices = np.where(valid_width_mask)[0]

            if len(candidate_indices) == 0:
                continue

            idx2 = rng.choice(candidate_indices)
            p2 = pts[idx2]
            n2 = normals[idx2]

            v_close = p2 - p1
            w = np.linalg.norm(v_close)
            v_close_unit = v_close / w

            # Friction cone antipodal test
            dot1 = np.dot(-n1, v_close_unit)
            dot2 = np.dot(n2, v_close_unit)

            theta1 = np.arccos(np.clip(dot1, -1.0, 1.0))
            theta2 = np.arccos(np.clip(dot2, -1.0, 1.0))

            if theta1 <= self.friction_angle and theta2 <= self.friction_angle:
                # Valid antipodal pair
                grasp = self._create_grasp_from_pair(p1, p2, n1, n2, theta1, theta2, w)
                if grasp is not None:
                    candidates.append(grasp)

        # 3. Filter candidates through IK solver and rank by quality
        evaluated_candidates = []
        for g in candidates:
            if verify_ik:
                # Check IK at grasp pose
                ik_ok, q_sol, _, _ = self.kinematics.inverse_kinematics(
                    target_pos=g.position,
                    target_rot=g.orientation_matrix,
                    q_init=q_current,
                    max_iter=80,
                    pos_tol=2e-3,
                    rot_tol=5e-3,
                )
                if not ik_ok:
                    continue

                # Check IK at pre-grasp standoff pose
                ik_pre_ok, _, _, _ = self.kinematics.inverse_kinematics(
                    target_pos=g.pre_grasp_position,
                    target_rot=g.pre_grasp_orientation,
                    q_init=q_sol,
                    max_iter=60,
                    pos_tol=2e-3,
                    rot_tol=5e-3,
                )
                if not ik_pre_ok:
                    continue

                g.ik_joint_solution = q_sol
                evaluated_candidates.append(g)
            else:
                evaluated_candidates.append(g)

        # Rank descending by quality score
        evaluated_candidates.sort(key=lambda x: x.quality_score, reverse=True)
        return evaluated_candidates

    def _generate_obb_grasps(self, obj: SegmentedObject) -> List[GraspCandidate]:
        """Generate canonical grasps aligned with object principal axes."""
        c = obj.centroid
        axes = obj.principal_axes  # 3 principal orthogonal vectors
        dims = obj.dimensions  # lengths along the 3 axes

        grasps = []
        # Sort dimensions to find graspable width (must be < max_gripper_width)
        for i in range(3):
            width = dims[i]
            if self.min_gripper_width <= width <= self.max_gripper_width:
                # Finger closing axis is along axes[:, i]
                closing_axis = axes[:, i] / np.linalg.norm(axes[:, i])

                # Preferred approach vector: vertical downwards [-Z] or along remaining principal axis
                z_down = np.array([0.0, 0.0, -1.0])
                # Approach axis should be perpendicular to closing axis
                approach_axis = z_down - np.dot(z_down, closing_axis) * closing_axis
                norm_app = np.linalg.norm(approach_axis)
                if norm_app < 0.2:
                    # Closing axis is almost vertical, choose horizontal approach
                    for j in range(3):
                        if j != i:
                            approach_axis = axes[:, j]
                            break
                    norm_app = np.linalg.norm(approach_axis)

                approach_axis = approach_axis / norm_app

                # Orthogonal y axis: y = z x x
                y_axis = np.cross(approach_axis, closing_axis)
                y_axis = y_axis / np.linalg.norm(y_axis)

                # Recompute closing axis to guarantee exact orthonormality: x = y x z
                closing_axis = np.cross(y_axis, approach_axis)

                R_grasp = np.column_stack([closing_axis, y_axis, approach_axis])
                q_grasp = rotation_matrix_to_quaternion(R_grasp)

                grasp_pos = c.copy()
                pre_grasp_pos = grasp_pos - self.standoff_distance * approach_axis

                p1 = c - 0.5 * width * closing_axis
                p2 = c + 0.5 * width * closing_axis

                # Quality metric: alignment with top-down approach + centered width
                quality = 0.85 + 0.15 * (1.0 - abs(width - 0.04) / self.max_gripper_width)

                grasps.append(
                    GraspCandidate(
                        position=grasp_pos,
                        orientation_matrix=R_grasp,
                        quaternion=q_grasp,
                        pre_grasp_position=pre_grasp_pos,
                        pre_grasp_orientation=R_grasp,
                        contact_p1=p1,
                        contact_p2=p2,
                        grasp_width=float(width),
                        quality_score=float(quality),
                    )
                )

        return grasps

    def _create_grasp_from_pair(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        n1: np.ndarray,
        n2: np.ndarray,
        theta1: float,
        theta2: float,
        width: float,
    ) -> Optional[GraspCandidate]:
        """Construct SE(3) grasp frame from antipodal contact points."""
        closing_axis = (p2 - p1) / width
        center = 0.5 * (p1 + p2)

        # Approach axis: prefer downward z-axis
        z_down = np.array([0.0, 0.0, -1.0])
        approach_axis = z_down - np.dot(z_down, closing_axis) * closing_axis
        norm_app = np.linalg.norm(approach_axis)
        if norm_app < 1e-4:
            return None

        approach_axis = approach_axis / norm_app
        y_axis = np.cross(approach_axis, closing_axis)
        norm_y = np.linalg.norm(y_axis)
        if norm_y < 1e-4:
            return None
        y_axis = y_axis / norm_y
        closing_axis = np.cross(y_axis, approach_axis)

        R_grasp = np.column_stack([closing_axis, y_axis, approach_axis])
        q_grasp = rotation_matrix_to_quaternion(R_grasp)

        pre_grasp_pos = center - self.standoff_distance * approach_axis

        # Composite quality score
        friction_term = 1.0 - (theta1 + theta2) / (2.0 * self.friction_angle)
        width_term = 1.0 - abs(width - 0.04) / self.max_gripper_width
        quality = 0.6 * friction_term + 0.4 * width_term

        return GraspCandidate(
            position=center,
            orientation_matrix=R_grasp,
            quaternion=q_grasp,
            pre_grasp_position=pre_grasp_pos,
            pre_grasp_orientation=R_grasp,
            contact_p1=p1,
            contact_p2=p2,
            grasp_width=float(width),
            quality_score=float(np.clip(quality, 0.0, 1.0)),
        )
