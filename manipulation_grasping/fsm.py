"""
Autonomous Pick-and-Place State Sequencer:
Finite State Machine (FSM) orchestrating perception, grasp planning,
trajectory generation, and gripper execution.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import time
import numpy as np

from manipulation_grasping.kinematics import KinematicsEngine
from manipulation_grasping.trajectory import TrajectoryGenerator, TrajectoryPoint
from manipulation_grasping.perception import PerceptionPipeline, PointCloud, SegmentedObject
from manipulation_grasping.grasp_planner import GraspPlanner, GraspCandidate


class FSMState(Enum):
    IDLE = auto()
    PERCEIVE = auto()
    SEGMENT = auto()
    PLAN_GRASP = auto()
    APPROACH_PRE_GRASP = auto()
    DESCEND_TO_GRASP = auto()
    CLOSE_GRIPPER = auto()
    LIFT_OBJECT = auto()
    TRANSFER = auto()
    DESCEND_TO_PLACE = auto()
    OPEN_GRIPPER = auto()
    RETRACT_HOME = auto()
    SUCCESS = auto()
    FAILED = auto()


@dataclass
class PickAndPlaceContext:
    """Shared blackboard state for execution cycle."""
    current_q: np.ndarray = field(default_factory=lambda: np.array([0.0, -np.pi / 4, np.pi / 3, -np.pi / 6, np.pi / 2, 0.0]))
    gripper_state: float = 0.04  # finger position (0.0 = closed, 0.04 = fully open)
    home_q: np.ndarray = field(default_factory=lambda: np.array([0.0, -np.pi / 4, np.pi / 3, -np.pi / 6, np.pi / 2, 0.0]))
    place_target_pos: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.45, 0.25]))
    place_target_rot: np.ndarray = field(default_factory=lambda: np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]
    ]))

    raw_cloud: Optional[PointCloud] = None
    table_plane: Optional[np.ndarray] = None
    detected_objects: List[SegmentedObject] = field(default_factory=list)
    target_object: Optional[SegmentedObject] = None
    active_grasp: Optional[GraspCandidate] = None
    active_trajectory: Optional[List[TrajectoryPoint]] = None

    cycle_count: int = 0
    start_time: float = 0.0
    elapsed_time: float = 0.0
    error_message: str = ""
    state_history: List[str] = field(default_factory=list)


class PickAndPlaceFSM:
    """Production-grade finite state machine for autonomous manipulation."""

    def __init__(
        self,
        kinematics: Optional[KinematicsEngine] = None,
        trajectory_gen: Optional[TrajectoryGenerator] = None,
        perception: Optional[PerceptionPipeline] = None,
        grasp_planner: Optional[GraspPlanner] = None,
    ):
        self.kinematics = kinematics or KinematicsEngine()
        self.trajectory_gen = trajectory_gen or TrajectoryGenerator(self.kinematics)
        self.perception = perception or PerceptionPipeline()
        self.grasp_planner = grasp_planner or GraspPlanner(self.kinematics)

        self.state = FSMState.IDLE
        self.context = PickAndPlaceContext()

    def set_state(self, new_state: FSMState) -> None:
        """State transition with logging."""
        self.state = new_state
        self.context.state_history.append(new_state.name)

    def step(self) -> FSMState:
        """Advance state machine by one cycle/transition."""
        if self.state == FSMState.IDLE:
            self._handle_idle()
        elif self.state == FSMState.PERCEIVE:
            self._handle_perceive()
        elif self.state == FSMState.SEGMENT:
            self._handle_segment()
        elif self.state == FSMState.PLAN_GRASP:
            self._handle_plan_grasp()
        elif self.state == FSMState.APPROACH_PRE_GRASP:
            self._handle_approach_pre_grasp()
        elif self.state == FSMState.DESCEND_TO_GRASP:
            self._handle_descend_to_grasp()
        elif self.state == FSMState.CLOSE_GRIPPER:
            self._handle_close_gripper()
        elif self.state == FSMState.LIFT_OBJECT:
            self._handle_lift_object()
        elif self.state == FSMState.TRANSFER:
            self._handle_transfer()
        elif self.state == FSMState.DESCEND_TO_PLACE:
            self._handle_descend_to_place()
        elif self.state == FSMState.OPEN_GRIPPER:
            self._handle_open_gripper()
        elif self.state == FSMState.RETRACT_HOME:
            self._handle_retract_home()

        return self.state

    def run_full_sequence(
        self,
        point_cloud: Optional[PointCloud] = None,
        max_steps: int = 25,
    ) -> Tuple[bool, List[str], float]:
        """Run complete pick-and-place sequence until SUCCESS or FAILED."""
        self.context = PickAndPlaceContext()
        self.context.start_time = time.time()
        self.context.raw_cloud = point_cloud
        self.set_state(FSMState.IDLE)

        steps = 0
        while self.state not in (FSMState.SUCCESS, FSMState.FAILED) and steps < max_steps:
            self.step()
            steps += 1

        self.context.elapsed_time = time.time() - self.context.start_time
        success = (self.state == FSMState.SUCCESS)
        return success, self.context.state_history, self.context.elapsed_time

    # ── State Handler Methods ─────────────────────────────────────────

    def _handle_idle(self) -> None:
        """Initialize cycle and check robot ready state."""
        self.context.cycle_count += 1
        self.set_state(FSMState.PERCEIVE)

    def _handle_perceive(self) -> None:
        """Acquire and downsample point cloud within workspace."""
        if self.context.raw_cloud is None:
            # Generate synthetic tabletop scene if not provided
            self.context.raw_cloud = self._generate_synthetic_scene()

        filtered_cloud = self.perception.pass_through_filter(self.context.raw_cloud)
        downsampled = self.perception.voxel_downsample(filtered_cloud)
        self.context.raw_cloud = downsampled
        self.set_state(FSMState.SEGMENT)

    def _handle_segment(self) -> None:
        """Segment table plane and cluster objects."""
        plane, inliers, outliers = self.perception.ransac_plane_segmentation(self.context.raw_cloud)
        self.context.table_plane = plane

        if len(outliers) < 10:
            self.context.error_message = "No object clusters detected above table plane"
            self.set_state(FSMState.FAILED)
            return

        objects = self.perception.extract_segmented_objects(self.context.raw_cloud, outliers)
        if not objects:
            self.context.error_message = "Clustering failed to form valid object"
            self.set_state(FSMState.FAILED)
            return

        self.context.detected_objects = objects
        # Target closest object to robot base in workspace
        self.context.target_object = sorted(
            objects, key=lambda o: np.linalg.norm(o.centroid[:2])
        )[0]
        self.set_state(FSMState.PLAN_GRASP)

    def _handle_plan_grasp(self) -> None:
        """Synthesize antipodal grasp candidate and verify reachability."""
        obj = self.context.target_object
        grasps = self.grasp_planner.sample_antipodal_grasps(
            obj, num_samples=50, verify_ik=True, q_current=self.context.current_q
        )

        if not grasps:
            self.context.error_message = "No kinematically feasible antipodal grasp found"
            self.set_state(FSMState.FAILED)
            return

        self.context.active_grasp = grasps[0]
        self.set_state(FSMState.APPROACH_PRE_GRASP)

    def _handle_approach_pre_grasp(self) -> None:
        """Compute and execute trajectory to standoff pose above object."""
        grasp = self.context.active_grasp
        ok, q_pre, _, _ = self.kinematics.inverse_kinematics(
            target_pos=grasp.pre_grasp_position,
            target_rot=grasp.pre_grasp_orientation,
            q_init=self.context.current_q,
        )
        if not ok:
            self.context.error_message = "IK failed for pre-grasp standoff"
            self.set_state(FSMState.FAILED)
            return

        traj = self.trajectory_gen.generate_quintic_joint_trajectory(
            q_start=self.context.current_q,
            q_goal=q_pre,
            duration=2.5,
        )
        self.context.active_trajectory = traj
        self.context.current_q = q_pre
        self.set_state(FSMState.DESCEND_TO_GRASP)

    def _handle_descend_to_grasp(self) -> None:
        """Execute straight Cartesian descent from standoff to grasp point."""
        grasp = self.context.active_grasp
        ok, traj = self.trajectory_gen.generate_cartesian_linear_path(
            p_start=grasp.pre_grasp_position,
            p_goal=grasp.position,
            R_start=grasp.pre_grasp_orientation,
            R_goal=grasp.orientation_matrix,
            q_init=self.context.current_q,
            num_waypoints=20,
            duration=1.5,
        )
        if not ok:
            self.context.error_message = "Cartesian descend trajectory failed IK"
            self.set_state(FSMState.FAILED)
            return

        self.context.active_trajectory = traj
        self.context.current_q = traj[-1].positions
        self.set_state(FSMState.CLOSE_GRIPPER)

    def _handle_close_gripper(self) -> None:
        """Simulate gripper closing onto object."""
        grasp = self.context.active_grasp
        target_width = max(0.005, grasp.grasp_width * 0.5)
        self.context.gripper_state = target_width
        self.set_state(FSMState.LIFT_OBJECT)

    def _handle_lift_object(self) -> None:
        """Perform vertical linear lift along world z-axis."""
        T_curr, _ = self.kinematics.forward_kinematics(self.context.current_q)
        p_curr = T_curr[:3, 3]
        R_curr = T_curr[:3, :3]

        p_lifted = p_curr + np.array([0.0, 0.0, 0.12])  # 12 cm lift
        ok, traj = self.trajectory_gen.generate_cartesian_linear_path(
            p_start=p_curr,
            p_goal=p_lifted,
            R_start=R_curr,
            R_goal=R_curr,
            q_init=self.context.current_q,
            num_waypoints=20,
            duration=1.5,
        )
        if not ok:
            self.context.error_message = "Cartesian lift trajectory failed IK"
            self.set_state(FSMState.FAILED)
            return

        self.context.active_trajectory = traj
        self.context.current_q = traj[-1].positions
        self.set_state(FSMState.TRANSFER)

    def _handle_transfer(self) -> None:
        """Transfer object over target bin."""
        p_place_pre = self.context.place_target_pos + np.array([0.0, 0.0, 0.12])
        ok, q_place_pre, _, _ = self.kinematics.inverse_kinematics(
            target_pos=p_place_pre,
            target_rot=self.context.place_target_rot,
            q_init=self.context.current_q,
        )
        if not ok:
            self.context.error_message = "IK failed for transfer target"
            self.set_state(FSMState.FAILED)
            return

        traj = self.trajectory_gen.generate_quintic_joint_trajectory(
            q_start=self.context.current_q,
            q_goal=q_place_pre,
            duration=3.0,
        )
        self.context.active_trajectory = traj
        self.context.current_q = q_place_pre
        self.set_state(FSMState.DESCEND_TO_PLACE)

    def _handle_descend_to_place(self) -> None:
        """Lower object into bin."""
        T_curr, _ = self.kinematics.forward_kinematics(self.context.current_q)
        p_curr = T_curr[:3, 3]
        R_curr = T_curr[:3, :3]

        p_place = self.context.place_target_pos.copy()
        ok, traj = self.trajectory_gen.generate_cartesian_linear_path(
            p_start=p_curr,
            p_goal=p_place,
            R_start=R_curr,
            R_goal=R_curr,
            q_init=self.context.current_q,
            num_waypoints=20,
            duration=1.5,
        )
        if not ok:
            self.context.error_message = "Cartesian descend to place failed IK"
            self.set_state(FSMState.FAILED)
            return

        self.context.active_trajectory = traj
        self.context.current_q = traj[-1].positions
        self.set_state(FSMState.OPEN_GRIPPER)

    def _handle_open_gripper(self) -> None:
        """Release object into bin."""
        self.context.gripper_state = 0.04  # fully opened
        self.set_state(FSMState.RETRACT_HOME)

    def _handle_retract_home(self) -> None:
        """Return manipulator arm to home configuration."""
        traj = self.trajectory_gen.generate_quintic_joint_trajectory(
            q_start=self.context.current_q,
            q_goal=self.context.home_q,
            duration=2.5,
        )
        self.context.active_trajectory = traj
        self.context.current_q = self.context.home_q.copy()
        self.set_state(FSMState.SUCCESS)

    @staticmethod
    def _generate_synthetic_scene() -> PointCloud:
        """Generate high-density synthetic tabletop point cloud for tests."""
        rng = np.random.default_rng(42)

        # 1. Tabletop plane: 0.8m x 0.8m at z = 0.0
        n_table = 2500
        x_table = rng.uniform(0.25, 0.75, n_table)
        y_table = rng.uniform(-0.35, 0.35, n_table)
        z_table = rng.normal(0.0, 0.002, n_table)
        table_pts = np.column_stack([x_table, y_table, z_table])

        # 2. Target pick object: 4cm x 4cm x 6cm box at (x=0.45, y=0.0, z=0.03)
        n_box = 800
        x_box = rng.uniform(0.43, 0.47, n_box)
        y_box = rng.uniform(-0.02, 0.02, n_box)
        z_box = rng.uniform(0.005, 0.065, n_box)
        box_pts = np.column_stack([x_box, y_box, z_box])

        pts = np.vstack([table_pts, box_pts])
        return PointCloud(points=pts)
