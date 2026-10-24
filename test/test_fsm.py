import pytest
import numpy as np

from manipulation_grasping.fsm import PickAndPlaceFSM, FSMState
from manipulation_grasping.perception import PointCloud


def test_fsm_initial_state():
    fsm = PickAndPlaceFSM()
    assert fsm.state == FSMState.IDLE
    assert fsm.context.cycle_count == 0


def test_fsm_full_successful_execution():
    fsm = PickAndPlaceFSM()
    success, history, elapsed = fsm.run_full_sequence()

    assert success, f"FSM failed to complete successfully! Error: {fsm.context.error_message}"
    assert fsm.state == FSMState.SUCCESS
    assert elapsed > 0.0

    # Ensure key manipulation states are traversed in correct order
    expected_states = [
        "PERCEIVE",
        "SEGMENT",
        "PLAN_GRASP",
        "APPROACH_PRE_GRASP",
        "DESCEND_TO_GRASP",
        "CLOSE_GRIPPER",
        "LIFT_OBJECT",
        "TRANSFER",
        "DESCEND_TO_PLACE",
        "OPEN_GRIPPER",
        "RETRACT_HOME",
        "SUCCESS"
    ]
    for state_name in expected_states:
        assert state_name in history, f"Expected state {state_name} not found in history: {history}"


def test_fsm_error_handling_empty_scene():
    fsm = PickAndPlaceFSM()
    # Provide empty point cloud to trigger perception failure
    empty_cloud = PointCloud(points=np.empty((0, 3)))
    success, history, _ = fsm.run_full_sequence(point_cloud=empty_cloud)

    assert not success
    assert fsm.state == FSMState.FAILED
    assert len(fsm.context.error_message) > 0
