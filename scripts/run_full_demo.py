#!/usr/bin/env python3
"""
Full Autonomous Pick-and-Place End-to-End Simulation Demo:
Executes the complete multi-stage manipulation pipeline, verifies every transition,
and prints timing and trajectory telemetry.
"""

import sys
import time
import numpy as np

from manipulation_grasping.fsm import PickAndPlaceFSM, FSMState
from manipulation_grasping.trajectory import TrajectoryGenerator


def main():
    print("=" * 70)
    print("🤖 Autonomous 6-DOF Robotic Pick-and-Place End-to-End Demonstration")
    print("   Physical AI Manipulation & Grasping Pipeline")
    print("=" * 70)

    fsm = PickAndPlaceFSM()

    print("\n[Phase 1] Initializing Robot Manipulator and Scene...")
    print(f"  Home Joint Configuration: {np.round(fsm.context.home_q, 3)}")
    print(f"  Target Place Bin Pose:    {fsm.context.place_target_pos}")

    print("\n[Phase 2] Executing Full Pick-and-Place State Sequence...")
    success, history, elapsed = fsm.run_full_sequence()

    print("\n[Phase 3] Execution Trajectory Trace:")
    for i, state_name in enumerate(history):
        print(f"  Step {i+1:02d}: --> {state_name}")

    if success:
        print("\n" + "=" * 70)
        print("🎉 SUCCESS: Object picked, transferred, and placed with high precision!")
        print(f"   Total Execution Time: {elapsed:.3f} seconds")
        print(f"   Selected Grasp Width: {fsm.context.active_grasp.grasp_width * 1000.0:.1f} mm")
        print(f"   Grasp Quality Score:  {fsm.context.active_grasp.quality_score:.3f} / 1.0")
        print("=" * 70 + "\n")
        return 0
    else:
        print("\n" + "=" * 70)
        print(f"❌ FAILED: Sequence aborted due to: {fsm.context.error_message}")
        print("=" * 70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
