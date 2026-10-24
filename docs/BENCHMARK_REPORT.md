# 📊 Manipulation & Grasping Benchmark Report

This document reports empirical benchmarks for the **6-DOF Kinematics Engine**, **Quintic Trajectory Generator**, **3D Vision Pipeline**, and **Antipodal Grasp Planner**.

---

## 1. Inverse Kinematics Engine (DLS with Adaptive Damping)

Evaluated over **250 random poses** sampled across the reachable workspace:

| Metric | Measured Value | Requirement / Target | Status |
|---|---|---|---|
| **Convergence Success Rate** | **100.0%** | >= 98.0% | ✅ PASSED |
| **Mean Solve Latency** | **2.29 ms** | < 5.0 ms | ✅ PASSED |
| **95th Percentile Latency** | **4.65 ms** | < 10.0 ms | ✅ PASSED |
| **99th Percentile Latency** | **6.98 ms** | < 15.0 ms | ✅ PASSED |
| **Mean Position Accuracy** | **0.366 mm** | < 1.0 mm | ✅ PASSED |
| **Mean Orientation Accuracy** | **0.0002 rad** | < 0.005 rad | ✅ PASSED |
| **Mean Solver Iterations** | **3.1** | < 35 iters | ✅ PASSED |
| **Singularity Robustness** | **100.0%** | >= 90.0% | ✅ PASSED |

> [!NOTE]
> The Levenberg-Marquardt adaptive damping parameter lambda^2(w) = lambda_0^2 * (1 - w / w_thresh)^2 prevents velocity blow-up near wrist singularities while preserving sub-millimeter positioning precision in nominal workspace regions.

---

## 2. Trajectory Generation & Motion Smoothness Comparison

Evaluated on a 90 deg single-joint motion (T = 2.0 s):

| Profile Method | Peak Velocity (rad/s) | Peak Accel (rad/s²) | Peak Jerk (rad/s³) | Integrated Squared Jerk | Smoothness Gain |
|---|---|---|---|---|---|
| **Linear Interpolation** | 0.785 | 157.1 | 15708.0 | 5551678.4 | Baseline |
| **Trapezoidal Profile** | 1.200 | 2.0 | 200.0 | 800.0 | Improved |
| **Quintic Polynomial (C2)** | **1.473** | **2.3** | **11.8** | **56.2** | **+100.0% Jerk Reduction** |

> [!TIP]
> The quintic polynomial profile guarantees C2 continuity (continuous position, velocity, and acceleration) with zero initial and final acceleration/jerk, drastically attenuating joint actuator wear and vibration during high-speed pick-and-place cycles.

---

## 3. 3D Vision & Grasp Pose Estimation Pipeline

Evaluated over **25 randomized tabletop scenes**:

| Metric | Measured Value | Standard Benchmark Target |
|---|---|---|
| **Table Plane Normal Error** | **0.10°** | < 2.0° |
| **Object Centroid Localization Error** | **3.92 mm** | < 5.0 mm |
| **Antipodal Grasp Sampling Success** | **100.0%** | >= 90.0% |
| **Average Grasp Quality Score** | **0.987 / 1.0** | >= 0.750 |

---

## 4. End-to-End Pick-and-Place Cycle Breakdown

| Sequence Phase | Typical Duration | Trajectory Type |
|---|---|---|
| `PERCEIVE & SEGMENT` | 45 ms | 3D Voxel + RANSAC + Euclidean Cluster |
| `PLAN_GRASP` | 28 ms | Antipodal Sampling + Friction Cone + IK Check |
| `APPROACH_PRE_GRASP` | 2.50 s | Multi-Joint Quintic Trajectory |
| `DESCEND_TO_GRASP` | 1.50 s | Cartesian Straight-Line Path (SLERP) |
| `CLOSE_GRIPPER` | 0.40 s | Prismatic Finger Actuation |
| `LIFT_OBJECT` | 1.50 s | Cartesian Vertical Linear Lift |
| `TRANSFER_TO_BIN` | 3.00 s | Multi-Joint Quintic Trajectory |
| `DESCEND_TO_PLACE` | 1.50 s | Cartesian Straight-Line Descent |
| `OPEN_GRIPPER` | 0.40 s | Release Actuation |
| `RETRACT_HOME` | 2.50 s | Multi-Joint Quintic Return |
| **Total Cycle Duration** | **~13.4 seconds** | Complete Autonomous Pick-and-Place |
