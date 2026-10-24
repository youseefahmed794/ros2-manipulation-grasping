# 🦾 ROS 2 Robotic Manipulation & Physical AI Grasping Stack (6-DOF + Gazebo Harmonic)

[![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy_Jalisco-3498db?logo=ros)](https://docs.ros.org/en/jazzy/)
[![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic-ff6f00)](https://gazebosim.org/)
[![C++17](https://img.shields.io/badge/C++-17-00599C?logo=c%2B%2B)](https://en.cppreference.com/w/cpp/17)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-23%20Passed%20(100%25)-brightgreen)](https://github.com/youseefahmed794/ros2-manipulation-grasping)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, high-performance **Robotic Manipulation and Physical AI Grasping Stack** built on **ROS 2 Jazzy** and **Gazebo Harmonic**. Featuring an analytical/numerical **Damped Least Squares (Levenberg-Marquardt) 6-DOF Inverse Kinematics Engine**, $C^2$-continuous **Quintic Polynomial & SLERP Trajectory Generator**, **3D Point Cloud Perception Pipeline** (Voxel filtering, RANSAC planar table segmentation, and Euclidean clustering), an **Antipodal Grasp Pose Planner**, and an **Autonomous Pick-and-Place State Sequencer**.

Developed by **[Youseef Abdel Tawwab](https://github.com/youseefahmed794)** as part of the [Robotics & Embodied AI Graduate Portfolio Roadmap](https://github.com/youseefahmed794).

---

## 📑 Table of Contents

- [Key Highlights](#-key-highlights)
- [System Architecture](#-system-architecture)
- [Theoretical & Mathematical Formulation](#-theoretical--mathematical-formulation)
  - [1. Kinematics & Manipulability](#1-kinematics--manipulability)
  - [2. Damped Least Squares (DLS) Inverse Kinematics](#2-damped-least-squares-dls-inverse-kinematics)
  - [3. $C^2$ Quintic Trajectory Optimization](#3-c2-quintic-trajectory-optimization)
  - [4. 3D Vision & Antipodal Grasp Sampling](#4-3d-vision--antipodal-grasp-sampling)
- [Package Structure](#-package-structure)
- [Installation & Build](#-installation--build)
- [Usage Guide](#-usage-guide)
  - [1. Launch Gazebo Harmonic Simulation](#1-launch-gazebo-harmonic-simulation)
  - [2. Launch RViz 2 Visualization & Telemetry](#2-launch-rviz-2-visualization--telemetry)
  - [3. Run 3D Perception & Segmentation](#3-run-3d-perception--segmentation)
  - [4. Execute Autonomous Pick-and-Place Sequencer](#4-execute-autonomous-pick-and-place-sequencer)
  - [5. Run Empirical Benchmarking Suite](#5-run-empirical-benchmarking-suite)
- [Empirical Benchmark Results](#-empirical-benchmark-results)
- [Automated Testing & CI/CD](#-automated-testing--cicd)
- [Scientific References](#-scientific-references)
- [License & Contact](#-license--contact)

---

## 🌟 Key Highlights

- **C++17 High-Speed Kinematics Core (`Eigen3`)**:
  - Analytical Forward Kinematics utilizing Denavit-Hartenberg (DH) parameters.
  - Geometric Jacobian derivation and Yoshikawa Manipulability Measure tracking.
  - Levenberg-Marquardt adaptive damping DLS IK solver achieving **sub-millimeter precision (0.366 mm)** at **2.29 ms mean latency** with **100% convergence**.
- **Jerk-Minimized Trajectory Generation**:
  - $C^2$-continuous Quintic Polynomial trajectory generator eliminating joint acceleration discontinuities and delivering a **>99.9% reduction in integrated squared jerk**.
  - Spherical Linear Quaternion Interpolation (SLERP) for smooth $SO(3)$ end-effector orientation transitions.
- **3D RGB-D Perception Pipeline**:
  - Point cloud pass-through and 3D voxel grid downsampling.
  - RANSAC-based parametric plane extraction removing tabletop surfaces with **0.10° normal vector accuracy**.
  - Euclidean clustering and Principal Component Analysis (PCA) for 3D Oriented Bounding Box (OBB) and centroid estimation (**3.92 mm error**).
- **Physics-Informed Antipodal Grasp Sampling**:
  - Evaluates antipodal contact pairs, surface normal alignment, approach clearance, and friction cone boundaries to generate optimal 6-DOF grasp candidates.
- **Autonomous Pick-and-Place State Sequencer**:
  - Robust Finite State Machine (FSM) orchestrating perception, pre-grasp approach, Cartesian descent, gripper actuation, linear lift, transfer, place, and home retraction.
- **Gazebo Harmonic Multi-Body Simulation**:
  - Detailed URDF/Xacro models of 6-DOF manipulator, parallel-jaw prismatic gripper, and RealSense RGB-D depth sensor in a realistic tabletop workspace.

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Simulation["Gazebo Harmonic Simulation"]
        GZ_ARM["6-DOF Manipulator & Gripper (URDF)"]
        GZ_RGBD["RGB-D Depth Camera Sensor"]
        GZ_WORLD["Tabletop Workspace SDF"]
    end

    subgraph Bridge["ROS-Gz Transport Bridge"]
        BR_JS["/joint_states"]
        BR_CMD["/arm_controller/joint_trajectory"]
        BR_GRIP["/gripper_controller/command"]
        BR_PC["/camera/depth/points"]
    end

    subgraph Perception["3D Perception Engine"]
        VOXEL["Voxel Grid Filter"]
        RANSAC["RANSAC Plane Segmentation"]
        CLUSTER["Euclidean Cluster Extraction"]
        PCA["PCA & OBB Orientation"]
    end

    subgraph Planning["Planning & Kinematics (C++ / Python)"]
        GRASP["Antipodal Grasp Planner"]
        IK["DLS Adaptive Damping IK (Eigen)"]
        TRAJ["Quintic & SLERP Trajectory Generator"]
    end

    subgraph Executive["Task Sequencer"]
        FSM["Pick-and-Place State Machine"]
    end

    GZ_ARM --> BR_JS
    GZ_RGBD --> BR_PC
    BR_PC --> VOXEL --> RANSAC --> CLUSTER --> PCA
    PCA --> GRASP
    GRASP --> FSM
    FSM --> TRAJ
    TRAJ --> IK
    IK --> BR_CMD --> GZ_ARM
    FSM --> BR_GRIP --> GZ_ARM
```

---

## 🔬 Theoretical & Mathematical Formulation

### 1. Kinematics & Manipulability

Given joint coordinate vector $\boldsymbol{q} = [q_1, q_2, \dots, q_6]^T \in \mathbb{R}^6$, the forward kinematics mapping end-effector pose $\boldsymbol{x} \in SE(3)$ is obtained by the product of successive Denavit-Hartenberg transformation matrices:

$$T_0^6(\boldsymbol{q}) = \prod_{i=1}^6 A_i(q_i) = \begin{bmatrix} R(\boldsymbol{q}) & \boldsymbol{p}(\boldsymbol{q}) \\ \boldsymbol{0}^T & 1 \end{bmatrix}$$

The geometric Jacobian $\boldsymbol{J}(\boldsymbol{q}) \in \mathbb{R}^{6 \times 6}$ relates joint velocities $\dot{\boldsymbol{q}}$ to Cartesian linear and angular velocities $\boldsymbol{v} = [\dot{\boldsymbol{p}}^T, \boldsymbol{\omega}^T]^T$:

$$\boldsymbol{v} = \boldsymbol{J}(\boldsymbol{q}) \dot{\boldsymbol{q}} = \begin{bmatrix} \boldsymbol{z}_{i-1} \times (\boldsymbol{p}_e - \boldsymbol{p}_{i-1}) \\ \boldsymbol{z}_{i-1} \end{bmatrix} \dot{\boldsymbol{q}}$$

Proximity to kinematic singularities is quantified via the **Yoshikawa Manipulability Measure**:

$$w(\boldsymbol{q}) = \sqrt{\det\left(\boldsymbol{J}(\boldsymbol{q}) \boldsymbol{J}^T(\boldsymbol{q})\right)} = \prod_{i=1}^6 \sigma_i$$

where $\sigma_i$ are the singular values of $\boldsymbol{J}(\boldsymbol{q})$.

### 2. Damped Least Squares (DLS) Inverse Kinematics

Near kinematic singularities ($w(\boldsymbol{q}) \to 0$), standard pseudo-inverse solvers $\boldsymbol{J}^\dagger$ suffer from joint velocity blowup. We formulate a Levenberg-Marquardt Damped Least Squares problem:

$$\min_{\Delta \boldsymbol{q}} \left( \|\boldsymbol{e} - \boldsymbol{J} \Delta \boldsymbol{q}\|^2 + \lambda^2 \|\Delta \boldsymbol{q}\|^2 \right)$$

yielding the optimal damped update step:

$$\Delta \boldsymbol{q} = \boldsymbol{J}^T \left( \boldsymbol{J} \boldsymbol{J}^T + \lambda^2 \boldsymbol{I}_6 \right)^{-1} \boldsymbol{e}$$

To maintain sub-millimeter tracking accuracy away from singularities while ensuring robust numerical conditioning inside singular regions, the damping factor $\lambda^2(w)$ is dynamically adapted:

$$\lambda^2(w) = \begin{cases} 0, & w \ge w_{\text{thresh}} \\ \lambda_0^2 \left( 1 - \frac{w}{w_{\text{thresh}}} \right)^2, & w < w_{\text{thresh}} \end{cases}$$

### 3. $C^2$ Quintic Trajectory Optimization

To prevent actuator wear, mechanical vibrations, and gear stress, trajectories are synthesized via 5th-order polynomials $s(\tau) = a_0 + a_1 \tau + a_2 \tau^2 + a_3 \tau^3 + a_4 \tau^4 + a_5 \tau^5$ with normalized time $\tau = t / T \in [0, 1]$ satisfying boundary conditions:

$$\begin{aligned}
s(0) = q_{\text{start}}, \quad \dot{s}(0) = 0, \quad \ddot{s}(0) = 0 \\
s(1) = q_{\text{target}}, \quad \dot{s}(1) = 0, \quad \ddot{s}(1) = 0
\end{aligned}$$

Solving this linear boundary value problem yields:

$$s(\tau) = q_{\text{start}} + (q_{\text{target}} - q_{\text{start}}) \left( 10 \tau^3 - 15 \tau^4 + 6 \tau^5 \right)$$

This profile guarantees continuous jerk $\dddot{s}(\tau)$ and vanishes at both path boundaries, yielding a **100% reduction in step jerk discontinuities** over standard trapezoidal velocity profiles.

### 4. 3D Vision & Antipodal Grasp Sampling

Given scene point cloud $\mathcal{P} \subset \mathbb{R}^3$, the dominant support plane $\boldsymbol{\pi} = [a, b, c, d]^T$ (satisfying $ax + by + cz + d = 0$) is segmented via RANSAC with consensus threshold $\delta_{\text{inlier}} = 10\,\text{mm}$. Segmented object clusters $\mathcal{O}_k$ are processed via Principal Component Analysis (PCA) on covariance matrix $\boldsymbol{\Sigma}_k$:

$$\boldsymbol{\Sigma}_k = \frac{1}{N} \sum_{i=1}^N (\boldsymbol{p}_i - \boldsymbol{\mu}_k)(\boldsymbol{p}_i - \boldsymbol{\mu}_k)^T = \boldsymbol{V} \boldsymbol{\Lambda} \boldsymbol{V}^T$$

Antipodal grasp candidates are sampled across principal surface normals and scored according to:

$$Q_{\text{grasp}} = \alpha \cos(\theta_{\text{normal}}) + \beta \left(1 - \frac{|\boldsymbol{d}_{\text{approach}} \cdot \boldsymbol{z}_{\text{table}}|}{d_{\max}}\right) + \gamma Q_{\text{clearance}}$$

---

## 📦 Package Structure

```
ros2_manipulation_grasping/
├── CMakeLists.txt                       # Build rules (C++17 core + Python + tests)
├── package.xml                          # Package dependencies & metadata
├── LICENSE                              # MIT License
├── README.md                            # Complete technical documentation
├── .github/
│   └── workflows/
│       └── ci.yml                       # GitHub Actions CI (Ubuntu 24.04 + ROS 2 Jazzy)
├── config/
│   ├── manipulation_params.yaml         # Kinematics, trajectory, and grasp parameters
│   └── ros_gz_bridge.yaml               # ROS 2 <-> Gazebo Harmonic bridge mappings
├── docs/
│   └── BENCHMARK_REPORT.md              # Empirical benchmark logs and analysis
├── include/manipulation_grasping/
│   ├── kinematics_engine.hpp            # C++17 Eigen DH & DLS IK header
│   └── trajectory_generator.hpp         # C++17 Quintic & SLERP trajectory header
├── src/
│   ├── kinematics_engine.cpp            # C++ DLS IK implementation
│   ├── trajectory_generator.cpp         # C++ Quintic polynomial trajectory implementation
│   └── manipulation_controller_node.cpp # ROS 2 Lifecycle C++ controller node
├── manipulation_grasping/
│   ├── __init__.py
│   ├── kinematics.py                    # Analytical FK, Jacobian, Yoshikawa & DLS IK
│   ├── trajectory.py                    # Quintic polynomial, trapezoidal, and SLERP
│   ├── perception.py                    # Voxel grid, RANSAC plane, Euclidean clustering
│   ├── grasp_planner.py                 # Antipodal grasp sampler & friction cone scorer
│   ├── fsm.py                           # Pick-and-place finite state machine
│   └── benchmarks.py                    # Quantitative benchmark evaluation engine
├── launch/
│   ├── gazebo_manipulation.launch.py    # Gazebo Harmonic tabletop world launcher
│   ├── rviz_view.launch.py              # RViz 2 visualizer with robot model & TF
│   └── pick_and_place.launch.py         # Complete autonomous execution pipeline
├── scripts/
│   ├── pick_and_place_node.py           # ROS 2 Autonomous task sequencer node
│   ├── perception_node.py               # Point cloud processing & grasp candidate node
│   ├── benchmark_manipulation.py        # Automated benchmark execution script
│   └── run_full_demo.py                 # Interactive demonstration runner
├── test/
│   ├── test_kinematics.py               # FK, Jacobian, DLS convergence & singularity tests
│   ├── test_trajectory.py               # Quintic, trapezoidal, SLERP & jerk tests
│   ├── test_perception.py               # Voxel, RANSAC plane & Euclidean cluster tests
│   ├── test_grasp_planner.py            # Antipodal grasp sampling & friction cone tests
│   ├── test_fsm.py                      # State machine transitions & error recovery tests
│   └── test_benchmarks.py               # Full benchmark execution test
├── urdf/
│   ├── arm_6dof.urdf.xacro              # 6-DOF robotic manipulator kinematics & dynamics
│   ├── gripper.urdf.xacro               # Parallel-jaw prismatic gripper
│   ├── camera_mount.urdf.xacro          # RGB-D depth camera optical frame definition
│   └── manipulation_scene.urdf.xacro    # Integrated system URDF
├── worlds/
│   └── tabletop_manipulation.sdf        # Gazebo Harmonic world (table, bins, target objects)
└── rviz/
    └── manipulation.rviz                # Preconfigured RViz 2 display configuration
```

---

## 🛠️ Installation & Build

### Prerequisites

- **OS**: Ubuntu 24.04 LTS (Noble Numbat)
- **ROS 2**: Jazzy Jalisco (`ros-jazzy-desktop` or `ros-jazzy-ros-base`)
- **Simulator**: Gazebo Harmonic (`ros_gz_sim`, `ros_gz_bridge`)
- **Compilers & Dependencies**: `g++` (C++17), `libeigen3-dev`, Python 3.12, NumPy, SciPy

### Setup & Compilation

```bash
# 1. Source ROS 2 Jazzy environment
source /opt/ros/jazzy/setup.bash

# 2. Clone repository into your ROS 2 workspace
cd ~/ros2_ws/src
git clone https://github.com/youseefahmed794/ros2-manipulation-grasping.git manipulation_grasping

# 3. Install required ROS 2 dependencies
sudo apt-get update && sudo apt-get install -y \
  ros-jazzy-ros-gz-sim \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-joint-state-publisher \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-rviz2 \
  ros-jazzy-xacro \
  libeigen3-dev \
  python3-scipy \
  python3-pytest

# 4. Build workspace with symlink install
cd ~/ros2_ws
colcon build --packages-select manipulation_grasping --symlink-install

# 5. Source local workspace
source install/setup.bash
```

---

## 🚀 Usage Guide

### 1. Launch Gazebo Harmonic Simulation

Launches the high-fidelity physics environment with the 6-DOF arm, tabletop workspace, target objects, and depth sensor:

```bash
ros2 launch manipulation_grasping gazebo_manipulation.launch.py
```

### 2. Launch RViz 2 Visualization & Telemetry

Visualizes robot joint states, end-effector frames, 3D point clouds, segmented bounding boxes, and grasp candidates:

```bash
ros2 launch manipulation_grasping rviz_view.launch.py
```

### 3. Run 3D Perception & Segmentation

Executes real-time point cloud processing, tabletop planar removal, and object pose estimation:

```bash
ros2 run manipulation_grasping perception_node.py
```

### 4. Execute Autonomous Pick-and-Place Sequencer

Runs the complete closed-loop pick-and-place FSM:

```bash
ros2 launch manipulation_grasping pick_and_place.launch.py
```

### 5. Run Empirical Benchmarking Suite

Executes quantitative evaluations of the kinematics engine, trajectory smoothness, and vision pipeline:

```bash
ros2 run manipulation_grasping benchmark_manipulation.py
```

---

## 📊 Empirical Benchmark Results

Evaluated on **Ubuntu 24.04 (Linux 6.8.0, x86_64)** across 250 test poses and 25 randomized tabletop perception scenes:

### 1. 6-DOF DLS Inverse Kinematics Engine

| Metric | Measured Value | Standard Target | Status |
|---|---|---|---|
| **Convergence Success Rate** | **100.0%** | $\ge 98.0\%$ | ✅ PASSED |
| **Mean Solve Latency** | **2.29 ms** | $< 5.0\text{ ms}$ | ✅ PASSED |
| **95th Percentile Latency** | **4.65 ms** | $< 10.0\text{ ms}$ | ✅ PASSED |
| **99th Percentile Latency** | **6.98 ms** | $< 15.0\text{ ms}$ | ✅ PASSED |
| **Mean Position Accuracy** | **0.366 mm** | $< 1.0\text{ mm}$ | ✅ PASSED |
| **Mean Orientation Accuracy** | **0.0002 rad** | $< 0.005\text{ rad}$ | ✅ PASSED |
| **Mean Solver Iterations** | **3.1 iters** | $< 35\text{ iters}$ | ✅ PASSED |
| **Singularity Robustness** | **100.0%** | $\ge 90.0\%$ | ✅ PASSED |

### 2. Trajectory Profile Smoothness & Jerk Comparison

Comparison over a 90° single-joint transition ($T = 2.0\,\text{s}$):

| Profile Method | Peak Velocity ($\text{rad/s}$) | Peak Accel ($\text{rad/s}^2$) | Peak Jerk ($\text{rad/s}^3$) | Integrated $\int \dddot{q}^2 dt$ | Jerk Attenuation |
|---|---|---|---|---|---|
| **Linear Interpolation** | 0.785 | 157.1 | 15708.0 | 5,551,678.4 | Baseline |
| **Trapezoidal Profile** | 1.200 | 2.0 | 200.0 | 800.0 | Improved |
| **Quintic Polynomial ($C^2$)** | **1.473** | **2.3** | **11.8** | **56.2** | **+99.99% Jerk Reduction** |

### 3. 3D Vision & Antipodal Grasp Pipeline

| Metric | Measured Value | Benchmark Target | Status |
|---|---|---|---|
| **Table Plane Normal Error** | **0.10°** | $< 2.0^\circ$ | ✅ PASSED |
| **Object Centroid Localization Error** | **3.92 mm** | $< 5.0\text{ mm}$ | ✅ PASSED |
| **Antipodal Grasp Sampling Success** | **100.0%** | $\ge 90.0\%$ | ✅ PASSED |
| **Average Grasp Quality Score** | **0.987 / 1.0** | $\ge 0.750$ | ✅ PASSED |

---

## 🧪 Automated Testing & CI/CD

The package contains an automated unit and integration test suite with **23 passed tests (100% pass rate)**:

```bash
# Execute unit test suite with pytest
PYTHONPATH=. pytest test/ -v

# Run colcon integration test harness
colcon test --packages-select manipulation_grasping --event-handlers console_direct+
colcon test-result --verbose
```

### GitHub Actions CI Workflow

Every commit and pull request is automatically validated inside a containerized `ros:jazzy-ros-base` environment across:
1. System dependency resolution and Eigen3 integration.
2. Full pytest automated test suite.
3. CMake C++17 library and executable compilation.
4. Colcon test execution and test result logging.

---

## 📚 Scientific References

1. **Buss, S. R.** (2004). *Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares methods*. IEEE Journal of Robotics and Automation.
2. **Yoshikawa, T.** (1985). *Manipulability of robotic mechanisms*. The International Journal of Robotics Research, 4(2), 3-9.
3. **Craig, J. J.** (2005). *Introduction to Robotics: Mechanics and Control*. Pearson Education India.
4. **Sundermeyer, M., Mousavian, A., Triebel, R., & Fox, D.** (2021). *Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes*. IEEE International Conference on Robotics and Automation (ICRA).
5. **Rusu, R. B., & Cousins, S.** (2011). *3D is here: Point Cloud Library (PCL)*. IEEE International Conference on Robotics and Automation (ICRA).
6. **Garrett, C. R., Chitnis, R., Holladay, R., Kim, B., Silver, T., Kaelbling, L. P., & Lozano-Pérez, T.** (2021). *Integrated task and motion planning*. Annual Review of Control, Robotics, and Autonomous Systems, 4, 265-293.

---

## 📄 License & Contact

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

**Author**: Youseef Abdel Tawwab  
**Email**: [youseefahmed794@gmail.com](mailto:youseefahmed794@gmail.com)  
**GitHub**: [@youseefahmed794](https://github.com/youseefahmed794)  
**Portfolio**: [Robotics Portfolio Roadmap](https://github.com/youseefahmed794)
