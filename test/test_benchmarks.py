import pytest
from manipulation_grasping.benchmarks import ManipulationBenchmarkSuite


def test_benchmark_suite_execution():
    suite = ManipulationBenchmarkSuite()

    # Fast IK benchmark
    ik_res = suite.run_ik_benchmark(num_samples=40)
    assert ik_res.success_rate_percent >= 90.0
    assert ik_res.mean_latency_ms < 50.0  # ms
    assert ik_res.mean_position_error_mm < 2.0  # mm

    # Trajectory comparison benchmark
    traj_res = suite.run_trajectory_benchmark()
    assert len(traj_res) == 3

    linear_res = [r for r in traj_res if "Linear" in r.method][0]
    quintic_res = [r for r in traj_res if "Quintic" in r.method][0]

    # Quintic must be smoother than linear (lower integrated squared jerk)
    assert quintic_res.integrated_squared_jerk < linear_res.integrated_squared_jerk

    # Perception and grasp benchmark
    perc_res = suite.run_perception_and_grasp_benchmark(num_trials=8)
    assert perc_res.plane_normal_error_deg < 5.0
    assert perc_res.grasp_success_rate_percent >= 80.0

    # Markdown report string generation
    report = suite.generate_markdown_report()
    assert "Manipulation & Grasping Benchmark Report" in report
    assert "Inverse Kinematics Engine" in report
    assert "Quintic Polynomial" in report
