#!/usr/bin/env python3
"""
Manipulation Benchmark CLI Runner:
Runs full IK accuracy/latency, trajectory smoothness, and perception benchmarks,
and outputs a graduate-level benchmark report.
"""

import sys
import os
from manipulation_grasping.benchmarks import ManipulationBenchmarkSuite


def main():
    print("=" * 70)
    print("🚀 ROS 2 Robotic Manipulation & Physical AI Benchmark Suite")
    print("   Author: Youseef Abdel Tawwab")
    print("=" * 70)

    suite = ManipulationBenchmarkSuite()

    print("\n[1/3] Benchmarking 6-DOF Inverse Kinematics Engine (250 poses)...")
    ik_res = suite.run_ik_benchmark(num_samples=250)
    print(f"  ✓ IK Success Rate:     {ik_res.success_rate_percent:.1f}%")
    print(f"  ✓ Mean Latency:         {ik_res.mean_latency_ms:.2f} ms")
    print(f"  ✓ 99th Percentile:      {ik_res.p99_latency_ms:.2f} ms")
    print(f"  ✓ Mean Position Error:  {ik_res.mean_position_error_mm:.4f} mm")
    print(f"  ✓ Singularity Success:  {ik_res.singularity_success_rate:.1f}%")

    print("\n[2/3] Benchmarking Trajectory Profiles (Linear vs Trapezoidal vs Quintic)...")
    traj_res = suite.run_trajectory_benchmark()
    for tr in traj_res:
        print(f"  - {tr.method:32s} | Peak Jerk: {tr.peak_jerk_rad_s3:6.1f} | Int Jerk^2: {tr.integrated_squared_jerk:8.1f}")

    print("\n[3/3] Benchmarking 3D Vision & Grasp Sampling (25 scenes)...")
    perc_res = suite.run_perception_and_grasp_benchmark(num_trials=25)
    print(f"  ✓ Plane Normal Error:   {perc_res.plane_normal_error_deg:.2f}°")
    print(f"  ✓ Centroid Error:       {perc_res.centroid_error_mm:.2f} mm")
    print(f"  ✓ Grasp Success Rate:   {perc_res.grasp_success_rate_percent:.1f}%")
    print(f"  ✓ Mean Grasp Quality:   {perc_res.mean_grasp_quality:.3f}")

    # Generate full Markdown report
    report = suite.generate_markdown_report()
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
    os.makedirs(docs_dir, exist_ok=True)
    report_path = os.path.join(docs_dir, "BENCHMARK_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n✅ Benchmark report saved to: {report_path}\n")


if __name__ == "__main__":
    main()
