#include "manipulation_grasping/trajectory_generator.hpp"
#include <cmath>
#include <algorithm>

namespace manipulation_grasping
{

QuinticSegment::QuinticSegment(
  double q0, double q1, double v0, double v1, double a0, double a1, double duration)
: T_(duration)
{
  double T = T_;
  double T2 = T * T;
  double T3 = T2 * T;
  double T4 = T3 * T;
  double T5 = T4 * T;
  double h = q1 - q0;

  a0_ = q0;
  a1_ = v0;
  a2_ = 0.5 * a0;

  a3_ = (20.0 * h - (8.0 * v1 + 12.0 * v0) * T - (3.0 * a0 - a1) * T2) / (2.0 * T3);
  a4_ = (-30.0 * h + (14.0 * v1 + 16.0 * v0) * T + (3.0 * a0 - 2.0 * a1) * T2) / (2.0 * T4);
  a5_ = (12.0 * h - 6.0 * (v1 + v0) * T - (a0 - a1) * T2) / (2.0 * T5);
}

void QuinticSegment::eval(
  double t, double & pos, double & vel, double & acc, double & jerk) const
{
  double tc = std::clamp(t, 0.0, T_);
  double t2 = tc * tc;
  double t3 = t2 * tc;
  double t4 = t3 * tc;
  double t5 = t4 * tc;

  pos = a0_ + a1_ * tc + a2_ * t2 + a3_ * t3 + a4_ * t4 + a5_ * t5;
  vel = a1_ + 2.0 * a2_ * tc + 3.0 * a3_ * t2 + 4.0 * a4_ * t3 + 5.0 * a5_ * t4;
  acc = 2.0 * a2_ + 6.0 * a3_ * tc + 12.0 * a4_ * t2 + 20.0 * a5_ * t3;
  jerk = 6.0 * a3_ + 24.0 * a4_ * tc + 60.0 * a5_ * t2;
}

std::vector<TrajectoryWaypoint> TrajectoryGenerator::generate_quintic_joint_trajectory(
  const Eigen::Matrix<double, 6, 1> & q_start,
  const Eigen::Matrix<double, 6, 1> & q_goal,
  double duration,
  double dt) const
{
  std::vector<QuinticSegment> segments;
  for (int i = 0; i < 6; ++i) {
    segments.emplace_back(q_start(i), q_goal(i), 0.0, 0.0, 0.0, 0.0, duration);
  }

  std::vector<TrajectoryWaypoint> trajectory;
  int num_steps = static_cast<int>(std::ceil(duration / dt)) + 1;

  for (int step = 0; step < num_steps; ++step) {
    double t = step * dt;
    TrajectoryWaypoint wp;
    wp.time = t;

    for (int i = 0; i < 6; ++i) {
      double p, v, a, j;
      segments[i].eval(t, p, v, a, j);
      wp.positions(i) = p;
      wp.velocities(i) = v;
      wp.accelerations(i) = a;
    }
    trajectory.push_back(wp);
  }

  return trajectory;
}

}  // namespace manipulation_grasping
