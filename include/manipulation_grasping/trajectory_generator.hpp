#ifndef MANIPULATION_GRASPING_TRAJECTORY_GENERATOR_HPP_
#define MANIPULATION_GRASPING_TRAJECTORY_GENERATOR_HPP_

#include <vector>
#include <Eigen/Dense>

namespace manipulation_grasping
{

struct TrajectoryWaypoint
{
  double time;
  Eigen::Matrix<double, 6, 1> positions;
  Eigen::Matrix<double, 6, 1> velocities;
  Eigen::Matrix<double, 6, 1> accelerations;
};

class QuinticSegment
{
public:
  QuinticSegment(double q0, double q1, double v0 = 0.0, double v1 = 0.0,
                 double a0 = 0.0, double a1 = 0.0, double duration = 1.0);

  void eval(double t, double & pos, double & vel, double & acc, double & jerk) const;

private:
  double T_;
  double a0_, a1_, a2_, a3_, a4_, a5_;
};

class TrajectoryGenerator
{
public:
  TrajectoryGenerator() = default;

  std::vector<TrajectoryWaypoint> generate_quintic_joint_trajectory(
    const Eigen::Matrix<double, 6, 1> & q_start,
    const Eigen::Matrix<double, 6, 1> & q_goal,
    double duration = 3.0,
    double dt = 0.02) const;
};

}  // namespace manipulation_grasping

#endif  // MANIPULATION_GRASPING_TRAJECTORY_GENERATOR_HPP_
