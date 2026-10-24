#ifndef MANIPULATION_GRASPING_KINEMATICS_ENGINE_HPP_
#define MANIPULATION_GRASPING_KINEMATICS_ENGINE_HPP_

#include <vector>
#include <array>
#include <string>
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace manipulation_grasping
{

struct DHParam
{
  double a;
  double alpha;
  double d;
  double theta_offset;
};

struct ArmLimits
{
  Eigen::Matrix<double, 6, 1> q_min;
  Eigen::Matrix<double, 6, 1> q_max;
  Eigen::Matrix<double, 6, 1> q_vel_max;
};

struct IKSolution
{
  bool success;
  Eigen::Matrix<double, 6, 1> q;
  int iterations;
  double position_error;
  double orientation_error;
};

class KinematicsEngine
{
public:
  KinematicsEngine();
  explicit KinematicsEngine(const std::vector<DHParam> & dh_table, double tcp_offset_z = 0.14);

  // Forward Kinematics
  Eigen::Matrix4d forward_kinematics(
    const Eigen::Matrix<double, 6, 1> & q,
    std::vector<Eigen::Matrix4d> * all_frames = nullptr) const;

  // Geometric Jacobian: J(q) in R^(6x6)
  Eigen::Matrix<double, 6, 6> compute_jacobian(const Eigen::Matrix<double, 6, 1> & q) const;

  // Yoshikawa Manipulability measure: w(q) = sqrt(det(J * J^T))
  double manipulability(const Eigen::Matrix<double, 6, 1> & q) const;

  // Damped Least Squares Inverse Kinematics (Levenberg-Marquardt with null-space optimization)
  IKSolution inverse_kinematics(
    const Eigen::Vector3d & target_pos,
    const Eigen::Matrix3d & target_rot,
    const Eigen::Matrix<double, 6, 1> & q_init,
    int max_iter = 120,
    double pos_tol = 1e-3,
    double rot_tol = 2e-3,
    double damping_lambda = 0.04,
    bool adaptive_damping = true) const;

  const ArmLimits & get_limits() const { return limits_; }

private:
  std::vector<DHParam> dh_table_;
  double tcp_offset_z_;
  ArmLimits limits_;

  static Eigen::Matrix4d dh_transform(double theta, double d, double a, double alpha);
};

}  // namespace manipulation_grasping

#endif  // MANIPULATION_GRASPING_KINEMATICS_ENGINE_HPP_
