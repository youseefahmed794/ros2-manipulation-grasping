#include "manipulation_grasping/kinematics_engine.hpp"
#include <cmath>
#include <algorithm>

namespace manipulation_grasping
{

KinematicsEngine::KinematicsEngine()
: tcp_offset_z_(0.14)
{
  // Standard 6-DOF DH Parameters
  dh_table_ = {
    {0.0, M_PI_2, 0.1625, 0.0},
    {-0.425, 0.0, 0.0, 0.0},
    {-0.3922, 0.0, 0.0, 0.0},
    {0.0, M_PI_2, 0.1333, 0.0},
    {0.0, -M_PI_2, 0.0997, 0.0},
    {0.0, 0.0, 0.0996, 0.0}
  };

  limits_.q_min = Eigen::Matrix<double, 6, 1>::Constant(-M_PI);
  limits_.q_max = Eigen::Matrix<double, 6, 1>::Constant(M_PI);
  limits_.q_vel_max = Eigen::Matrix<double, 6, 1>::Constant(3.14);
}

KinematicsEngine::KinematicsEngine(const std::vector<DHParam> & dh_table, double tcp_offset_z)
: dh_table_(dh_table), tcp_offset_z_(tcp_offset_z)
{
  limits_.q_min = Eigen::Matrix<double, 6, 1>::Constant(-M_PI);
  limits_.q_max = Eigen::Matrix<double, 6, 1>::Constant(M_PI);
  limits_.q_vel_max = Eigen::Matrix<double, 6, 1>::Constant(3.14);
}

Eigen::Matrix4d KinematicsEngine::dh_transform(double theta, double d, double a, double alpha)
{
  double ct = std::cos(theta);
  double st = std::sin(theta);
  double ca = std::cos(alpha);
  double sa = std::sin(alpha);

  Eigen::Matrix4d T;
  T << ct, -st * ca,  st * sa, a * ct,
       st,  ct * ca, -ct * sa, a * st,
      0.0,       sa,       ca,      d,
      0.0,      0.0,      0.0,    1.0;
  return T;
}

Eigen::Matrix4d KinematicsEngine::forward_kinematics(
  const Eigen::Matrix<double, 6, 1> & q,
  std::vector<Eigen::Matrix4d> * all_frames) const
{
  Eigen::Matrix4d T_curr = Eigen::Matrix4d::Identity();
  if (all_frames) {
    all_frames->clear();
    all_frames->push_back(T_curr);
  }

  for (size_t i = 0; i < dh_table_.size(); ++i) {
    double theta = q(static_cast<int>(i)) + dh_table_[i].theta_offset;
    Eigen::Matrix4d T_i = dh_transform(
      theta, dh_table_[i].d, dh_table_[i].a, dh_table_[i].alpha);
    T_curr = T_curr * T_i;
    if (all_frames) {
      all_frames->push_back(T_curr);
    }
  }

  Eigen::Matrix4d T_tcp = T_curr;
  T_tcp(2, 3) += tcp_offset_z_;
  if (all_frames) {
    all_frames->push_back(T_tcp);
  }

  return T_tcp;
}

Eigen::Matrix<double, 6, 6> KinematicsEngine::compute_jacobian(
  const Eigen::Matrix<double, 6, 1> & q) const
{
  std::vector<Eigen::Matrix4d> frames;
  Eigen::Matrix4d T_tcp = forward_kinematics(q, &frames);
  Eigen::Vector3d p_tcp = T_tcp.block<3, 1>(0, 3);

  Eigen::Matrix<double, 6, 6> J = Eigen::Matrix<double, 6, 6>::Zero();

  for (int i = 0; i < 6; ++i) {
    Eigen::Vector3d z_i = frames[i].block<3, 1>(0, 2);
    Eigen::Vector3d p_i = frames[i].block<3, 1>(0, 3);

    J.block<3, 1>(0, i) = z_i.cross(p_tcp - p_i);
    J.block<3, 1>(3, i) = z_i;
  }

  return J;
}

double KinematicsEngine::manipulability(const Eigen::Matrix<double, 6, 1> & q) const
{
  Eigen::Matrix<double, 6, 6> J = compute_jacobian(q);
  double det = (J * J.transpose()).determinant();
  return (det > 0.0) ? std::sqrt(det) : 0.0;
}

IKSolution KinematicsEngine::inverse_kinematics(
  const Eigen::Vector3d & target_pos,
  const Eigen::Matrix3d & target_rot,
  const Eigen::Matrix<double, 6, 1> & q_init,
  int max_iter,
  double pos_tol,
  double rot_tol,
  double damping_lambda,
  bool adaptive_damping) const
{
  Eigen::Matrix<double, 6, 1> q = q_init;
  IKSolution sol;
  sol.success = false;

  for (int it = 0; it < max_iter; ++it) {
    Eigen::Matrix4d T_curr = forward_kinematics(q);
    Eigen::Vector3d p_curr = T_curr.block<3, 1>(0, 3);
    Eigen::Matrix3d R_curr = T_curr.block<3, 3>(0, 0);

    Eigen::Vector3d pos_err = target_pos - p_curr;

    // Orientation error: 0.5 * (n_c x n_d + s_c x s_d + a_c x a_d)
    Eigen::Vector3d rot_err = 0.5 * (
      R_curr.col(0).cross(target_rot.col(0)) +
      R_curr.col(1).cross(target_rot.col(1)) +
      R_curr.col(2).cross(target_rot.col(2))
    );

    double p_norm = pos_err.norm();
    double r_norm = rot_err.norm();

    if (p_norm < pos_tol && r_norm < rot_tol) {
      sol.success = true;
      sol.q = q;
      sol.iterations = it;
      sol.position_error = p_norm;
      sol.orientation_error = r_norm;
      return sol;
    }

    Eigen::Matrix<double, 6, 1> err;
    err << pos_err, rot_err;

    Eigen::Matrix<double, 6, 6> J = compute_jacobian(q);
    double w = manipulability(q);

    double lam2 = damping_lambda * damping_lambda;
    if (adaptive_damping) {
      double w_thresh = 0.03;
      if (w < w_thresh) {
        double ratio = 1.0 - (w / w_thresh);
        lam2 = (damping_lambda * damping_lambda) * (ratio * ratio);
      } else {
        lam2 = 1e-6;
      }
    }

    Eigen::Matrix<double, 6, 6> dls = (J * J.transpose()) + lam2 * Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::Matrix<double, 6, 1> dq = J.transpose() * dls.ldlt().solve(err);

    // Limit maximum step
    double max_step = 0.25;
    double max_val = dq.cwiseAbs().maxCoeff();
    if (max_val > max_step) {
      dq *= (max_step / max_val);
    }

    q += dq;

    // Clamp limits
    for (int i = 0; i < 6; ++i) {
      q(i) = std::clamp(q(i), limits_.q_min(i), limits_.q_max(i));
    }
  }

  Eigen::Matrix4d T_final = forward_kinematics(q);
  sol.position_error = (target_pos - T_final.block<3, 1>(0, 3)).norm();
  sol.orientation_error = 0.0;
  sol.success = (sol.position_error < pos_tol * 2.0);
  sol.q = q;
  sol.iterations = max_iter;
  return sol;
}

}  // namespace manipulation_grasping
