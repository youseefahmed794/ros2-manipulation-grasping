#include <iostream>
#include <iomanip>
#include <cmath>
#include <Eigen/Dense>
#include "manipulation_grasping/kinematics_engine.hpp"

int main()
{
  double tcp_offset_z = 0.14;
  manipulation_grasping::KinematicsEngine engine; // default uses tcp_offset_z = 0.14

  // Choose a tilted pose where the wrist/flange is rotated (not pointing along +Z)
  Eigen::Matrix<double, 6, 1> q;
  q << 0.0, -M_PI / 4.0, M_PI / 3.0, -M_PI / 6.0, M_PI / 2.0, 0.0;

  std::vector<Eigen::Matrix4d> frames;
  Eigen::Matrix4d T_tcp = engine.forward_kinematics(q, &frames);

  // frames contains: T_0, T_1, T_2, T_3, T_4, T_5, T_6 (flange), T_tcp
  // Link 6 frame is index 6
  if (frames.size() < 7) {
    std::cerr << "FAIL: frames vector does not contain link 6" << std::endl;
    return 1;
  }

  Eigen::Matrix4d T_flange = frames[6];
  Eigen::Vector3d p_flange = T_flange.block<3, 1>(0, 3);
  Eigen::Matrix3d R_flange = T_flange.block<3, 3>(0, 0);

  // Hand-computed expected value: p_flange + z_tcp * R_flange * e3
  Eigen::Vector3d e3(0.0, 0.0, 1.0);
  Eigen::Vector3d p_expected = p_flange + tcp_offset_z * (R_flange * e3);
  Eigen::Vector3d p_actual = T_tcp.block<3, 1>(0, 3);

  double error = (p_actual - p_expected).norm();

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "=== Standalone Kinematics TCP Offset Verification ===" << std::endl;
  std::cout << "q: " << q.transpose() << std::endl;
  std::cout << "p_flange:   [" << p_flange.transpose() << "]" << std::endl;
  std::cout << "R_flange*e3:[" << (R_flange * e3).transpose() << "]" << std::endl;
  std::cout << "p_expected: [" << p_expected.transpose() << "]" << std::endl;
  std::cout << "p_actual:   [" << p_actual.transpose() << "]" << std::endl;
  std::cout << "Error norm: " << error << " m" << std::endl;

  if (error > 1e-4) {
    std::cout << "RESULT: FAIL (Discrepancy: " << error << " m)" << std::endl;
    return 1;
  } else {
    std::cout << "RESULT: PASS" << std::endl;
    return 0;
  }
}
