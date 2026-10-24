#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"

#include "manipulation_grasping/kinematics_engine.hpp"
#include "manipulation_grasping/trajectory_generator.hpp"

using namespace std::chrono_literals;

namespace manipulation_grasping
{

class ManipulationControllerNode : public rclcpp::Node
{
public:
  ManipulationControllerNode()
  : Node("manipulation_controller_node"),
    kinematics_(),
    trajectory_gen_()
  {
    // Parameters
    this->declare_parameter<std::string>("arm_base_frame", "base_link");
    this->declare_parameter<std::string>("tool_frame", "tool0");
    this->declare_parameter<double>("trajectory_duration", 3.0);

    joint_names_ = {
      "shoulder_pan_joint",
      "shoulder_lift_joint",
      "elbow_joint",
      "wrist_1_joint",
      "wrist_2_joint",
      "wrist_3_joint"
    };

    current_q_ << 0.0, -M_PI_4, M_PI_2, -M_PI_4, M_PI_2, 0.0;

    // Publishers
    traj_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/joint_trajectory_controller/joint_trajectory", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/manipulation/trajectory_markers", 10);
    joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
      "/joint_states", 10);

    // Subscribers
    target_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/target_pose", 10,
      std::bind(&ManipulationControllerNode::target_pose_callback, this, std::placeholders::_1));

    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/arm_joint_states", 10,
      std::bind(&ManipulationControllerNode::joint_state_callback, this, std::placeholders::_1));

    // Periodic heartbeat timer
    timer_ = this->create_wall_timer(
      50ms, std::bind(&ManipulationControllerNode::timer_callback, this));

    RCLCPP_INFO(this->get_logger(), "ManipulationControllerNode initialized and ready.");
  }

private:
  void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    for (size_t i = 0; i < msg->name.size(); ++i) {
      for (size_t j = 0; j < joint_names_.size(); ++j) {
        if (msg->name[i] == joint_names_[j] && i < msg->position.size()) {
          current_q_(static_cast<int>(j)) = msg->position[i];
        }
      }
    }
  }

  void target_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    Eigen::Vector3d target_pos(
      msg->pose.position.x,
      msg->pose.position.y,
      msg->pose.position.z
    );

    Eigen::Quaterniond q_rot(
      msg->pose.orientation.w,
      msg->pose.orientation.x,
      msg->pose.orientation.y,
      msg->pose.orientation.z
    );
    Eigen::Matrix3d target_rot = q_rot.toRotationMatrix();

    RCLCPP_INFO(
      this->get_logger(),
      "Received target pose: pos=[%.3f, %.3f, %.3f]",
      target_pos.x(), target_pos.y(), target_pos.z());

    // Solve Inverse Kinematics
    auto ik_sol = kinematics_.inverse_kinematics(
      target_pos, target_rot, current_q_, 100, 1e-3, 2e-3, 0.04, true);

    if (!ik_sol.success) {
      RCLCPP_WARN(this->get_logger(), "IK failed to converge for target pose!");
      return;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "IK converged in %d iters, err=%.4f mm",
      ik_sol.iterations, ik_sol.position_error * 1000.0);

    // Plan Quintic Trajectory
    double duration = this->get_parameter("trajectory_duration").as_double();
    auto waypoints = trajectory_gen_.generate_quintic_joint_trajectory(
      current_q_, ik_sol.q, duration, 0.02);

    // Publish JointTrajectory
    trajectory_msgs::msg::JointTrajectory traj_msg;
    traj_msg.header.stamp = this->now();
    traj_msg.joint_names = joint_names_;

    for (const auto & wp : waypoints) {
      trajectory_msgs::msg::JointTrajectoryPoint pt;
      pt.time_from_start = rclcpp::Duration::from_seconds(wp.time);
      for (int i = 0; i < 6; ++i) {
        pt.positions.push_back(wp.positions(i));
        pt.velocities.push_back(wp.velocities(i));
        pt.accelerations.push_back(wp.accelerations(i));
      }
      traj_msg.points.push_back(pt);
    }

    traj_pub_->publish(traj_msg);
    publish_trajectory_markers(waypoints);
  }

  void publish_trajectory_markers(const std::vector<TrajectoryWaypoint> & waypoints)
  {
    visualization_msgs::msg::MarkerArray marker_array;

    visualization_msgs::msg::Marker line_strip;
    line_strip.header.frame_id = "base_link";
    line_strip.header.stamp = this->now();
    line_strip.ns = "tcp_trajectory";
    line_strip.id = 0;
    line_strip.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line_strip.action = visualization_msgs::msg::Marker::ADD;
    line_strip.scale.x = 0.008;
    line_strip.color.r = 0.0f;
    line_strip.color.g = 0.8f;
    line_strip.color.b = 1.0f;
    line_strip.color.a = 0.9f;

    for (const auto & wp : waypoints) {
      Eigen::Matrix4d T = kinematics_.forward_kinematics(wp.positions);
      geometry_msgs::msg::Point p;
      p.x = T(0, 3);
      p.y = T(1, 3);
      p.z = T(2, 3);
      line_strip.points.push_back(p);
    }

    marker_array.markers.push_back(line_strip);
    marker_pub_->publish(marker_array);
  }

  void timer_callback()
  {
    sensor_msgs::msg::JointState js_msg;
    js_msg.header.stamp = this->now();
    js_msg.name = joint_names_;
    for (int i = 0; i < 6; ++i) {
      js_msg.position.push_back(current_q_(i));
      js_msg.velocity.push_back(0.0);
      js_msg.effort.push_back(0.0);
    }
    joint_state_pub_->publish(js_msg);
  }

  KinematicsEngine kinematics_;
  TrajectoryGenerator trajectory_gen_;
  Eigen::Matrix<double, 6, 1> current_q_;
  std::vector<std::string> joint_names_;

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr traj_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace manipulation_grasping

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<manipulation_grasping::ManipulationControllerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
