#ifndef _RIGID_BODY_MANAGER_HPP_
#define _RIGID_BODY_MANAGER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "ParticleGeometry.hpp"
#include "Quaternion.hpp"
#include "Vec3.hpp"
#include "geometry.hpp"

class rigid_body_manager {
private:
  std::vector<Vec3> rigid_body_position;
  std::vector<Vec3> rigid_body_orientation;
  std::vector<quaternion> rigid_body_quaternion;
  std::vector<Vec3> rigid_body_velocity;
  std::vector<Vec3> rigid_body_angular_velocity;
  std::vector<Vec3> rigid_body_acceleration;
  std::vector<Vec3> rigid_body_angular_acceleration;
  std::vector<Vec3> rigid_body_force;
  std::vector<Vec3> rigid_body_torque;
  std::vector<Vec3> rigid_body_velocity_force_switch;
  std::vector<Vec3> rigid_body_angvelocity_torque_switch;

  std::vector<int> rigid_body_type;
  std::vector<std::vector<double>> rigid_body_size;

  std::shared_ptr<ParticleGeometry> geometry_mgr;

public:
  rigid_body_manager() {}

  ~rigid_body_manager() {}

  void init(std::string rigid_body_input_file_name, int dim);

  void init_geometry_manager(std::shared_ptr<ParticleGeometry> mgr) {
    geometry_mgr = mgr;
  }

  int get_rigid_body_num() { return rigid_body_position.size(); }

  std::vector<int> &get_rigid_body_type() { return rigid_body_type; }

  int get_rigid_body_type(const int _rigid_body_index) {
    if (_rigid_body_index < rigid_body_position.size()) {
      return rigid_body_type[_rigid_body_index];
    } else {
      return -1;
    }
  }

  std::vector<std::vector<double>> &get_rigid_body_size() {
    return rigid_body_size;
  }

  std::vector<double> &get_rigid_body_size(const int _rigid_body_index) {
    return rigid_body_size[_rigid_body_index];
  }

  std::vector<Vec3> &get_position() { return rigid_body_position; }

  Vec3 get_position(const int _rigid_body_index) {
    if (_rigid_body_index < rigid_body_position.size()) {
      return rigid_body_position[_rigid_body_index];
    } else {
      return Vec3(0.0, 0.0, 0.0);
    }
  }

  std::vector<Vec3> &get_orientation() { return rigid_body_orientation; }

  Vec3 get_orientation(const int _rigid_body_index) {
    if (_rigid_body_index < rigid_body_position.size()) {
      return rigid_body_orientation[_rigid_body_index];
    } else {
      return Vec3(0.0, 0.0, 0.0);
    }
  }

  std::vector<quaternion> &get_quaternion() { return rigid_body_quaternion; }

  quaternion &get_quaternion(const int _rigid_body_index) {
    return rigid_body_quaternion[_rigid_body_index];
  }

  std::vector<Vec3> &get_velocity() { return rigid_body_velocity; }

  std::vector<Vec3> &get_angular_velocity() {
    return rigid_body_angular_velocity;
  }

  std::vector<Vec3> &get_velocity_force_switch() {
    return rigid_body_velocity_force_switch;
  }

  std::vector<Vec3> &get_angvelocity_torque_switch() {
    return rigid_body_angvelocity_torque_switch;
  }

  std::vector<Vec3> &get_force() { return rigid_body_force; }

  std::vector<Vec3> &get_torque() { return rigid_body_torque; }

  void set_position(int _rigid_body_index, Vec3 &_position) {
    if (_rigid_body_index < rigid_body_position.size()) {
      rigid_body_position[_rigid_body_index] = _position;
    }
  }

  void set_orientation(int _rigid_body_index, Vec3 &_orientation) {
    if (_rigid_body_index < rigid_body_position.size()) {
      rigid_body_orientation[_rigid_body_index] = _orientation;
    }
  }

  void set_velocity(int _rigid_body_index, Vec3 &_velocity) {
    if (_rigid_body_index < rigid_body_velocity.size()) {
      rigid_body_velocity[_rigid_body_index] = _velocity;
    }
  }

  void set_angular_velocity(int _rigid_body_index, Vec3 &_angular_velocity) {
    if (_rigid_body_index < rigid_body_angular_velocity.size()) {
      rigid_body_angular_velocity[_rigid_body_index] = _angular_velocity;
    }
  }

  bool rigid_body_collision_detection();
  bool rigid_body_collision_detection(double &min_dis);
};

#endif