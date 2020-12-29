#ifndef _RIGID_BODY_MANAGER_HPP_
#define _RIGID_BODY_MANAGER_HPP_

#include <string>
#include <vector>

#include "vec3.hpp"

class rigid_body_manager {
private:
  std::vector<vec3> rigid_body_position;
  std::vector<vec3> rigid_body_orientation;
  std::vector<vec3> rigid_body_velocity;
  std::vector<vec3> rigid_body_angular_velocity;
  std::vector<vec3> rigid_body_acceleration;
  std::vector<vec3> rigid_body_angular_acceleration;

  std::vector<int> rigid_body_type;
  std::vector<double> rigid_body_size;

public:
  rigid_body_manager() {}

  ~rigid_body_manager() {}

  void init(std::string rigid_body_input_file_name, int dim);

  const int get_rigid_body_num() { return rigid_body_position.size(); }

  std::vector<int> &get_rigid_body_type() { return rigid_body_type; }

  const int get_rigid_body_type(const int _rigid_body_index) {
    if (_rigid_body_index < rigid_body_position.size()) {
      return rigid_body_type[_rigid_body_index];
    } else {
      return -1;
    }
  }

  std::vector<double> &get_rigid_body_size() { return rigid_body_size; }

  const double get_rigid_body_size(const int _rigid_body_index) {
    if (_rigid_body_index < rigid_body_position.size()) {
      return rigid_body_size[_rigid_body_index];
    } else {
      return 0.0;
    }
  }

  std::vector<vec3> &get_position() { return rigid_body_position; }

  const vec3 &get_position(const int _rigid_body_index) {
    if (_rigid_body_index < rigid_body_position.size()) {
      return rigid_body_position[_rigid_body_index];
    } else {
      return vec3(0.0, 0.0, 0.0);
    }
  }

  std::vector<vec3> &get_orientation() { return rigid_body_orientation; }

  const vec3 &get_orientation(const int _rigid_body_index) {
    if (_rigid_body_index < rigid_body_position.size()) {
      return rigid_body_orientation[_rigid_body_index];
    } else {
      return vec3(0.0, 0.0, 0.0);
    }
  }

  std::vector<vec3> &get_velocity() { return rigid_body_velocity; }

  std::vector<vec3> &get_angular_velocity() {
    return rigid_body_angular_velocity;
  }

  void set_position(int _rigid_body_index, vec3 &_position) {
    if (_rigid_body_index < rigid_body_position.size()) {
      rigid_body_position[_rigid_body_index] = _position;
    }
  }

  void set_orientation(int _rigid_body_index, vec3 &_orientation) {
    if (_rigid_body_index < rigid_body_position.size()) {
      rigid_body_orientation[_rigid_body_index] = _orientation;
    }
  }

  void set_velocity(int _rigid_body_index, vec3 &_velocity) {
    if (_rigid_body_index < rigid_body_velocity.size()) {
      rigid_body_velocity[_rigid_body_index] = _velocity;
    }
  }

  void set_angular_velocity(int _rigid_body_index, vec3 &_angular_velocity) {
    if (_rigid_body_index < rigid_body_angular_velocity.size()) {
      rigid_body_angular_velocity[_rigid_body_index] = _angular_velocity;
    }
  }

  bool rigid_body_collision_detection();
};

#endif