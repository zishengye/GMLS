#include "rigid_body_manager.hpp"
#include "PetscWrapper.hpp"

#include <fstream>

using namespace std;

void rigid_body_manager::init(string rigid_body_input_file_name, int dim) {
  ifstream input(rigid_body_input_file_name, ios::in);
  if (!input.is_open()) {
    PetscPrintf(PETSC_COMM_WORLD, "rigid body input file does not exist\n");
    return;
  }

  while (!input.eof()) {
    Vec3 xyz;
    Vec3 rxyz;
    double size;
    int type;
    type = -1;
    input >> type;
    input >> size;
    vector<double> size_list;
    size_list.push_back(size);
    int end_of_input = 0;
    switch (type) {
    case 2:
      input >> size;
      size_list.push_back(size);
      break;
    case 4:
      // y
      input >> size;
      size_list.push_back(size);
      input >> size;
      size_list.push_back(size);
      break;
    case 5:
      // r2
      input >> size;
      size_list.push_back(size);
      // d
      input >> size;
      size_list.push_back(size);
      break;
    case -1:
      end_of_input = 1;
      break;
    }
    if (end_of_input == 1)
      break;
    for (int i = 0; i < dim; i++) {
      input >> xyz[i];
    }

    int rotation_dof = (dim == 3) ? 3 : 1;
    for (int i = 0; i < rotation_dof; i++) {
      input >> rxyz[i];
    }

    Vec3 vf_switch;
    Vec3 rt_switch;

    Vec3 vxyz;
    Vec3 fxyz;
    Vec3 rvxyz;
    Vec3 txyz;

    for (int i = 0; i < dim; i++) {
      input >> vf_switch[i];
      if (vf_switch[i] == 1) {
        input >> vxyz[i];
        fxyz[i] = 0.0;
      } else {
        input >> fxyz[i];
        vxyz[i] = 0.0;
      }
    }
    for (int i = 0; i < rotation_dof; i++) {
      input >> rt_switch[i];
      if (rt_switch[i] == 1) {
        input >> rvxyz[i];
        txyz[i] = 0.0;
      } else {
        input >> txyz[i];
        rvxyz[i] = 0.0;
      }
    }

    rigid_body_type.push_back(type);
    rigid_body_size.push_back(size_list);
    rigid_body_position.push_back(xyz);
    rigid_body_orientation.push_back(rxyz);
    if (dim == 3)
      rigid_body_quaternion.push_back(quaternion(rxyz[0], rxyz[1], rxyz[2]));
    else
      rigid_body_quaternion.push_back(quaternion(Vec3(0.0, 0.0, 1.0), rxyz[0]));

    rigid_body_velocity_force_switch.push_back(vf_switch);
    rigid_body_angvelocity_torque_switch.push_back(rt_switch);
    rigid_body_velocity.push_back(vxyz);
    rigid_body_angular_velocity.push_back(rvxyz);
    rigid_body_acceleration.push_back(Vec3(0.0, 0.0, 0.0));
    rigid_body_angular_acceleration.push_back(Vec3(0.0, 0.0, 0.0));
    rigid_body_force.push_back(fxyz);
    rigid_body_torque.push_back(txyz);

    /* rigid body type
    type 1: circle in 2d, sphere in 3d
    type 2: square in 2d, cubic in 3d
    type 3: equilateral triangle in 2d, tetrahedron in 3d
     */
  }

  input.close();
}

bool rigid_body_manager::rigid_body_collision_detection() {
  double min_dis;
  return rigid_body_collision_detection(min_dis);
}

bool rigid_body_manager::rigid_body_collision_detection(double &min_dis) {
  min_dis = 1.0;

  // distance between colloids
  for (int i = 0; i < rigid_body_position.size(); i++) {
    for (int j = i + 1; j < rigid_body_position.size(); j++) {
      if (rigid_body_type[i] == 1 && rigid_body_type[j] == 1) {
        Vec3 dist = rigid_body_position[i] - rigid_body_position[j];
        if (min_dis >
            dist.mag() - rigid_body_size[i][0] - rigid_body_size[j][0]) {
          min_dis = dist.mag() - rigid_body_size[i][0] - rigid_body_size[j][0];
        }
      }
      if (rigid_body_type[i] == 2 && rigid_body_type[j] == 2) {
        Vec3 dist = rigid_body_position[i] - rigid_body_position[j];

        double theta1 = rigid_body_orientation[i][0];
        double theta2 = rigid_body_orientation[j][0];

        if (dist.mag() - rigid_body_size[i][0] - rigid_body_size[j][0] < 0.0) {
          double half_length = rigid_body_size[i][0];
          Vec3 x11 = Vec3(-half_length, -half_length, 0.0);
          Vec3 x12 = Vec3(-half_length, half_length, 0.0);
          Vec3 x21 = Vec3(half_length, -half_length, 0.0);
          Vec3 x22 = Vec3(half_length, half_length, 0.0);
        }
      }
    }
  }

  // distance to the boundary
  for (int i = 0; i < rigid_body_position.size(); i++) {
    if (rigid_body_type[i] == 1) {
    }
    if (rigid_body_type[i] == 2) {
    }
  }

  return (min_dis < 0.0);
}