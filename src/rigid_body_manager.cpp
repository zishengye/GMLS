#include "rigid_body_manager.hpp"
#include "petsc_wrapper.hpp"

#include <fstream>

using namespace std;

void rigid_body_manager::init(string rigid_body_input_file_name, int dim) {
  ifstream input(rigid_body_input_file_name, ios::in);
  if (!input.is_open()) {
    PetscPrintf(PETSC_COMM_WORLD, "rigid body input file not exist\n");
    return;
  }

  while (!input.eof()) {
    vec3 xyz;
    vec3 rxyz;
    double size;
    int type;
    input >> type;
    input >> size;
    for (int i = 0; i < dim; i++) {
      input >> xyz[i];
    }

    int rotation_dof = (dim == 3) ? 3 : 1;
    for (int i = 0; i < rotation_dof; i++) {
      input >> rxyz[i];
    }

    rigid_body_type.push_back(type);
    rigid_body_size.push_back(size);
    rigid_body_position.push_back(xyz);
    rigid_body_orientation.push_back(rxyz);

    rigid_body_velocity.push_back(vec3(0.0, 0.0, 0.0));
    rigid_body_angular_velocity.push_back(vec3(0.0, 0.0, 0.0));
    rigid_body_acceleration.push_back(vec3(0.0, 0.0, 0.0));
    rigid_body_angular_acceleration.push_back(vec3(0.0, 0.0, 0.0));

    /* rigid body type
    type 1: circle in 2d, sphere in 3d
    type 2: square in 2d, cubic in 3d
    type 3: equilateral triangle in 2d, tetrahedron in 3d
     */
  }
}