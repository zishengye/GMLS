#include <cstdio>
#include <cstdlib>
#include <string>

#include "solver.hpp"

using namespace std;

void solver::attach_parser(std::shared_ptr<parser> par) {
  string output_file_name;
  string time_integration_method;
  string equation_type;
  string scheme_type;

  int dimension;

  int polynomial_order;
  int output_data_level;

  int adaptive_refinement_flag;
  double adaptive_refinement_tolerance;
  int maximum_adaptive_level;

  string rigid_body_input_file_name;
  bool rigid_body_inclusion;

  int nx, ny, nz;

  double final_time;
  double initial_time_step;

  if (par->search("-OutputFile", output_file_name)) {
    if (_net->get_id() == 0) {
      freopen(output_file_name.data(), "w", stdout);
    }
  }

  // default dimension is 3
  if (!par->search("-Dim", dimension)) {
    dimension = 3;
  }

  // default time integration method is forward euler method
  if (!par->search("-TimeIntegration", time_integration_method)) {
    time_integration_method = "ForwardEuler";
  }

  // default governing equation is Stokes equation
  if (!par->search("-EquationType", equation_type)) {
    equation_type = "Stokes";
  }

  // default particle scheme is Eulerian particles
  if (!par->search("-Scheme", scheme_type)) {
    scheme_type = "Eulerian";
  }

  // defalut discretization order is 2
  if (!par->search("-PolynomialOrder", polynomial_order)) {
    polynomial_order = 2;
  }

  // default output level is 0
  if (!par->search("-WriteData", output_data_level)) {
    output_data_level = 0;
  }

  if (!par->search("-ft", final_time)) {
    final_time = 0.0;
  }

  if (!par->search("-dt", initial_time_step)) {
    initial_time_step = 0.1;
  }

  // discretization parameter
  par->search("-Mx", nx);
  par->search("-My", ny);
  par->search("-Mz", nz);

  // [summary of problem setup]
  PetscPrintf(PETSC_COMM_WORLD, "===============================\n");
  PetscPrintf(PETSC_COMM_WORLD, "==== Problem setup summary ====\n");
  PetscPrintf(PETSC_COMM_WORLD, "===============================\n");
  PetscPrintf(PETSC_COMM_WORLD, "==> Dimension: %d\n", dimension);
  PetscPrintf(PETSC_COMM_WORLD, "==> Governing equation: %s\n",
              equation_type.c_str());
  PetscPrintf(PETSC_COMM_WORLD, "==> Time interval: %fs\n", initial_time_step);
  PetscPrintf(PETSC_COMM_WORLD, "==> Final time: %fs\n", final_time);
  PetscPrintf(PETSC_COMM_WORLD, "==> Polynomial order: %d\n", polynomial_order);

  PetscPrintf(PETSC_COMM_WORLD, "==> Particle count in X axis: %d\n", nx);
  if (dimension > 1)
    PetscPrintf(PETSC_COMM_WORLD, "==> Particle count in Y axis: %d\n", ny);
  if (dimension > 2)
    PetscPrintf(PETSC_COMM_WORLD, "==> Particle count in Z axis: %d\n", nz);

  if (adaptive_refinement_flag) {
    PetscPrintf(PETSC_COMM_WORLD, "==> Adaptive refinement: on\n");
    PetscPrintf(PETSC_COMM_WORLD, "==> Adaptive refinement tolerance:  %f\n",
                adaptive_refinement_tolerance);
    PetscPrintf(PETSC_COMM_WORLD, "==> Maximum adaptive refinement level: %d\n",
                maximum_adaptive_level);
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "==> Adaptive refinement: off\n");
  }

  // initialize the solver by the command collected above
  if (equation_type == "Eulerian") {
    _gm.set_discretization_update_scheme(eulerian);
  }

  // use particle discretization by default
  _gm.set_discretization_type(particle_discretization);
  _gm.set_discretization_size(nx, ny, nz);
  _gm.set_output_level(output_data_level);
  _gm.set_dimension(dimension);

  _gm.initialization();
}