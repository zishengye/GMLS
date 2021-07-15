#ifndef _GMLS_SOLVER_HPP_
#define _GMLS_SOLVER_HPP_

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Compadre_Config.h>
#include <Compadre_Evaluator.hpp>
#include <Compadre_GMLS.hpp>
#include <Compadre_PointCloudSearch.hpp>

#include <petscksp.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>

#include "info.hpp"
#include "particle_geometry.hpp"
#include "petsc_sparse_matrix.hpp"
#include "search_command.hpp"
#include "stokes_multilevel.hpp"
#include "vec3.hpp"

#include "stokes_equation.hpp"

template <typename T>
int SearchCommand(int argc, char **argv, const std::string &commandName,
                  T &res);

class gmls_solver {
private:
  // MPI setting
  int rank;
  int size;

  int use_viewer;

  // solver control parameter
  std::string equation_type;
  std::string time_integration_method;
  std::string scheme_type;
  int polynomial_order;
  int weight_func_order;
  int write_data;
  int batch_size;
  int refinement_method;
  int refinement_field;
  int max_refinement_level;
  double refinement_tolerance;
  int current_refinement_step;
  std::string adaptive_base_field;
  double max_particle_num;

  bool initialization_status;

  int dim;

  double final_time;
  double max_dt;

  int current_time_integration_step;
  double current_simulation_time;
  double current_time_period;

  double epsilon_multiplier;
  double spacing;

  int coordinate_system;
  // 1 cartesian coordinate system
  // 2 cylindrical coordinate system
  // 3 spherical coordinate system

  // solver physics parameter
  double eta;

  // rigid body info
  std::string rigid_body_input_file_name;
  std::string trajectory_output_file_name;
  std::string velocity_output_file_name;
  std::string force_output_file_name;
  bool rigid_body_inclusion;

  std::shared_ptr<stokes_equation> equation_mgr;

  // time integration scheme
  void foward_euler_integration();
  void adaptive_runge_kutta_intagration();
  void implicit_midpoint_integration();

  double t, dt, dtMin, rtol, atol, err, norm_y;

  std::vector<vec3> position0;
  std::vector<vec3> orientation0;
  std::vector<quaternion> quaternion0;

  // operator
  template <typename Func> void serial_operation(Func operation) {
    for (int i = 0; i < size; i++) {
      if (i == rank) {
        operation();
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  template <typename Func> void master_operation(int master, Func operation) {
    if (master == rank) {
      operation();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  void write_time_step_data();
  void write_refinement_data(std::vector<int> &split_tag,
                             std::vector<double> &h_gradient);
  void write_refinement_data_geometry_only();
  void write_geometry_ghost();

  // geometry manager
  std::shared_ptr<particle_geometry> geo_mgr;
  std::shared_ptr<rigid_body_manager> rb_mgr;

  bool refinement();

public:
  gmls_solver(int argc, char **argv);

  void time_integration();

  bool is_initialized() { return initialization_status; }

  void implicit_midpoint_integration_sub(Vec, Vec);
};

#endif