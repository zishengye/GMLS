#ifndef _STOKES_EQUATION_HPP_
#define _STOKES_EQUATION_HPP_

#include <memory>

#include <Compadre_Config.h>
#include <Compadre_Evaluator.hpp>
#include <Compadre_GMLS.hpp>
#include <Compadre_PointCloudSearch.hpp>

#include "particle_geometry.hpp"
#include "petsc_wrapper.hpp"
#include "rigid_body_manager.hpp"
#include "stokes_multilevel.hpp"

#define VELOCITY_ERROR_EST 1
#define PRESSURE_ERROR_EST 2

class stokes_equation {
private:
  std::shared_ptr<particle_geometry> geo_mgr;
  std::shared_ptr<rigid_body_manager> rb_mgr;

  std::shared_ptr<stokes_multilevel> multi_mgr;

  void build_coefficient_matrix();
  void build_rhs();
  void solve_step();
  void check_solution();
  void calculate_error();

  std::shared_ptr<Compadre::GMLS> pressure_basis;
  std::shared_ptr<Compadre::GMLS> velocity_basis;
  std::shared_ptr<Compadre::GMLS> pressure_neumann_basis;

  std::vector<double> epsilon;
  std::vector<double> ghost_epsilon;

  std::vector<double> rhs;
  std::vector<double> res;
  std::vector<double> error;
  std::vector<int> idx_colloid;

  std::vector<int> neumann_map;

  std::vector<vec3> velocity;
  std::vector<double> pressure;

  std::vector<int> invert_row_index;

  std::vector<std::vector<double>> gradient;
  int gradient_component_num;

  int poly_order;
  int dim;
  int error_esimation_method;
  double epsilon_multiplier;
  double eta;

  double global_error;

  int rank, size;

  int current_refinement_level;

  bool use_viewer;

public:
  stokes_equation() { use_viewer = false; }

  void init(std::shared_ptr<particle_geometry> _geo_mgr,
            std::shared_ptr<rigid_body_manager> _rb_mgr, const int _poly_order,
            const int _dim,
            const int _error_estimation_method = VELOCITY_ERROR_EST,
            const double _epsilon_multiplier = 0.0, const double _eta = 1.0);
  void reset();
  void update();

  void set_viewer() { use_viewer = true; }

  std::shared_ptr<Compadre::GMLS> get_pressure_basis() {
    return pressure_basis;
  }

  std::shared_ptr<Compadre::GMLS> get_velocity_basis() {
    return velocity_basis;
  }

  std::vector<vec3> &get_velocity() { return velocity; }

  std::vector<double> &get_pressure() { return pressure; }

  std::vector<double> &get_epsilon() { return epsilon; }

  std::vector<std::vector<double>> &get_gradient() { return gradient; }

  std::vector<double> &get_error() { return error; }

  double get_estimated_error() { return global_error; }
};

#endif