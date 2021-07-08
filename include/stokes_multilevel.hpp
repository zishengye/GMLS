#ifndef _STOKES_MULTILEVEL_HPP_
#define _STOKES_MULTILEVEL_HPP_

#include <memory>
#include <vector>

#include "particle_geometry.hpp"
#include "petsc_wrapper.hpp"

class stokes_multilevel {
public:
  typedef std::shared_ptr<petsc_sparse_matrix> matrix_type;
  typedef std::shared_ptr<petsc_block_matrix> block_matrix_type;
  typedef std::shared_ptr<petsc_vector> vector_type;
  typedef std::shared_ptr<petsc_is> is_type;
  typedef std::shared_ptr<petsc_ksp> ksp_type;
  typedef std::shared_ptr<petsc_vecscatter> vecscatter_type;

private:
  std::vector<block_matrix_type> A_list; // coefficient matrix list
  std::vector<matrix_type> I_list;       // interpolation matrix list
  std::vector<matrix_type> R_list;       // restriction matrix list
  std::vector<matrix_type> ff_list;      // field sub-matrix list
  std::vector<matrix_type> nn_list;      // nearfield sub-matrix list
  std::vector<matrix_type> nw_list;      // nearfield-whole sub-matrix list
  std::vector<matrix_type> pp_list;      // pressure sub-matrix list
  std::vector<matrix_type> pw_list;      // pressure-whole sub-matrix list
  std::vector<is_type> isg_field_list;
  std::vector<is_type> isg_colloid_list;
  std::vector<is_type> isg_pressure_list;

  // vector list
  std::vector<vector_type> x_list;
  std::vector<vector_type> y_list;
  std::vector<vector_type> b_list;
  std::vector<vector_type> r_list;
  std::vector<vector_type> t_list;

  std::vector<vector_type> x_field_list;
  std::vector<vector_type> y_field_list;
  std::vector<vector_type> b_field_list;
  std::vector<vector_type> r_field_list;
  std::vector<vector_type> t_field_list;

  std::vector<vector_type> x_colloid_list;
  std::vector<vector_type> y_colloid_list;
  std::vector<vector_type> b_colloid_list;
  std::vector<vector_type> r_colloid_list;
  std::vector<vector_type> t_colloid_list;

  std::vector<vector_type> x_pressure_list;
  std::vector<vector_type> y_pressure_list;

  std::vector<vecscatter_type> field_scatter_list;
  std::vector<vecscatter_type> colloid_scatter_list;
  std::vector<vecscatter_type> pressure_scatter_list;

  vector_type x_colloid, y_colloid;

  // relaxation list
  std::vector<ksp_type> field_relaxation_list;
  std::vector<ksp_type> colloid_relaxation_list;
  std::vector<ksp_type> pressure_relaxation_list;

  std::vector<int> local_particle_num_list;
  std::vector<int> global_particle_num_list;

  ksp_type ksp_field_base, ksp_colloid_base;

  int mpi_rank, mpi_size;

  int dimension, num_rigid_body;

  bool base_level_initialized;

  int current_refinement_level;

  std::shared_ptr<particle_geometry> geo_mgr;

public:
  stokes_multilevel()
      : base_level_initialized(false), current_refinement_level(-1) {}

  ~stokes_multilevel() { clear(); }

  void init(int _dimension, std::shared_ptr<particle_geometry> _geo_mgr) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    set_dimension(_dimension);

    geo_mgr = _geo_mgr;
  }

  void reset() { clear(); }

  inline void set_dimension(int _dimension) { dimension = _dimension; }

  inline void set_num_rigid_body(int _num_rigid_body) {
    num_rigid_body = _num_rigid_body;

    x_colloid = std::make_shared<petsc_vector>();
    y_colloid = std::make_shared<petsc_vector>();
  }

  inline int get_dimension() { return dimension; }

  inline int get_num_rigid_body() { return num_rigid_body; }

  block_matrix_type getA(int num_level) { return A_list[num_level]; }
  matrix_type getI(int num_level) { return I_list[num_level]; }
  matrix_type getR(int num_level) { return R_list[num_level]; }
  ksp_type get_field_relaxation(int num_level) {
    return field_relaxation_list[num_level];
  }
  ksp_type get_colloid_relaxation(int num_level) {
    return colloid_relaxation_list[num_level];
  }
  ksp_type get_pressure_relaxation(int num_level) {
    return pressure_relaxation_list[num_level];
  }
  ksp_type get_field_base() { return ksp_field_base; }
  ksp_type get_colloid_base() { return ksp_colloid_base; }

  matrix_type get_field_mat(int num_level) { return ff_list[num_level]; }
  matrix_type get_colloid_whole_mat(int num_level) {
    return nw_list[num_level];
  }
  matrix_type get_colloid_mat(int num_level) { return nn_list[num_level]; }
  matrix_type get_pressure_whole_mat(int num_level) {
    return pw_list[num_level];
  }

  vector_type get_colloid_x() { return x_colloid; }
  vector_type get_colloid_y() { return y_colloid; }

  void add_new_level() {
    if (!base_level_initialized) {
      ksp_field_base = std::make_shared<petsc_ksp>();
      ksp_colloid_base = std::make_shared<petsc_ksp>();
    }

    A_list.push_back(std::make_shared<petsc_block_matrix>(2, 2));
    if (base_level_initialized) {
      I_list.push_back(std::make_shared<petsc_sparse_matrix>());
      R_list.push_back(std::make_shared<petsc_sparse_matrix>());
    }

    base_level_initialized = true;
    current_refinement_level++;

    isg_field_list.push_back(std::make_shared<petsc_is>());
    isg_colloid_list.push_back(std::make_shared<petsc_is>());
    isg_pressure_list.push_back(std::make_shared<petsc_is>());

    ff_list.push_back(std::make_shared<petsc_sparse_matrix>());
    nn_list.push_back(std::make_shared<petsc_sparse_matrix>());
    nw_list.push_back(std::make_shared<petsc_sparse_matrix>());
    pp_list.push_back(std::make_shared<petsc_sparse_matrix>());
    pw_list.push_back(std::make_shared<petsc_sparse_matrix>());
  }

  void clear();

  void initial_guess_from_previous_adaptive_step(
      std::vector<double> &initial_guess, std::vector<vec3> &velocity,
      std::vector<double> &pressure, std::vector<vec3> &rb_velocity,
      std::vector<vec3> &rb_angular_velocity);
  void build_interpolation_restriction(const int _num_rigid_body,
                                       const int _dim, const int _poly_order);

  std::vector<matrix_type> &get_interpolation_list() { return I_list; }
  std::vector<matrix_type> &get_restriction_list() { return R_list; }

  std::vector<vector_type> &get_x_list() { return x_list; }
  std::vector<vector_type> &get_y_list() { return y_list; }
  std::vector<vector_type> &get_b_list() { return b_list; }
  std::vector<vector_type> &get_r_list() { return r_list; }
  std::vector<vector_type> &get_t_list() { return t_list; }

  std::vector<vector_type> &get_x_field_list() { return x_field_list; }
  std::vector<vector_type> &get_y_field_list() { return y_field_list; }
  std::vector<vector_type> &get_b_field_list() { return b_field_list; }
  std::vector<vector_type> &get_r_field_list() { return r_field_list; }
  std::vector<vector_type> &get_t_field_list() { return t_field_list; }

  std::vector<vector_type> &get_x_colloid_list() { return x_colloid_list; }
  std::vector<vector_type> &get_y_colloid_list() { return y_colloid_list; }
  std::vector<vector_type> &get_b_colloid_list() { return b_colloid_list; }
  std::vector<vector_type> &get_r_colloid_list() { return r_colloid_list; }
  std::vector<vector_type> &get_t_colloid_list() { return t_colloid_list; }

  std::vector<vector_type> &get_x_pressure_list() { return x_pressure_list; }
  std::vector<vector_type> &get_y_pressure_list() { return y_pressure_list; }

  std::vector<vecscatter_type> &get_field_scatter_list() {
    return field_scatter_list;
  }

  std::vector<vecscatter_type> &get_colloid_scatter_list() {
    return colloid_scatter_list;
  }

  std::vector<vecscatter_type> &get_pressure_scatter_list() {
    return pressure_scatter_list;
  }

  int get_local_particle_num(int level_num) {
    return local_particle_num_list[level_num];
  }

  int get_global_particle_num(int level_num) {
    return global_particle_num_list[level_num];
  }

  int solve(std::vector<double> &rhs, std::vector<double> &x,
            std::vector<int> &idx_colloid);
};

#endif