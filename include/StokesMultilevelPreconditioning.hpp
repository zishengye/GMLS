#ifndef _STOKES_MULTILEVEL_HPP_
#define _STOKES_MULTILEVEL_HPP_

#include <memory>
#include <vector>

#include "ParticleGeometry.hpp"
#include "PetscNestedVec.hpp"
#include "PetscWrapper.hpp"
#include "StokesMatrix.hpp"

class StokesMultilevelPreconditioning {
public:
  typedef std::shared_ptr<StokesMatrix> StokesMatrixType;
  typedef std::shared_ptr<PetscNestedMatrix> NestedMatrixType;
  typedef std::shared_ptr<PetscVector> vector_type;
  typedef std::shared_ptr<petsc_is> is_type;
  typedef std::shared_ptr<petsc_ksp> ksp_type;
  typedef std::shared_ptr<petsc_vecscatter> vecscatter_type;

private:
  std::vector<StokesMatrixType> A_list; // coefficient matrix list
  std::vector<NestedMatrixType> I_list; // interpolation matrix list
  std::vector<NestedMatrixType> R_list; // restriction matrix list

  // vector list
  std::vector<std::shared_ptr<PetscNestedVec>> x_list;
  std::vector<std::shared_ptr<PetscNestedVec>> b_list;
  std::vector<std::shared_ptr<PetscNestedVec>> r_list;
  std::vector<std::shared_ptr<PetscNestedVec>> t_list;

  std::vector<vector_type> x_field_list;
  std::vector<vector_type> y_field_list;
  std::vector<vector_type> b_field_list;
  std::vector<vector_type> r_field_list;

  // relaxation list
  std::vector<ksp_type> field_relaxation_list;
  std::vector<ksp_type> colloid_relaxation_list;

  std::vector<int> local_particle_num_list;
  std::vector<int> global_particle_num_list;

  ksp_type ksp_field_base, ksp_colloid_base;

  int mpi_rank, mpi_size;

  int dimension, num_rigid_body;

  bool base_level_initialized;

  int current_refinement_level;

  std::shared_ptr<ParticleGeometry> geo_mgr;

public:
  StokesMultilevelPreconditioning()
      : base_level_initialized(false), current_refinement_level(-1) {}

  ~StokesMultilevelPreconditioning() { clear(); }

  void init(int _dimension, std::shared_ptr<ParticleGeometry> _geo_mgr) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    set_dimension(_dimension);

    geo_mgr = _geo_mgr;
  }

  void reset() { clear(); }

  inline void set_dimension(int _dimension) { dimension = _dimension; }

  inline void set_num_rigid_body(int _num_rigid_body) {
    num_rigid_body = _num_rigid_body;
  }

  inline int get_dimension() { return dimension; }

  inline int get_num_rigid_body() { return num_rigid_body; }

  StokesMatrixType getA(int num_level) { return A_list[num_level]; }
  NestedMatrixType getI(int num_level) { return I_list[num_level]; }
  NestedMatrixType getR(int num_level) { return R_list[num_level]; }
  ksp_type get_field_relaxation(int num_level) {
    return field_relaxation_list[num_level];
  }
  ksp_type get_colloid_relaxation(int num_level) {
    return colloid_relaxation_list[num_level];
  }
  ksp_type get_field_base() { return ksp_field_base; }
  ksp_type get_colloid_base() { return ksp_colloid_base; }

  void add_new_level() {
    if (!base_level_initialized) {
      ksp_field_base = std::make_shared<petsc_ksp>();
      ksp_colloid_base = std::make_shared<petsc_ksp>();
    }

    A_list.push_back(std::make_shared<StokesMatrix>(dimension));
    if (base_level_initialized) {
      I_list.push_back(std::make_shared<PetscNestedMatrix>(2, 2));
      R_list.push_back(std::make_shared<PetscNestedMatrix>(2, 2));
    }

    base_level_initialized = true;
    current_refinement_level++;
  }

  void clear();

  void initial_guess_from_previous_adaptive_step(
      std::vector<double> &initial_guess, std::vector<Vec3> &velocity,
      std::vector<double> &pressure, std::vector<Vec3> &rb_velocity,
      std::vector<Vec3> &rb_angular_velocity);
  void build_interpolation_restriction(const int _num_rigid_body,
                                       const int _dim, const int _poly_order);

  std::vector<NestedMatrixType> &get_interpolation_list() { return I_list; }
  std::vector<NestedMatrixType> &get_restriction_list() { return R_list; }

  std::vector<std::shared_ptr<PetscNestedVec>> &get_x_list() { return x_list; }
  std::vector<std::shared_ptr<PetscNestedVec>> &get_b_list() { return b_list; }
  std::vector<std::shared_ptr<PetscNestedVec>> &get_r_list() { return r_list; }
  std::vector<std::shared_ptr<PetscNestedVec>> &get_t_list() { return t_list; }

  std::vector<vector_type> &get_x_field_list() { return x_field_list; }
  std::vector<vector_type> &get_y_field_list() { return y_field_list; }
  std::vector<vector_type> &get_b_field_list() { return b_field_list; }
  std::vector<vector_type> &get_r_field_list() { return r_field_list; }

  int get_local_particle_num(int level_num) {
    return local_particle_num_list[level_num];
  }

  int get_global_particle_num(int level_num) {
    return global_particle_num_list[level_num];
  }

  int Solve(std::vector<double> &rhs1, std::vector<double> &x1,
            std::vector<double> &rhs2, std::vector<double> &x2);
};

#endif