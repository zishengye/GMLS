#ifndef _PETSC_SPARSE_MATRIX_HPP_
#define _PETSC_SPARSE_MATRIX_HPP_

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <utility>
#include <vector>

#include "petsc_vector.hpp"

inline bool compare_index(std::pair<int, double> i, std::pair<int, double> j) {
  return (i.first < j.first);
}

struct fluid_colloid_matrix_context {
  bool use_raw_fluid_part = false;
  Mat *fluid_part;
  Mat fluid_raw_part;
  Mat colloid_part;
  Mat fluid_colloid_part;
  Vec fluid_vec1;
  Vec fluid_vec2;
  Vec fluid_vec_local;
  Vec colloid_vec;
  Vec colloid_vec_local;
  bool use_local_vec = true;

  bool use_vec_scatter = false;
  VecScatter fluid_scatter;
  VecScatter colloid_scatter;

  PetscInt fluid_local_size;
  PetscInt rigid_body_size;

  PetscInt local_fluid_particle_num;
  PetscInt global_fluid_particle_num;
  PetscInt field_dof;
  PetscInt pressure_offset;

  std::vector<PetscInt> fluid_colloid_part_i;
  std::vector<PetscInt> fluid_colloid_part_j;
  std::vector<PetscReal> fluid_colloid_part_val;

  int myid, mpisize;

  double matmult_duration = 0.0;
};

PetscErrorCode fluid_colloid_matrix_mult(Mat mat, Vec x, Vec y);
PetscErrorCode fluid_colloid_matrix_mult2(Mat mat, Vec x, Vec y);
PetscErrorCode fluid_matrix_mult(Mat mat, Vec x, Vec y);

class petsc_sparse_matrix {
public:
  fluid_colloid_matrix_context __ctx;

private:
  bool is_assembled, is_shell_assembled, is_ctx_assembled;
  bool is_transpose_assemble;

  typedef std::pair<PetscInt, double> entry;
  std::vector<std::vector<entry>> __matrix;
  std::vector<std::vector<entry>> __out_process_matrix;

  PetscInt __row, __col, __nnz, __Col, __out_process_row,
      __out_process_reduction;

  std::vector<PetscInt> __i;
  std::vector<PetscInt> __j;
  std::vector<PetscReal> __val;

  Mat __mat, __shell_mat;
  MatNullSpace null_space;

public:
  petsc_sparse_matrix()
      : is_ctx_assembled(false), is_shell_assembled(false), is_assembled(false),
        __row(0), __col(0), __Col(0), __out_process_row(0),
        __out_process_reduction(0), __mat(PETSC_NULL), __shell_mat(PETSC_NULL),
        null_space(PETSC_NULL) {}

  // only for square matrix
  petsc_sparse_matrix(PetscInt m /* local # of rows */,
                      PetscInt N /* global # of cols */,
                      PetscInt out_process_row = 0,
                      PetscInt out_process_row_reduction = 0)
      : is_ctx_assembled(false), is_shell_assembled(false), is_assembled(false),
        __row(m), __col(m), __Col(N), __out_process_row(out_process_row),
        __out_process_reduction(out_process_row_reduction) {
    __matrix.resize(m);
    __out_process_matrix.resize(out_process_row);
  }

  petsc_sparse_matrix(PetscInt m /* local # of rows */,
                      PetscInt n /* local # of cols */,
                      PetscInt N /* global # of cols */,
                      PetscInt out_process_row = 0,
                      PetscInt out_process_row_reduction = 0)
      : is_ctx_assembled(false), is_shell_assembled(false), is_assembled(false),
        __row(m), __col(n), __Col(N), __out_process_row(out_process_row),
        __out_process_reduction(out_process_row_reduction) {
    __matrix.resize(m);
    __out_process_matrix.resize(out_process_row);
  }

  ~petsc_sparse_matrix() {
    if (null_space != PETSC_NULL) {
      MatSetNullSpace(__mat, PETSC_NULL);
      MatNullSpaceDestroy(&null_space);
    }
    if (is_assembled || __mat != PETSC_NULL) {
      MatSetNearNullSpace(__mat, NULL);
      MatDestroy(&__mat);
    }
    if (is_shell_assembled || __shell_mat != PETSC_NULL) {
      MatSetNearNullSpace(__shell_mat, NULL);
      MatDestroy(&__shell_mat);
    }
    if (is_ctx_assembled) {
      if (__ctx.use_raw_fluid_part)
        MatDestroy(&__ctx.fluid_raw_part);
      MatDestroy(&__ctx.colloid_part);
      MatDestroy(&__ctx.fluid_colloid_part);
      VecDestroy(&__ctx.colloid_vec);
      VecDestroy(&__ctx.fluid_vec1);
      VecDestroy(&__ctx.fluid_vec2);
      if (__ctx.use_local_vec) {
        VecDestroy(&__ctx.fluid_vec_local);
        VecDestroy(&__ctx.colloid_vec_local);
      }
      if (__ctx.use_vec_scatter) {
        VecScatterDestroy(&__ctx.fluid_scatter);
        VecScatterDestroy(&__ctx.colloid_scatter);
      }
    }
  }

  void resize(PetscInt m, PetscInt n) {
    __row = m;
    __col = n;
    __Col = 0;
    __matrix.resize(m);

    __out_process_row = 0;
    __out_process_reduction = 0;
  }

  void resize(PetscInt m, PetscInt n, PetscInt N, PetscInt out_process_row = 0,
              PetscInt out_process_row_reduction = 0) {
    __row = m;
    __col = n;
    __Col = N;
    __matrix.resize(m);

    __out_process_row = out_process_row;
    __out_process_reduction = out_process_row_reduction;
    __out_process_matrix.resize(out_process_row);
  }

  const PetscInt get_row() { return __row; }

  const PetscInt get_col() { return __col; }

  const PetscInt get_Col() { return __Col; }

  Mat &get_reference() { return __mat; }

  Mat *get_pointer() { return &__mat; }

  Mat &get_shell_reference() { return __shell_mat; }

  Mat *get_shell_pointer() { return &__shell_mat; }

  Mat &get_operator_reference() {
    if (is_shell_assembled)
      return __shell_mat;
    else
      return __mat;
  }

  Mat *get_operator_pointer() {
    if (is_shell_assembled)
      return &__shell_mat;
    else
      return &__mat;
  }

  void set_null_space() {
    MatNullSpaceCreate(MPI_COMM_WORLD, PETSC_TRUE, 0, PETSC_NULL, &null_space);
    MatSetNullSpace(__mat, null_space);
    MatSetTransposeNullSpace(__mat, null_space);
  }

  inline void set_col_index(const PetscInt row, std::vector<PetscInt> &index);
  inline void set_out_process_col_index(const PetscInt row,
                                        std::vector<PetscInt> &index);
  inline void zero_row(const PetscInt i);
  inline void increment(const PetscInt i, const PetscInt j, double daij);
  inline void set(const PetscInt i, const PetscInt j, double daij);
  inline void out_process_increment(const PetscInt i, const PetscInt j,
                                    double daij);

  inline double get_entity(const PetscInt i, const PetscInt j);

  inline void invert_row(const PetscInt i);

  int write(std::string filename);

  int assemble(bool reserve_data = false);
  int assemble(int blockSize, bool reserve_data = false);
  int assemble(int blockSize, int num_rigid_body, int rigid_body_dof);
  int assemble(petsc_sparse_matrix &mat, int blockSize, int num_rigid_body,
               int rigid_body_dof);
  int out_process_assemble(bool reserve_data = false);

  int extract_neighbor_index(std::vector<int> &idx_colloid, int dimension,
                             int num_rigid_body, int field_dof);

  // (*this) * x = rhs
  void solve(std::vector<double> &rhs,
             std::vector<double> &x); // simple solver
  void solve(std::vector<double> &rhs, std::vector<double> &x,
             PetscInt blockSize); // simple solver
  void solve(std::vector<double> &rhs, std::vector<double> &x, int dimension,
             int numRigidBody);
  void solve(std::vector<double> &rhs, std::vector<double> &x,
             std::vector<int> &idx_colloid, int dimension, int numRigidBody,
             int adaptive_step, petsc_sparse_matrix &I, petsc_sparse_matrix &R);
  // two field solver with rigid body inclusion

  // [A Bt; B C] * [x; y] = [f; g]
  friend void solve(petsc_sparse_matrix &A, petsc_sparse_matrix &Bt,
                    petsc_sparse_matrix &B, petsc_sparse_matrix &C,
                    std::vector<double> &f, std::vector<double> &g,
                    std::vector<double> &x, std::vector<double> &y,
                    int numRigidBody, int rigidBodyDof);
};

void petsc_sparse_matrix::set_col_index(const PetscInt row,
                                        std::vector<PetscInt> &index) {
  sort(index.begin(), index.end());
  __matrix[row].resize(index.size());
  size_t counter = 0;
  for (std::vector<entry>::iterator it = __matrix[row].begin();
       it != __matrix[row].end(); it++) {
    if (index[counter] > __Col) {
      std::cout << row << ' ' << index[counter]
                << " index setting with wrong column index" << std::endl;
      counter++;
      continue;
    }

    it->first = index[counter++];
    it->second = 0.0;
  }
}

void petsc_sparse_matrix::set_out_process_col_index(
    const PetscInt row, std::vector<PetscInt> &index) {
  sort(index.begin(), index.end());
  __out_process_matrix[row - __out_process_reduction].resize(index.size());
  size_t counter = 0;
  for (std::vector<entry>::iterator it =
           __out_process_matrix[row - __out_process_reduction].begin();
       it != __out_process_matrix[row - __out_process_reduction].end(); it++) {
    if (index[counter] > __Col) {
      std::cout << row << ' ' << index[counter]
                << " out process index setting with wrong column index"
                << std::endl;
      counter++;
      continue;
    }

    it->first = index[counter++];
    it->second = 0.0;
  }
}

void petsc_sparse_matrix::zero_row(const PetscInt i) {
  for (auto it = __matrix[i].begin(); it != __matrix[i].end(); it++) {
    it->second = 0.0;
  }
}

void petsc_sparse_matrix::increment(const PetscInt i, const PetscInt j,
                                    const double daij) {
  if (std::abs(daij) > 1e-15) {
    auto it = lower_bound(__matrix[i].begin(), __matrix[i].end(),
                          entry(j, daij), compare_index);
    if (j > __Col) {
      std::cout << i << ' ' << j << " increment wrong column index"
                << std::endl;
      return;
    }

    if (it->first == j)
      it->second += daij;
    else
      std::cout << i << ' ' << j << " increment misplacement" << std::endl;
  }
}

void petsc_sparse_matrix::set(const PetscInt i, const PetscInt j,
                              const double daij) {
  if (std::abs(daij) > 1e-15) {
    auto it = lower_bound(__matrix[i].begin(), __matrix[i].end(),
                          entry(j, daij), compare_index);
    if (j > __Col) {
      std::cout << i << ' ' << j << " increment wrong column index"
                << std::endl;
      return;
    }

    if (it->first == j)
      it->second = daij;
    else
      std::cout << i << ' ' << j << " increment misplacement" << std::endl;
  }
}

void petsc_sparse_matrix::out_process_increment(const PetscInt i,
                                                const PetscInt j,
                                                const double daij) {
  if (std::abs(daij) > 1e-15) {
    PetscInt in = i - __out_process_reduction;
    auto it = lower_bound(__out_process_matrix[in].begin(),
                          __out_process_matrix[in].end(), entry(j, daij),
                          compare_index);
    if (j > __Col) {
      std::cout << i << ' ' << j << " out process wrong column index"
                << std::endl;
      return;
    }

    if (it != __out_process_matrix[in].end() && it->first == j)
      it->second += daij;
    else
      std::cout << in << ' ' << j << " out process increament misplacement"
                << std::endl;
  }
}

double petsc_sparse_matrix::get_entity(const PetscInt i, const PetscInt j) {
  auto it = lower_bound(__matrix[i].begin(), __matrix[i].end(), entry(j, 0.0),
                        compare_index);
  if (j > __Col) {
    std::cout << i << ' ' << j << " wrong matrix index access" << std::endl;
    return 0.0;
  }

  if (it->first == j)
    return it->second;
  else
    return 0.0;
}

void petsc_sparse_matrix::invert_row(const PetscInt i) {
  for (auto it = __matrix[i].begin(); it != __matrix[i].end(); it++) {
    it->second = -it->second;
  }
}

#endif