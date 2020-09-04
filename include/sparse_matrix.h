#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

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

#include <petscksp.h>

inline bool compare_index(std::pair<int, double> i, std::pair<int, double> j) {
  return (i.first < j.first);
}

struct fluid_colloid_matrix_context {
  Mat fluid_part;
  Mat colloid_part;
  VecScatter fluid_part_vec;
  Vec fluid_vec;
  Vec colloid_vec;

  PetscInt fluid_local_size;
  PetscInt rigid_body_size;

  int myid, mpisize;
};

PetscErrorCode fluid_colloid_matrix_mult(Mat mat, Vec x, Vec y);

class PetscSparseMatrix {
private:
  fluid_colloid_matrix_context __ctx;

  bool __isAssembled;

  typedef std::pair<PetscInt, double> entry;
  std::vector<std::vector<entry>> __matrix;
  std::vector<std::vector<entry>> __out_process_matrix;

  PetscInt __row, __col, __nnz, __Col, __out_process_row,
      __out_process_reduction;

  std::vector<PetscInt> __i;
  std::vector<PetscInt> __j;
  std::vector<PetscReal> __val;

public:
  Mat __mat, __shell_mat;

  PetscSparseMatrix()
      : __isAssembled(false), __row(0), __col(0), __Col(0),
        __out_process_row(0), __out_process_reduction(0) {}

  // only for square matrix
  PetscSparseMatrix(PetscInt m /* local # of rows */,
                    PetscInt N /* global # of cols */,
                    PetscInt out_process_row = 0,
                    PetscInt out_process_row_reduction = 0)
      : __isAssembled(false), __row(m), __col(m), __Col(N),
        __out_process_row(out_process_row),
        __out_process_reduction(out_process_row_reduction) {
    __matrix.resize(m);
    __out_process_matrix.resize(out_process_row);
  }

  PetscSparseMatrix(PetscInt m /* local # of rows */,
                    PetscInt n /* local # of cols */,
                    PetscInt N /* global # of cols */,
                    PetscInt out_process_row = 0,
                    PetscInt out_process_row_reduction = 0)
      : __isAssembled(false), __row(m), __col(n), __Col(N),
        __out_process_row(out_process_row),
        __out_process_reduction(out_process_row_reduction) {
    __matrix.resize(m);
    __out_process_matrix.resize(out_process_row);
  }

  ~PetscSparseMatrix() {
    if (__isAssembled)
      MatDestroy(&__mat);
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

  inline void setColIndex(const PetscInt row, std::vector<PetscInt> &index);
  inline void setOutProcessColIndex(const PetscInt row,
                                    std::vector<PetscInt> &index);
  inline void increment(const PetscInt i, const PetscInt j, double daij);
  inline void outProcessIncrement(const PetscInt i, const PetscInt j,
                                  double daij);

  int Write(std::string filename);

  int StructureAssemble();
  int FinalAssemble();
  int FinalAssemble(int blockSize);
  int FinalAssemble(int blockSize, int num_rigid_body, int rigid_body_dof);
  int FinalAssemble(Mat &mat, int blockSize, int num_rigid_body,
                    int rigid_body_dof);

  int ExtractNeighborIndex(std::vector<int> &idx_neighbor, int dimension,
                           int num_rigid_body, int local_rigid_body_offset,
                           int global_rigid_body_offset);

  // (*this) * x = rhs
  void Solve(std::vector<double> &rhs,
             std::vector<double> &x); // simple solver
  void Solve(std::vector<double> &rhs, std::vector<double> &x,
             PetscInt blockSize); // simple solver
  void Solve(std::vector<double> &rhs, std::vector<double> &x, int dimension,
             int numRigidBody);
  void Solve(std::vector<double> &rhs, std::vector<double> &x,
             std::vector<int> &idx_neighbor, int dimension, int numRigidBody,
             int adaptive_step, PetscSparseMatrix &I, PetscSparseMatrix &R);
  // two field solver with rigid body inclusion

  // [A Bt; B C] * [x; y] = [f; g]
  friend void Solve(PetscSparseMatrix &A, PetscSparseMatrix &Bt,
                    PetscSparseMatrix &B, PetscSparseMatrix &C,
                    std::vector<double> &f, std::vector<double> &g,
                    std::vector<double> &x, std::vector<double> &y,
                    int numRigidBody, int rigidBodyDof);
};

void PetscSparseMatrix::setColIndex(const PetscInt row,
                                    std::vector<PetscInt> &index) {
  sort(index.begin(), index.end());
  __matrix[row].resize(index.size());
  size_t counter = 0;
  for (std::vector<entry>::iterator it = __matrix[row].begin();
       it != __matrix[row].end(); it++) {
    it->first = index[counter++];
    it->second = 0.0;
  }
}

void PetscSparseMatrix::setOutProcessColIndex(const PetscInt row,
                                              std::vector<PetscInt> &index) {
  sort(index.begin(), index.end());
  __out_process_matrix[row - __out_process_reduction].resize(index.size());
  size_t counter = 0;
  for (std::vector<entry>::iterator it =
           __out_process_matrix[row - __out_process_reduction].begin();
       it != __out_process_matrix[row - __out_process_reduction].end(); it++) {
    it->first = index[counter++];
    it->second = 0.0;
  }
}

void PetscSparseMatrix::increment(const PetscInt i, const PetscInt j,
                                  const double daij) {
  if (std::abs(daij) > 1e-15) {
    auto it = lower_bound(__matrix[i].begin(), __matrix[i].end(),
                          entry(j, daij), compare_index);
    if (it->first == j)
      it->second += daij;
    else
      std::cout << i << ' ' << j << " increment misplacement" << std::endl;
  }
}

void PetscSparseMatrix::outProcessIncrement(const PetscInt i, const PetscInt j,
                                            const double daij) {
  if (std::abs(daij) > 1e-15) {
    PetscInt in = i - __out_process_reduction;
    auto it = lower_bound(__out_process_matrix[in].begin(),
                          __out_process_matrix[in].end(), entry(j, daij),
                          compare_index);
    if (it != __out_process_matrix[in].end() && it->first == j)
      it->second += daij;
    else
      std::cout << in << ' ' << j << " out process increament misplacement"
                << std::endl;
  }
}

#endif