#ifndef _PETSC_BLOCK_MATRIX_HPP_
#define _PETSC_BLOCK_MATRIX_HPP_

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "petsc_index_set.hpp"
#include "petsc_sparse_matrix.hpp"

class petsc_block_matrix;

struct mask_matrix_wrapper {
  petsc_block_matrix *ptr;
  int mask_idx;
};

PetscErrorCode petsc_block_matrix_matmult_wrapper(Mat mat, Vec x, Vec y);

PetscErrorCode petsc_mask_matrix_matmult_wrapper(Mat mat, Vec x, Vec y);

class petsc_block_matrix {
private:
  std::vector<std::shared_ptr<petsc_sparse_matrix>> block_matrix;
  std::vector<petsc_sparse_matrix> mask_matrix;
  std::vector<mask_matrix_wrapper> mask_wrapper;
  std::vector<PetscInt> block_offset;
  std::vector<std::vector<int>> mask_index;
  std::vector<std::vector<PetscInt>> mask_block_offset;
  std::vector<std::vector<petsc_is>> mask_is;
  std::vector<petsc_vector> x_list;
  std::vector<petsc_vector> y_list;
  std::vector<petsc_vector> b_list;

  PetscInt Row, Col;

  Mat mat;

  void matmult(Vec &x, Vec &y);
  void mask_matmult(int mask_idx, Vec &x, Vec &y);

public:
  petsc_block_matrix();

  ~petsc_block_matrix();

  void resize(PetscInt M, PetscInt N);

  std::shared_ptr<petsc_sparse_matrix> get_matrix(PetscInt i, PetscInt j) {
    return block_matrix[i * Col + j];
  }

  void assemble();
  void assemble_mask_matrix(std::vector<int> idx);

  Mat &get_reference() { return mat; }

  Mat &get_mask_matrix(int idx) { return mask_matrix[idx].get_reference(); }

  IS &get_mask_matrix_sub_is(int idx1, int idx2) {
    return mask_is[idx1][idx2].get_reference();
  }

  friend PetscErrorCode petsc_block_matrix_matmult_wrapper(Mat mat, Vec x,
                                                           Vec y);
  friend PetscErrorCode petsc_mask_matrix_matmult_wrapper(Mat mat, Vec x,
                                                          Vec y);
  friend struct mask_matrix_wrapper;
};

#endif