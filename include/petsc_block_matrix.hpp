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

#include "petsc_sparse_matrix.hpp"

PetscErrorCode petsc_block_matrix_matmult_wrapper(Mat mat, Vec x, Vec y);

class petsc_block_matrix {
private:
  std::vector<std::shared_ptr<petsc_sparse_matrix>> block_matrix;
  std::vector<PetscInt> block_offset;
  std::vector<petsc_vector> x_list;
  std::vector<petsc_vector> y_list;
  std::vector<petsc_vector> b_list;

  PetscInt Row, Col;

  Mat mat;

  void matmult(Vec &x, Vec &y);

public:
  petsc_block_matrix();

  ~petsc_block_matrix();

  void resize(PetscInt M, PetscInt N);

  std::shared_ptr<petsc_sparse_matrix> get_matrix(PetscInt i, PetscInt j) {
    return block_matrix[i * Col + j];
  }

  void assemble();

  Mat &get_reference() { return mat; }

  friend PetscErrorCode petsc_block_matrix_matmult_wrapper(Mat mat, Vec x,
                                                           Vec y);
};

#endif