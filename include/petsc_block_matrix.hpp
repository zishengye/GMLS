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

class petsc_block_matrix {
private:
  std::vector<std::shared_ptr<petsc_sparse_matrix>> block_matrix;

  PetscInt Row, Col;

public:
  petsc_block_matrix();

  ~petsc_block_matrix();

  void resize(PetscInt M, PetscInt N);

  std::shared_ptr<petsc_sparse_matrix> get_matrix(PetscInt i, PetscInt j) {
    return block_matrix[i * Col + j];
  }
};

#endif