#include "petsc_block_matrix.hpp"

using namespace std;

petsc_block_matrix::petsc_block_matrix() {}

petsc_block_matrix::~petsc_block_matrix() {}

void petsc_block_matrix::resize(PetscInt M, PetscInt N) {
  Row = M;
  Col = N;
  block_matrix.resize(M * N);

  for (int i = 0; i < Row * Col; i++) {
    block_matrix[i] = make_shared<petsc_sparse_matrix>();
  }
}