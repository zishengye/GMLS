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

void petsc_block_matrix::assemble() {
  PetscInt local_row = 0;
  for (int i = 0; i < Row; i++) {
    local_row += block_matrix[i * Col]->get_row();
  }

  PetscInt local_col = 0;
  for (int i = 0; i < Row; i++) {
    local_col += block_matrix[i * Col]->get_col();
  }

  PetscInt global_col = 0;
  for (int i = 0; i < Row; i++) {
    global_col += block_matrix[i * Col]->get_Col();
  }

  MatCreateShell(MPI_COMM_WORLD, local_row, local_col, PETSC_DETERMINE,
                 PETSC_DETERMINE, PETSC_NULL, &mat);
}