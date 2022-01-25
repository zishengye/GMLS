#include "petsc_block_matrix.hpp"

using namespace std;

PetscErrorCode petsc_block_matrix_matmult_wrapper(Mat mat, Vec x, Vec y) {
  petsc_block_matrix *ctx;
  MatShellGetContext(mat, &ctx);

  ctx->matmult(x, y);

  return 0;
}

void petsc_block_matrix::matmult(Vec &x, Vec &y) {
  PetscReal *a, *b;

  // move in vecs
  VecGetArray(x, &a);
  for (int i = 0; i < Row; i++) {
    VecGetArray(x_list[i].get_reference(), &b);
    for (int offset = block_offset[i]; offset < block_offset[i + 1]; offset++) {
      b[offset - block_offset[i]] = a[offset];
    }
    VecRestoreArray(x_list[i].get_reference(), &b);
  }
  VecRestoreArray(x, &a);

  for (int i = 0; i < Row; i++) {
    VecZeroEntries(y_list[i].get_reference());
  }

  for (int i = 0; i < Row; i++) {
    for (int j = 0; j < Col; j++) {
      MatMultAdd(block_matrix[i * Col + j]->get_reference(),
                 x_list[j].get_reference(), y_list[i].get_reference(),
                 y_list[i].get_reference());
    }
  }

  VecGetArray(y, &a);
  for (int i = 0; i < Row; i++) {
    VecGetArray(y_list[i].get_reference(), &b);
    for (int offset = block_offset[i]; offset < block_offset[i + 1]; offset++) {
      a[offset] = b[offset - block_offset[i]];
    }
    VecRestoreArray(y_list[i].get_reference(), &b);
  }
  VecRestoreArray(y, &a);
}

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
    local_row += block_matrix[i * Col + i]->get_row();
  }

  PetscInt local_col = 0;
  for (int i = 0; i < Row; i++) {
    local_col += block_matrix[i * Col + i]->get_col();
  }

  MatCreateShell(MPI_COMM_WORLD, local_row, local_col, PETSC_DETERMINE,
                 PETSC_DETERMINE, this, &mat);

  MatShellSetOperation(mat, MATOP_MULT,
                       (void (*)(void))petsc_block_matrix_matmult_wrapper);

  block_offset.resize(Row + 1);
  block_offset[0] = 0;
  for (int i = 0; i < Row; i++) {
    block_offset[i + 1] =
        block_offset[i] + block_matrix[i * Col + i]->get_row();
  }

  x_list.resize(Row);
  y_list.resize(Row);
  b_list.resize(Row);
  for (int i = 0; i < Row; i++) {
    x_list[i].create(block_matrix[i * Col + i]->get_row());
    y_list[i].create(block_matrix[i * Col + i]->get_row());
    b_list[i].create(block_matrix[i * Col + i]->get_row());
  }
}