#include "petsc_block_matrix.hpp"

using namespace std;

PetscErrorCode petsc_block_matrix_matmult_wrapper(Mat mat, Vec x, Vec y) {
  petsc_block_matrix *ctx;
  MatShellGetContext(mat, &ctx);

  ctx->matmult(x, y);

  return 0;
}

PetscErrorCode petsc_mask_matrix_matmult_wrapper(Mat mat, Vec x, Vec y) {
  mask_matrix_wrapper *ctx;
  MatShellGetContext(mat, &ctx);

  ctx->ptr->mask_matmult(ctx->mask_idx, x, y);

  PetscPrintf(PETSC_COMM_WORLD, "mask matmult wrapper\n");
  MPI_Barrier(MPI_COMM_WORLD);

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

void petsc_block_matrix::mask_matmult(int mask_idx, Vec &x, Vec &y) {
  PetscReal *a, *b;

  // move in vecs
  VecGetArray(x, &a);
  for (int i = 0; i < mask_index[mask_idx].size(); i++) {
    VecGetArray(x_list[mask_index[mask_idx][i]].get_reference(), &b);
    for (int offset = mask_block_offset[mask_idx][i];
         offset < mask_block_offset[mask_idx][i + 1]; offset++) {
      b[offset - mask_block_offset[mask_idx][i]] =
          a[block_offset[mask_index[mask_idx][i]] + offset -
            mask_block_offset[mask_idx][i]];
    }
    VecRestoreArray(x_list[mask_index[mask_idx][i]].get_reference(), &b);
  }
  VecRestoreArray(x, &a);

  for (int i = 0; i < mask_index[mask_idx].size(); i++) {
    VecZeroEntries(y_list[mask_index[mask_idx][i]].get_reference());
  }

  for (int i = 0; i < mask_index[mask_idx].size(); i++) {
    for (int j = 0; j < mask_index[mask_idx].size(); j++) {
      MatMultAdd(
          block_matrix[mask_index[mask_idx][i] * Col + mask_index[mask_idx][j]]
              ->get_reference(),
          x_list[mask_index[mask_idx][j]].get_reference(),
          y_list[mask_index[mask_idx][i]].get_reference(),
          y_list[mask_index[mask_idx][i]].get_reference());
    }
  }

  VecGetArray(y, &a);
  for (int i = 0; i < mask_index[mask_idx].size(); i++) {
    VecGetArray(y_list[mask_index[mask_idx][i]].get_reference(), &b);
    for (int offset = mask_block_offset[mask_idx][i];
         offset < mask_block_offset[mask_idx][i + 1]; offset++) {
      a[block_offset[mask_index[mask_idx][i]] + offset -
        mask_block_offset[mask_idx][i]] =
          b[offset - mask_block_offset[mask_idx][i]];
    }
    VecRestoreArray(y_list[mask_index[mask_idx][i]].get_reference(), &b);
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

void petsc_block_matrix::assemble_mask_matrix(std::vector<int> idx) {
  int mask_idx = mask_matrix.size();
  mask_matrix.push_back(petsc_sparse_matrix());
  mask_block_offset.push_back(vector<PetscInt>());
  mask_is.push_back(vector<petsc_is>());

  mask_matrix_wrapper wrapper;
  wrapper.ptr = this;
  wrapper.mask_idx = mask_idx;
  mask_wrapper.push_back(wrapper);
  mask_index.push_back(idx);

  PetscInt local_row = 0;
  for (int i = 0; i < idx.size(); i++) {
    local_row += block_matrix[idx[i] * Col + idx[i]]->get_row();
  }

  PetscInt local_col = 0;
  for (int i = 0; i < idx.size(); i++) {
    local_col += block_matrix[idx[i] * Col + idx[i]]->get_col();
  }

  MatCreateShell(MPI_COMM_WORLD, local_row, local_col, PETSC_DETERMINE,
                 PETSC_DETERMINE, &(mask_wrapper[0]),
                 mask_matrix[mask_idx].get_pointer());

  MatShellSetOperation(mask_matrix[mask_idx].get_reference(), MATOP_MULT,
                       (void (*)(void))petsc_mask_matrix_matmult_wrapper);

  mask_block_offset[mask_idx].resize(idx.size() + 1);
  mask_block_offset[mask_idx][0] = 0;
  for (int i = 0; i < idx.size(); i++) {
    mask_block_offset[mask_idx][i + 1] =
        mask_block_offset[mask_idx][i] +
        block_matrix[idx[i] * Col + idx[i]]->get_row();
  }

  mask_is[mask_idx].resize(idx.size());
  for (int i = 0; i < idx.size(); i++) {
    vector<int> index_set;
    index_set.resize(mask_block_offset[mask_idx][i + 1] -
                     mask_block_offset[mask_idx][i]);
    for (int j = 0; j < index_set.size(); j++)
      index_set[j] = j + mask_block_offset[mask_idx][i];
    mask_is[mask_idx][i].create(index_set);
  }
}