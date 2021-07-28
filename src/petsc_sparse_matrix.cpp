#include "stokes_composite_preconditioner.hpp"

#include "petsc_ksp.hpp"
#include "petsc_sparse_matrix.hpp"

#include <fstream>
#include <string>

using namespace std;

PetscErrorCode MultTranspose(Mat mat, Vec x, Vec y);

PetscErrorCode MultTransposeAdd(Mat mat, Vec x, Vec y, Vec z);

int petsc_sparse_matrix::write(string fileName) {
  ofstream output;

  if (rank == 0) {
    output.open(fileName, ios::trunc);
    output.close();
  }

  for (int process = 0; process < size; process++) {
    if (process == rank) {
      output.open(fileName, ios::app);
      for (int i = 0; i < mat_i.size(); i++) {
        for (int j = mat_i[i]; j < mat_i[i + 1]; j++) {
          output << (i + 1) + range_col1 << '\t' << mat_j[j] << '\t' << mat_a[j]
                 << endl;
        }
        for (int j = mat_oi[i]; j < mat_oi[i + 1]; j++) {
          output << (i + 1) + range_col1 << '\t' << mat_oj[j] << '\t'
                 << mat_oa[j] << endl;
        }
      }
      output.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  return 0;
}

void petsc_sparse_matrix::set_col_index(const PetscInt i,
                                        vector<PetscInt> &idx) {
  sort(idx.begin(), idx.end());

  if (mat_i[i + 1] - mat_i[i] > 0)
    cout << i << " index has been set" << endl;

  for (auto it = idx.begin(); it != idx.end(); it++) {
    if (*it > Col) {
      cout << i << ' ' << *it << " index setting with wrong column index"
           << endl;

      return;
    }
  }

  vector<PetscInt> index, o_index;
  if (!is_transpose)
    for (auto it = idx.begin(); it != idx.end(); it++) {
      if (*it >= range_row1 && *it < range_row2) {
        index.push_back(*it);
      } else {
        o_index.push_back(*it);
      }
    }
  else
    index = idx;

  if (!is_transpose) {
    for (int it = i; it < row; it++) {
      mat_i[it + 1] += index.size();
    }
    for (int it = i; it < row; it++) {
      mat_oi[it + 1] += o_index.size();
    }
  } else {
    for (int it = i; it < Row; it++) {
      mat_i[it + 1] += index.size();
    }
  }

  if (!is_transpose) {
    mat_j.insert(mat_j.begin() + mat_i[i], index.begin(), index.end());
    mat_oj.insert(mat_oj.begin() + mat_oi[i], o_index.begin(), o_index.end());
  } else {
    mat_j.insert(mat_j.begin() + mat_i[i], index.begin(), index.end());
  }
}

void petsc_sparse_matrix::set_block_col_index(const PetscInt i,
                                              vector<PetscInt> &idx) {
  sort(idx.begin(), idx.end());

  if (mat_i[i + 1] - mat_i[i] > 0)
    cout << i << " block index has been set" << endl;

  for (int it = i; it < block_row; it++) {
    mat_i[it + 1] += idx.size();
  }

  for (auto it = idx.begin(); it != idx.end(); it++) {
    if (*it > block_Col) {
      cout << i << ' ' << *it << " block index setting with wrong column index"
           << endl;

      return;
    }
  }

  mat_j.insert(mat_j.begin() + mat_i[i], idx.begin(), idx.end());
}

void petsc_sparse_matrix::increment(const PetscInt i, const PetscInt j,
                                    const double daij) {
  if (j > Col) {
    cout << rank << ' ' << i << ' ' << j << " increment wrong column index"
         << endl;
    return;
  }
  if (!is_transpose) {
    if (block_size == 1) {
      if (j >= range_row1 && j < range_row2) {
        auto it = lower_bound(mat_j.begin() + mat_i[i],
                              mat_j.begin() + mat_i[i + 1], j);

        if (*it == j) {
          int offset = it - mat_j.begin();
          if (offset > mat_a.size())
            cout << " diagonal exceed value array size" << endl;
          else
            mat_a[it - mat_j.begin()] += daij;
        } else
          cout << rank << ' ' << i << ' ' << j
               << " diagonal increment misplacement" << endl;
      } else {
        auto it = lower_bound(mat_oj.begin() + mat_oi[i],
                              mat_oj.begin() + mat_oi[i + 1], j);

        if (*it == j)
          mat_oa[it - mat_oj.begin()] += daij;
        else
          cout << rank << ' ' << i << ' ' << j
               << " off-diagonal increment misplacement" << endl;
      }
    } else {
      int block_i = i / block_size;
      int block_j = j / block_size;
      int block_row_idx = i % block_size;
      int block_col_idx = j % block_size;

      auto it = lower_bound(mat_j.begin() + mat_i[block_i],
                            mat_j.begin() + mat_i[block_i + 1], block_j);

      if (*it == block_j)
        mat_a[(it - mat_j.begin()) * block_size * block_size +
              block_row_idx * block_size + block_col_idx] += daij;
      else
        cout << rank << ' ' << block_i << ' ' << block_j << ' '
             << " block single increment misplacement" << endl;
    }
  } else {
    auto it =
        lower_bound(mat_j.begin() + mat_i[i], mat_j.begin() + mat_i[i + 1], j);

    if (*it == j)
      mat_a[it - mat_j.begin()] += daij;
    else
      cout << rank << ' ' << i << ' ' << j
           << " transpose increment misplacement" << endl;
  }
}

double petsc_sparse_matrix::get_entity(const PetscInt i, const PetscInt j) {
  if (j > Col) {
    cout << i << ' ' << j << " wrong matrix index access" << endl;
    return 0.0;
  }

  if (block_size == 1) {
    if (j >= range_col1 && j < range_col2) {
      auto it = lower_bound(mat_j.begin() + mat_i[i],
                            mat_j.begin() + mat_i[i + 1], j);

      if (*it == j)
        return mat_a[it - mat_j.begin()];
      else
        return 0.0;
    } else {
      auto it = lower_bound(mat_oj.begin() + mat_oi[i],
                            mat_oj.begin() + mat_oi[i + 1], j);

      if (*it == j)
        return mat_oa[it - mat_oj.begin()];
      else
        return 0.0;
    }
  } else {
    int block_i = i / block_size;
    int block_row_idx = i % block_size;
    int block_j = j / block_size;
    int block_col_idx = j % block_size;
    auto it = lower_bound(mat_j.begin() + mat_i[block_i],
                          mat_j.begin() + mat_i[block_i + 1], block_j);

    if (*it == block_j)
      return mat_a[(it - mat_j.begin()) * block_size * block_size +
                   block_row_idx * block_size + block_col_idx];
    else
      return 0.0;
  }

  return 0.0;
}

int petsc_sparse_matrix::graph_assemble() {
  int nnz, o_nnz;

  if (block_size == 1) {
    if (!is_transpose) {
      nnz = mat_i[row];
      o_nnz = mat_oi[row];

      mat_a.resize(nnz);
      mat_oa.resize(o_nnz);
      fill(mat_a.begin(), mat_a.end(), 0.0);
      fill(mat_oa.begin(), mat_oa.end(), 0.0);
    } else {
      nnz = mat_i[Row];
      o_nnz = 0;

      mat_a.resize(nnz);
      fill(mat_a.begin(), mat_a.end(), 0.0);
    }
  } else {
    nnz = mat_i[block_row] * block_size * block_size;
    o_nnz = 0;

    mat_a.resize(nnz);
    fill(mat_a.begin(), mat_a.end(), 0.0);
  }

  return nnz + o_nnz;
}

int petsc_sparse_matrix::assemble() {
  if (block_size == 1) {
    MatCreateMPIAIJWithSplitArrays(PETSC_COMM_WORLD, row, col, PETSC_DECIDE,
                                   Col, mat_i.data(), mat_j.data(),
                                   mat_a.data(), mat_oi.data(), mat_oj.data(),
                                   mat_oa.data(), mat);

    is_assembled = true;
  } else {
    MatCreate(MPI_COMM_WORLD, mat);
    MatSetSizes(*mat, row, row, PETSC_DECIDE, Col);
    MatSetType(*mat, MATMPIBAIJ);
    MatSetBlockSize(*mat, block_size);
    MatSetUp(*mat);
    MatMPIBAIJSetPreallocationCSR(*mat, block_size, mat_i.data(), mat_j.data(),
                                  mat_a.data());
    is_assembled = true;
    {
      decltype(mat_i)().swap(mat_i);
      decltype(mat_j)().swap(mat_j);
      decltype(mat_a)().swap(mat_a);
    }
  }

  return 0;
}

int petsc_sparse_matrix::transpose_assemble() {
  int send_count = col;
  std::vector<int> recv_count;
  recv_count.resize(size);

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Allgather(&send_count, 1, MPI_INT, recv_count.data(), 1, MPI_INT,
                MPI_COMM_WORLD);

  std::vector<int> col_range;
  std::vector<int> displs;
  displs.resize(size + 1);
  col_range.resize(size + 1);
  col_range[0] = 0;
  for (int i = 1; i <= size; i++) {
    col_range[i] = col_range[i - 1] + recv_count[i - 1];
  }

  // move matrix
  vector<PetscInt> sendmat_i, sendmat_j;
  vector<PetscReal> sendmat_a;
  vector<PetscInt> recvmat_i, recvmat_j;
  vector<PetscReal> recvmat_a;
  for (int i = 0; i < size; i++) {
    send_count = 0;
    if (i != rank) {
      sendmat_i.clear();
      sendmat_j.clear();
      sendmat_a.clear();
      // prepare send data
      for (int row_it = 0; row_it < Row; row_it++) {
        for (int j = mat_i[row_it]; j < mat_i[row_it + 1]; j++) {
          if (mat_j[j] >= col_range[i] && mat_j[j] < col_range[i + 1]) {
            sendmat_i.push_back(row_it);
            sendmat_j.push_back(mat_j[j]);
            sendmat_a.push_back(mat_a[j]);
          }
        }
      }

      send_count = sendmat_i.size();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(&send_count, 1, MPI_INT, recv_count.data(), 1, MPI_INT, i,
               MPI_COMM_WORLD);

    if (rank == i) {
      displs[0] = 0;
      for (int i = 1; i <= size; i++) {
        displs[i] = displs[i - 1] + recv_count[i - 1];
      }

      recvmat_i.resize(displs[size]);
      recvmat_j.resize(displs[size]);
      recvmat_a.resize(displs[size]);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(sendmat_i.data(), send_count, MPI_UNSIGNED, recvmat_i.data(),
                recv_count.data(), displs.data(), MPI_UNSIGNED, i,
                MPI_COMM_WORLD);
    MPI_Gatherv(sendmat_j.data(), send_count, MPI_UNSIGNED, recvmat_j.data(),
                recv_count.data(), displs.data(), MPI_UNSIGNED, i,
                MPI_COMM_WORLD);
    MPI_Gatherv(sendmat_a.data(), send_count, MPI_DOUBLE, recvmat_a.data(),
                recv_count.data(), displs.data(), MPI_DOUBLE, i,
                MPI_COMM_WORLD);
  }

  // merge matrix
  int col_range1 = col_range[rank];
  vector<PetscInt> old_local_mat_i, old_local_mat_j;
  vector<PetscReal> old_local_mat_a;
  old_local_mat_i.resize(Row + 1);
  old_local_mat_i[0] = 0;
  for (int row_it = 0; row_it < Row; row_it++) {
    vector<PetscInt> index;
    vector<PetscReal> row_a;
    for (int j = mat_i[row_it]; j < mat_i[row_it + 1]; j++) {
      if (mat_j[j] >= col_range[rank] && mat_j[j] < col_range[rank + 1]) {
        index.push_back(mat_j[j]);
        row_a.push_back(mat_a[j]);
      }
    }
    old_local_mat_i[row_it + 1] = old_local_mat_i[row_it] + index.size();
    old_local_mat_j.insert(old_local_mat_j.end(), index.begin(), index.end());
    old_local_mat_a.insert(old_local_mat_a.end(), row_a.begin(), row_a.end());
  }

  vector<vector<PetscInt>> new_mat;
  new_mat.resize(col);
  for (int i = 0; i < recvmat_i.size(); i++) {
    new_mat[recvmat_j[i] - col_range1].push_back(recvmat_i[i]);
  }
  for (int row_it = 0; row_it < Row; row_it++) {
    for (int j = old_local_mat_i[row_it]; j < old_local_mat_i[row_it + 1];
         j++) {
      new_mat[old_local_mat_j[j] - col_range1].push_back(row_it);
    }
  }

  for (int i = 0; i < col; i++) {
    sort(new_mat[i].begin(), new_mat[i].end());
    new_mat[i].erase(unique(new_mat[i].begin(), new_mat[i].end()),
                     new_mat[i].end());
  }

  // set up transpose matrix
  send_count = col;
  MPI_Allgather(&send_count, 1, MPI_INT, recv_count.data(), 1, MPI_INT,
                MPI_COMM_WORLD);

  displs.resize(size + 1);
  displs[0] = 0;
  for (int i = 1; i <= size; i++) {
    displs[i] = displs[i - 1] + recv_count[i - 1];
  }

  range_col1 = displs[rank];
  range_col2 = displs[rank + 1];

  mat_i.resize(col + 1);
  mat_oi.resize(col + 1);
  mat_j.clear();
  mat_oj.clear();
  mat_i[0] = 0;
  mat_oi[0] = 0;
  for (int i = 0; i < col; i++) {
    int diag_count = 0;
    int offdiag_count = 0;
    for (auto it = new_mat[i].begin(); it != new_mat[i].end(); it++) {
      if (*it >= range_col1 && *it < range_col2) {
        mat_j.push_back(*it);
        diag_count++;
      } else {
        mat_oj.push_back(*it);
        offdiag_count++;
      }
    }
    mat_i[i + 1] = mat_i[i] + diag_count;
    mat_oi[i + 1] = mat_oi[i] + offdiag_count;
  }
  mat_a.resize(mat_j.size());
  mat_oa.resize(mat_oj.size());
  fill(mat_a.begin(), mat_a.end(), 0.0);
  fill(mat_oa.begin(), mat_oa.end(), 0.0);
  for (int i = 0; i < recvmat_i.size(); i++) {
    if (recvmat_i[i] >= range_col1 && recvmat_i[i] < range_col2) {
      auto it = lower_bound(
          mat_j.begin() + mat_i[recvmat_j[i] - col_range1],
          mat_j.begin() + mat_i[recvmat_j[i] - col_range1 + 1], recvmat_i[i]);
      if (*it == recvmat_i[i]) {
        mat_a[it - mat_j.begin()] += recvmat_a[i];
      } else {
        cout << rank << ' ' << recvmat_j[i] - col_range1 << ' ' << recvmat_i[i]
             << " recv mat transpose matrix assembly error" << endl;
      }
    } else {
      auto it = lower_bound(
          mat_oj.begin() + mat_oi[recvmat_j[i] - col_range1],
          mat_oj.begin() + mat_oi[recvmat_j[i] - col_range1 + 1], recvmat_i[i]);
      if (*it == recvmat_i[i]) {
        mat_oa[it - mat_oj.begin()] += recvmat_a[i];
      } else {
        cout << rank << ' ' << recvmat_j[i] - col_range1 << ' ' << recvmat_i[i]
             << " recv mat off-diag transpose matrix assembly error" << endl;
      }
    }
  }
  for (int row_it = 0; row_it < Row; row_it++) {
    for (int j = old_local_mat_i[row_it]; j < old_local_mat_i[row_it + 1];
         j++) {
      if (row_it >= range_col1 && row_it < range_col2) {
        auto it = lower_bound(
            mat_j.begin() + mat_i[old_local_mat_j[j] - col_range1],
            mat_j.begin() + mat_i[old_local_mat_j[j] - col_range1 + 1], row_it);
        if (*it == row_it) {
          mat_a[it - mat_j.begin()] += old_local_mat_a[j];
        } else {
          cout << "transpose matrix assembly error" << endl;
        }
      } else {
        auto it = lower_bound(
            mat_oj.begin() + mat_oi[old_local_mat_j[j] - col_range1],
            mat_oj.begin() + mat_oi[old_local_mat_j[j] - col_range1 + 1],
            row_it);
        if (*it == row_it) {
          mat_oa[it - mat_oj.begin()] += old_local_mat_a[j];
        } else {
          cout << "transpose matrix assembly error" << endl;
        }
      }
    }
  }

  MatCreateMPIAIJWithSplitArrays(
      PETSC_COMM_WORLD, col, row, PETSC_DECIDE, Row, mat_i.data(), mat_j.data(),
      mat_a.data(), mat_oi.data(), mat_oj.data(), mat_oa.data(), &shell_mat);

  MatCreateShell(PETSC_COMM_WORLD, row, col, Row, Col, &shell_mat, mat);

  MatShellSetOperation(*mat, MATOP_MULT, (void (*)(void))MultTranspose);
  MatShellSetOperation(*mat, MATOP_MULT_ADD, (void (*)(void))MultTransposeAdd);

  is_assembled = true;
  is_transpose = true;

  return 0;
}

void extract_neighbor_index(
    petsc_sparse_matrix &A, petsc_sparse_matrix &B, petsc_sparse_matrix &C,
    petsc_sparse_matrix &D, petsc_sparse_matrix &nn_A,
    petsc_sparse_matrix &nn_B, petsc_sparse_matrix &nn_C,
    petsc_sparse_matrix &nn_D, petsc_sparse_matrix &nw_A,
    petsc_sparse_matrix &nw_B, petsc_sparse_matrix &nw_C,
    petsc_sparse_matrix &nw_D, std::vector<int> &idx_colloid,
    const int field_dof) {
  // extract idx_colloid
  idx_colloid.clear();
  for (int i = 0; i < C.col; i++) {
    if (C.mat_i[i + 1] - C.mat_i[i] > 0 || C.mat_oi[i + 1] - C.mat_oi[i] > 0)
      idx_colloid.push_back(i);
  }
  for (int i = 0; i < B.row; i++) {
    if (B.mat_i[i + 1] - B.mat_i[i] > 0 || B.mat_oi[i + 1] - B.mat_oi[i] > 0)
      idx_colloid.push_back(i);
  }
  for (auto it = idx_colloid.begin(); it != idx_colloid.end(); it++) {
    *it = *it / field_dof;
  }

  sort(idx_colloid.begin(), idx_colloid.end());
  idx_colloid.erase(unique(idx_colloid.begin(), idx_colloid.end()),
                    idx_colloid.end());

  vector<int> idx_colloid_global;
  for (int i = 0; i < idx_colloid.size(); i++) {
    for (int j = 0; j < field_dof; j++) {
      idx_colloid_global.push_back(idx_colloid[i] * field_dof + j +
                                   A.range_col1);
    }
  }

  IS isg_colloid;
  ISCreateGeneral(MPI_COMM_WORLD, idx_colloid_global.size(),
                  idx_colloid_global.data(), PETSC_COPY_VALUES, &isg_colloid);

  // build nn
  MatCreateSubMatrix(A.get_reference(), isg_colloid, isg_colloid,
                     MAT_INITIAL_MATRIX, &(nn_A.get_reference()));
  MatCreateSubMatrix(B.get_reference(), isg_colloid, NULL, MAT_INITIAL_MATRIX,
                     &(nn_B.get_reference()));
  MatCreateSubMatrix(C.get_shell_reference(), isg_colloid, NULL,
                     MAT_INITIAL_MATRIX, &(nn_C.get_shell_reference()));
  MatCreateShell(PETSC_COMM_WORLD, C.row, idx_colloid_global.size(),
                 PETSC_DECIDE, PETSC_DECIDE, &(nn_C.get_shell_reference()),
                 &(nn_C.get_reference()));
  MatShellSetOperation(nn_C.get_reference(), MATOP_MULT,
                       (void (*)(void))MultTranspose);
  MatShellSetOperation(nn_C.get_reference(), MATOP_MULT_ADD,
                       (void (*)(void))MultTransposeAdd);
  MatDuplicate(D.get_reference(), MAT_COPY_VALUES, &(nn_D.get_reference()));

  nn_A.is_assembled = true;
  nn_B.is_assembled = true;
  nn_C.is_assembled = true;
  nn_C.is_transpose = true;
  nn_D.is_assembled = true;

  MPI_Barrier(MPI_COMM_WORLD);

  // build nw
  MatCreateSubMatrix(A.get_reference(), isg_colloid, NULL, MAT_INITIAL_MATRIX,
                     &(nw_A.get_reference()));
  MatCreateSubMatrix(B.get_reference(), isg_colloid, NULL, MAT_INITIAL_MATRIX,
                     &(nw_B.get_reference()));
  MatDuplicate(C.get_shell_reference(), MAT_COPY_VALUES,
               &(nw_C.get_shell_reference()));
  MatCreateShell(PETSC_COMM_WORLD, C.row, C.col, C.Row, C.Col,
                 &(nw_C.get_shell_reference()), &(nw_C.get_reference()));
  MatShellSetOperation(nw_C.get_reference(), MATOP_MULT,
                       (void (*)(void))MultTranspose);
  MatShellSetOperation(nw_C.get_reference(), MATOP_MULT_ADD,
                       (void (*)(void))MultTransposeAdd);
  MatDuplicate(D.get_reference(), MAT_COPY_VALUES, &(nw_D.get_reference()));

  nw_A.is_assembled = true;
  nw_B.is_assembled = true;
  nw_C.is_assembled = true;
  nw_C.is_transpose = true;
  nw_D.is_assembled = true;

  MPI_Barrier(MPI_COMM_WORLD);

  PetscInt Row_offset, local_n1, local_n2;
  MatGetOwnershipRange(D.get_reference(), &local_n1, &local_n2);
  MatGetSize(A.get_reference(), &Row_offset, NULL);

  for (int i = local_n1; i < local_n2; i++) {
    idx_colloid_global.push_back(i + Row_offset);
  }
  idx_colloid = move(idx_colloid_global);

  ISDestroy(&isg_colloid);
}

PetscErrorCode null_space_matrix_mult(Mat mat, Vec x, Vec y) {
  petsc_sparse_matrix *ctx;
  MatShellGetContext(mat, &ctx);

  Vec z;
  VecDuplicate(x, &z);
  VecCopy(x, z);

  PetscReal *a;
  PetscReal sum, average;

  VecGetArray(z, &a);
  sum = 0.0;
  for (auto it = ctx->null_space_ptr->begin(); it != ctx->null_space_ptr->end();
       it++) {
    sum += a[*it];
  }
  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  average = sum / (double)ctx->null_space_size;
  for (auto it = ctx->null_space_ptr->begin(); it != ctx->null_space_ptr->end();
       it++) {
    a[*it] -= average;
  }
  VecRestoreArray(z, &a);

  MatMult(*(ctx->mat), z, y);

  VecGetArray(y, &a);
  sum = 0.0;
  for (auto it = ctx->null_space_ptr->begin(); it != ctx->null_space_ptr->end();
       it++) {
    sum += a[*it];
  }
  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  average = sum / ctx->null_space_size;
  for (auto it = ctx->null_space_ptr->begin(); it != ctx->null_space_ptr->end();
       it++) {
    a[*it] -= average;
  }
  VecRestoreArray(y, &a);

  VecDestroy(&z);

  return 0;
}

PetscErrorCode MultTranspose(Mat mat, Vec x, Vec y) {
  Mat *ctx;
  MatShellGetContext(mat, &ctx);

  return MatMultTranspose(*ctx, x, y);
}

PetscErrorCode MultTransposeAdd(Mat mat, Vec x, Vec y, Vec z) {
  Mat *ctx;
  MatShellGetContext(mat, &ctx);

  return MatMultTransposeAdd(*ctx, x, y, z);
}