#include "stokes_composite_preconditioner.hpp"

#include "petsc_ksp.hpp"
#include "petsc_sparse_matrix.hpp"

#include <fstream>
#include <string>

using namespace std;

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
          output << (i + 1) + range_row1 << '\t' << mat_j[j] << '\t' << mat_a[j]
                 << endl;
        }
        for (int j = mat_oi[i]; j < mat_oi[i + 1]; j++) {
          output << (i + 1) + range_row1 << '\t' << mat_oj[j] << '\t'
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

  for (int it = i; it < mat_i.size(); it++) {
    mat_i[it + 1] += index.size();
  }
  if (!is_transpose)
    for (int it = i; it < row; it++) {
      mat_oi[it + 1] += o_index.size();
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
  if (abs(daij) > 1e-15) {
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
      auto it = lower_bound(mat_j.begin() + mat_i[i],
                            mat_j.begin() + mat_i[i + 1], j);

      if (*it == j)
        mat_a[it - mat_j.begin()] += daij;
      else
        cout << rank << ' ' << i << ' ' << j
             << " transpose increment misplacement" << endl;
    }
  }
}

double petsc_sparse_matrix::get_entity(const PetscInt i, const PetscInt j) {
  if (j > Col) {
    cout << i << ' ' << j << " wrong matrix index access" << endl;
    return 0.0;
  }

  if (block_size == 1) {
    if (j >= range_row1 && j < range_row2) {
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
}

int petsc_sparse_matrix::graph_assemble() {
  int nnz, o_nnz;

  if (block_size == 1) {
    if (!is_transpose) {
      nnz = mat_i[block_row];
      o_nnz = mat_oi[block_row];

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
  if (is_transpose)
    return transpose_assemble();

  if (mat == PETSC_NULL) {
    mat = new Mat;
  }
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

  mat_i.resize(col + 1);
  mat_j.clear();
  mat_i[0] = 0;
  for (int i = 0; i < col; i++) {
    mat_i[i + 1] = mat_i[i] + new_mat[i].size();
    mat_j.insert(mat_j.end(), new_mat[i].begin(), new_mat[i].end());
  }
  mat_a.resize(mat_j.size());
  fill(mat_a.begin(), mat_a.end(), 0.0);
  for (int i = 0; i < recvmat_i.size(); i++) {
    auto it = lower_bound(mat_j.begin() + mat_i[recvmat_j[i] - col_range1],
                          mat_j.begin() + mat_i[recvmat_j[i] - col_range1 + 1],
                          recvmat_i[i]);
    if (*it == recvmat_i[i]) {
      mat_a[it - mat_j.begin()] += recvmat_a[i];
    }
  }
  for (int row_it = 0; row_it < Row; row_it++) {
    for (int j = old_local_mat_i[row_it]; j < old_local_mat_i[row_it + 1];
         j++) {
      auto it = lower_bound(
          mat_j.begin() + mat_i[old_local_mat_j[j] - col_range1],
          mat_j.begin() + mat_i[old_local_mat_j[j] - col_range1 + 1], row_it);
      if (*it == row_it) {
        mat_a[it - mat_j.begin()] += old_local_mat_a[j];
      }
    }
  }

  MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, col, row, PETSC_DECIDE, Row,
                            mat_i.data(), mat_j.data(), mat_a.data(), mat);
  {
    decltype(mat_i)().swap(mat_i);
    decltype(mat_j)().swap(mat_j);
    decltype(mat_a)().swap(mat_a);
  }

  is_assembled = true;

  return 0;
}

int petsc_sparse_matrix::extract_neighbor_index(
    vector<int> &idx_colloid, int dimension, int num_rigid_body,
    int local_rigid_body_offset, int global_rigid_body_offset,
    petsc_sparse_matrix &nn, petsc_sparse_matrix &nw) {

  // int size, myId;
  // MPI_Comm_rank(MPI_COMM_WORLD, &myId);
  // MPI_Comm_size(MPI_COMM_WORLD, &size);

  // int rigid_body_dof = (dimension == 2) ? 3 : 6;
  // int field_dof = dimension + 1;

  // idx_colloid.clear();

  // vector<int> neighbor_inclusion;

  // MPI_Barrier(MPI_COMM_WORLD);
  // idx_colloid.clear();

  // PetscInt local_N1;
  // MatGetOwnershipRange(__shell_mat, &local_N1, NULL);

  // neighbor_inclusion.clear();

  // PetscInt Col_block, row_block, col_block;
  // Col_block = __Col - num_rigid_body * rigid_body_dof;
  // if (myId == size - 1) {
  //   row_block = __row - num_rigid_body * rigid_body_dof;
  //   col_block = __col - num_rigid_body * rigid_body_dof;
  // } else {
  //   row_block = __row;
  //   col_block = __col;
  // }

  // for (int i = 0; i < row_block; i++) {
  //   auto it = lower_bound(__matrix[i].begin(), __matrix[i].end(),
  //                         entry(global_rigid_body_offset, 0.0),
  //                         compare_index);
  //   if (it != __matrix[i].end()) {
  //     neighbor_inclusion.push_back((local_N1 + i) / field_dof);
  //   }
  // }

  // sort(neighbor_inclusion.begin(), neighbor_inclusion.end());
  // neighbor_inclusion.erase(
  //     unique(neighbor_inclusion.begin(), neighbor_inclusion.end()),
  //     neighbor_inclusion.end());

  // // move data
  // vector<int> local_neighbor_inclusion_num(size);
  // for (int i = 0; i < size; i++) {
  //   local_neighbor_inclusion_num[i] = 0;
  // }
  // local_neighbor_inclusion_num[myId] = neighbor_inclusion.size();

  // MPI_Allreduce(MPI_IN_PLACE, local_neighbor_inclusion_num.data(), size,
  //               MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  // int recv_num = 0;
  // vector<int> displs;
  // vector<int> recv_neighbor_inclusion;
  // if (myId == size - 1) {
  //   displs.resize(size + 1);
  //   displs[0] = 0;
  //   for (int i = 0; i < size; i++) {
  //     recv_num += local_neighbor_inclusion_num[i];
  //     displs[i + 1] = displs[i] + local_neighbor_inclusion_num[i];
  //   }

  //   recv_neighbor_inclusion.resize(recv_num);
  // }

  // MPI_Gatherv(neighbor_inclusion.data(), neighbor_inclusion.size(), MPI_INT,
  //             recv_neighbor_inclusion.data(),
  //             local_neighbor_inclusion_num.data(), displs.data(), MPI_INT,
  //             size - 1, MPI_COMM_WORLD);

  // if (myId == size - 1) {
  //   sort(recv_neighbor_inclusion.begin(), recv_neighbor_inclusion.end());
  //   recv_neighbor_inclusion.erase(
  //       unique(recv_neighbor_inclusion.begin(),
  //       recv_neighbor_inclusion.end()), recv_neighbor_inclusion.end());
  // }

  // vector<vector<entry>>().swap(__matrix);

  // if (myId == size - 1) {
  //   neighbor_inclusion.clear();
  //   neighbor_inclusion.insert(
  //       neighbor_inclusion.end(), __j.begin() + __i[local_rigid_body_offset],
  //       __j.begin() +
  //           __i[local_rigid_body_offset + num_rigid_body * rigid_body_dof]);

  //   for (int i = 0; i < neighbor_inclusion.size(); i++) {
  //     if (neighbor_inclusion[i] < global_rigid_body_offset)
  //       neighbor_inclusion[i] /= field_dof;
  //   }

  //   neighbor_inclusion.insert(neighbor_inclusion.end(),
  //                             recv_neighbor_inclusion.begin(),
  //                             recv_neighbor_inclusion.end());

  //   sort(neighbor_inclusion.begin(), neighbor_inclusion.end());

  //   neighbor_inclusion.erase(
  //       unique(neighbor_inclusion.begin(), neighbor_inclusion.end()),
  //       neighbor_inclusion.end());

  //   auto it = neighbor_inclusion.begin();
  //   for (; it != neighbor_inclusion.end(); it++) {
  //     if (*it >= global_rigid_body_offset)
  //       break;
  //   }

  //   neighbor_inclusion.erase(it, neighbor_inclusion.end());
  // }

  // int neighbor_inclusionSize, offset = 0;

  // vector<int> recvneighbor_inclusion;

  // for (int i = 0; i < size; i++) {
  //   if (i != size - 1) {
  //     if (myId == size - 1) {
  //       neighbor_inclusionSize = neighbor_inclusion.size() / size;
  //       neighbor_inclusionSize +=
  //           (neighbor_inclusion.size() % size > i) ? 1 : 0;

  //       MPI_Send(&neighbor_inclusionSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
  //       MPI_Send(neighbor_inclusion.data() + offset, neighbor_inclusionSize,
  //                MPI_INT, i, 1, MPI_COMM_WORLD);
  //     }
  //     if (myId == i) {
  //       MPI_Status stat;
  //       MPI_Recv(&neighbor_inclusionSize, 1, MPI_INT, size - 1, 0,
  //                MPI_COMM_WORLD, &stat);
  //       recvneighbor_inclusion.resize(neighbor_inclusionSize);
  //       MPI_Recv(recvneighbor_inclusion.data(), neighbor_inclusionSize,
  //       MPI_INT,
  //                size - 1, 1, MPI_COMM_WORLD, &stat);
  //     }
  //   } else {
  //     if (myId == size - 1) {
  //       recvneighbor_inclusion.clear();
  //       recvneighbor_inclusion.insert(recvneighbor_inclusion.end(),
  //                                     neighbor_inclusion.begin() + offset,
  //                                     neighbor_inclusion.end());
  //     }
  //   }

  //   if (myId == size - 1) {
  //     offset += neighbor_inclusionSize;
  //   }

  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  // MPI_Barrier(MPI_COMM_WORLD);

  // idx_colloid.clear();
  // for (int i = 0; i < recvneighbor_inclusion.size(); i++) {
  //   for (int j = 0; j < field_dof; j++) {
  //     idx_colloid.push_back(recvneighbor_inclusion[i] * field_dof + j);
  //   }
  // }

  // // split colloid rigid body dof to each process
  // int avg_rigid_body_num = num_rigid_body / size;
  // int rigid_body_idx_low = 0;
  // int rigid_body_idx_high = 0;
  // for (int i = 0; i < myId; i++) {
  //   if (i < num_rigid_body % size) {
  //     rigid_body_idx_low += avg_rigid_body_num + 1;
  //   } else {
  //     rigid_body_idx_low += avg_rigid_body_num;
  //   }
  // }
  // if (myId < num_rigid_body % size) {
  //   rigid_body_idx_high = rigid_body_idx_low + avg_rigid_body_num + 1;
  // } else {
  //   rigid_body_idx_high = rigid_body_idx_low + avg_rigid_body_num;
  // }

  // for (int i = rigid_body_idx_low; i < rigid_body_idx_high; i++) {
  //   for (int j = 0; j < rigid_body_dof; j++) {
  //     idx_colloid.push_back(global_rigid_body_offset + i * rigid_body_dof +
  //     j);
  //   }
  // }

  // MPI_Barrier(MPI_COMM_WORLD);

  // {
  //   decltype(__i)().swap(__i);
  //   decltype(__j)().swap(__j);
  //   decltype(__val)().swap(__val);
  // }

  // petsc_is isg_colloid;
  // isg_colloid.create(idx_colloid);

  // MatCreateSubMatrix(mat, isg_colloid.get_reference(),
  //                    isg_colloid.get_reference(), MAT_INITIAL_MATRIX,
  //                    nn.get_pointer());
  // MatCreateSubMatrix(mat, isg_colloid.get_reference(), NULL,
  // MAT_INITIAL_MATRIX,
  //                    nw.get_pointer());

  // MatDestroy(&mat);

  // int mpi_rank = myId;
  // int mpi_size = size;

  // vector<int> idx_colloid_sub_field;
  // vector<int> idx_colloid_sub_colloid;
  // vector<int> idx_colloid_field;

  // vector<int> idx_colloid_offset, idx_colloid_global_size;
  // idx_colloid_offset.resize(mpi_size + 1);
  // idx_colloid_global_size.resize(mpi_size);

  // int idx_colloid_local_size = idx_colloid.size();
  // MPI_Allgather(&idx_colloid_local_size, 1, MPI_INT,
  //               idx_colloid_global_size.data(), 1, MPI_INT, MPI_COMM_WORLD);

  // idx_colloid_offset[0] = 0;
  // for (int i = 0; i < mpi_size; i++) {
  //   idx_colloid_offset[i + 1] =
  //       idx_colloid_offset[i] + idx_colloid_global_size[i];
  // }

  // for (int i = 0; i < idx_colloid.size(); i++) {
  //   if (idx_colloid[i] < Col_block) {
  //     idx_colloid_sub_field.push_back(i + idx_colloid_offset[mpi_rank]);
  //     idx_colloid_field.push_back(idx_colloid[i]);
  //   } else {
  //     idx_colloid_sub_colloid.push_back(i + idx_colloid_offset[mpi_rank]);
  //   }
  // }

  // IS isg_colloid_sub_field, isg_colloid_sub_colloid, isg_colloid_field;

  // Mat sub_ff, sub_fc, sub_cf;

  // ISCreateGeneral(MPI_COMM_WORLD, idx_colloid_sub_field.size(),
  //                 idx_colloid_sub_field.data(), PETSC_COPY_VALUES,
  //                 &isg_colloid_sub_field);
  // ISCreateGeneral(MPI_COMM_WORLD, idx_colloid_sub_colloid.size(),
  //                 idx_colloid_sub_colloid.data(), PETSC_COPY_VALUES,
  //                 &isg_colloid_sub_colloid);
  // ISCreateGeneral(MPI_COMM_WORLD, idx_colloid_field.size(),
  //                 idx_colloid_field.data(), PETSC_COPY_VALUES,
  //                 &isg_colloid_field);

  // MatCreateSubMatrix(*(ctx.fluid_part), isg_colloid_field, isg_colloid_field,
  //                    MAT_INITIAL_MATRIX, &sub_ff);
  // // MatCreateSubMatrix(nn.get_reference(), isg_colloid_sub_field,
  // //                    isg_colloid_sub_field, MAT_INITIAL_MATRIX, &sub_ff);
  // MatCreateSubMatrix(nn.get_reference(), isg_colloid_sub_field,
  //                    isg_colloid_sub_colloid, MAT_INITIAL_MATRIX, &sub_fc);
  // MatCreateSubMatrix(nn.get_reference(), isg_colloid_sub_colloid, NULL,
  //                    MAT_INITIAL_MATRIX, &sub_cf);

  // MatConvert(sub_ff, MATSAME, MAT_INITIAL_MATRIX, &(nn.ctx.fluid_raw_part));
  // MatConvert(sub_fc, MATSAME, MAT_INITIAL_MATRIX,
  // &(nn.ctx.fluid_colloid_part)); MatTranspose(sub_cf, MAT_INITIAL_MATRIX,
  // &(nn.ctx.colloid_part));

  // VecCreateMPI(PETSC_COMM_WORLD, idx_colloid_sub_field.size(), PETSC_DECIDE,
  //              &(nn.ctx.fluid_vec1));
  // VecCreateMPI(PETSC_COMM_WORLD, idx_colloid_sub_field.size(), PETSC_DECIDE,
  //              &(nn.ctx.fluid_vec2));
  // VecCreateMPI(PETSC_COMM_WORLD, idx_colloid_sub_colloid.size(),
  // PETSC_DECIDE,
  //              &(nn.ctx.colloid_vec));

  // Vec x;
  // MatCreateVecs(nn.get_reference(), &x, NULL);

  // VecScatterCreate(x, isg_colloid_sub_field, nn.ctx.fluid_vec1, NULL,
  //                  &(nn.ctx.fluid_scatter));
  // VecScatterCreate(x, isg_colloid_sub_colloid, nn.ctx.colloid_vec, NULL,
  //                  &(nn.ctx.colloid_scatter));

  // Mat &shell_mat = nn.get_shell_reference();
  // MatCreateShell(PETSC_COMM_WORLD, idx_colloid.size(), idx_colloid.size(),
  //                PETSC_DECIDE, PETSC_DECIDE, &nn.ctx, &shell_mat);
  // MatShellSetOperation(shell_mat, MATOP_MULT,
  //                      (void (*)(void))fluid_colloid_matrix_mult2);

  // nn.is_shell_assembled = true;
  // nn.is_ctx_assembled = true;
  // nn.ctx.use_vec_scatter = true;
  // nn.ctx.use_raw_fluid_part = true;
  // nn.ctx.use_local_vec = false;

  // ISDestroy(&isg_colloid_sub_field);
  // ISDestroy(&isg_colloid_sub_colloid);
  // ISDestroy(&isg_colloid_field);

  // MatDestroy(&sub_ff);
  // MatDestroy(&sub_fc);
  // MatDestroy(&sub_cf);

  // VecDestroy(&x);

  return 0;
}

PetscErrorCode null_space_matrix_mult(Mat mat, Vec x, Vec y) {
  petsc_sparse_matrix *ctx;
  MatShellGetContext(mat, &ctx);

  PetscReal *a;
  PetscReal sum, average;

  VecGetArray(x, &a);
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
  VecRestoreArray(x, &a);

  MatMult(*(ctx->mat), x, y);

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

  return 0;
}