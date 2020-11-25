#include "sparse_matrix.h"
#include "composite_preconditioner.h"

#include <fstream>
#include <string>

using namespace std;

// group size comparison
inline bool compare_group(vector<int> group1, vector<int> group2) {
  return group1.size() > group2.size();
}

int PetscSparseMatrix::Write(string fileName) {
  ofstream output;
  int MPIsize, myID;
  MPI_Comm_rank(MPI_COMM_WORLD, &myID);
  MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

  if (myID == 0) {
    output.open(fileName, ios::trunc);
    output.close();
  }

  int send_count = __row;
  vector<int> recv_count(MPIsize);

  MPI_Allgather(&send_count, 1, MPI_INT, recv_count.data(), 1, MPI_INT,
                MPI_COMM_WORLD);

  vector<int> displs(MPIsize + 1);
  displs[0] = 0;
  for (int i = 1; i <= MPIsize; i++) {
    displs[i] = displs[i - 1] + recv_count[i - 1];
  }

  for (int process = 0; process < MPIsize; process++) {
    if (process == myID) {
      output.open(fileName, ios::app);
      for (int i = 0; i < __row; i++) {
        for (vector<entry>::iterator it = __matrix[i].begin();
             it != __matrix[i].end(); it++) {
          output << (i + 1) + displs[myID] << '\t' << (it->first + 1) << '\t'
                 << it->second << endl;
        }
      }
    }
    output.close();

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

int PetscSparseMatrix::FinalAssemble() {
  // move data from outProcessIncrement
  int myid, MPIsize;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

  for (PetscInt row = 0; row < __out_process_row; row++) {
    int send_count = __out_process_matrix[row].size();
    vector<int> recv_count(MPIsize);

    MPI_Gather(&send_count, 1, MPI_INT, recv_count.data(), 1, MPI_INT,
               MPIsize - 1, MPI_COMM_WORLD);

    vector<int> displs(MPIsize + 1);
    if (myid == MPIsize - 1) {
      displs[0] = 0;
      for (int i = 1; i <= MPIsize; i++) {
        displs[i] = displs[i - 1] + recv_count[i - 1];
      }
    }

    vector<PetscInt> recv_j;
    vector<PetscReal> recv_val;

    recv_j.resize(displs[MPIsize]);
    recv_val.resize(displs[MPIsize]);

    vector<PetscInt> send_j(send_count);
    vector<PetscReal> send_val(send_count);

    size_t n = 0;
    n = __out_process_matrix[row].size();
    send_j.resize(n);
    send_val.resize(n);
    for (auto i = 0; i < n; i++) {
      send_j[i] = __out_process_matrix[row][i].first;
      send_val[i] = __out_process_matrix[row][i].second;
    }

    MPI_Gatherv(send_j.data(), send_count, MPI_UNSIGNED, recv_j.data(),
                recv_count.data(), displs.data(), MPI_UNSIGNED, MPIsize - 1,
                MPI_COMM_WORLD);
    MPI_Gatherv(send_val.data(), send_count, MPI_DOUBLE, recv_val.data(),
                recv_count.data(), displs.data(), MPI_DOUBLE, MPIsize - 1,
                MPI_COMM_WORLD);

    // merge data
    if (myid == MPIsize - 1) {
      vector<int> sorted_recv_j = recv_j;
      sort(sorted_recv_j.begin(), sorted_recv_j.end());
      sorted_recv_j.erase(unique(sorted_recv_j.begin(), sorted_recv_j.end()),
                          sorted_recv_j.end());

      __matrix[row + __out_process_reduction].resize(sorted_recv_j.size());
      for (int i = 0; i < sorted_recv_j.size(); i++) {
        __matrix[row + __out_process_reduction][i] =
            entry(sorted_recv_j[i], 0.0);
      }

      for (int i = 0; i < recv_j.size(); i++) {
        auto it = lower_bound(__matrix[row + __out_process_reduction].begin(),
                              __matrix[row + __out_process_reduction].end(),
                              entry(recv_j[i], recv_val[i]), compare_index);

        it->second += recv_val[i];
      }
    }
  }

  __i.resize(__row + 1);

  __nnz = 0;
  for (int i = 0; i < __row; i++) {
    __i[i] = 0;
    __nnz += __matrix[i].size();
  }

  __j.resize(__nnz);
  __val.resize(__nnz);

  for (int i = 1; i <= __row; i++) {
    if (__i[i - 1] == 0) {
      __i[i] = 0;
      for (int j = i - 1; j >= 0; j--) {
        if (__i[j] == 0) {
          __i[i] += __matrix[j].size();
        } else {
          __i[i] += __i[j] + __matrix[j].size();
          break;
        }
      }
    } else {
      __i[i] = __i[i - 1] + __matrix[i - 1].size();
    }
  }

  for (int i = 0; i < __row; i++) {
    for (auto n = 0; n < __matrix[i].size(); n++) {
      __j[__i[i] + n] = __matrix[i][n].first;
      __val[__i[i] + n] = __matrix[i][n].second;
    }
  }

  if (__Col != 0)
    MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, __row, __col, PETSC_DECIDE,
                              __Col, __i.data(), __j.data(), __val.data(),
                              &__mat);
  else
    MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, __row, __col, PETSC_DECIDE,
                              PETSC_DECIDE, __i.data(), __j.data(),
                              __val.data(), &__mat);

  __isAssembled = true;

  return __nnz;
}

int PetscSparseMatrix::FinalAssemble(int blockSize) {
  // move data from outProcessIncrement
  int myid, MPIsize;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

  for (PetscInt row = 0; row < __out_process_row; row++) {
    int send_count = __out_process_matrix[row].size();
    vector<int> recv_count(MPIsize);

    MPI_Gather(&send_count, 1, MPI_INT, recv_count.data(), 1, MPI_INT,
               MPIsize - 1, MPI_COMM_WORLD);

    vector<int> displs(MPIsize + 1);
    if (myid == MPIsize - 1) {
      displs[0] = 0;
      for (int i = 1; i <= MPIsize; i++) {
        displs[i] = displs[i - 1] + recv_count[i - 1];
      }
    }

    vector<PetscInt> recv_j;
    vector<PetscReal> recv_val;

    recv_j.resize(displs[MPIsize]);
    recv_val.resize(displs[MPIsize]);

    vector<PetscInt> send_j(send_count);
    vector<PetscReal> send_val(send_count);

    size_t n = 0;
    n = __out_process_matrix[row].size();
    send_j.resize(n);
    send_val.resize(n);
    for (auto i = 0; i < n; i++) {
      send_j[i] = __out_process_matrix[row][i].first;
      send_val[i] = __out_process_matrix[row][i].second;
    }

    MPI_Gatherv(send_j.data(), send_count, MPI_UNSIGNED, recv_j.data(),
                recv_count.data(), displs.data(), MPI_UNSIGNED, MPIsize - 1,
                MPI_COMM_WORLD);
    MPI_Gatherv(send_val.data(), send_count, MPI_DOUBLE, recv_val.data(),
                recv_count.data(), displs.data(), MPI_DOUBLE, MPIsize - 1,
                MPI_COMM_WORLD);

    // merge data
    if (myid == MPIsize - 1) {
      vector<int> sorted_recv_j = recv_j;
      sort(sorted_recv_j.begin(), sorted_recv_j.end());
      sorted_recv_j.erase(unique(sorted_recv_j.begin(), sorted_recv_j.end()),
                          sorted_recv_j.end());

      __matrix[row + __out_process_reduction].resize(sorted_recv_j.size());
      for (int i = 0; i < sorted_recv_j.size(); i++) {
        __matrix[row + __out_process_reduction][i] =
            entry(sorted_recv_j[i], 0.0);
      }

      for (int i = 0; i < recv_j.size(); i++) {
        auto it = lower_bound(__matrix[row + __out_process_reduction].begin(),
                              __matrix[row + __out_process_reduction].end(),
                              entry(recv_j[i], recv_val[i]), compare_index);

        it->second += recv_val[i];
      }
    }
  }

  // block version matrix
  auto block_row = __row / blockSize;

  __i.resize(block_row + 1);
  __j.clear();

  __nnz = 0;
  auto nnz_block = __nnz;
  vector<PetscInt> block_col_indices;
  __i[0] = 0;
  for (int i = 0; i < block_row; i++) {
    block_col_indices.clear();
    for (int j = 0; j < blockSize; j++) {
      for (int k = 0; k < __matrix[i * blockSize + j].size(); k++) {
        if (__matrix[i * blockSize + j][k].first < __Col)
          block_col_indices.push_back(__matrix[i * blockSize + j][k].first /
                                      blockSize);
      }
    }

    sort(block_col_indices.begin(), block_col_indices.end());
    block_col_indices.erase(
        unique(block_col_indices.begin(), block_col_indices.end()),
        block_col_indices.end());

    nnz_block += block_col_indices.size();

    __i[i + 1] = __i[i] + block_col_indices.size();

    __j.insert(__j.end(), block_col_indices.begin(), block_col_indices.end());
  }

  auto blockStorage = blockSize * blockSize;

  __val.resize(nnz_block * blockStorage);

  for (int i = 0; i < nnz_block * blockStorage; i++) {
    __val[i] = 0.0;
  }

  for (int i = 0; i < __row; i++) {
    int block_row_index = i / blockSize;
    int local_row_index = i % blockSize;
    for (int j = 0; j < __matrix[i].size(); j++) {
      if (__matrix[i][j].first < __Col) {
        int block_col_index = __matrix[i][j].first / blockSize;
        int local_col_index = __matrix[i][j].first % blockSize;

        auto it = lower_bound(__j.begin() + __i[block_row_index],
                              __j.begin() + __i[block_row_index + 1],
                              block_col_index);

        auto disp = it - __j.begin();
        __val[blockStorage * disp + local_col_index +
              local_row_index * blockSize] = __matrix[i][j].second;
      }
    }
  }

  // MatCreateMPIBAIJWithArray is incompatible with current code setup
  if (__Col != 0) {
    MatCreate(MPI_COMM_WORLD, &__mat);
    MatSetSizes(__mat, __row, __col, PETSC_DECIDE, __Col);
    MatSetType(__mat, MATMPIBAIJ);
    MatSetBlockSize(__mat, blockSize);
    MatSetUp(__mat);
    MatMPIBAIJSetPreallocationCSR(__mat, blockSize, __i.data(), __j.data(),
                                  __val.data());
  } else {
    MatCreate(MPI_COMM_WORLD, &__mat);
    MatSetSizes(__mat, __row, __col, PETSC_DECIDE, PETSC_DECIDE);
    MatSetType(__mat, MATMPIBAIJ);
    MatSetBlockSize(__mat, blockSize);
    MatSetUp(__mat);
    MatMPIBAIJSetPreallocationCSR(__mat, blockSize, __i.data(), __j.data(),
                                  __val.data());
  }

  __isAssembled = true;

  __i.clear();
  __j.clear();
  __val.clear();

  __matrix.clear();
  __out_process_matrix.clear();

  return __nnz;
}

int PetscSparseMatrix::FinalAssemble(int blockSize, int num_rigid_body,
                                     int rigid_body_dof) {}

int PetscSparseMatrix::FinalAssemble(Mat &mat, int blockSize,
                                     int num_rigid_body, int rigid_body_dof) {
  // move data from outProcessIncrement
  int myid, MPIsize;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

  for (PetscInt row = 0; row < __out_process_row; row++) {
    int send_count = __out_process_matrix[row].size();
    vector<int> recv_count(MPIsize);

    MPI_Gather(&send_count, 1, MPI_INT, recv_count.data(), 1, MPI_INT,
               MPIsize - 1, MPI_COMM_WORLD);

    vector<int> displs(MPIsize + 1);
    if (myid == MPIsize - 1) {
      displs[0] = 0;
      for (int i = 1; i <= MPIsize; i++) {
        displs[i] = displs[i - 1] + recv_count[i - 1];
      }
    }

    vector<PetscInt> recv_j;
    vector<PetscReal> recv_val;

    recv_j.resize(displs[MPIsize]);
    recv_val.resize(displs[MPIsize]);

    vector<PetscInt> send_j(send_count);
    vector<PetscReal> send_val(send_count);

    size_t n = 0;
    n = __out_process_matrix[row].size();
    send_j.resize(n);
    send_val.resize(n);
    for (auto i = 0; i < n; i++) {
      send_j[i] = __out_process_matrix[row][i].first;
      send_val[i] = __out_process_matrix[row][i].second;
    }

    MPI_Gatherv(send_j.data(), send_count, MPI_UNSIGNED, recv_j.data(),
                recv_count.data(), displs.data(), MPI_UNSIGNED, MPIsize - 1,
                MPI_COMM_WORLD);
    MPI_Gatherv(send_val.data(), send_count, MPI_DOUBLE, recv_val.data(),
                recv_count.data(), displs.data(), MPI_DOUBLE, MPIsize - 1,
                MPI_COMM_WORLD);

    // merge data
    if (myid == MPIsize - 1) {
      vector<int> sorted_recv_j = recv_j;
      sort(sorted_recv_j.begin(), sorted_recv_j.end());
      sorted_recv_j.erase(unique(sorted_recv_j.begin(), sorted_recv_j.end()),
                          sorted_recv_j.end());

      __matrix[row + __out_process_reduction].resize(sorted_recv_j.size());
      for (int i = 0; i < sorted_recv_j.size(); i++) {
        __matrix[row + __out_process_reduction][i] =
            entry(sorted_recv_j[i], 0.0);
      }

      for (int i = 0; i < recv_j.size(); i++) {
        auto it = lower_bound(__matrix[row + __out_process_reduction].begin(),
                              __matrix[row + __out_process_reduction].end(),
                              entry(recv_j[i], recv_val[i]), compare_index);

        it->second += recv_val[i];
      }
    }
  }

  // get block version matrix for field submatrix
  PetscInt Col_block, row_block, col_block;
  Col_block = __Col - num_rigid_body * rigid_body_dof;
  if (myid == MPIsize - 1) {
    row_block = __row - num_rigid_body * rigid_body_dof;
    col_block = __col - num_rigid_body * rigid_body_dof;
  } else {
    row_block = __row;
    col_block = __col;
  }

  auto block_row = row_block / blockSize;

  __i.resize(block_row + 1);
  __j.clear();

  vector<PetscInt> block_col_indices;
  __i[0] = 0;
  int nnz_block = 0;
  for (int i = 0; i < block_row; i++) {
    block_col_indices.clear();
    for (int j = 0; j < blockSize; j++) {
      for (int k = 0; k < __matrix[i * blockSize + j].size(); k++) {
        if (__matrix[i * blockSize + j][k].first < Col_block)
          block_col_indices.push_back(__matrix[i * blockSize + j][k].first /
                                      blockSize);
      }
    }

    sort(block_col_indices.begin(), block_col_indices.end());
    block_col_indices.erase(
        unique(block_col_indices.begin(), block_col_indices.end()),
        block_col_indices.end());

    nnz_block += block_col_indices.size();

    __i[i + 1] = __i[i] + block_col_indices.size();

    __j.insert(__j.end(), block_col_indices.begin(), block_col_indices.end());
  }

  auto blockStorage = blockSize * blockSize;

  __val.resize(nnz_block * blockStorage);

  for (int i = 0; i < nnz_block * blockStorage; i++) {
    __val[i] = 0.0;
  }

  for (int i = 0; i < row_block; i++) {
    int block_row_index = i / blockSize;
    int local_row_index = i % blockSize;
    for (int j = 0; j < __matrix[i].size(); j++) {
      if (__matrix[i][j].first < Col_block) {
        int block_col_index = __matrix[i][j].first / blockSize;
        int local_col_index = __matrix[i][j].first % blockSize;

        auto it = lower_bound(__j.begin() + __i[block_row_index],
                              __j.begin() + __i[block_row_index + 1],
                              block_col_index);

        auto disp = it - __j.begin();
        __val[blockStorage * disp + local_col_index +
              local_row_index * blockSize] = __matrix[i][j].second;
      }
    }
  }

  if (Col_block != 0) {
    MatCreate(MPI_COMM_WORLD, &mat);
    MatSetSizes(mat, row_block, col_block, PETSC_DECIDE, Col_block);
    MatSetType(mat, MATMPIBAIJ);
    MatSetBlockSize(mat, blockSize);
    MatSetUp(mat);
    MatMPIBAIJSetPreallocationCSR(mat, blockSize, __i.data(), __j.data(),
                                  __val.data());
  } else {
    MatCreate(MPI_COMM_WORLD, &mat);
    MatSetSizes(mat, row_block, col_block, PETSC_DECIDE, PETSC_DECIDE);
    MatSetType(mat, MATMPIBAIJ);
    MatSetBlockSize(mat, blockSize);
    MatSetUp(mat);
    MatMPIBAIJSetPreallocationCSR(mat, blockSize, __i.data(), __j.data(),
                                  __val.data());
  }

  // non-block version
  __i.resize(__row + 1);

  __nnz = 0;
  for (int i = 0; i < __row; i++) {
    __i[i] = 0;
    __nnz += __matrix[i].size();
  }

  __j.resize(__nnz);
  __val.resize(__nnz);

  for (int i = 1; i <= __row; i++) {
    if (__i[i - 1] == 0) {
      __i[i] = 0;
      for (int j = i - 1; j >= 0; j--) {
        if (__i[j] == 0) {
          __i[i] += __matrix[j].size();
        } else {
          __i[i] += __i[j] + __matrix[j].size();
          break;
        }
      }
    } else {
      __i[i] = __i[i - 1] + __matrix[i - 1].size();
    }
  }

  for (int i = 0; i < __row; i++) {
    for (auto n = 0; n < __matrix[i].size(); n++) {
      __j[__i[i] + n] = __matrix[i][n].first;
      __val[__i[i] + n] = __matrix[i][n].second;
    }
  }

  if (__Col != 0)
    MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, __row, __col, PETSC_DECIDE,
                              __Col, __i.data(), __j.data(), __val.data(),
                              &__mat);
  else
    MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, __row, __col, PETSC_DECIDE,
                              PETSC_DECIDE, __i.data(), __j.data(),
                              __val.data(), &__mat);

  MatCreateShell(PETSC_COMM_WORLD, __row, __col, PETSC_DECIDE, __Col, &__ctx,
                 &__shell_mat);
  MatShellSetOperation(__shell_mat, MATOP_MULT,
                       (void (*)(void))fluid_colloid_matrix_mult);

  vector<PetscInt> colloid_part_i;
  vector<PetscInt> colloid_part_j;
  vector<PetscReal> colloid_part_val;

  colloid_part_i.resize(__out_process_row + 1);
  colloid_part_i[0] = 0;
  for (int i = 0; i < __out_process_row; i++) {
    colloid_part_i[i + 1] = colloid_part_i[i] + __out_process_matrix[i].size();
  }
  colloid_part_j.resize(colloid_part_i[__out_process_row]);
  colloid_part_val.resize(colloid_part_i[__out_process_row]);

  PetscInt local_size = __row;
  if (myid == MPIsize - 1)
    local_size -= num_rigid_body * rigid_body_dof;

  auto colloid_part_j_it = colloid_part_j.begin();
  auto colloid_part_val_it = colloid_part_val.begin();
  for (int i = 0; i < __out_process_row; i++) {
    for (int j = 0; j < __out_process_matrix[i].size(); j++) {
      *colloid_part_j_it = __out_process_matrix[i][j].first;
      *colloid_part_val_it = __out_process_matrix[i][j].second;

      colloid_part_j_it++;
      colloid_part_val_it++;
    }
  }

  VecCreateMPI(PETSC_COMM_WORLD, num_rigid_body * rigid_body_dof, PETSC_DECIDE,
               &(__ctx.colloid_vec));
  VecCreateMPI(PETSC_COMM_WORLD, local_size, PETSC_DECIDE, &(__ctx.fluid_vec));
  __ctx.fluid_local_size = local_size;
  __ctx.rigid_body_size = num_rigid_body * rigid_body_dof;

  __ctx.local_fluid_particle_num = local_size / blockSize;
  __ctx.field_dof = blockSize;
  MPI_Allreduce(&(__ctx.local_fluid_particle_num),
                &(__ctx.global_fluid_particle_num), 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  __ctx.pressure_offset = blockSize - 1;

  __ctx.myid = myid;
  __ctx.mpisize = MPIsize;

  MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, __out_process_row, __col,
                            PETSC_DECIDE, __Col, colloid_part_i.data(),
                            colloid_part_j.data(), colloid_part_val.data(),
                            &(__ctx.colloid_part));

  MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, local_size, __col, PETSC_DECIDE,
                            __Col, __i.data(), __j.data(), __val.data(),
                            &(__ctx.fluid_part));

  __isAssembled = true;
  __shellIsAssembled = true;
  __isCtxAssembled = true;

  return __nnz;
}

int PetscSparseMatrix::ExtractNeighborIndex(vector<int> &idx_neighbor,
                                            int dimension, int num_rigid_body,
                                            int local_rigid_body_offset,
                                            int global_rigid_body_offset) {
  idx_neighbor.clear();

  int MPIsize, myId;
  MPI_Comm_rank(MPI_COMM_WORLD, &myId);
  MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

  int rigid_body_dof = (dimension == 2) ? 3 : 6;
  int field_dof = dimension + 1;

  // first transpose the matrix
  vector<int> neighborInclusion;

  vector<vector<int>> rigid_body_block_distribution(MPIsize);

  if (myId == MPIsize - 1) {
    neighborInclusion.clear();
    neighborInclusion.insert(
        neighborInclusion.end(), __j.begin() + __i[local_rigid_body_offset],
        __j.begin() +
            __i[local_rigid_body_offset + num_rigid_body * rigid_body_dof]);

    for (int i = 0; i < neighborInclusion.size(); i++) {
      if (neighborInclusion[i] > global_rigid_body_offset)
        neighborInclusion[i] = neighborInclusion[0];
      else
        neighborInclusion[i] /= field_dof;
    }

    sort(neighborInclusion.begin(), neighborInclusion.end());

    neighborInclusion.erase(
        unique(neighborInclusion.begin(), neighborInclusion.end()),
        neighborInclusion.end());

    vector<vector<int>> transpose_mat;
    transpose_mat.resize(neighborInclusion.size());

    for (int i = 0; i < num_rigid_body; i++) {
      for (int j = 0; j < rigid_body_dof; j++) {
        for (int k = __i[local_rigid_body_offset + i * rigid_body_dof + j];
             k < __i[local_rigid_body_offset + i * rigid_body_dof + j + 1];
             k++) {
          size_t neighbor_index =
              lower_bound(neighborInclusion.begin(), neighborInclusion.end(),
                          __j[k] / field_dof) -
              neighborInclusion.begin();

          if (neighbor_index >= 0 &&
              neighbor_index < neighborInclusion.size()) {
            if (transpose_mat[neighbor_index].size() == 0) {
              transpose_mat[neighbor_index].push_back(i);
            } else {
              auto it = lower_bound(transpose_mat[neighbor_index].begin(),
                                    transpose_mat[neighbor_index].end(), i);
              if (it == transpose_mat[neighbor_index].end() || *it != i) {
                transpose_mat[neighbor_index].insert(it, i);
              }
            }
          }
        }
      }
    }

    // get the connectivity of rigid body here
    vector<vector<int>> connectivity;
    connectivity.resize(num_rigid_body);
    for (int i = 0; i < transpose_mat.size(); i++) {
      if (transpose_mat[i].size() > 1) {
        for (int j = 0; j < transpose_mat[i].size() - 1; j++) {
          int rigid_body_index1 = transpose_mat[i][j];
          for (int k = j + 1; k < transpose_mat[i].size(); k++) {
            int rigid_body_index2 = transpose_mat[i][k];

            // make connection between two rigid body
            if (connectivity[rigid_body_index1].size() == 0) {
              connectivity[rigid_body_index1].push_back(rigid_body_index2);
            } else {
              auto it = lower_bound(connectivity[rigid_body_index1].begin(),
                                    connectivity[rigid_body_index1].end(),
                                    rigid_body_index2);
              if (it == connectivity[rigid_body_index1].end() ||
                  *it != rigid_body_index2) {
                connectivity[rigid_body_index1].insert(it, rigid_body_index2);
              }
            }

            if (connectivity[rigid_body_index2].size() == 0) {
              connectivity[rigid_body_index2].push_back(rigid_body_index1);
            } else {
              auto it = lower_bound(connectivity[rigid_body_index2].begin(),
                                    connectivity[rigid_body_index2].end(),
                                    rigid_body_index1);
              if (it == connectivity[rigid_body_index2].end() ||
                  *it != rigid_body_index1) {
                connectivity[rigid_body_index2].insert(it, rigid_body_index1);
              }
            }
          }
        }
      }
    }

    // make rigid body group split
    vector<vector<int>> connected_group;
    vector<int> ungroup_rigid_body;

    ungroup_rigid_body.reserve(num_rigid_body);
    for (int i = 0; i < num_rigid_body; i++)
      ungroup_rigid_body.push_back(i);

    while (ungroup_rigid_body.size() > 0) {
      vector<int> maximum_group;
      vector<int> search_stack;
      maximum_group.push_back(ungroup_rigid_body[0]);
      search_stack.push_back(ungroup_rigid_body[0]);
      ungroup_rigid_body.erase(ungroup_rigid_body.begin());

      int search_stack_index = 0;
      while (search_stack_index < search_stack.size()) {
        for (auto item : connectivity[search_stack[search_stack_index]]) {
          auto it =
              lower_bound(maximum_group.begin(), maximum_group.end(), item);
          if (it == maximum_group.end() || *it != item) {
            maximum_group.insert(it, item);

            if (ungroup_rigid_body.size() != 0) {
              auto rm_it = lower_bound(ungroup_rigid_body.begin(),
                                       ungroup_rigid_body.end(), item);
              if (*rm_it == item) {
                ungroup_rigid_body.erase(rm_it);
                search_stack.push_back(item);
              }
            }
          }
        }

        search_stack_index++;
      }

      connected_group.push_back(maximum_group);
    }

    sort(connected_group.begin(), connected_group.end(), compare_group);

    PetscInt *ptr;
    MatGetOwnershipRanges(__mat, (const PetscInt **)(&ptr));
    vector<int> row_range;
    row_range.resize(MPIsize);
    for (int i = 0; i < MPIsize; i++)
      row_range[i] = ptr[i];

    // distribute rigid body among processes
    for (int i = 0; i < connected_group.size(); i++) {
      neighborInclusion.clear();

      // select neighbor column
      for (int j = 0; j < connected_group[i].size(); j++) {
        neighborInclusion.insert(
            neighborInclusion.end(),
            __j.begin() + __i[local_rigid_body_offset +
                              connected_group[i][j] * rigid_body_dof],
            __j.begin() + __i[local_rigid_body_offset +
                              (connected_group[i][j] + 1) * rigid_body_dof]);
      }

      for (int j = 0; j < neighborInclusion.size(); j++) {
        if (neighborInclusion[j] > global_rigid_body_offset)
          neighborInclusion[j] = neighborInclusion[0];
        else
          neighborInclusion[j] /= field_dof;
      }

      sort(neighborInclusion.begin(), neighborInclusion.end());

      neighborInclusion.erase(
          unique(neighborInclusion.begin(), neighborInclusion.end()),
          neighborInclusion.end());

      vector<int> tempNeighborInclusion = move(neighborInclusion);

      neighborInclusion.reserve(tempNeighborInclusion.size() * field_dof);

      for (auto neighbor : tempNeighborInclusion) {
        for (int j = 0; j < field_dof; j++) {
          neighborInclusion.push_back(neighbor * field_dof + j);
        }
      }

      sort(neighborInclusion.begin(), neighborInclusion.end());

      vector<int> count_per_process(MPIsize);
      for (int j = 0; j < MPIsize; j++) {
        count_per_process[j] = 0;
      }
      for (int j = 0; j < neighborInclusion.size(); j++) {
        auto pos = upper_bound(row_range.begin(), row_range.end(),
                               neighborInclusion[j]);
        if (pos == row_range.end())
          count_per_process[MPIsize - 1]++;
        else if (*pos == neighborInclusion[j])
          count_per_process[pos - row_range.begin()]++;
        else
          count_per_process[pos - row_range.begin() - 1]++;
      }

      int process_index = 0;
      int max_count = 0;
      for (int j = 0; j < MPIsize; j++) {
        if (count_per_process[j] > max_count) {
          process_index = j;
          max_count = count_per_process[j];
        }
      }

      for (int j = 0; j < connected_group[i].size(); j++) {
        rigid_body_block_distribution[process_index].push_back(
            connected_group[i][j]);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  int neighborInclusionSize;

  for (int i = 0; i < MPIsize; i++) {
    if (myId == MPIsize - 1) {
      neighborInclusion.clear();

      // select neighbor column
      for (int j = 0; j < rigid_body_block_distribution[i].size(); j++) {
        neighborInclusion.insert(
            neighborInclusion.end(),
            __j.begin() +
                __i[local_rigid_body_offset +
                    rigid_body_block_distribution[i][j] * rigid_body_dof],
            __j.begin() + __i[local_rigid_body_offset +
                              (rigid_body_block_distribution[i][j] + 1) *
                                  rigid_body_dof]);
      }

      sort(neighborInclusion.begin(), neighborInclusion.end());

      for (int j = 0; j < neighborInclusion.size(); j++) {
        if (neighborInclusion[j] > global_rigid_body_offset) {
          neighborInclusion[j] = neighborInclusion[0];
        } else
          neighborInclusion[j] /= field_dof;
      }

      sort(neighborInclusion.begin(), neighborInclusion.end());

      neighborInclusion.erase(
          unique(neighborInclusion.begin(), neighborInclusion.end()),
          neighborInclusion.end());

      vector<int> tempNeighborInclusion = move(neighborInclusion);

      neighborInclusion.reserve(tempNeighborInclusion.size() * field_dof +
                                rigid_body_block_distribution[i].size() *
                                    rigid_body_dof);

      for (auto neighbor : tempNeighborInclusion) {
        for (int j = 0; j < field_dof; j++) {
          neighborInclusion.push_back(neighbor * field_dof + j);
        }
      }

      for (int j = 0; j < rigid_body_block_distribution[i].size(); j++) {
        for (int k = 0; k < rigid_body_dof; k++) {
          neighborInclusion.push_back(
              global_rigid_body_offset +
              rigid_body_block_distribution[i][j] * rigid_body_dof + k);
        }
      }

      sort(neighborInclusion.begin(), neighborInclusion.end());

      neighborInclusionSize = neighborInclusion.size();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (i != MPIsize - 1) {
      if (myId == MPIsize - 1) {
        MPI_Send(&neighborInclusionSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        MPI_Send(neighborInclusion.data(), neighborInclusionSize, MPI_INT, i, 1,
                 MPI_COMM_WORLD);
      }
      if (myId == i) {
        MPI_Status stat;
        MPI_Recv(&neighborInclusionSize, 1, MPI_INT, MPIsize - 1, 0,
                 MPI_COMM_WORLD, &stat);
        idx_neighbor.resize(neighborInclusionSize);
        MPI_Recv(idx_neighbor.data(), neighborInclusionSize, MPI_INT,
                 MPIsize - 1, 1, MPI_COMM_WORLD, &stat);
      }
    } else {
      if (myId == MPIsize - 1)
        idx_neighbor = move(neighborInclusion);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  return 0;
}

void PetscSparseMatrix::Solve(vector<double> &rhs, vector<double> &x) {
  if (__isAssembled) {
    Vec _rhs, _x;
    VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, rhs.size(), PETSC_DECIDE,
                          rhs.data(), &_rhs);
    VecDuplicate(_rhs, &_x);

    KSP _ksp;
    KSPCreate(PETSC_COMM_WORLD, &_ksp);
    KSPSetOperators(_ksp, __mat, __mat);
    KSPSetFromOptions(_ksp);

    KSPSetUp(_ksp);

    PC _pc;
    KSPGetPC(_ksp, &_pc);
    PCSetType(_pc, PCLU);
    PCSetUp(_pc);

    PetscPrintf(PETSC_COMM_WORLD, "final solving of linear system\n");
    KSPSolve(_ksp, _rhs, _x);
    MPI_Barrier(MPI_COMM_WORLD);

    KSPDestroy(&_ksp);

    PetscScalar *a;
    VecGetArray(_x, &a);
    for (size_t i = 0; i < rhs.size(); i++) {
      x[i] = a[i];
    }

    VecDestroy(&_rhs);
    VecDestroy(&_x);
  }
}

void PetscSparseMatrix::Solve(vector<double> &rhs, vector<double> &x,
                              PetscInt blockSize) {
  if (__isAssembled) {
    Vec _rhs, _x, null;
    VecCreateMPIWithArray(PETSC_COMM_WORLD, blockSize, rhs.size(), PETSC_DECIDE,
                          rhs.data(), &_rhs);
    VecDuplicate(_rhs, &_x);
    VecDuplicate(_rhs, &null);

    PetscScalar *a;
    VecGetArray(null, &a);
    for (size_t i = 0; i < rhs.size(); i++) {
      if (i % blockSize == blockSize - 1)
        a[i] = 1.0;
      else
        a[i] = 0.0;
    }
    VecRestoreArray(null, &a);

    MatNullSpace nullspace;
    MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &null, &nullspace);
    MatSetNullSpace(__shell_mat, nullspace);

    KSP _ksp;
    KSPCreate(PETSC_COMM_WORLD, &_ksp);
    KSPSetOperators(_ksp, __shell_mat, __shell_mat);
    KSPSetFromOptions(_ksp);

    PC _pc;
    KSPGetPC(_ksp, &_pc);
    PCSetType(_pc, PCSHELL);

    HypreConstConstraintPC *shell_ctx;
    HypreConstConstraintPCCreate(&shell_ctx);

    PCShellSetApply(_pc, HypreConstConstraintPCApply);
    PCShellSetContext(_pc, shell_ctx);
    PCShellSetDestroy(_pc, HypreConstConstraintPCDestroy);

    HypreConstConstraintPCSetUp(_pc, &__mat, blockSize);

    KSPSetUp(_ksp);

    PetscPrintf(PETSC_COMM_WORLD, "final solving of linear system\n");
    KSPSolve(_ksp, _rhs, _x);
    MPI_Barrier(MPI_COMM_WORLD);

    KSPDestroy(&_ksp);

    VecGetArray(_x, &a);
    for (size_t i = 0; i < rhs.size(); i++) {
      x[i] = a[i];
    }
    VecRestoreArray(_x, &a);

    VecDestroy(&_rhs);
    VecDestroy(&_x);
    VecDestroy(&null);

    MatSetNearNullSpace(__mat, NULL);
    MatNullSpaceDestroy(&nullspace);
  }
}

void PetscSparseMatrix::Solve(vector<double> &rhs, vector<double> &x,
                              int dimension, int numRigidBody) {
  int fieldDof = dimension + 1;
  int velocityDof = dimension;
  int pressureDof = 1;
  int rigidBodyDof = (dimension == 3) ? 6 : 3;

  vector<int> idx_field, idx_rigid;
  vector<int> idx_velocity, idx_pressure;

  int MPIsize, myId;
  MPI_Comm_rank(MPI_COMM_WORLD, &myId);
  MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

  PetscInt localN1, localN2;
  MatGetOwnershipRange(__mat, &localN1, &localN2);

  if (myId != MPIsize - 1) {
    int localParticleNum = (localN2 - localN1) / fieldDof;
    idx_field.resize(fieldDof * localParticleNum);
    idx_velocity.resize(velocityDof * localParticleNum);
    idx_pressure.resize(localParticleNum);

    for (int i = 0; i < localParticleNum; i++) {
      for (int j = 0; j < dimension; j++) {
        idx_field[fieldDof * i + j] = localN1 + fieldDof * i + j;
        idx_velocity[velocityDof * i + j] = localN1 + fieldDof * i + j;
      }
      idx_field[fieldDof * i + velocityDof] =
          localN1 + fieldDof * i + velocityDof;
      idx_pressure[i] = localN1 + fieldDof * i + velocityDof;
    }
  } else {
    int localParticleNum =
        (localN2 - localN1 - 1 - numRigidBody * rigidBodyDof) / fieldDof;
    idx_field.resize(fieldDof * localParticleNum + 1);
    idx_velocity.resize(velocityDof * localParticleNum);
    idx_pressure.resize(localParticleNum + 1);
    idx_rigid.resize(rigidBodyDof * numRigidBody);

    for (int i = 0; i < localParticleNum; i++) {
      for (int j = 0; j < dimension; j++) {
        idx_field[fieldDof * i + j] = localN1 + fieldDof * i + j;
        idx_velocity[velocityDof * i + j] = localN1 + fieldDof * i + j;
      }
      idx_field[fieldDof * i + velocityDof] =
          localN1 + fieldDof * i + velocityDof;
      idx_pressure[i] = localN1 + fieldDof * i + velocityDof;
    }

    idx_field[fieldDof * localParticleNum] =
        localN1 + fieldDof * localParticleNum;
    idx_pressure[localParticleNum] = localN1 + fieldDof * localParticleNum;

    for (int i = 0; i < numRigidBody; i++) {
      for (int j = 0; j < rigidBodyDof; j++) {
        idx_rigid[rigidBodyDof * i + j] =
            localN1 + fieldDof * localParticleNum + 1 + i * rigidBodyDof + j;
      }
    }
  }

  IS isg_field, isg_rigid;
  IS isg_velocity, isg_pressure;

  ISCreateGeneral(MPI_COMM_WORLD, idx_velocity.size(), idx_velocity.data(),
                  PETSC_COPY_VALUES, &isg_velocity);
  ISCreateGeneral(MPI_COMM_WORLD, idx_pressure.size(), idx_pressure.data(),
                  PETSC_COPY_VALUES, &isg_pressure);
  ISCreateGeneral(MPI_COMM_WORLD, idx_field.size(), idx_field.data(),
                  PETSC_COPY_VALUES, &isg_field);
  ISCreateGeneral(MPI_COMM_WORLD, idx_rigid.size(), idx_rigid.data(),
                  PETSC_COPY_VALUES, &isg_rigid);

  Mat ff, fr, rf, fr_s;

  Mat uu, up, pu, pp, up_s;

  MatCreateSubMatrix(__mat, isg_field, isg_field, MAT_INITIAL_MATRIX, &ff);
  MatCreateSubMatrix(__mat, isg_field, isg_rigid, MAT_INITIAL_MATRIX, &fr);
  MatCreateSubMatrix(__mat, isg_rigid, isg_field, MAT_INITIAL_MATRIX, &rf);

  MatCreateSubMatrix(__mat, isg_velocity, isg_velocity, MAT_INITIAL_MATRIX,
                     &uu);
  MatCreateSubMatrix(__mat, isg_pressure, isg_velocity, MAT_INITIAL_MATRIX,
                     &pu);
  MatCreateSubMatrix(__mat, isg_velocity, isg_pressure, MAT_INITIAL_MATRIX,
                     &up);
  MatCreateSubMatrix(__mat, isg_pressure, isg_pressure, MAT_INITIAL_MATRIX,
                     &pp);

  Vec diag;

  MatCreateVecs(ff, &diag, NULL);
  MatGetDiagonal(ff, diag);
  VecReciprocal(diag);
  MatDiagonalScale(fr, diag, NULL);
  MatMatMult(rf, fr, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &fr_s);
  MatScale(fr_s, -1.0);

  MatCreateVecs(uu, &diag, NULL);
  MatGetDiagonal(uu, diag);
  VecReciprocal(diag);
  MatDiagonalScale(up, diag, NULL);
  MatMatMult(pu, up, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &up_s);
  MatScale(up_s, -1.0);
  MatAXPY(up_s, 1.0, pp, DIFFERENT_NONZERO_PATTERN);

  Vec _rhs, _x;
  VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, rhs.size(), PETSC_DECIDE,
                        rhs.data(), &_rhs);
  VecDuplicate(_rhs, &_x);

  KSP _ksp;
  KSPCreate(PETSC_COMM_WORLD, &_ksp);
  KSPSetOperators(_ksp, __mat, __mat);
  KSPSetFromOptions(_ksp);

  PC _pc;

  KSPGetPC(_ksp, &_pc);
  PCFieldSplitSetIS(_pc, "0", isg_field);
  PCFieldSplitSetIS(_pc, "1", isg_rigid);

  PCFieldSplitSetSchurPre(_pc, PC_FIELDSPLIT_SCHUR_PRE_USER, fr_s);

  KSP *_fieldKsp;
  PetscInt n = 1;
  PCSetUp(_pc);
  PCFieldSplitGetSubKSP(_pc, &n, &_fieldKsp);
  KSPSetOperators(_fieldKsp[1], fr_s, fr_s);
  KSPSetFromOptions(_fieldKsp[0]);
  KSPSetOperators(_fieldKsp[0], ff, ff);

  PC subpc;
  KSPGetPC(_fieldKsp[0], &subpc);

  PCFieldSplitSetIS(subpc, "0", isg_velocity);
  PCFieldSplitSetIS(subpc, "1", isg_pressure);

  PCFieldSplitSetSchurPre(subpc, PC_FIELDSPLIT_SCHUR_PRE_USER, up_s);

  // setup sub solver
  KSP *_fieldsubKsp;
  PCSetFromOptions(subpc);
  PCSetUp(subpc);
  PCFieldSplitGetSubKSP(subpc, &n, &_fieldsubKsp);
  KSPSetOperators(_fieldsubKsp[1], up_s, up_s);
  KSPSetFromOptions(_fieldsubKsp[0]);
  PetscFree(_fieldsubKsp);

  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "final solving of linear system\n");
  KSPSolve(_ksp, _rhs, _x);
  PetscPrintf(PETSC_COMM_WORLD, "ksp solving finished\n");

  KSPDestroy(&_ksp);

  PetscScalar *a;
  VecGetArray(_x, &a);
  for (size_t i = 0; i < rhs.size(); i++) {
    x[i] = a[i];
  }

  VecDestroy(&_rhs);
  VecDestroy(&_x);
  VecDestroy(&diag);

  MatDestroy(&ff);
  MatDestroy(&fr);
  MatDestroy(&rf);
  MatDestroy(&fr_s);

  MatDestroy(&uu);
  MatDestroy(&up);
  MatDestroy(&pu);
  MatDestroy(&pp);
  MatDestroy(&up_s);

  ISDestroy(&isg_field);
  ISDestroy(&isg_rigid);
  ISDestroy(&isg_velocity);
  ISDestroy(&isg_pressure);
}

void Solve(PetscSparseMatrix &A, PetscSparseMatrix &Bt, PetscSparseMatrix &B,
           PetscSparseMatrix &C, vector<double> &f, vector<double> &g,
           vector<double> &x, vector<double> &y, int numRigid,
           int rigidBodyDof) {
  if (!A.__isAssembled || !Bt.__isAssembled || !B.__isAssembled ||
      !C.__isAssembled)
    return;

  // setup rhs
  Vec _bpSub[2], _xpSub[2], _bp, _xp;
  VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, f.size(), PETSC_DECIDE, f.data(),
                        &_bpSub[0]);
  VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, g.size(), PETSC_DECIDE, g.data(),
                        &_bpSub[1]);

  // setup matrix
  Mat _ASub[5], _A;

  VecCreateNest(PETSC_COMM_WORLD, 2, NULL, _bpSub, &_bp);
  MatDuplicate(A.__mat, MAT_COPY_VALUES, &_ASub[0]);
  MatDuplicate(Bt.__mat, MAT_COPY_VALUES, &_ASub[1]);
  MatDuplicate(B.__mat, MAT_COPY_VALUES, &_ASub[2]);
  MatDuplicate(C.__mat, MAT_COPY_VALUES, &_ASub[3]);
  MatCreateNest(MPI_COMM_WORLD, 2, NULL, 2, NULL, _ASub, &_A);

  // setup schur complement approximation
  Vec _diag;
  MatCreateVecs(_ASub[0], &_diag, NULL);

  MatGetDiagonal(_ASub[0], _diag);
  VecReciprocal(_diag);
  MatDiagonalScale(_ASub[1], _diag, NULL);

  MatMatMult(_ASub[2], _ASub[1], MAT_INITIAL_MATRIX, PETSC_DEFAULT, _ASub + 4);

  MatScale(_ASub[4], -1.0);
  MatAXPY(_ASub[4], 1.0, _ASub[3], DIFFERENT_NONZERO_PATTERN);

  MatGetDiagonal(_ASub[0], _diag);
  MatDiagonalScale(_ASub[1], _diag, NULL);
  VecDestroy(&_diag);

  PetscPrintf(PETSC_COMM_WORLD, "Setup Schur Complement Preconditioner\n");

  // setup ksp
  KSP _ksp;
  PC _pc;

  KSPCreate(PETSC_COMM_WORLD, &_ksp);
  KSPSetOperators(_ksp, _A, _A);
  KSPSetFromOptions(_ksp);

  IS _isg[2];

  PetscFunctionBeginUser;

  // setup index sets
  MatNestGetISs(_A, _isg, NULL);

  PetscFunctionBeginUser;
  KSPGetPC(_ksp, &_pc);
  PCFieldSplitSetIS(_pc, "0", _isg[0]);
  PCFieldSplitSetIS(_pc, "1", _isg[1]);

  PCFieldSplitSetSchurPre(_pc, PC_FIELDSPLIT_SCHUR_PRE_USER, _ASub[4]);

  // setup sub solver
  KSP *_fieldKsp;
  PetscInt n = 1;
  PCSetUp(_pc);
  PCFieldSplitGetSubKSP(_pc, &n, &_fieldKsp);
  KSPSetOperators(_fieldKsp[1], _ASub[4], _ASub[4]);
  KSPSetFromOptions(_fieldKsp[0]);
  KSPSetOperators(_fieldKsp[0], _ASub[0], _ASub[0]);

  int MPIsize, myId;
  MPI_Comm_rank(MPI_COMM_WORLD, &myId);
  MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

  vector<PetscInt> idx1, idx2;
  PetscInt localN1, localN2;
  MatGetOwnershipRange(_ASub[0], &localN1, &localN2);
  if (myId != MPIsize - 1) {
    idx1.resize(localN2 - localN1);
    idx2.resize(0);
    for (int i = 0; i < localN2 - localN1; i++) {
      idx1[i] = localN1 + i;
    }
  } else {
    idx1.resize(localN2 - localN1 - rigidBodyDof * numRigid);
    idx2.resize(rigidBodyDof * numRigid);

    for (int i = 0; i < localN2 - localN1 - rigidBodyDof * numRigid; i++) {
      idx1[i] = localN1 + i;
    }
    for (int i = 0; i < rigidBodyDof * numRigid; i++) {
      idx2[i] = localN2 - rigidBodyDof * numRigid + i;
    }
  }

  IS isg1, isg2;
  ISCreateGeneral(MPI_COMM_WORLD, idx1.size(), idx1.data(), PETSC_COPY_VALUES,
                  &isg1);
  ISCreateGeneral(MPI_COMM_WORLD, idx2.size(), idx2.data(), PETSC_COPY_VALUES,
                  &isg2);

  Mat sub_A, sub_Bt, sub_B, sub_S;

  MatCreateSubMatrix(_ASub[0], isg1, isg1, MAT_INITIAL_MATRIX, &sub_A);
  MatCreateSubMatrix(_ASub[0], isg1, isg2, MAT_INITIAL_MATRIX, &sub_Bt);
  MatCreateSubMatrix(_ASub[0], isg2, isg1, MAT_INITIAL_MATRIX, &sub_B);

  MatCreateVecs(sub_A, &_diag, NULL);

  MatGetDiagonal(sub_A, _diag);
  VecReciprocal(_diag);
  MatDiagonalScale(sub_Bt, _diag, NULL);

  MatMatMult(sub_B, sub_Bt, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &sub_S);

  MatScale(sub_S, -1.0);

  PC subpc;
  KSPGetPC(_fieldKsp[0], &subpc);

  PCFieldSplitSetIS(subpc, "0", isg1);
  PCFieldSplitSetIS(subpc, "1", isg2);

  PCFieldSplitSetSchurPre(subpc, PC_FIELDSPLIT_SCHUR_PRE_USER, sub_S);

  // setup sub solver
  KSP *_fieldsubKsp;
  PCSetFromOptions(subpc);
  PCSetUp(subpc);
  PCFieldSplitGetSubKSP(subpc, &n, &_fieldsubKsp);
  KSPSetOperators(_fieldsubKsp[1], sub_S, sub_S);
  KSPSetFromOptions(_fieldsubKsp[0]);
  PetscFree(_fieldsubKsp);

  // vector<PetscInt> idx3, idx4;
  // MatGetOwnershipRange(_ASub[4], &localN1, &localN2);
  // if (myId != MPIsize - 1) {
  //   idx3.resize(localN2 - localN1);
  //   idx4.resize(0);
  //   for (int i = 0; i < localN2 - localN1; i++) {
  //     idx3[i] = localN1 + i;
  //   }
  // } else {
  //   idx3.resize(localN2 - localN1 - 1);
  //   idx4.resize(1);

  //   for (int i = 0; i < localN2 - localN1 - 1; i++) {
  //     idx3[i] = localN1 + i;
  //   }
  //   idx4[0] = localN2 - 1;
  // }

  // IS isg3, isg4;
  // ISCreateGeneral(MPI_COMM_WORLD, idx3.size(), idx3.data(),
  // PETSC_COPY_VALUES,
  //                 &isg3);
  // ISCreateGeneral(MPI_COMM_WORLD, idx4.size(), idx4.data(),
  // PETSC_COPY_VALUES,
  //                 &isg4);

  // Mat S_A, S_Bt, S_B, S_S;

  // MatCreateSubMatrix(_ASub[4], isg3, isg3, MAT_INITIAL_MATRIX, &S_A);
  // MatCreateSubMatrix(_ASub[4], isg3, isg4, MAT_INITIAL_MATRIX, &S_Bt);
  // MatCreateSubMatrix(_ASub[4], isg4, isg3, MAT_INITIAL_MATRIX, &S_B);

  // MatCreateVecs(S_A, &_diag, NULL);

  // MatGetDiagonal(S_A, _diag);
  // VecReciprocal(_diag);
  // MatDiagonalScale(S_Bt, _diag, NULL);

  // MatMatMult(S_B, S_Bt, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &S_S);

  // MatScale(S_S, -1.0);

  // PC subpc_s;
  // KSPGetPC(_fieldKsp[1], &subpc_s);

  // PCFieldSplitSetIS(subpc_s, "0", isg3);
  // PCFieldSplitSetIS(subpc_s, "1", isg4);

  // PCFieldSplitSetSchurPre(subpc_s, PC_FIELDSPLIT_SCHUR_PRE_USER, S_S);

  // // setup sub solver
  // KSP *_fieldsubKsp_s;
  // PCSetFromOptions(subpc_s);
  // PCSetUp(subpc_s);
  // PCFieldSplitGetSubKSP(subpc_s, &n, &_fieldsubKsp_s);
  // KSPSetOperators(_fieldsubKsp_s[1], S_S, S_S);
  // KSPSetFromOptions(_fieldsubKsp_s[0]);
  // PetscFree(_fieldsubKsp_s);
  PetscFree(_fieldKsp);

  MPI_Barrier(MPI_COMM_WORLD);

  VecDuplicate(_bp, &_xp);

  // final solve
  PetscPrintf(PETSC_COMM_WORLD, "final solving of linear system\n");
  KSPSolve(_ksp, _bp, _xp);
  PetscPrintf(PETSC_COMM_WORLD, "finish ksp solving\n");

  KSPDestroy(&_ksp);

  // copy result
  VecGetSubVector(_xp, _isg[0], &_xpSub[0]);
  VecGetSubVector(_xp, _isg[1], &_xpSub[1]);

  double *p;
  VecGetArray(_xpSub[0], &p);
  for (size_t i = 0; i < f.size(); i++) {
    x[i] = p[i];
  }
  VecRestoreArray(_xpSub[0], &p);
  VecGetArray(_xpSub[1], &p);
  for (size_t i = 0; i < g.size(); i++) {
    y[i] = p[i];
  }
  VecRestoreArray(_xpSub[1], &p);

  VecRestoreSubVector(_xp, _isg[0], &_xpSub[0]);
  VecRestoreSubVector(_xp, _isg[1], &_xpSub[1]);

  // destroy objects
  VecDestroy(_bpSub);
  VecDestroy(_bpSub + 1);
  VecDestroy(&_bp);
  VecDestroy(&_xp);

  for (int i = 0; i < 5; i++) {
    MatDestroy(_ASub + i);
  }

  MatDestroy(&_A);

  MatDestroy(&sub_A);
  MatDestroy(&sub_Bt);
  MatDestroy(&sub_B);
  MatDestroy(&sub_S);

  ISDestroy(&isg1);
  ISDestroy(&isg2);

  VecDestroy(&_diag);
}

void PetscSparseMatrix::Solve(vector<double> &rhs, vector<double> &x,
                              vector<int> &idx_neighbor, int dimension,
                              int numRigidBody, int adatptive_step,
                              PetscSparseMatrix &I, PetscSparseMatrix &R) {
  int fieldDof = dimension + 1;
  int velocityDof = dimension;
  int pressureDof = 1;
  int rigidBodyDof = (dimension == 3) ? 6 : 3;

  vector<int> idx_field;
  vector<int> idx_global;

  int MPIsize, myId;
  MPI_Comm_rank(MPI_COMM_WORLD, &myId);
  MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

  PetscInt localN1, localN2;
  MatGetOwnershipRange(__mat, &localN1, &localN2);

  if (myId != MPIsize - 1) {
    int localParticleNum = (localN2 - localN1) / fieldDof;
    idx_field.resize(fieldDof * localParticleNum);

    for (int i = 0; i < localParticleNum; i++) {
      for (int j = 0; j < dimension; j++) {
        idx_field[fieldDof * i + j] = localN1 + fieldDof * i + j;
      }
      idx_field[fieldDof * i + velocityDof] =
          localN1 + fieldDof * i + velocityDof;
    }

    idx_global = idx_field;
  } else {
    int localParticleNum =
        (localN2 - localN1 - 1 - numRigidBody * rigidBodyDof) / fieldDof + 1;
    idx_field.resize(fieldDof * localParticleNum);

    for (int i = 0; i < localParticleNum; i++) {
      for (int j = 0; j < dimension; j++) {
        idx_field[fieldDof * i + j] = localN1 + fieldDof * i + j;
      }
      idx_field[fieldDof * i + velocityDof] =
          localN1 + fieldDof * i + velocityDof;
    }

    idx_global = idx_field;

    // idx_field.push_back(localN1 + fieldDof * localParticleNum +
    // velocityDof);
  }

  IS isg_field, isg_neighbor;
  IS isg_global;

  ISCreateGeneral(MPI_COMM_WORLD, idx_field.size(), idx_field.data(),
                  PETSC_COPY_VALUES, &isg_field);
  ISCreateGeneral(MPI_COMM_WORLD, idx_neighbor.size(), idx_neighbor.data(),
                  PETSC_COPY_VALUES, &isg_neighbor);
  ISCreateGeneral(MPI_COMM_WORLD, idx_global.size(), idx_global.data(),
                  PETSC_COPY_VALUES, &isg_global);

  Vec _rhs, _x;
  VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, rhs.size(), PETSC_DECIDE,
                        rhs.data(), &_rhs);
  VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, x.size(), PETSC_DECIDE, x.data(),
                        &_x);

  Mat ff, nn, gg;

  MatCreateSubMatrix(__mat, isg_field, isg_field, MAT_INITIAL_MATRIX, &ff);
  MatCreateSubMatrix(__mat, isg_neighbor, isg_neighbor, MAT_INITIAL_MATRIX,
                     &nn);
  MatCreateSubMatrix(__mat, isg_global, isg_global, MAT_INITIAL_MATRIX, &gg);

  MatSetBlockSize(ff, fieldDof);

  KSP _ksp;
  KSPCreate(PETSC_COMM_WORLD, &_ksp);
  KSPSetOperators(_ksp, __mat, __mat);
  KSPSetFromOptions(_ksp);

  PC _pc;

  KSPGetPC(_ksp, &_pc);
  PCSetType(_pc, PCSHELL);

  HypreLUShellPC *shell_ctx;
  HypreLUShellPCCreate(&shell_ctx);
  if (adatptive_step == 0) {
    PCShellSetApply(_pc, HypreLUShellPCApply);
    PCShellSetContext(_pc, shell_ctx);
    PCShellSetDestroy(_pc, HypreLUShellPCDestroy);

    // HypreLUShellPCSetUp(_pc, &__mat, &ff, &nn, &isg_field, &isg_neighbor,
    // _x);
  } else {
    PCShellSetApply(_pc, HypreLUShellPCApply);
    PCShellSetContext(_pc, shell_ctx);
    PCShellSetDestroy(_pc, HypreLUShellPCDestroy);

    // HypreLUShellPCSetUp(_pc, &__mat, &ff, &nn, &isg_field, &isg_neighbor,
    // _x);
  }

  if (adatptive_step > 0) {
    KSP smoother_ksp;
    KSPCreate(PETSC_COMM_WORLD, &smoother_ksp);
    KSPSetOperators(smoother_ksp, nn, nn);
    KSPSetType(smoother_ksp, KSPPREONLY);

    PC smoother_pc;
    KSPGetPC(smoother_ksp, &smoother_pc);
    PCSetType(smoother_pc, PCLU);
    PCSetFromOptions(smoother_pc);

    PCSetUp(smoother_pc);
    KSPSetUp(smoother_ksp);

    Vec r, delta_x;
    VecDuplicate(_x, &r);
    VecDuplicate(_x, &delta_x);
    MatMult(__mat, _x, r);
    VecAXPY(r, -1.0, _rhs);

    // KSPSolve(smoother_ksp, r, delta_x);
    // VecAXPY(_x, -1.0, delta_x);

    // KSPSetInitialGuessNonzero(smoother_ksp, PETSC_TRUE);

    Vec r_f, x_f, delta_x_f;
    VecGetSubVector(r, isg_neighbor, &r_f);
    VecGetSubVector(_x, isg_neighbor, &x_f);
    VecDuplicate(x_f, &delta_x_f);
    KSPSolve(smoother_ksp, r_f, delta_x_f);
    VecAXPY(x_f, -1.0, delta_x_f);
    VecRestoreSubVector(_rhs, isg_neighbor, &r_f);
    VecRestoreSubVector(_x, isg_neighbor, &x_f);
  }

  Vec x_initial;
  if (adatptive_step > 0) {
    VecDuplicate(_x, &x_initial);
    VecCopy(_x, x_initial);
  }

  KSPSetInitialGuessNonzero(_ksp, PETSC_TRUE);

  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "final solving of linear system\n");
  KSPSolve(_ksp, _rhs, _x);
  PetscPrintf(PETSC_COMM_WORLD, "ksp solving finished\n");

  // if (adatptive_step > 0) {
  //   VecAXPY(_x, -1.0, x_initial);
  //   VecAbs(_x);
  // }

  KSPDestroy(&_ksp);

  PetscScalar *a;
  VecGetArray(_x, &a);
  for (size_t i = 0; i < rhs.size(); i++) {
    x[i] = a[i];
  }
  VecRestoreArray(_x, &a);

  VecDestroy(&_rhs);
  VecDestroy(&_x);

  ISDestroy(&isg_field);
  ISDestroy(&isg_neighbor);
  ISDestroy(&isg_global);

  MatDestroy(&ff);
  MatDestroy(&nn);
  MatDestroy(&gg);
}

PetscErrorCode fluid_colloid_matrix_mult(Mat mat, Vec x, Vec y) {
  fluid_colloid_matrix_context *ctx;
  MatShellGetContext(mat, &ctx);

  PetscReal *a, *b;

  VecGetArray(y, &a);

  MatMult(ctx->colloid_part, x, ctx->colloid_vec);

  VecGetArray(ctx->colloid_vec, &b);

  MPI_Reduce(b, a + ctx->fluid_local_size, ctx->rigid_body_size, MPI_DOUBLE,
             MPI_SUM, ctx->mpisize - 1, MPI_COMM_WORLD);

  VecRestoreArray(ctx->colloid_vec, &b);

  MatMult(ctx->fluid_part, x, ctx->fluid_vec);

  VecGetArray(ctx->fluid_vec, &b);
  for (int i = 0; i < ctx->fluid_local_size; i++)
    a[i] = b[i];

  PetscReal pressure_sum = 0.0;
  for (int i = 0; i < ctx->local_fluid_particle_num; i++) {
    pressure_sum += a[i * ctx->field_dof + ctx->pressure_offset];
  }
  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  PetscReal average_pressure = pressure_sum / ctx->global_fluid_particle_num;
  for (int i = 0; i < ctx->local_fluid_particle_num; i++) {
    a[i * ctx->field_dof + ctx->pressure_offset] -= average_pressure;
  }

  VecRestoreArray(y, &a);
  VecRestoreArray(ctx->fluid_vec, &b);

  return 0;
}