#include "stokes_composite_preconditioner.hpp"

#include "petsc_ksp.hpp"
#include "petsc_sparse_matrix.hpp"

#include <fstream>
#include <string>

using namespace std;

// group size comparison
inline bool compare_group(vector<int> group1, vector<int> group2) {
  return group1.size() > group2.size();
}

int petsc_sparse_matrix::write(string fileName) {
  ofstream output;
  int MPIsize, myID;
  MPI_Comm_rank(MPI_COMM_WORLD, &myID);
  MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

  if (myID == 0) {
    output.open(fileName, ios::trunc);
    output.close();
  }

  int send_count = __row;
  vector<int> recv_count;
  recv_count.resize(MPIsize);

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Allgather(&send_count, 1, MPI_INT, recv_count.data(), 1, MPI_INT,
                MPI_COMM_WORLD);

  vector<int> displs;
  displs.resize(MPIsize + 1);
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
      output.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  return 0;
}

bool petsc_sparse_matrix::increment(const PetscInt i, const PetscInt j,
                                    const double daij) {
  auto it = lower_bound(__matrix[i].begin(), __matrix[i].end(), entry(j, daij),
                        compare_index);
  if (j > __Col) {
    std::cout << i << ' ' << j << " increment wrong column index" << std::endl;
    return false;
  }

  if (it->first == j)
    it->second += daij;
  else {
    std::cout << i << ' ' << j << " increment misplacement" << std::endl;
    return false;
  }

  return true;
}

void petsc_sparse_matrix::set(const PetscInt i, const PetscInt j,
                              const double daij) {
  if (std::abs(daij) > 1e-15) {
    auto it = lower_bound(__matrix[i].begin(), __matrix[i].end(),
                          entry(j, daij), compare_index);
    if (j > __Col) {
      std::cout << i << ' ' << j << " increment wrong column index"
                << std::endl;
      return;
    }

    if (it->first == j)
      it->second = daij;
    else
      std::cout << i << ' ' << j << " increment misplacement" << std::endl;
  }
}

void petsc_sparse_matrix::out_process_increment(const PetscInt i,
                                                const PetscInt j,
                                                const double daij) {
  if (std::abs(daij) > 1e-15) {
    PetscInt in = i - __out_process_reduction;
    auto it = lower_bound(__out_process_matrix[in].begin(),
                          __out_process_matrix[in].end(), entry(j, daij),
                          compare_index);
    if (j > __Col) {
      std::cout << i << ' ' << j << " out process wrong column index"
                << std::endl;
      return;
    }

    if (it != __out_process_matrix[in].end() && it->first == j)
      it->second += daij;
    else
      std::cout << in << ' ' << j << " out process increment misplacement"
                << std::endl;
  }
}

int petsc_sparse_matrix::assemble() {
  // move data from out_process_increment
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

  is_assembled = true;

  {
    decltype(__i)().swap(__i);
    decltype(__j)().swap(__j);
    decltype(__val)().swap(__val);

    decltype(__matrix)().swap(__matrix);
  }

  return __nnz;
}

int petsc_sparse_matrix::assemble(int block_size) {
  // move data from out_process_increment
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
  auto block_row = __row / block_size;

  __i.resize(block_row + 1);
  __j.clear();

  __nnz = 0;
  auto nnz_block = __nnz;
  vector<PetscInt> block_col_indices;
  __i[0] = 0;
  for (int i = 0; i < block_row; i++) {
    block_col_indices.clear();
    for (int j = 0; j < block_size; j++) {
      for (int k = 0; k < __matrix[i * block_size + j].size(); k++) {
        if (__matrix[i * block_size + j][k].first < __Col)
          block_col_indices.push_back(__matrix[i * block_size + j][k].first /
                                      block_size);
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

  auto blockStorage = block_size * block_size;

  __val.resize(nnz_block * blockStorage);

  for (int i = 0; i < nnz_block * blockStorage; i++) {
    __val[i] = 0.0;
  }

  for (int i = 0; i < __row; i++) {
    int block_row_index = i / block_size;
    int local_row_index = i % block_size;
    for (int j = 0; j < __matrix[i].size(); j++) {
      if (__matrix[i][j].first < __Col) {
        int block_col_index = __matrix[i][j].first / block_size;
        int local_col_index = __matrix[i][j].first % block_size;

        auto it = lower_bound(__j.begin() + __i[block_row_index],
                              __j.begin() + __i[block_row_index + 1],
                              block_col_index);

        auto disp = it - __j.begin();
        __val[blockStorage * disp + local_col_index +
              local_row_index * block_size] = __matrix[i][j].second;
      }
    }
  }

  // MatCreateMPIBAIJWithArray is incompatible with current code setup
  if (__Col != 0) {
    MatCreate(MPI_COMM_WORLD, &__mat);
    MatSetSizes(__mat, __row, __col, PETSC_DECIDE, __Col);
    MatSetType(__mat, MATMPIBAIJ);
    MatSetBlockSize(__mat, block_size);
    MatSetUp(__mat);
    MatMPIBAIJSetPreallocationCSR(__mat, block_size, __i.data(), __j.data(),
                                  __val.data());
  } else {
    MatCreate(MPI_COMM_WORLD, &__mat);
    MatSetSizes(__mat, __row, __col, PETSC_DECIDE, PETSC_DECIDE);
    MatSetType(__mat, MATMPIBAIJ);
    MatSetBlockSize(__mat, block_size);
    MatSetUp(__mat);
    MatMPIBAIJSetPreallocationCSR(__mat, block_size, __i.data(), __j.data(),
                                  __val.data());
  }

  is_assembled = true;

  {
    decltype(__i)().swap(__i);
    decltype(__j)().swap(__j);
    decltype(__val)().swap(__val);

    decltype(__matrix)().swap(__matrix);
  }

  return __nnz;
}

int petsc_sparse_matrix::assemble(int block_size, int num_rigid_body,
                                  int rigid_body_dof) {}

int petsc_sparse_matrix::assemble(petsc_sparse_matrix &pmat, int block_size,
                                  int num_rigid_body, int rigid_body_dof) {
  // move data from out_process_increment
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

  auto block_row = row_block / block_size;

  __i.resize(block_row + 1);
  __j.clear();

  vector<PetscInt> block_col_indices;
  __i[0] = 0;
  int nnz_block = 0;
  for (int i = 0; i < block_row; i++) {
    block_col_indices.clear();
    for (int j = 0; j < block_size; j++) {
      for (int k = 0; k < __matrix[i * block_size + j].size(); k++) {
        if (__matrix[i * block_size + j][k].first < Col_block)
          block_col_indices.push_back(__matrix[i * block_size + j][k].first /
                                      block_size);
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

  auto blockStorage = block_size * block_size;

  __val.resize(nnz_block * blockStorage);

  for (int i = 0; i < nnz_block * blockStorage; i++) {
    __val[i] = 0.0;
  }

  for (int i = 0; i < row_block; i++) {
    int block_row_index = i / block_size;
    int local_row_index = i % block_size;
    for (int j = 0; j < __matrix[i].size(); j++) {
      if (__matrix[i][j].first < Col_block) {
        int block_col_index = __matrix[i][j].first / block_size;
        int local_col_index = __matrix[i][j].first % block_size;

        auto it = lower_bound(__j.begin() + __i[block_row_index],
                              __j.begin() + __i[block_row_index + 1],
                              block_col_index);

        auto disp = it - __j.begin();
        __val[blockStorage * disp + local_col_index +
              local_row_index * block_size] = __matrix[i][j].second;
      }
    }
  }

  Mat &mat = pmat.get_reference();

  PetscLogDouble mem;

  if (Col_block != 0) {
    MatCreate(MPI_COMM_WORLD, &mat);
    MatSetSizes(mat, row_block, col_block, PETSC_DECIDE, Col_block);
    MatSetType(mat, MATMPIBAIJ);
    MatSetBlockSize(mat, block_size);
    MatSetUp(mat);
    MatMPIBAIJSetPreallocationCSR(mat, block_size, __i.data(), __j.data(),
                                  __val.data());

    __ctx.fluid_part = &mat;
    pmat.__ctx.fluid_part = &mat;
  } else {
    MatCreate(MPI_COMM_WORLD, &mat);
    MatSetSizes(mat, row_block, col_block, PETSC_DECIDE, PETSC_DECIDE);
    MatSetType(mat, MATMPIBAIJ);
    MatSetBlockSize(mat, block_size);
    MatSetUp(mat);
    MatMPIBAIJSetPreallocationCSR(mat, block_size, __i.data(), __j.data(),
                                  __val.data());

    __ctx.fluid_part = &mat;
    pmat.__ctx.fluid_part = &mat;
  }

  pmat.is_assembled = true;

  Mat &shell_mat = pmat.get_shell_reference();
  MatCreateShell(PETSC_COMM_WORLD, row_block, col_block, PETSC_DECIDE,
                 Col_block, &pmat.__ctx, &shell_mat);
  MatShellSetOperation(shell_mat, MATOP_MULT,
                       (void (*)(void))fluid_matrix_mult);

  pmat.__ctx.fluid_local_size = row_block;
  pmat.__ctx.local_fluid_particle_num = block_row;
  pmat.__ctx.field_dof = block_size;
  MPI_Allreduce(&(pmat.__ctx.local_fluid_particle_num),
                &(pmat.__ctx.global_fluid_particle_num), 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  pmat.__ctx.pressure_offset = block_size - 1;

  pmat.__ctx.myid = myid;
  pmat.__ctx.mpisize = MPIsize;

  pmat.is_shell_assembled = true;

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
  VecCreateSeq(PETSC_COMM_SELF, num_rigid_body * rigid_body_dof,
               &(__ctx.colloid_vec_local));
  VecCreateMPI(PETSC_COMM_WORLD, local_size, PETSC_DECIDE, &(__ctx.fluid_vec1));
  VecCreateMPI(PETSC_COMM_WORLD, local_size, PETSC_DECIDE, &(__ctx.fluid_vec2));
  VecCreateSeq(PETSC_COMM_SELF, local_size, &(__ctx.fluid_vec_local));
  __ctx.fluid_local_size = local_size;
  __ctx.rigid_body_size = num_rigid_body * rigid_body_dof;

  __ctx.local_fluid_particle_num = local_size / block_size;
  __ctx.field_dof = block_size;
  MPI_Allreduce(&(__ctx.local_fluid_particle_num),
                &(__ctx.global_fluid_particle_num), 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  __ctx.pressure_offset = block_size - 1;

  __ctx.myid = myid;
  __ctx.mpisize = MPIsize;

  MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, __out_process_row, __col,
                            PETSC_DECIDE, __Col, colloid_part_i.data(),
                            colloid_part_j.data(), colloid_part_val.data(),
                            &(__ctx.colloid_part));

  __ctx.fluid_colloid_part_i.resize(local_size + 1);

  size_t nnz = 0;
  __ctx.fluid_colloid_part_i[0] = 0;
  for (int i = 0; i < local_size; i++) {
    for (int k = 0; k < __matrix[i].size(); k++) {
      if (__matrix[i][k].first >= Col_block) {
        nnz++;

        __ctx.fluid_colloid_part_j.push_back(__matrix[i][k].first - Col_block);
        __ctx.fluid_colloid_part_val.push_back(__matrix[i][k].second);
      }
    }
    __ctx.fluid_colloid_part_i[i + 1] = nnz;
  }

  // since this function doesn't copy the array, they could not be released at
  // this time
  MatCreateSeqAIJWithArrays(
      PETSC_COMM_SELF, local_size, num_rigid_body * rigid_body_dof,
      __ctx.fluid_colloid_part_i.data(), __ctx.fluid_colloid_part_j.data(),
      __ctx.fluid_colloid_part_val.data(), &(__ctx.fluid_colloid_part));

  MPI_Barrier(MPI_COMM_WORLD);
  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "Current memory usage after assembly of shell mat %.2f GB\n",
              mem / 1e9);

  is_shell_assembled = true;
  is_ctx_assembled = true;

  { vector<vector<entry>>().swap(__out_process_matrix); }

  MPI_Barrier(MPI_COMM_WORLD);
  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "Current memory usage after shrinking of matrix and out process "
              "matrix %.2f GB\n",
              mem / 1e9);

  return __nnz;
}

int petsc_sparse_matrix::extract_neighbor_index(
    vector<int> &idx_colloid, int dimension, int num_rigid_body,
    int local_rigid_body_offset, int global_rigid_body_offset,
    petsc_sparse_matrix &nn, petsc_sparse_matrix &nw) {

  int MPIsize, myId;
  MPI_Comm_rank(MPI_COMM_WORLD, &myId);
  MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

  int rigid_body_dof = (dimension == 2) ? 3 : 6;
  int field_dof = dimension + 1;

  idx_colloid.clear();

  vector<int> neighbor_inclusion;

  MPI_Barrier(MPI_COMM_WORLD);
  idx_colloid.clear();

  PetscInt local_N1;
  MatGetOwnershipRange(__shell_mat, &local_N1, NULL);

  neighbor_inclusion.clear();

  PetscInt Col_block, row_block, col_block;
  Col_block = __Col - num_rigid_body * rigid_body_dof;
  if (myId == MPIsize - 1) {
    row_block = __row - num_rigid_body * rigid_body_dof;
    col_block = __col - num_rigid_body * rigid_body_dof;
  } else {
    row_block = __row;
    col_block = __col;
  }

  for (int i = 0; i < row_block; i++) {
    auto it = lower_bound(__matrix[i].begin(), __matrix[i].end(),
                          entry(global_rigid_body_offset, 0.0), compare_index);
    if (it != __matrix[i].end()) {
      neighbor_inclusion.push_back((local_N1 + i) / field_dof);
    }
  }

  sort(neighbor_inclusion.begin(), neighbor_inclusion.end());
  neighbor_inclusion.erase(
      unique(neighbor_inclusion.begin(), neighbor_inclusion.end()),
      neighbor_inclusion.end());

  // move data
  vector<int> local_neighbor_inclusion_num(MPIsize);
  for (int i = 0; i < MPIsize; i++) {
    local_neighbor_inclusion_num[i] = 0;
  }
  local_neighbor_inclusion_num[myId] = neighbor_inclusion.size();

  MPI_Allreduce(MPI_IN_PLACE, local_neighbor_inclusion_num.data(), MPIsize,
                MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  int recv_num = 0;
  vector<int> displs;
  vector<int> recv_neighbor_inclusion;
  if (myId == MPIsize - 1) {
    displs.resize(MPIsize + 1);
    displs[0] = 0;
    for (int i = 0; i < MPIsize; i++) {
      recv_num += local_neighbor_inclusion_num[i];
      displs[i + 1] = displs[i] + local_neighbor_inclusion_num[i];
    }

    recv_neighbor_inclusion.resize(recv_num);
  }

  MPI_Gatherv(neighbor_inclusion.data(), neighbor_inclusion.size(), MPI_INT,
              recv_neighbor_inclusion.data(),
              local_neighbor_inclusion_num.data(), displs.data(), MPI_INT,
              MPIsize - 1, MPI_COMM_WORLD);

  if (myId == MPIsize - 1) {
    sort(recv_neighbor_inclusion.begin(), recv_neighbor_inclusion.end());
    recv_neighbor_inclusion.erase(
        unique(recv_neighbor_inclusion.begin(), recv_neighbor_inclusion.end()),
        recv_neighbor_inclusion.end());
  }

  vector<vector<entry>>().swap(__matrix);

  if (myId == MPIsize - 1) {
    neighbor_inclusion.clear();
    neighbor_inclusion.insert(
        neighbor_inclusion.end(), __j.begin() + __i[local_rigid_body_offset],
        __j.begin() +
            __i[local_rigid_body_offset + num_rigid_body * rigid_body_dof]);

    for (int i = 0; i < neighbor_inclusion.size(); i++) {
      if (neighbor_inclusion[i] < global_rigid_body_offset)
        neighbor_inclusion[i] /= field_dof;
    }

    neighbor_inclusion.insert(neighbor_inclusion.end(),
                              recv_neighbor_inclusion.begin(),
                              recv_neighbor_inclusion.end());

    sort(neighbor_inclusion.begin(), neighbor_inclusion.end());

    neighbor_inclusion.erase(
        unique(neighbor_inclusion.begin(), neighbor_inclusion.end()),
        neighbor_inclusion.end());

    auto it = neighbor_inclusion.begin();
    for (; it != neighbor_inclusion.end(); it++) {
      if (*it >= global_rigid_body_offset)
        break;
    }

    neighbor_inclusion.erase(it, neighbor_inclusion.end());
  }

  int neighbor_inclusionSize, offset = 0;

  vector<int> recvneighbor_inclusion;

  for (int i = 0; i < MPIsize; i++) {
    if (i != MPIsize - 1) {
      if (myId == MPIsize - 1) {
        neighbor_inclusionSize = neighbor_inclusion.size() / MPIsize;
        neighbor_inclusionSize +=
            (neighbor_inclusion.size() % MPIsize > i) ? 1 : 0;

        MPI_Send(&neighbor_inclusionSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        MPI_Send(neighbor_inclusion.data() + offset, neighbor_inclusionSize,
                 MPI_INT, i, 1, MPI_COMM_WORLD);
      }
      if (myId == i) {
        MPI_Status stat;
        MPI_Recv(&neighbor_inclusionSize, 1, MPI_INT, MPIsize - 1, 0,
                 MPI_COMM_WORLD, &stat);
        recvneighbor_inclusion.resize(neighbor_inclusionSize);
        MPI_Recv(recvneighbor_inclusion.data(), neighbor_inclusionSize, MPI_INT,
                 MPIsize - 1, 1, MPI_COMM_WORLD, &stat);
      }
    } else {
      if (myId == MPIsize - 1) {
        recvneighbor_inclusion.clear();
        recvneighbor_inclusion.insert(recvneighbor_inclusion.end(),
                                      neighbor_inclusion.begin() + offset,
                                      neighbor_inclusion.end());
      }
    }

    if (myId == MPIsize - 1) {
      offset += neighbor_inclusionSize;
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  idx_colloid.clear();
  for (int i = 0; i < recvneighbor_inclusion.size(); i++) {
    for (int j = 0; j < field_dof; j++) {
      idx_colloid.push_back(recvneighbor_inclusion[i] * field_dof + j);
    }
  }

  // split colloid rigid body dof to each process
  int avg_rigid_body_num = num_rigid_body / MPIsize;
  int rigid_body_idx_low = 0;
  int rigid_body_idx_high = 0;
  for (int i = 0; i < myId; i++) {
    if (i < num_rigid_body % MPIsize) {
      rigid_body_idx_low += avg_rigid_body_num + 1;
    } else {
      rigid_body_idx_low += avg_rigid_body_num;
    }
  }
  if (myId < num_rigid_body % MPIsize) {
    rigid_body_idx_high = rigid_body_idx_low + avg_rigid_body_num + 1;
  } else {
    rigid_body_idx_high = rigid_body_idx_low + avg_rigid_body_num;
  }

  for (int i = rigid_body_idx_low; i < rigid_body_idx_high; i++) {
    for (int j = 0; j < rigid_body_dof; j++) {
      idx_colloid.push_back(global_rigid_body_offset + i * rigid_body_dof + j);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // // build nn & nw matrices
  // PetscInt localN1, localN2;
  // MatGetOwnershipRange(__shell_mat, &localN1, &localN2);

  // vector<int> range;
  // range.resize(MPIsize + 1);
  // for (int i = 0; i < MPIsize; i++) {
  //   range[i] = 0;
  // }
  // range[myId] = localN1;
  // if (myId == MPIsize - 1)
  //   range[MPIsize] = localN2;
  // MPI_Allreduce(MPI_IN_PLACE, range.data(), MPIsize + 1, MPI_INT, MPI_SUM,
  //               MPI_COMM_WORLD);

  // vector<int> index;
  // vector<vector<int>> out_index, in_index;
  // vector<int> out_graph, in_graph;

  // vector<int> in_num;
  // in_num.resize(MPIsize);
  // for (int i = 0; i < MPIsize; i++) {
  //   in_num[i] = 0;
  // }
  // for (int i = 0; i < idx_colloid.size(); i++) {
  //   auto it = lower_bound(range.begin(), range.end(), idx_colloid[i]);
  //   int rank_index = (int)(it - range.begin());
  //   if (rank_index != myId)
  //     in_num[rank_index]++;
  // }
  // for (int i = 0; i < MPIsize; i++) {
  //   if (in_num[i] != 0)
  //     in_graph.push_back(i);
  // }

  // for (int i = 0; i < MPIsize; i++) {
  //   int index_num;
  //   if (i == myId) {
  //     index = idx_colloid;
  //     index_num = idx_colloid.size();
  //   }
  //   MPI_Bcast(&index_num, 1, MPI_INT, i, MPI_COMM_WORLD);
  //   if (i != myId)
  //     index.resize(index_num);
  //   MPI_Bcast(index.data(), index_num, MPI_INT, i, MPI_COMM_WORLD);

  //   if (i != myId) {
  //     for (auto it = index.begin(); it != index.end(); it++) {
  //       if (*it >= localN1 && *it < localN2) {
  //         out_index[i].push_back(*it);
  //       }
  //     }

  //     if (out_index[i].size() != 0)
  //       out_graph.push_back(i);
  //   }

  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  // vector<vector<int>> out_matrix_i, out_matrix_j;
  // vector<vector<double>> out_matrix_val;
  // vector<vector<int>> in_matrix_i, in_matrix_j;
  // vector<vector<double>> in_matrix_val;

  // for (int i = 0; i < MPIsize; i++) {
  //   out_matrix_i[i].resize(out_index[i].size() + 1);
  //   out_matrix_i[i][0] = 0;
  //   for (int j = 0; j < out_index[i].size(); j++) {
  //     int local_row_index = out_index[i][j] - localN1;
  //     out_matrix_i[i][j + 1] =
  //         out_matrix_i[i][j] + __i[local_row_index + 1] -
  //         __i[local_row_index];
  //   }
  //   out_matrix_j[i].resize(out_matrix_i[i][out_index[i].size()]);
  //   out_matrix_val[i].resize(out_matrix_i[i][out_index[i].size()]);
  //   for (int j = 0; j < out_index[i].size(); j++) {
  //     int local_row_index = out_index[i][j] - localN1;
  //     for (int k = __i[local_row_index]; k < __i[local_row_index + 1]; k++) {
  //       out_matrix_j[i][out_matrix_i[i][j] + k - __i[local_row_index]] =
  //       __j[k]; out_matrix_val[i][out_matrix_i[i][j] + k -
  //       __i[local_row_index]] =
  //           __val[k];
  //     }
  //   }
  // }

  // // collect from other processes
  // vector<int> in_collector;
  // in_collector.resize(MPIsize);
  // for (int i = 0; i < MPIsize; i++) {
  //   for (int j = 0; j < MPIsize; j++) {
  //     in_collector[j] = 0;
  //   }
  //   if (out_matrix_i[i].size() != 0)
  //     in_collector[i] = out_matrix_i[i].size();

  //   MPI_Allreduce(MPI_IN_PLACE, in_collector.data(), 1, MPI_INT, MPI_SUM,
  //                 MPI_COMM_WORLD);
  //   if (i == myId) {
  //     for (int j = 0; j < MPIsize; j++)
  //       in_matrix_i[j].resize(in_collector[j]);
  //   }

  //   if (out_matrix_i[i].size() != 0)
  //     in_collector[i] = out_matrix_i[i][out_matrix_i[i].size()];

  //   MPI_Allreduce(MPI_IN_PLACE, in_collector.data(), 1, MPI_INT, MPI_SUM,
  //                 MPI_COMM_WORLD);
  //   if (i == myId) {
  //     for (int j = 0; j < MPIsize; j++) {
  //       in_matrix_j[j].resize(in_collector[j]);
  //       in_matrix_val[j].resize(in_collector[j]);
  //     }
  //   }
  // }

  // // move data
  // vector<MPI_Request> send_request;
  // vector<MPI_Request> recv_request;

  // vector<MPI_Status> send_status;
  // vector<MPI_Status> recv_status;

  // send_request.resize(out_graph.size());
  // send_status.resize(out_graph.size());
  // recv_request.resize(in_graph.size());
  // recv_status.resize(in_graph.size());

  // // move i
  // for (int i = 0; i < out_graph.size(); i++) {
  //   MPI_Isend(out_matrix_i[out_graph[i]].data(),
  //             out_matrix_i[out_graph[i]].size(), MPI_INT, out_graph[i], 0,
  //             MPI_COMM_WORLD, send_request.data() + i);
  // }

  // for (int i = 0; i < in_graph.size(); i++) {
  //   MPI_Irecv(in_matrix_i[in_graph[i]].data(),
  //   in_matrix_i[in_graph[i]].size(),
  //             MPI_INT, in_graph[i], 0, MPI_COMM_WORLD, recv_request.data() +
  //             i);
  // }

  // MPI_Waitall(send_request.size(), send_request.data(), send_status.data());
  // MPI_Waitall(recv_request.size(), recv_request.data(), recv_status.data());
  // MPI_Barrier(MPI_COMM_WORLD);

  {
    decltype(__i)().swap(__i);
    decltype(__j)().swap(__j);
    decltype(__val)().swap(__val);
  }

  petsc_is isg_colloid;
  isg_colloid.create(idx_colloid);

  MatCreateSubMatrix(__mat, isg_colloid.get_reference(),
                     isg_colloid.get_reference(), MAT_INITIAL_MATRIX,
                     nn.get_pointer());
  MatCreateSubMatrix(__mat, isg_colloid.get_reference(), NULL,
                     MAT_INITIAL_MATRIX, nw.get_pointer());

  MatDestroy(&__mat);

  int mpi_rank = myId;
  int mpi_size = MPIsize;

  vector<int> idx_colloid_sub_field;
  vector<int> idx_colloid_sub_colloid;
  vector<int> idx_colloid_field;

  vector<int> idx_colloid_offset, idx_colloid_global_size;
  idx_colloid_offset.resize(mpi_size + 1);
  idx_colloid_global_size.resize(mpi_size);

  int idx_colloid_local_size = idx_colloid.size();
  MPI_Allgather(&idx_colloid_local_size, 1, MPI_INT,
                idx_colloid_global_size.data(), 1, MPI_INT, MPI_COMM_WORLD);

  idx_colloid_offset[0] = 0;
  for (int i = 0; i < mpi_size; i++) {
    idx_colloid_offset[i + 1] =
        idx_colloid_offset[i] + idx_colloid_global_size[i];
  }

  for (int i = 0; i < idx_colloid.size(); i++) {
    if (idx_colloid[i] < Col_block) {
      idx_colloid_sub_field.push_back(i + idx_colloid_offset[mpi_rank]);
      idx_colloid_field.push_back(idx_colloid[i]);
    } else {
      idx_colloid_sub_colloid.push_back(i + idx_colloid_offset[mpi_rank]);
    }
  }

  IS isg_colloid_sub_field, isg_colloid_sub_colloid, isg_colloid_field;

  Mat sub_ff, sub_fc, sub_cf;

  ISCreateGeneral(MPI_COMM_WORLD, idx_colloid_sub_field.size(),
                  idx_colloid_sub_field.data(), PETSC_COPY_VALUES,
                  &isg_colloid_sub_field);
  ISCreateGeneral(MPI_COMM_WORLD, idx_colloid_sub_colloid.size(),
                  idx_colloid_sub_colloid.data(), PETSC_COPY_VALUES,
                  &isg_colloid_sub_colloid);
  ISCreateGeneral(MPI_COMM_WORLD, idx_colloid_field.size(),
                  idx_colloid_field.data(), PETSC_COPY_VALUES,
                  &isg_colloid_field);

  MatCreateSubMatrix(*(__ctx.fluid_part), isg_colloid_field, isg_colloid_field,
                     MAT_INITIAL_MATRIX, &sub_ff);
  // MatCreateSubMatrix(nn.get_reference(), isg_colloid_sub_field,
  //                    isg_colloid_sub_field, MAT_INITIAL_MATRIX, &sub_ff);
  MatCreateSubMatrix(nn.get_reference(), isg_colloid_sub_field,
                     isg_colloid_sub_colloid, MAT_INITIAL_MATRIX, &sub_fc);
  MatCreateSubMatrix(nn.get_reference(), isg_colloid_sub_colloid, NULL,
                     MAT_INITIAL_MATRIX, &sub_cf);

  MatConvert(sub_ff, MATSAME, MAT_INITIAL_MATRIX, &(nn.__ctx.fluid_raw_part));
  MatConvert(sub_fc, MATSAME, MAT_INITIAL_MATRIX,
             &(nn.__ctx.fluid_colloid_part));
  MatTranspose(sub_cf, MAT_INITIAL_MATRIX, &(nn.__ctx.colloid_part));

  VecCreateMPI(PETSC_COMM_WORLD, idx_colloid_sub_field.size(), PETSC_DECIDE,
               &(nn.__ctx.fluid_vec1));
  VecCreateMPI(PETSC_COMM_WORLD, idx_colloid_sub_field.size(), PETSC_DECIDE,
               &(nn.__ctx.fluid_vec2));
  VecCreateMPI(PETSC_COMM_WORLD, idx_colloid_sub_colloid.size(), PETSC_DECIDE,
               &(nn.__ctx.colloid_vec));

  Vec x;
  MatCreateVecs(nn.get_reference(), &x, NULL);

  VecScatterCreate(x, isg_colloid_sub_field, nn.__ctx.fluid_vec1, NULL,
                   &(nn.__ctx.fluid_scatter));
  VecScatterCreate(x, isg_colloid_sub_colloid, nn.__ctx.colloid_vec, NULL,
                   &(nn.__ctx.colloid_scatter));

  Mat &shell_mat = nn.get_shell_reference();
  MatCreateShell(PETSC_COMM_WORLD, idx_colloid.size(), idx_colloid.size(),
                 PETSC_DECIDE, PETSC_DECIDE, &nn.__ctx, &shell_mat);
  MatShellSetOperation(shell_mat, MATOP_MULT,
                       (void (*)(void))fluid_colloid_matrix_mult2);

  nn.is_shell_assembled = true;
  nn.is_ctx_assembled = true;
  nn.__ctx.use_vec_scatter = true;
  nn.__ctx.use_raw_fluid_part = true;
  nn.__ctx.use_local_vec = false;

  ISDestroy(&isg_colloid_sub_field);
  ISDestroy(&isg_colloid_sub_colloid);
  ISDestroy(&isg_colloid_field);

  MatDestroy(&sub_ff);
  MatDestroy(&sub_fc);
  MatDestroy(&sub_cf);

  VecDestroy(&x);

  return 0;
}

void petsc_sparse_matrix::solve(vector<double> &rhs, vector<double> &x) {
  if (is_assembled) {
    petsc_vector _rhs, _x;
    _rhs.create(rhs);
    _x.create(_rhs);

    petsc_ksp ksp;
    KSP &_ksp = ksp.get_reference();
    KSPCreate(PETSC_COMM_WORLD, &_ksp);
    KSPSetOperators(_ksp, __mat, __mat);
    KSPSetFromOptions(_ksp);

    KSPSetUp(_ksp);

    PC _pc;
    KSPGetPC(_ksp, &_pc);
    PCSetType(_pc, PCLU);
    PCSetUp(_pc);

    PetscPrintf(PETSC_COMM_WORLD, "final solving of linear system\n");
    KSPSolve(_ksp, _rhs.get_reference(), _x.get_reference());
    MPI_Barrier(MPI_COMM_WORLD);

    KSPDestroy(&_ksp);

    _x.copy(x);
  }
}

void petsc_sparse_matrix::solve(vector<double> &rhs, vector<double> &x,
                                PetscInt block_size) {
  if (is_assembled) {
    petsc_vector _rhs, _x, _null;
    _rhs.create(rhs);
    _x.create(_rhs);
    _null.create(_rhs);

    Vec &null = _null.get_reference();

    PetscScalar *a;
    VecGetArray(null, &a);
    for (size_t i = 0; i < rhs.size(); i++) {
      if (i % block_size == block_size - 1)
        a[i] = 1.0;
      else
        a[i] = 0.0;
    }
    VecRestoreArray(null, &a);

    MatNullSpace nullspace;
    MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &null, &nullspace);
    MatSetNullSpace(__shell_mat, nullspace);

    petsc_ksp ksp;
    ksp.setup(__shell_mat);

    KSP &_ksp = ksp.get_reference();

    PC _pc;
    KSPGetPC(_ksp, &_pc);
    PCSetType(_pc, PCSHELL);

    HypreConstConstraintPC *shell_ctx;
    HypreConstConstraintPCCreate(&shell_ctx);

    PCShellSetApply(_pc, HypreConstConstraintPCApply);
    PCShellSetContext(_pc, shell_ctx);
    PCShellSetDestroy(_pc, HypreConstConstraintPCDestroy);

    HypreConstConstraintPCSetUp(_pc, &__mat, block_size);

    PetscPrintf(PETSC_COMM_WORLD, "final solving of linear system\n");
    KSPSolve(_ksp, _rhs.get_reference(), _x.get_reference());
    MPI_Barrier(MPI_COMM_WORLD);

    _x.copy(x);

    MatSetNearNullSpace(__shell_mat, NULL);
    MatNullSpaceDestroy(&nullspace);
  }
}

void petsc_sparse_matrix::solve(vector<double> &rhs, vector<double> &x,
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

void solve(petsc_sparse_matrix &A, petsc_sparse_matrix &Bt,
           petsc_sparse_matrix &B, petsc_sparse_matrix &C, vector<double> &f,
           vector<double> &g, vector<double> &x, vector<double> &y,
           int numRigid, int rigidBodyDof) {
  if (!A.is_assembled || !Bt.is_assembled || !B.is_assembled || !C.is_assembled)
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

void petsc_sparse_matrix::solve(vector<double> &rhs, vector<double> &x,
                                vector<int> &idx_colloid, int dimension,
                                int numRigidBody, int adatptive_step,
                                petsc_sparse_matrix &I,
                                petsc_sparse_matrix &R) {
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
  ISCreateGeneral(MPI_COMM_WORLD, idx_colloid.size(), idx_colloid.data(),
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

  PetscReal *a, *b, *c, *d;

  PetscReal pressure_sum = 0.0;
  PetscReal average_pressure;

  VecGetArray(x, &a);

  pressure_sum = 0.0;
  for (int i = 0; i < ctx->local_fluid_particle_num; i++) {
    pressure_sum += a[i * ctx->field_dof + ctx->pressure_offset];
  }
  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  average_pressure = pressure_sum / ctx->global_fluid_particle_num;
  for (int i = 0; i < ctx->local_fluid_particle_num; i++) {
    a[i * ctx->field_dof + ctx->pressure_offset] -= average_pressure;
  }

  VecRestoreArray(x, &a);

  VecGetArray(y, &a);

  MatMult(ctx->colloid_part, x, ctx->colloid_vec);

  VecGetArray(ctx->colloid_vec, &b);

  MPI_Reduce(b, a + ctx->fluid_local_size, ctx->rigid_body_size, MPI_DOUBLE,
             MPI_SUM, ctx->mpisize - 1, MPI_COMM_WORLD);

  VecRestoreArray(ctx->colloid_vec, &b);

  VecGetArray(x, &c);
  VecGetArray(ctx->fluid_vec1, &d);

  for (int i = 0; i < ctx->fluid_local_size; i++)
    d[i] = c[i];

  VecRestoreArray(x, &c);
  VecRestoreArray(ctx->fluid_vec1, &d);

  MatMult(*(ctx->fluid_part), ctx->fluid_vec1, ctx->fluid_vec2);

  VecGetArray(ctx->colloid_vec_local, &d);

  if (ctx->myid == ctx->mpisize - 1) {
    VecGetArray(x, &c);
    for (int i = 0; i < ctx->rigid_body_size; i++) {
      d[i] = c[i + ctx->fluid_local_size];
    }
    VecRestoreArray(x, &c);
  }

  MPI_Bcast(d, ctx->rigid_body_size, MPI_DOUBLE, ctx->mpisize - 1,
            MPI_COMM_WORLD);
  VecRestoreArray(ctx->colloid_vec_local, &d);

  MatMult(ctx->fluid_colloid_part, ctx->colloid_vec_local,
          ctx->fluid_vec_local);

  VecGetArray(ctx->fluid_vec_local, &b);
  VecGetArray(ctx->fluid_vec2, &c);
  for (int i = 0; i < ctx->fluid_local_size; i++)
    a[i] = b[i] + c[i];

  pressure_sum = 0.0;
  for (int i = 0; i < ctx->local_fluid_particle_num; i++) {
    pressure_sum += a[i * ctx->field_dof + ctx->pressure_offset];
  }
  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  average_pressure = pressure_sum / ctx->global_fluid_particle_num;
  for (int i = 0; i < ctx->local_fluid_particle_num; i++) {
    a[i * ctx->field_dof + ctx->pressure_offset] -= average_pressure;
  }

  VecRestoreArray(y, &a);
  VecRestoreArray(ctx->fluid_vec2, &c);
  VecRestoreArray(ctx->fluid_vec_local, &b);

  return 0;
}

PetscErrorCode fluid_colloid_matrix_mult2(Mat mat, Vec x, Vec y) {
  fluid_colloid_matrix_context *ctx;
  MatShellGetContext(mat, &ctx);

  VecScatterBegin(ctx->fluid_scatter, x, ctx->fluid_vec1, INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(ctx->fluid_scatter, x, ctx->fluid_vec1, INSERT_VALUES,
                SCATTER_FORWARD);
  VecScatterBegin(ctx->colloid_scatter, x, ctx->colloid_vec, INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(ctx->colloid_scatter, x, ctx->colloid_vec, INSERT_VALUES,
                SCATTER_FORWARD);

  double timer1, timer2;
  timer1 = MPI_Wtime();
  MatMult(ctx->fluid_raw_part, ctx->fluid_vec1, ctx->fluid_vec2);
  MatMult(ctx->fluid_colloid_part, ctx->colloid_vec, ctx->fluid_vec1);
  timer2 = MPI_Wtime();
  VecAXPY(ctx->fluid_vec2, 1.0, ctx->fluid_vec1);
  MatMultTranspose(ctx->colloid_part, x, ctx->colloid_vec);
  ctx->matmult_duration += timer2 - timer1;

  VecScatterBegin(ctx->fluid_scatter, ctx->fluid_vec2, y, INSERT_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(ctx->fluid_scatter, ctx->fluid_vec2, y, INSERT_VALUES,
                SCATTER_REVERSE);
  VecScatterBegin(ctx->colloid_scatter, ctx->colloid_vec, y, INSERT_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(ctx->colloid_scatter, ctx->colloid_vec, y, INSERT_VALUES,
                SCATTER_REVERSE);

  return 0;
}

PetscErrorCode fluid_matrix_mult(Mat mat, Vec x, Vec y) {
  fluid_colloid_matrix_context *ctx;
  MatShellGetContext(mat, &ctx);

  PetscReal *a;

  PetscReal pressure_sum = 0.0;
  PetscReal average_pressure;

  VecGetArray(x, &a);

  pressure_sum = 0.0;
  for (int i = 0; i < ctx->local_fluid_particle_num; i++) {
    pressure_sum += a[i * ctx->field_dof + ctx->pressure_offset];
  }
  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  average_pressure = pressure_sum / ctx->global_fluid_particle_num;
  for (int i = 0; i < ctx->local_fluid_particle_num; i++) {
    a[i * ctx->field_dof + ctx->pressure_offset] -= average_pressure;
  }

  VecRestoreArray(x, &a);

  double timer1, timer2;
  timer1 = MPI_Wtime();
  MatMult(*(ctx->fluid_part), x, y);
  timer2 = MPI_Wtime();
  ctx->matmult_duration += timer2 - timer1;

  VecGetArray(y, &a);

  pressure_sum = 0.0;
  for (int i = 0; i < ctx->local_fluid_particle_num; i++) {
    pressure_sum += a[i * ctx->field_dof + ctx->pressure_offset];
  }
  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  average_pressure = pressure_sum / ctx->global_fluid_particle_num;
  for (int i = 0; i < ctx->local_fluid_particle_num; i++) {
    a[i * ctx->field_dof + ctx->pressure_offset] -= average_pressure;
  }

  VecRestoreArray(y, &a);

  return 0;
}

void petsc_sparse_matrix::set_col_index(const PetscInt row,
                                        std::vector<PetscInt> &index) {
  sort(index.begin(), index.end());
  __matrix[row].resize(index.size());
  size_t counter = 0;
  for (std::vector<entry>::iterator it = __matrix[row].begin();
       it != __matrix[row].end(); it++) {
    if (index[counter] > __Col) {
      std::cout << row << ' ' << index[counter]
                << " index setting with wrong column index" << std::endl;
      counter++;
      continue;
    }

    it->first = index[counter++];
    it->second = 0.0;
  }
}

void petsc_sparse_matrix::set_out_process_col_index(
    const PetscInt row, std::vector<PetscInt> &index) {
  sort(index.begin(), index.end());
  __out_process_matrix[row - __out_process_reduction].resize(index.size());
  size_t counter = 0;
  for (std::vector<entry>::iterator it =
           __out_process_matrix[row - __out_process_reduction].begin();
       it != __out_process_matrix[row - __out_process_reduction].end(); it++) {
    if (index[counter] > __Col) {
      std::cout << row << ' ' << index[counter]
                << " out process index setting with wrong column index"
                << std::endl;
      counter++;
      continue;
    }

    it->first = index[counter++];
    it->second = 0.0;
  }
}

void petsc_sparse_matrix::zero_row(const PetscInt i) {
  for (auto it = __matrix[i].begin(); it != __matrix[i].end(); it++) {
    it->second = 0.0;
  }
}

double petsc_sparse_matrix::get_entity(const PetscInt i, const PetscInt j) {
  auto it = lower_bound(__matrix[i].begin(), __matrix[i].end(), entry(j, 0.0),
                        compare_index);
  if (j > __Col) {
    std::cout << i << ' ' << j << " wrong matrix index access" << std::endl;
    return 0.0;
  }

  if (it->first == j)
    return it->second;
  else
    return 0.0;
}

void petsc_sparse_matrix::invert_row(const PetscInt i) {
  for (auto it = __matrix[i].begin(); it != __matrix[i].end(); it++) {
    it->second = -it->second;
  }
}