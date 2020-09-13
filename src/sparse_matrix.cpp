#include "sparse_matrix.hpp"

#include <fstream>
#include <string>

using namespace std;

int sparse_matrix::Write(string fileName) {
  ofstream output(fileName, ios::trunc);

  for (int i = 0; i < __row; i++) {
    for (vector<entry>::iterator it = __matrix[i].begin();
         it != __matrix[i].end(); it++) {
      output << (i + 1) << '\t' << (it->first + 1) << '\t' << it->second
             << endl;
    }
  }

  output.close();
}

int sparse_matrix::FinalAssemble() {
  // move data from outProcessIncrement
  int myid, MPIsize;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

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

int sparse_matrix::FinalAssemble(int blockSize) {
  // move data from outProcessIncrement
  int myid, MPIsize;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

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

  return __nnz;
}

void sparse_matrix::Solve(vector<double> &rhs, vector<double> &x) {
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
    PCSetFromOptions(_pc);
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