#ifndef _PETSC_SPARSE_MATRIX_HPP_
#define _PETSC_SPARSE_MATRIX_HPP_

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

#include "petsc_vector.hpp"

PetscErrorCode null_space_matrix_mult(Mat mat, Vec x, Vec y);

class petsc_sparse_matrix {
private:
  bool is_assembled;
  bool is_self_contained;
  bool is_set_null_space;
  bool is_transpose;

  PetscInt row, col, Row, Col;
  PetscInt range_row1, range_row2;
  PetscInt block_size, block_row, block_col, block_Row, block_Col,
      block_range_row1, block_range_row2;

  std::vector<PetscInt> mat_i;
  std::vector<PetscInt> mat_j;
  std::vector<PetscReal> mat_a;

  std::vector<PetscInt> mat_oi;
  std::vector<PetscInt> mat_oj;
  std::vector<PetscReal> mat_oa;

  std::shared_ptr<std::vector<int>> null_space_ptr;
  PetscReal null_space_size;

  Mat *mat;
  Mat shell_mat;

  int rank, size;

  friend PetscErrorCode null_space_matrix_mult(Mat mat, Vec x, Vec y);

public:
  petsc_sparse_matrix()
      : is_assembled(false), is_self_contained(true), is_set_null_space(false),
        is_transpose(false), row(0), col(0), Col(0), mat(PETSC_NULL) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  }

  petsc_sparse_matrix(PetscInt m /* local # of rows */,
                      PetscInt n /* local # of cols */,
                      PetscInt N /* global # of cols */, PetscInt bs = 1)
      : is_assembled(false), is_self_contained(true), is_set_null_space(false),
        is_transpose(false), row(m), col(n), Col(N), block_size(bs),
        mat(PETSC_NULL) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int send_count = row;
    std::vector<int> recv_count;
    recv_count.resize(size);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allgather(&send_count, 1, MPI_INT, recv_count.data(), 1, MPI_INT,
                  MPI_COMM_WORLD);

    std::vector<int> displs;
    displs.resize(size + 1);
    displs[0] = 0;
    for (int i = 1; i <= size; i++) {
      displs[i] = displs[i - 1] + recv_count[i - 1];
    }

    range_row1 = displs[rank];
    range_row2 = displs[rank + 1];

    Row = displs[size];

    block_row = row / block_size;
    block_col = col / block_size;
    block_range_row1 = range_row1 / block_size;
    block_range_row2 = range_row2 / block_size;

    block_Row = Row / block_size;
    block_Col = Col / block_size;

    mat_i.resize(block_row + 1);
    mat_oi.resize(block_row + 1);

    std::fill(mat_i.begin(), mat_i.end(), 0);
    std::fill(mat_oi.begin(), mat_oi.end(), 0);
  }

  ~petsc_sparse_matrix() {
    if (is_assembled && is_self_contained) {
      MatDestroy(mat);
      delete mat;
    }
    if (is_set_null_space)
      MatDestroy(&shell_mat);
  }

  void resize(PetscInt m, PetscInt n, PetscInt N, PetscInt bs = 1) {
    row = m;
    col = n;
    Col = N;

    block_size = bs;

    int send_count = row;
    std::vector<int> recv_count;
    recv_count.resize(size);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allgather(&send_count, 1, MPI_INT, recv_count.data(), 1, MPI_INT,
                  MPI_COMM_WORLD);

    std::vector<int> displs;
    displs.resize(size + 1);
    displs[0] = 0;
    for (int i = 1; i <= size; i++) {
      displs[i] = displs[i - 1] + recv_count[i - 1];
    }

    range_row1 = displs[rank];
    range_row2 = displs[rank + 1];

    Row = displs[size];

    block_row = row / block_size;
    block_range_row1 = range_row1 / block_size;
    block_range_row2 = range_row2 / block_size;

    block_Row = Row / block_size;
    block_Col = Col / block_size;

    mat_i.resize(block_row + 1);
    mat_oi.resize(block_row + 1);

    mat_j.clear();
    mat_a.clear();
    mat_oj.clear();
    mat_oa.clear();

    std::fill(mat_i.begin(), mat_i.end(), 0);
    std::fill(mat_oi.begin(), mat_oi.end(), 0);
  }

  void set_null_space(std::vector<int> &null_space) {
    null_space_ptr = std::make_shared<std::vector<int>>(null_space);
    is_set_null_space = true;

    MatCreateShell(PETSC_COMM_WORLD, block_row, block_col, PETSC_DECIDE,
                   block_Col, this, &shell_mat);
    MatShellSetOperation(shell_mat, MATOP_MULT,
                         (void (*)(void))null_space_matrix_mult);

    null_space_size = null_space_ptr->size();
    MPI_Allreduce(MPI_IN_PLACE, &null_space_size, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
  }

  void set_transpose() {
    is_transpose = true;

    mat_i.resize(Row + 1);
    std::fill(mat_i.begin(), mat_i.end(), 0);
  }

  Mat &get_reference() { return *mat; }

  Mat &get_operator() {
    if (is_set_null_space)
      return shell_mat;
    else
      return *mat;
  }

  void link(Mat *A) {
    mat = A;
    is_self_contained = false;
  }

  void set_col_index(const PetscInt i, std::vector<PetscInt> &idx);
  void set_block_col_index(const PetscInt i, std::vector<PetscInt> &idx);
  void increment(const PetscInt i, const PetscInt j, double a);
  void increment_row(const PetscInt i, std::vector<PetscInt> &j,
                     std::vector<PetscReal> &a);
  void increment_row_block(const PetscInt i, std::vector<PetscInt> &j,
                           std::vector<PetscReal> &a);

  double get_entity(const PetscInt i, const PetscInt j);

  int write(std::string filename);

  int graph_assemble();
  int graph_assemble(Mat *A);
  int assemble();
  int transpose_assemble();

  int extract_neighbor_index(std::vector<int> &idx_colloid, int dimension,
                             int num_rigid_body, int local_rigid_body_offset,
                             int global_rigid_body_offset,
                             petsc_sparse_matrix &nn, petsc_sparse_matrix &nw);
};

#endif