#ifndef _PETSC_BLOCK_MATRIX_HPP_
#define _PETSC_BLOCK_MATRIX_HPP_

#include <vector>

#include "petsc_sparse_matrix.hpp"

class petsc_block_matrix {
private:
  std::vector<Mat> mat_list;
  std::vector<std::shared_ptr<petsc_sparse_matrix>> pmat_list;
  Mat mat;

  int Row, Col;

  bool is_assembled;

public:
  petsc_block_matrix() : is_assembled(false) {}
  petsc_block_matrix(const PetscInt M, const PetscInt N)
      : Row(M), Col(N), is_assembled(false) {
    mat_list.resize(Row * Col);
    for (int i = 0; i < Row * Col; i++) {
      pmat_list.push_back(std::make_shared<petsc_sparse_matrix>());
      pmat_list[i]->link(&(mat_list[i]));
    }
  }

  ~petsc_block_matrix();

  void resize(const PetscInt M, const PetscInt N) {
    Row = M;
    Col = N;
    mat_list.resize(Row * Col);
    for (int i = 0; i < Row * Col; i++) {
      pmat_list.push_back(std::make_shared<petsc_sparse_matrix>());
      pmat_list[i]->link(&(mat_list[i]));
    }
  }

  void assemble() {
    MatCreateNest(MPI_COMM_WORLD, Row, NULL, Col, NULL, mat_list.data(), &mat);
    is_assembled = true;
  }

  petsc_sparse_matrix &get_reference(const PetscInt I, const PetscInt J) {
    return *(pmat_list[I * Col + J].get());
  }

  Mat &get_operator() { return mat; }
};

#endif