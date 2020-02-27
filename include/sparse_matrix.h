#pragma once

#include <assert.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <utility>
#include <vector>

#include <petscksp.h>

inline bool compare_index(std::pair<int, double> i, std::pair<int, double> j) {
  return (i.first < j.first);
}

class PetscSparseMatrix {
 private:
  bool __isAssembled;

  typedef std::pair<PetscInt, double> entry;
  std::vector<std::list<entry>> __matrix;
  std::vector<std::list<entry>> __out_process_matrix;

  PetscInt __row, __col, __nnz, __Col, __out_process_row,
      __out_process_reduction;

  Mat __mat;

  inline void sortbyj();

 public:
  std::vector<PetscInt> __i;
  std::vector<PetscInt> __j;
  std::vector<PetscReal> __val;

  PetscSparseMatrix() : __isAssembled(false), __row(0), __col(0), __Col(0) {}

  // only for square matrix
  PetscSparseMatrix(PetscInt m /* local # of rows */,
                    PetscInt N /* global # of cols */,
                    PetscInt out_process_row = 0,
                    PetscInt out_process_row_reduction = 0)
      : __isAssembled(false),
        __row(m),
        __col(m),
        __Col(N),
        __out_process_row(out_process_row),
        __out_process_reduction(out_process_row_reduction) {
    __matrix.resize(m);
    __out_process_matrix.resize(out_process_row);
  }

  PetscSparseMatrix(PetscInt m /* local # of rows */,
                    PetscInt n /* local # of cols */,
                    PetscInt N /* global # of cols */,
                    PetscInt out_process_row = 0,
                    PetscInt out_process_row_reduction = 0)
      : __isAssembled(false),
        __row(m),
        __col(n),
        __Col(N),
        __out_process_row(out_process_row),
        __out_process_reduction(out_process_row_reduction) {
    __matrix.resize(m);
    __out_process_matrix.resize(out_process_row);
  }

  ~PetscSparseMatrix() {
    if (__isAssembled) MatDestroy(&__mat);
  }

  void resize(PetscInt m, PetscInt n) {
    __row = m;
    __col = m;
    __Col = n;
    __matrix.resize(m);
  }

  inline void setRowSize(const PetscInt row, const size_t size);
  inline void setColIndex(const PetscInt row, std::vector<PetscInt> &index);
  inline void increment(const PetscInt i, const PetscInt j, double daij);
  inline void outProcessIncrement(const PetscInt i, const PetscInt j,
                                  double daij);

  int Write(std::string filename);

  int StructureAssemble();
  int FinalAssemble();

  // (*this) * x = rhs
  void Solve(std::vector<double> &rhs,
             std::vector<double> &x);  // simple solver
  void Solve(std::vector<double> &rhs, std::vector<double> &x,
             int dimension);  // two field solver
  void Solve(std::vector<double> &rhs, std::vector<double> &x, int dimension,
             int numRigidBody);
  void Solve(std::vector<double> &rhs, std::vector<double> &x,
             std::vector<int> &neighborInclusion, int dimension,
             int numRigidBody);
  // two field solver with rigid body inclusion

  // [A Bt; B C] * [x; y] = [f; g]
  friend void Solve(PetscSparseMatrix &A, PetscSparseMatrix &Bt,
                    PetscSparseMatrix &B, PetscSparseMatrix &C,
                    std::vector<double> &f, std::vector<double> &g,
                    std::vector<double> &x, std::vector<double> &y,
                    int numRigidBody, int rigidBodyDof);
};

void PetscSparseMatrix::setRowSize(const PetscInt row, const size_t size) {
  __matrix[row].resize(size);
}

void PetscSparseMatrix::setColIndex(const PetscInt row,
                                    std::vector<PetscInt> &index) {
  if (__matrix[row].size() == index.size()) {
    size_t counter = 0;
    for (std::list<entry>::iterator it = __matrix[row].begin();
         it != __matrix[row].end(); it++) {
      it->first = index[counter++];
      it->second = 0.0;
    }
  }
}

void PetscSparseMatrix::increment(const PetscInt i, const PetscInt j,
                                  const double daij) {
  if (std::abs(daij) > 1e-15) {
    bool inlist = false;

    for (std::list<entry>::iterator it = __matrix[i].begin();
         it != __matrix[i].end(); it++) {
      if (it->first == j) {
        it->second += daij;
        inlist = true;
        break;
      }
    }

    if (!inlist) {
      __matrix[i].push_back(entry(j, daij));
    }
  }
}

void PetscSparseMatrix::outProcessIncrement(const PetscInt i, const PetscInt j,
                                            const double daij) {
  if (std::abs(daij) > 1e-15) {
    bool inlist = false;

    PetscInt in = i - __out_process_reduction;

    for (std::list<entry>::iterator it = __out_process_matrix[in].begin();
         it != __out_process_matrix[in].end(); it++) {
      if (it->first == j) {
        it->second += daij;
        inlist = true;
        break;
      }
    }

    if (!inlist) {
      __out_process_matrix[in].push_back(entry(j, daij));
    }
  }
}

void PetscSparseMatrix::sortbyj() {
#pragma omp parallel for
  for (PetscInt i = 0; i < __row; i++) {
    __matrix[i].sort(compare_index);
  }

#pragma omp parallel for
  for (PetscInt i = 0; i < __out_process_row; i++) {
    __out_process_matrix[i].sort(compare_index);
  }
}