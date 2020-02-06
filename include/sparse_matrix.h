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

  PetscInt __row, __col, __nnz, __Col;

  Mat __mat;

  std::vector<PetscInt> __unsorted_i;
  std::vector<PetscInt> __unsorted_j;
  std::vector<PetscReal> __unsorted_val;

  inline void sortbyj();

 public:
  std::vector<PetscInt> __i;
  std::vector<PetscInt> __j;
  std::vector<PetscReal> __val;

  PetscSparseMatrix() : __isAssembled(false), __row(0), __col(0), __Col(0) {}

  // only for square matrix
  PetscSparseMatrix(PetscInt m /* local # of rows */,
                    PetscInt N /* global # of cols */)
      : __isAssembled(false), __row(m), __col(m), __Col(N) {
    __matrix.resize(m);
  }

  PetscSparseMatrix(PetscInt m /* local # of rows */,
                    PetscInt n /* local # of cols */,
                    PetscInt N /* global # of cols */)
      : __isAssembled(false), __row(m), __col(n), __Col(N) {
    __matrix.resize(m);
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
  __unsorted_i.push_back(i);
  __unsorted_j.push_back(j);
  __unsorted_val.push_back(daij);
}

void PetscSparseMatrix::sortbyj() {
  for (PetscInt i = 0; i < __row; i++) {
    __matrix[i].sort(compare_index);
  }
}