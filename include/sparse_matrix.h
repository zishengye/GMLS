#pragma once

#include <algorithm>
#include <assert.h>
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

  typedef std::pair<int, double> entry;
  std::vector<std::list<entry>> __matrix;

  PetscInt __row, __col, __nnz;

  Mat __mat;

  std::vector<PetscInt> __i;
  std::vector<PetscInt> __j;
  std::vector<PetscReal> __val;

  inline void sortbyj();

public:
  PetscSparseMatrix() : __isAssembled(false) {}

  PetscSparseMatrix(int m /* local # of rows */, int N /* global # of cols */)
      : __isAssembled(false), __row(m), __col(N) {
    __matrix.resize(m);
  }

  ~PetscSparseMatrix() {
    if (__isAssembled)
      MatDestroy(&__mat);
  }

  void resize(int m, int N) {
    __row = m;
    __col = N;
    __matrix.resize(m);
  }

  inline void increment(const int i, const int j, double daij);

  int FinalAssemble();

  void Solve(std::vector<double> &rhs, std::vector<double> &x);
};

void PetscSparseMatrix::increment(const int i, const int j, double daij) {
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

void PetscSparseMatrix::sortbyj() {
  for (int i = 0; i < __row; i++) {
    __matrix[i].sort(compare_index);
  }
}