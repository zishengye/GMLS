#ifndef _SPARSE_MATRIX_HPP_
#define _SPARSE_MATRIX_HPP_

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

#ifndef _COMPARE_
#define _COMPARE_

inline bool compare_index(std::pair<int, double> i, std::pair<int, double> j) {
  return (i.first < j.first);
}

// group size comparison
inline bool compare_group(std::vector<int> group1, std::vector<int> group2) {
  return group1.size() > group2.size();
}

#endif

class sparse_matrix {
protected:
  bool __isAssembled;

  typedef std::pair<PetscInt, double> entry;
  std::vector<std::vector<entry>> __matrix;

  PetscInt __row, __col, __nnz, __Col;

  std::vector<PetscInt> __i;
  std::vector<PetscInt> __j;
  std::vector<PetscReal> __val;

public:
  Mat __mat, __shell_mat;

  sparse_matrix() : __isAssembled(false), __row(0), __col(0), __Col(0) {}

  // only for square matrix
  sparse_matrix(PetscInt m /* local # of rows */,
                PetscInt N /* global # of cols */)
      : __isAssembled(false), __row(m), __col(m), __Col(N) {
    __matrix.resize(m);
  }

  ~sparse_matrix() {
    if (__isAssembled)
      MatDestroy(&__mat);
  }

  void resize(PetscInt m, PetscInt n) {
    __row = m;
    __col = n;
    __Col = 0;
    __matrix.resize(m);
  }

  void resize(PetscInt m, PetscInt n, PetscInt N) {
    __row = m;
    __col = n;
    __Col = N;
    __matrix.resize(m);
  }

  inline void setColIndex(const PetscInt row, std::vector<PetscInt> &index);
  inline void increment(const PetscInt i, const PetscInt j, double daij);

  int Write(std::string filename);

  int StructureAssemble();
  int FinalAssemble();
  int FinalAssemble(int blockSize);

  // (*this) * x = rhs
  void Solve(std::vector<double> &rhs,
             std::vector<double> &x); // simple solver
  void Solve(std::vector<double> &rhs, std::vector<double> &x,
             PetscInt blockSize); // simple solver
};

void sparse_matrix::setColIndex(const PetscInt row,
                                std::vector<PetscInt> &index) {
  sort(index.begin(), index.end());
  __matrix[row].resize(index.size());
  size_t counter = 0;
  for (std::vector<entry>::iterator it = __matrix[row].begin();
       it != __matrix[row].end(); it++) {
    it->first = index[counter++];
    it->second = 0.0;
  }
}

void sparse_matrix::increment(const PetscInt i, const PetscInt j,
                              const double daij) {
  if (std::abs(daij) > 1e-15) {
    auto it = lower_bound(__matrix[i].begin(), __matrix[i].end(),
                          entry(j, daij), compare_index);
    if (it->first == j)
      it->second += daij;
    else
      std::cout << i << ' ' << j << " increment misplacement" << std::endl;
  }
}

#endif