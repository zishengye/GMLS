#ifndef _PETSC_SPARSE_MATRIX_HPP_
#define _PETSC_SPARSE_MATRIX_HPP_

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

#include "PetscVector.hpp"
typedef std::pair<PetscInt, PetscReal> Entry;

inline bool CompareIndex(Entry entry1, Entry entry2) {
  return (entry1.first < entry2.first);
}

class PetscMatrix {
protected:
  std::vector<PetscInt> row_;
  std::vector<PetscInt> col_;
  std::vector<PetscReal> val_;

  PetscInt localRowSize_, localColSize_;

  std::vector<std::vector<Entry>> matrix_;

  int mpiRank_, mpiSize_;

  Mat mat_;

public:
  PetscMatrix();
  PetscMatrix(const PetscInt m, const PetscInt n);

  ~PetscMatrix();

  void Resize(const PetscInt m, const PetscInt n);

  const int GetRowSize();

  void SetColIndex(const PetscInt row, const std::vector<PetscInt> &index);
  void Increment(const PetscInt row, const PetscInt col, const PetscReal value);
  void Increment(const PetscInt row, const std::vector<PetscInt> &index,
                 const std::vector<PetscReal> &value);

  const unsigned long Assemble();
  const unsigned long Assemble(const PetscInt blockSize);
};

#endif