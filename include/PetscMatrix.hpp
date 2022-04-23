#ifndef _PetscMatrix_Hpp_
#define _PetscMatrix_Hpp_

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

class PetscMatrix {
protected:
  std::vector<PetscInt> diagRow_;
  std::vector<PetscInt> diagCol_;
  std::vector<PetscReal> diagVal_;

  std::vector<PetscInt> offDiagRow_;
  std::vector<PetscInt> offDiagCol_;
  std::vector<PetscReal> offDiagVal_;

  PetscInt localRowSize_, localColSize_;
  PetscInt globalRowSize_, globalColSize_;
  PetscInt colRangeLow, colRangeHigh;

  std::vector<std::vector<PetscInt>> diagMatrixCol_, offDiagMatrixCol_;

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

  const unsigned long GraphAssemble();

  const unsigned long Assemble();
  const unsigned long Assemble(const PetscInt blockSize);

  Mat &GetReference();
};

#endif