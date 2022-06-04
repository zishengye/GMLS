#ifndef _PetscMatrix_Hpp_
#define _PetscMatrix_Hpp_

#include <algorithm>
#include <cstdlib>
#include <list>
#include <string>
#include <utility>
#include <vector>

#include "PetscMatrixBase.hpp"

class PetscMatrix : public PetscMatrixBase {
protected:
  std::vector<PetscInt> diagRow_;
  std::vector<PetscInt> diagCol_;
  std::vector<PetscReal> diagVal_;

  std::vector<PetscInt> offDiagRow_;
  std::vector<PetscInt> offDiagCol_;
  std::vector<PetscReal> offDiagVal_;

  std::vector<unsigned long> rankColSize_, rankRowSize_;

  PetscInt localRowSize_, localColSize_;
  PetscInt globalRowSize_, globalColSize_;
  PetscInt colRangeLow_, colRangeHigh_;
  PetscInt rowRangeLow_, rowRangeHigh_;
  PetscInt blockSize_, blockStorage_;

  std::vector<std::vector<PetscInt>> diagMatrixCol_, offDiagMatrixCol_;

public:
  PetscMatrix();
  PetscMatrix(const PetscInt m, const PetscInt n, const PetscInt blockSize = 1);

  ~PetscMatrix();

  void Resize(const PetscInt m, const PetscInt n, const PetscInt blockSize = 1);

  PetscInt GetRowSize();
  PetscInt GetColSize();

  void SetColIndex(const PetscInt row, const std::vector<PetscInt> &index);
  void Increment(const PetscInt row, const PetscInt col, const PetscReal value);
  void Increment(const PetscInt row, const std::vector<PetscInt> &index,
                 const std::vector<PetscReal> &value);

  unsigned long GraphAssemble();

  unsigned long Assemble();
};

#endif