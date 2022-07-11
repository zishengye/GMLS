#ifndef _PetscMatrixBase_Hpp_
#define _PetscMatrixBase_Hpp_

#include "PetscVector.hpp"

/*!
 *
 */

class PetscMatrixBase {
protected:
  Mat mat_;

  int mpiRank_, mpiSize_;

  PetscInt localRowSize_, localColSize_;
  PetscInt globalRowSize_, globalColSize_;
  PetscInt colRangeLow_, colRangeHigh_;
  PetscInt blockSize_, blockStorage_;

public:
  PetscMatrixBase();
  ~PetscMatrixBase();

  void Clear();

  virtual int GetRowSize();
  virtual int GetColSize();

  virtual void Resize(const PetscInt m, const PetscInt n);

  virtual unsigned long GraphAssemble();
  virtual unsigned long Assemble();

  Mat &GetReference();
};

#endif