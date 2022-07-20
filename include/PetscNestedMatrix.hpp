#ifndef _PetscNestedMatrix_Hpp_
#define _PetscNestedMatrix_Hpp_

#include "PetscMatrix.hpp"

#include <memory>

PetscErrorCode NestedMatMultWrapper(Mat mat, Vec x, Vec y);

class PetscNestedMatrix : public PetscMatrixBase {
protected:
  PetscInt nestedRowBlockSize_, nestedColBlockSize_;

  std::vector<Mat> nestedMat_;
  std::vector<std::shared_ptr<PetscMatrix>> nestedWrappedMat_;

public:
  PetscNestedMatrix();
  PetscNestedMatrix(const PetscInt nestedRowBlockSize,
                    const PetscInt nestedColBlockSize);

  ~PetscNestedMatrix();

  void Resize(const PetscInt nestedRowBlockSize,
              const PetscInt nestColBlockSize);

  std::shared_ptr<PetscMatrix> GetMatrix(const PetscInt row,
                                         const PetscInt col);

  unsigned long GraphAssemble();
  unsigned long Assemble();

  PetscErrorCode NestedMatrixMult(Vec x, Vec y);
};

#endif