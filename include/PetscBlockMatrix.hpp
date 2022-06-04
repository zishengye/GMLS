#ifndef _PetscBlockMatrix_Hpp_
#define _PetscBlockMatrix_Hpp_

#include "PetscMatrix.hpp"

class PetscBlockMatrix : public PetscMatrixBase {
protected:
  std::vector<std::shared_ptr<PetscMatrixBase>> subMat_;

  PetscInt blockM_, blockN_;

public:
  PetscBlockMatrix();
  PetscBlockMatrix(const PetscInt blockM, const PetscInt blockN);

  ~PetscBlockMatrix();

  void Resize(const PetscInt blockM, const PetscInt blockN);

  PetscInt GetRowSize();
  PetscInt GetColSize();

  std::shared_ptr<PetscMatrixBase> &GetSubMat(const PetscInt blockI,
                                              const PetscInt blockJ);
  void SetSubMat(const PetscInt blockI, const PetscInt blockJ,
                 std::shared_ptr<PetscMatrixBase> mat);

  unsigned long Assemble();
};

#endif