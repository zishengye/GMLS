#ifndef _LinearAlgebra_Impl_Petsc_PetscBlockMatrix_Hpp_
#define _LinearAlgebra_Impl_Petsc_PetscBlockMatrix_Hpp_

#include <vector>

#include <petscksp.h>

#include "Core/Typedef.hpp"
#include "LinearAlgebra/BlockMatrix.hpp"
#include "LinearAlgebra/Impl/Petsc/Petsc.hpp"
#include "LinearAlgebra/Impl/Petsc/PetscMatrix.hpp"

PetscErrorCode PetscBlockMatrixMatMultWrapper(Mat mat, Vec x, Vec y);

namespace LinearAlgebra {
namespace Impl {
class PetscBlockMatrix : public PetscMatrix {
protected:
  std::vector<std::shared_ptr<PetscMatrix>> subMat_;

  std::vector<unsigned long> localLhsVectorOffset_;
  std::vector<unsigned long> localRhsVectorOffset_;

  std::vector<Vec> lhsVector_;
  std::vector<Vec> rhsVector_;

  PetscInt blockM_, blockN_;

  Mat schur_;

  KSP a00Ksp_, a11Ksp_;

  PetscReal a00Timer_, a11Timer_;

  LinearAlgebra::BlockMatrix<LinearAlgebra::Impl::PetscBackend> *callbackPtr_;

public:
  PetscBlockMatrix();

  ~PetscBlockMatrix();

  Void ClearTimer();

  PetscReal GetA00Timer();
  PetscReal GetA11Timer();

  virtual Void Resize(const PetscInt blockM, const PetscInt blockN,
                      const PetscInt blockSize = 1);

  std::shared_ptr<PetscMatrix> GetSubMat(const PetscInt blockI,
                                         const PetscInt blockJ);
  void SetSubMat(const PetscInt blockI, const PetscInt blockJ,
                 std::shared_ptr<PetscMatrix> mat);

  Void SetCallbackPointer(
      LinearAlgebra::BlockMatrix<LinearAlgebra::Impl::PetscBackend> *ptr);

  virtual Void Assemble();

  virtual Void MatrixVectorMultiplication(PetscVector &vec1, PetscVector &vec2);

  virtual Void PrepareSchurComplementPreconditioner();
  virtual Void ApplySchurComplementPreconditioningIteration(PetscVector &x,
                                                            PetscVector &y);

  friend class PetscKsp;
};
} // namespace Impl
} // namespace LinearAlgebra

#endif