#ifndef _LinearAlgebra_Impl_PetscMatrix_Hpp_
#define _LinearAlgebra_Impl_PetscMatrix_Hpp_

#include <vector>

#include <petscksp.h>

#include "Core/Typedef.hpp"
#include "LinearAlgebra/Impl/Petsc/Petsc.hpp"

/*! \brief Vector implementation based on Petsc.
 *! \date Sept 30, 2022
 *! \author Zisheng Ye <zisheng_ye@outlook.com>
 *
 *
 */

namespace LinearAlgebra {
namespace Impl {
class PetscMatrix {
protected:
  std::vector<unsigned long> rankColSize_, rankRowSize_;

  std::shared_ptr<Mat> matSharedPtr_;
  /*
    Development note: This repetition of the pointer is required to ensure that
    the pointer used by the function Increment
  */
  Mat *matPtr_;

  int mpiRank_, mpiSize_;

  PetscInt localColSize_, localRowSize_;
  PetscInt globalColSize_, globalRowSize_;
  PetscInt colRangeLow_, colRangeHigh_;
  PetscInt rowRangeLow_, rowRangeHigh_;
  PetscInt blockSize_, blockStorage_;

  std::vector<std::vector<PetscInt>> diagMatrixCol_, offDiagMatrixCol_;

public:
  PetscMatrix();

  ~PetscMatrix();

  virtual Void Resize(const PetscInt m, const PetscInt n,
                      const PetscInt blockSize = 1);

  Void Transpose(PetscMatrix &mat);

  Void Clear();

  PetscInt GetLocalColSize() const;
  PetscInt GetLocalRowSize() const;

  PetscInt GetGlobalColSize() const;
  PetscInt GetGlobalRowSize() const;

  inline virtual Void SetColIndex(const PetscInt row,
                                  const std::vector<PetscInt> &index);
  inline virtual Void Increment(const PetscInt row, const PetscInt col,
                                const PetscReal value);
  inline virtual Void Increment(const PetscInt row,
                                const std::vector<PetscInt> &index,
                                const std::vector<PetscReal> &value);

  virtual Void GraphAssemble();
  virtual Void Assemble();

  virtual Void MatrixVectorMultiplication(PetscVector &vec1, PetscVector &vec2);

  friend class PetscKsp;
  friend class PetscBlockMatrix;
};
} // namespace Impl
} // namespace LinearAlgebra

#endif