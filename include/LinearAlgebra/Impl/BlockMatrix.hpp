#ifndef _LinearAlgebra_Impl_BlockMatrix_Hpp_
#define _LinearAlgebra_Impl_BlockMatrix_Hpp_

#include <type_traits>

template <typename T> struct fail : std::false_type {};

#include "LinearAlgebra/BlockMatrix.hpp"
namespace LinearAlgebra {
template <class LinearAlgebraBackend>
BlockMatrix<LinearAlgebraBackend>::BlockMatrix()
    : Matrix<LinearAlgebraBackend>() {
  blockMat_ =
      std::make_shared<typename LinearAlgebraBackend::BlockMatrixBase>();
  Matrix<LinearAlgebraBackend>::mat_ =
      std::static_pointer_cast<typename LinearAlgebraBackend::MatrixBase>(
          blockMat_);
}

template <class LinearAlgebraBackend>
BlockMatrix<LinearAlgebraBackend>::~BlockMatrix() {}

template <class LinearAlgebraBackend>
Void BlockMatrix<LinearAlgebraBackend>::Clear() {}

template <class LinearAlgebraBackend>
Void BlockMatrix<LinearAlgebraBackend>::Resize(const Integer blockM,
                                               const Integer blockN,
                                               const Integer blockSize) {
  blockRow_ = blockM;
  blockCol_ = blockN;

  subMatrix_.clear();
  subMatrix_.resize(blockM * blockN);

  blockMat_->Resize(blockM, blockN, blockSize);
}

template <class LinearAlgebraBackend>
typename LinearAlgebraBackend::DefaultInteger
BlockMatrix<LinearAlgebraBackend>::GetLocalColSize() const {
  typename LinearAlgebraBackend::DefaultInteger localCol = 0;
  for (unsigned int i = 0; i < blockRow_; i++)
    localCol += subMatrix_[i * blockCol_ + i]->GetLocalColSize();

  return localCol;
}

template <class LinearAlgebraBackend>
typename LinearAlgebraBackend::DefaultInteger
BlockMatrix<LinearAlgebraBackend>::GetLocalRowSize() const {
  typename LinearAlgebraBackend::DefaultInteger localRow = 0;
  for (unsigned int i = 0; i < blockRow_; i++)
    localRow += subMatrix_[i * blockCol_ + i]->GetLocalRowSize();

  return localRow;
}

template <class LinearAlgebraBackend>
typename LinearAlgebraBackend::DefaultInteger
BlockMatrix<LinearAlgebraBackend>::GetGlobalColSize() const {
  typename LinearAlgebraBackend::DefaultInteger globalCol = 0;
  for (unsigned int i = 0; i < blockRow_; i++)
    globalCol += subMatrix_[i * blockCol_ + i]->GetGlobalColSize();

  return globalCol;
}

template <class LinearAlgebraBackend>
typename LinearAlgebraBackend::DefaultInteger
BlockMatrix<LinearAlgebraBackend>::GetGlobalRowSize() const {
  typename LinearAlgebraBackend::DefaultInteger globalRow = 0;
  for (unsigned int i = 0; i < blockRow_; i++)
    globalRow += subMatrix_[i * blockCol_ + i]->GetGlobalRowSize();

  return globalRow;
}

template <class LinearAlgebraBackend>
std::shared_ptr<Matrix<LinearAlgebraBackend>>
BlockMatrix<LinearAlgebraBackend>::GetSubMat(const unsigned int blockI,
                                             const unsigned int blockJ) {
  return subMatrix_[blockI * blockCol_ + blockJ];
}

template <class LinearAlgebraBackend>
Void BlockMatrix<LinearAlgebraBackend>::SetSubMat(
    const unsigned int blockI, const unsigned int blockJ,
    std::shared_ptr<Matrix<LinearAlgebraBackend>> mat) {
  subMatrix_[blockI * blockCol_ + blockJ] = mat;

  blockMat_->SetSubMat(blockI, blockJ, mat->GetMatrix());
}

template <class LinearAlgebraBackend>
Void BlockMatrix<LinearAlgebraBackend>::Assemble() {
  blockMat_->SetCallbackPointer(this);

  blockMat_->Assemble();

  localLhsVectorOffset_.resize(blockRow_ + 1);
  localRhsVectorOffset_.resize(blockCol_ + 1);

  localLhsVectorOffset_[0] = 0;
  for (int i = 0; i < blockRow_; i++)
    localLhsVectorOffset_[i + 1] =
        localLhsVectorOffset_[i] + subMatrix_[i * blockCol_]->GetLocalRowSize();

  localRhsVectorOffset_[0] = 0;
  for (int i = 0; i < blockCol_; i++)
    localRhsVectorOffset_[i + 1] =
        localRhsVectorOffset_[i] + subMatrix_[i]->GetLocalColSize();
}

template <class LinearAlgebraBackend>
Void BlockMatrix<LinearAlgebraBackend>::PrepareSchurComplementPreconditioner() {
  blockMat_->PrepareSchurComplementPreconditioner();
}

template <class LinearAlgebraBackend>
Void BlockMatrix<LinearAlgebraBackend>::
    ApplySchurComplementPreconditioningIteration(DefaultVector &x,
                                                 DefaultVector &y) {
  blockMat_->ApplySchurComplementPreconditioningIteration(x.vec_, y.vec_);
}
} // namespace LinearAlgebra

#endif