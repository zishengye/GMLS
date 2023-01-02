#include "LinearAlgebra/Impl/Default/DefaultBlockMatrix.hpp"
#include "Core/Typedef.hpp"
#include "LinearAlgebra/Impl/Default/DefaultBackend.hpp"
#include "LinearAlgebra/Impl/Default/DefaultMatrix.hpp"

LinearAlgebra::Impl::DefaultBlockMatrix::DefaultBlockMatrix() {}

LinearAlgebra::Impl::DefaultBlockMatrix::~DefaultBlockMatrix() {}

Void LinearAlgebra::Impl::DefaultBlockMatrix::ClearTimer() {
  a00Timer_ = 0;
  a11Timer_ = 0;
}

Scalar LinearAlgebra::Impl::DefaultBlockMatrix::GetA00Timer() {
  return a00Timer_;
}

Scalar LinearAlgebra::Impl::DefaultBlockMatrix::GetA11Timer() {
  return a11Timer_;
}

Void LinearAlgebra::Impl::DefaultBlockMatrix::Resize(
    const GlobalIndex blockM, const GlobalIndex blockN,
    const LocalIndex blockSize) {
  blockM_ = blockM;
  blockN_ = blockN;

  subMat_.resize(blockM * blockN);
}

std::shared_ptr<LinearAlgebra::Impl::DefaultMatrix>
LinearAlgebra::Impl::DefaultBlockMatrix::GetSubMat(const LocalIndex blockI,
                                                   const LocalIndex blockJ) {
  return subMat_[blockI * blockN_ + blockJ];
}

Void LinearAlgebra::Impl::DefaultBlockMatrix::SetSubMat(
    const LocalIndex blockI, const LocalIndex blockJ,
    std::shared_ptr<DefaultMatrix> mat) {
  subMat_[blockI * blockN_ + blockJ] = mat;
}

Void LinearAlgebra::Impl::DefaultBlockMatrix::SetCallbackPointer(
    LinearAlgebra::BlockMatrix<LinearAlgebra::Impl::DefaultBackend> *ptr) {}

Void LinearAlgebra::Impl::DefaultBlockMatrix::Assemble() {
  for (unsigned int i = 0; i < subMat_.size(); i++) {
    subMat_[i]->Assemble();
  }

  LocalIndex localRow = 0, localCol = 0;
  lhsVector_.resize(blockM_);
  rhsVector_.resize(blockN_);
  localLhsVectorOffset_.resize(blockM_ + 1);
  localRhsVectorOffset_.resize(blockN_ + 1);
  localLhsVectorOffset_[0] = 0;
  localRhsVectorOffset_[0] = 0;
  for (LocalIndex i = 0; i < blockM_; i++) {
    LocalIndex tLocalRow = subMat_[i * blockN_]->GetLocalRowSize();
    localRow += tLocalRow;
    localLhsVectorOffset_[i + 1] = localRow;
    lhsVector_[i].Resize(tLocalRow);
  }

  for (LocalIndex i = 0; i < blockN_; i++) {
    LocalIndex tLocalCol = subMat_[i]->GetLocalColSize();
    localCol += tLocalCol;
    localRhsVectorOffset_[i + 1] = localCol;
    lhsVector_[i].Resize(tLocalCol);
  }

  isAssembled_ = true;
}

Void LinearAlgebra::Impl::DefaultBlockMatrix::MatrixVectorMultiplication(
    DefaultVector &vec1, DefaultVector &vec2) {}

Void LinearAlgebra::Impl::DefaultBlockMatrix::
    MatrixVectorMultiplicationAddition(DefaultVector &vec1,
                                       DefaultVector &vec2) {}

Void LinearAlgebra::Impl::DefaultBlockMatrix::
    PrepareSchurComplementPreconditioner() {}

Void LinearAlgebra::Impl::DefaultBlockMatrix::
    ApplySchurComplementPreconditioningIteration(DefaultVector &vec1,
                                                 DefaultVector &vec2) {}