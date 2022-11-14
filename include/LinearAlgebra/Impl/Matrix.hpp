#ifndef _LinearAlgebra_Impl_Matrix_Hpp_
#define _LinearAlgebra_Impl_Matrix_Hpp_

namespace LinearAlgebra {
template <class LinearAlgebraBackend> Matrix<LinearAlgebraBackend>::Matrix() {
  mat_ = std::make_shared<typename LinearAlgebraBackend::MatrixBase>();
}

template <class LinearAlgebraBackend> Matrix<LinearAlgebraBackend>::~Matrix() {}

template <class LinearAlgebraBackend>
Void Matrix<LinearAlgebraBackend>::Clear() {
  mat_->Clear();
}

template <class LinearAlgebraBackend>
Void Matrix<LinearAlgebraBackend>::Resize(const Integer m, const Integer n,
                                          const Integer blockSize) {
  mat_->Resize(m, n, blockSize);
}

template <class LinearAlgebraBackend>
Void Matrix<LinearAlgebraBackend>::Transpose(
    Matrix<LinearAlgebraBackend> &mat) {
  mat_->Transpose(*(mat.mat_));
}

template <class LinearAlgebraBackend>
typename LinearAlgebraBackend::DefaultInteger
Matrix<LinearAlgebraBackend>::GetLocalColSize() const {
  return mat_->GetLocalColSize();
}

template <class LinearAlgebraBackend>
typename LinearAlgebraBackend::DefaultInteger
Matrix<LinearAlgebraBackend>::GetLocalRowSize() const {
  return mat_->GetLocalRowSize();
}

template <class LinearAlgebraBackend>
typename LinearAlgebraBackend::DefaultInteger
Matrix<LinearAlgebraBackend>::GetGlobalColSize() const {
  return mat_->GetGlobalColSize();
}

template <class LinearAlgebraBackend>
typename LinearAlgebraBackend::DefaultInteger
Matrix<LinearAlgebraBackend>::GetGlobalRowSize() const {
  return mat_->GetGlobalRowSize();
}

template <class LinearAlgebraBackend>
Void Matrix<LinearAlgebraBackend>::SetColIndex(
    const Integer row, const std::vector<Integer> &colIndex) {
  mat_->SetColIndex(row, colIndex);
}

template <class LinearAlgebraBackend>
Void Matrix<LinearAlgebraBackend>::Increment(const Integer row,
                                             const Integer col,
                                             const Scalar value) {
  mat_->Increment(row, col, value);
}

template <class LinearAlgebraBackend>
Void Matrix<LinearAlgebraBackend>::Increment(
    const Integer row, const std::vector<Integer> colIndex,
    const std::vector<Scalar> &value) {
  mat_->Increment(row, colIndex, value);
}

template <class LinearAlgebraBackend>
Void Matrix<LinearAlgebraBackend>::GraphAssemble() {
  mat_->GraphAssemble();
}

template <class LinearAlgebraBackend>
Void Matrix<LinearAlgebraBackend>::Assemble() {
  mat_->Assemble();
}

template <class LinearAlgebraBackend>
VectorEntry<LinearAlgebraBackend>
Matrix<LinearAlgebraBackend>::operator*(Vector<LinearAlgebraBackend> &vec) {
  VectorEntry<LinearAlgebraBackend> vectorEntry;

  vectorEntry.InsertLeftMatrixOperand(*this);
  vectorEntry.InsertRightVectorOperand(vec);

  return vectorEntry;
}

template <class LinearAlgebraBackend>
Void Matrix<LinearAlgebraBackend>::MatrixVectorMultiplication(
    Vector<LinearAlgebraBackend> &vec1, Vector<LinearAlgebraBackend> &vec2) {
  mat_->MatrixVectorMultiplication(vec1.vec_, vec2.vec_);
}

template <class LinearAlgebraBackend>
std::shared_ptr<typename LinearAlgebraBackend::MatrixBase>
Matrix<LinearAlgebraBackend>::GetMatrix() {
  return mat_;
}
} // namespace LinearAlgebra

#endif