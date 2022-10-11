#ifndef _LinearAlgebra_Impl_VectorEntry_Hpp_
#define _LinearAlgebra_Impl_VectorEntry_Hpp_

#include "LinearAlgebra/Matrix.hpp"
#include "LinearAlgebra/Vector.hpp"

namespace LinearAlgebra {
template <class LinearAlgebraBackend>
VectorEntry<LinearAlgebraBackend>::VectorEntry()
    : matOperandPtr_(nullptr), leftVectorOperand_(nullptr),
      rightVectorOperand_(nullptr), leftVectorEntryOperand_(nullptr),
      rightVectorEntryOperand_(nullptr) {}

template <class LinearAlgebraBackend>
VectorEntry<LinearAlgebraBackend>::~VectorEntry() {}

template <class LinearAlgebraBackend>
Void VectorEntry<LinearAlgebraBackend>::InsertLeftMatrixOperand(
    Matrix<LinearAlgebraBackend> &mat) {
  matOperandPtr_ = std::make_shared<Matrix<LinearAlgebraBackend>>(mat);
}

template <class LinearAlgebraBackend>
Void VectorEntry<LinearAlgebraBackend>::InsertLeftVectorOperand(
    Vector<LinearAlgebraBackend> &vec) {
  leftVectorOperand_ = std::make_shared<Vector<LinearAlgebraBackend>>(vec);
}

template <class LinearAlgebraBackend>
Void VectorEntry<LinearAlgebraBackend>::InsertRightVectorOperand(
    Vector<LinearAlgebraBackend> &vec) {
  rightVectorOperand_ = std::make_shared<Vector<LinearAlgebraBackend>>(vec);
}

template <class LinearAlgebraBackend>
Boolean VectorEntry<LinearAlgebraBackend>::ExistLeftMatrixOperand() {
  return matOperandPtr_ != nullptr;
}

template <class LinearAlgebraBackend>
Matrix<LinearAlgebraBackend> &
VectorEntry<LinearAlgebraBackend>::GetLeftMatrixOperand() {
  return *matOperandPtr_;
}

template <class LinearAlgebraBackend>
Vector<LinearAlgebraBackend> &
VectorEntry<LinearAlgebraBackend>::GetLeftVectorOperand() {
  return *leftVectorOperand_;
}

template <class LinearAlgebraBackend>
Vector<LinearAlgebraBackend> &
VectorEntry<LinearAlgebraBackend>::GetRightVectorOperand() {
  return *rightVectorOperand_;
}
} // namespace LinearAlgebra

#endif