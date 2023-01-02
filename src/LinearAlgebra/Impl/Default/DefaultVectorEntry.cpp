#include "LinearAlgebra/Impl/Default/DefaultVectorEntry.hpp"

#include "LinearAlgebra/Impl/Default/Default.hpp"
#include "LinearAlgebra/Impl/Default/DefaultMatrix.hpp"
#include "LinearAlgebra/Impl/Default/DefaultVector.hpp"

LinearAlgebra::Impl::DefaultVectorEntry::DefaultVectorEntry()
    : matOperandPtr_(nullptr), leftVectorOperandPtr_(nullptr),
      rightVectorOperandPtr_(nullptr), leftVectorEntryOperandPtr_(nullptr),
      rightVectorEntryOperandPtr_(nullptr), isRightOperandScalar_(false) {}

LinearAlgebra::Impl::DefaultVectorEntry::~DefaultVectorEntry() {}

Void LinearAlgebra::Impl::DefaultVectorEntry::InsertLeftMatrixOperand(
    const std::shared_ptr<DefaultMatrix> &matPtr) {
  matOperandPtr_ = matPtr;
}

Void LinearAlgebra::Impl::DefaultVectorEntry::InsertLeftVectorOperand(
    const std::shared_ptr<DefaultVector> &vecPtr) {
  leftVectorOperandPtr_ = vecPtr;
}

Void LinearAlgebra::Impl::DefaultVectorEntry::InsertRightVectorOperand(
    const std::shared_ptr<DefaultVector> &vecPtr) {
  rightVectorOperandPtr_ = vecPtr;
}

Void LinearAlgebra::Impl::DefaultVectorEntry::InsertRightScalarOperand(
    const Scalar scalar) {
  scalar_ = scalar;
  isRightOperandScalar_ = true;
}

Void LinearAlgebra::Impl::DefaultVectorEntry::InsertOperator(
    const char operatorName) {
  operatorName_ = operatorName;
}

Boolean
LinearAlgebra::Impl::DefaultVectorEntry::ExistLeftMatrixOperand() const {
  return matOperandPtr_ != nullptr;
}

Boolean
LinearAlgebra::Impl::DefaultVectorEntry::ExistRightScalarOperand() const {
  return isRightOperandScalar_;
}

const std::shared_ptr<LinearAlgebra::Impl::DefaultMatrix> &
LinearAlgebra::Impl::DefaultVectorEntry::GetLeftMatrixOperand() const {
  return matOperandPtr_;
}

const std::shared_ptr<LinearAlgebra::Impl::DefaultVector> &
LinearAlgebra::Impl::DefaultVectorEntry::GetLeftVectorOperand() const {
  return leftVectorOperandPtr_;
}

const std::shared_ptr<LinearAlgebra::Impl::DefaultVector> &
LinearAlgebra::Impl::DefaultVectorEntry::GetRightVectorOperand() const {
  return rightVectorOperandPtr_;
}

Scalar LinearAlgebra::Impl::DefaultVectorEntry::GetRightScalarOperand() const {
  return scalar_;
}

char LinearAlgebra::Impl::DefaultVectorEntry::GetOperatorName() const {
  return operatorName_;
}