#ifndef _LinearAlgebra_Impl_Vector_Hpp_
#define _LinearAlgebra_Impl_Vector_Hpp_

#include "LinearAlgebra/VectorEntry.hpp"

namespace LinearAlgebra {
template <class LinearAlgebraBackend>
Vector<LinearAlgebraBackend>::Vector() : vec_() {}

template <class LinearAlgebraBackend>
Vector<LinearAlgebraBackend>::Vector(const Integer localSize)
    : vec_(localSize) {}

template <class LinearAlgebraBackend> Vector<LinearAlgebraBackend>::~Vector() {}

template <class LinearAlgebraBackend>
typename LinearAlgebraBackend::DefaultInteger
Vector<LinearAlgebraBackend>::GetLocalSize() {
  return vec_.GetLocalSize();
}

template <class LinearAlgebraBackend>
Void Vector<LinearAlgebraBackend>::Create(const std::vector<Scalar> &vec) {
  vec_.Create(vec);
}

template <class LinearAlgebraBackend>
Void Vector<LinearAlgebraBackend>::Create(
    const typename LinearAlgebraBackend::VectorBase &vec) {
  vec_.Create(vec);
}

template <class LinearAlgebraBackend>
Void Vector<LinearAlgebraBackend>::Create(const Vector &vec) {
  vec_.Create(vec.vec_);
}

template <class LinearAlgebraBackend>
Void Vector<LinearAlgebraBackend>::Create(const HostRealVector &vec) {
  vec_.Create(vec);
}

template <class LinearAlgebraBackend>
Void Vector<LinearAlgebraBackend>::Copy(std::vector<Scalar> &vec) {
  vec_.Copy(vec);
}

template <class LinearAlgebraBackend>
Void Vector<LinearAlgebraBackend>::Copy(
    typename LinearAlgebraBackend::VectorBase &vec) {
  vec_.Copy(vec);
}

template <class LinearAlgebraBackend>
Void Vector<LinearAlgebraBackend>::Copy(HostRealVector &vec) {
  vec_.Copy(vec);
}

template <class LinearAlgebraBackend>
Void Vector<LinearAlgebraBackend>::Clear() {
  vec_.Clear();
}

template <class LinearAlgebraBackend>
Vector<LinearAlgebraBackend> &
Vector<LinearAlgebraBackend>::Scale(const Scalar scalar) {
  vec_.Scale(scalar);

  return *this;
}

template <class LinearAlgebraBackend>
typename Vector<LinearAlgebraBackend>::Scalar &
Vector<LinearAlgebraBackend>::operator()(const LocalIndex index) {
  return vec_(index);
}

template <class LinearAlgebraBackend>
Vector<LinearAlgebraBackend> &
Vector<LinearAlgebraBackend>::operator=(Vector &vec) {
  vec_ = vec.vec_;

  return *this;
}

template <class LinearAlgebraBackend>
Vector<LinearAlgebraBackend> &Vector<LinearAlgebraBackend>::operator=(
    VectorEntry<LinearAlgebraBackend> vecEntry) {
  if (vecEntry.ExistLeftMatrixOperand()) {
    vecEntry.GetLeftMatrixOperand().MatrixVectorMultiplication(
        vecEntry.GetRightVectorOperand(), *this);
  }

  return *this;
}

template <class LinearAlgebraBackend>
Vector<LinearAlgebraBackend> &Vector<LinearAlgebraBackend>::operator+=(
    const Vector<LinearAlgebraBackend> &vec) {
  vec_ += vec.vec_;

  return *this;
}

template <class LinearAlgebraBackend>
Vector<LinearAlgebraBackend> &Vector<LinearAlgebraBackend>::operator-=(
    const Vector<LinearAlgebraBackend> &vec) {
  vec_ -= vec.vec_;

  return *this;
}

template <class LinearAlgebraBackend>
Vector<LinearAlgebraBackend> &Vector<LinearAlgebraBackend>::operator*=(
    const typename LinearAlgebraBackend::DefaultScalar scalar) {
  vec_ *= scalar;

  return *this;
}

template <class LinearAlgebraBackend>
typename LinearAlgebraBackend::VectorBase &
Vector<LinearAlgebraBackend>::GetVector() {
  return vec_;
}

template <class LinearAlgebraBackend>
Void Vector<LinearAlgebraBackend>::OrthogonalizeToConstant(const Integer start,
                                                           const Integer end) {
  vec_.OrthogonalizeToConstant(start, end);
}
} // namespace LinearAlgebra

#endif