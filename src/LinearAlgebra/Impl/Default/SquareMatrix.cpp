#include "LinearAlgebra/Impl/Default/SquareMatrix.hpp"
#include "Core/Typedef.hpp"
#include <Cuda/Kokkos_Cuda_Team.hpp>

LinearAlgebra::Impl::SquareMatrix::SquareMatrix() {}

LinearAlgebra::Impl::SquareMatrix::SquareMatrix(const LocalIndex m) {
  rowSize_ = m;
  Kokkos::resize(mat_, rowSize_ * rowSize_);
}

Scalar &LinearAlgebra::Impl::SquareMatrix::operator()(const LocalIndex i,
                                                      const LocalIndex j) {
  return mat_(i * rowSize_ + j);
}

LinearAlgebra::Impl::UpperTriangleMatrix::UpperTriangleMatrix(
    const LocalIndex m)
    : SquareMatrix() {
  rowSize_ = m;
  Kokkos::resize(mat_, (rowSize_ + 1) * rowSize_ / 2);
}

Scalar &
LinearAlgebra::Impl::UpperTriangleMatrix::operator()(const LocalIndex i,
                                                     const LocalIndex j) {
  return mat_(j * (j + 1) / 2 + i);
}

Void LinearAlgebra::Impl::UpperTriangleMatrix::Solve(HostRealVector &b,
                                                     HostRealVector &x) {
  for (int i = 0; i < rowSize_; i++)
    x(i) = b(i);

  for (int i = rowSize_ - 1; i >= 0; i--) {
    x(i) = x(i) / operator()(i, i);

    if (i > 20)
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, i),
          [&](const int j) { x(j) -= operator()(j, i) * x(i); });
    else
      for (int j = 0; j < i; j++)
        x(j) -= operator()(j, i) * x(i);
  }
}

LinearAlgebra::Impl::UpperHessenbergMatrix::UpperHessenbergMatrix(
    const LocalIndex m)
    : SquareMatrix() {
  rowSize_ = m;
  Kokkos::resize(mat_, (rowSize_ + 3) * rowSize_ / 2);
}

Scalar &
LinearAlgebra::Impl::UpperHessenbergMatrix::operator()(const LocalIndex i,
                                                       const LocalIndex j) {
  return mat_(j * (j + 3) / 2 + i);
}