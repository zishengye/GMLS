#ifndef _LinearAlgebra_Impl_Default_SquareMatrix_Hpp_
#define _LinearAlgebra_Impl_Default_SquareMatrix_Hpp_

#include "Core/Typedef.hpp"

namespace LinearAlgebra {
namespace Impl {
class SquareMatrix {
protected:
  HostRealVector mat_;

  LocalIndex rowSize_;

public:
  SquareMatrix();
  SquareMatrix(const LocalIndex m);

  virtual Scalar &operator()(const LocalIndex i, const LocalIndex j);
};

class UpperTriangleMatrix : public SquareMatrix {
public:
  UpperTriangleMatrix(const LocalIndex m);

  virtual Scalar &operator()(const LocalIndex i, const LocalIndex j);

  virtual Void Solve(HostRealVector &b, HostRealVector &x);
};

class UpperHessenbergMatrix : public SquareMatrix {
public:
  UpperHessenbergMatrix(const LocalIndex m);

  virtual Scalar &operator()(const LocalIndex i, const LocalIndex j);
};
} // namespace Impl
} // namespace LinearAlgebra

#endif