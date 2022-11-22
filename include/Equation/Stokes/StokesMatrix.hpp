#ifndef _Equation_Stokes_StokesMatrix_Hpp_
#define _Equation_Stokes_StokesMatrix_Hpp_

#include "Core/Typedef.hpp"
#include "LinearAlgebra/BlockMatrix.hpp"
#include "LinearAlgebra/LinearAlgebra.hpp"

#include <memory>
#include <vector>

namespace Equation {
class StokesMatrix
    : public LinearAlgebra::BlockMatrix<DefaultLinearAlgebraBackend> {
protected:
  double matrixMultiplicationDuration_, matrixSmoothingDuration_;

public:
  typedef typename LinearAlgebra::Vector<DefaultLinearAlgebraBackend>
      DefaultVector;

public:
  Void ClearTimer();

  double GetMatrixMultiplicationDuration();
  double GetMatrixSmoothingDuration();
  double GetVelocitySmoothingDuration();
  double GetPressureSmoothingDuration();

  virtual Void MatrixVectorMultiplication(DefaultVector &vec1,
                                          DefaultVector &vec2);

  Void ApplyPreconditioningIteration(DefaultVector &x, DefaultVector &y);
};
} // namespace Equation

#endif