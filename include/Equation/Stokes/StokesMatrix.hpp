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
public:
  typedef typename LinearAlgebra::Vector<DefaultLinearAlgebraBackend>
      DefaultVector;

public:
  Void ApplyPreconditioningIteration(DefaultVector &x, DefaultVector &y);
};
} // namespace Equation

#endif