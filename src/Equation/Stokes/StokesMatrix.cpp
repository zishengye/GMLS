#include "Equation/Stokes/StokesMatrix.hpp"
#include "LinearAlgebra/BlockMatrix.hpp"

Void Equation::StokesMatrix::ApplyPreconditioningIteration(DefaultVector &x,
                                                           DefaultVector &y) {
  this->ApplySchurComplementPreconditioningIteration(x, y);
}