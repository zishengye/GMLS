#include "Equation/Stokes/StokesMatrix.hpp"
#include "LinearAlgebra/BlockMatrix.hpp"

Void Equation::StokesMatrix::MatrixVectorMultiplication(DefaultVector &x,
                                                        DefaultVector &y) {
  x.OrthogonalizeToConstant(localRhsVectorOffset_[1], localRhsVectorOffset_[2]);
  BlockMatrix::MatrixVectorMultiplication(x, y);
  y.OrthogonalizeToConstant(localLhsVectorOffset_[1], localLhsVectorOffset_[2]);
}

Void Equation::StokesMatrix::ApplyPreconditioningIteration(DefaultVector &x,
                                                           DefaultVector &y) {
  x.OrthogonalizeToConstant(localRhsVectorOffset_[1], localRhsVectorOffset_[2]);
  this->ApplySchurComplementPreconditioningIteration(x, y);
  y.OrthogonalizeToConstant(localLhsVectorOffset_[1], localLhsVectorOffset_[2]);
}