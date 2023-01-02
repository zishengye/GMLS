#ifndef _LinearAlgebra_Impl_Default_DefaultLinearSolver_Hpp_
#define _LinearAlgebra_Impl_Default_DefaultLinearSolver_Hpp_

#include <functional>
#include <memory>

#include "Core/Typedef.hpp"
#include "LinearAlgebra/Impl/Default/Default.hpp"
#include "LinearAlgebra/Impl/Default/DefaultMatrix.hpp"
#include "LinearAlgebra/Impl/Default/DefaultVector.hpp"
#include "LinearAlgebra/Impl/Default/SquareMatrix.hpp"
#include "LinearAlgebra/LinearSolverDescriptor.hpp"

namespace LinearAlgebra {
namespace Impl {
class DefaultLinearSolver {
protected:
  int mpiRank_, mpiSize_;

  std::shared_ptr<DefaultMatrix> linearSystemPtr_;

  Boolean postCheck_, monitor_;

  Void GmresIteration(DefaultVector &b, DefaultVector &x);
  Void FlexGmresIteration(DefaultVector &b, DefaultVector &x);
  Void ConjugateGradientIteration(DefaultVector &b, DefaultVector &x);
  Void RichardsonIteration(DefaultVector &b, DefaultVector &x);
  Void PreconditioningOnlyIteration(DefaultVector &b, DefaultVector &x);
  Void CustomPreconditioningIteration(DefaultVector &b, DefaultVector &x);

  Scalar UpdateHessenbergMatrix(UpperHessenbergMatrix &hessenberg,
                                const LocalIndex size, const Scalar beta,
                                HostRealVector &y);

  Void UpdateSolution(HostRealVector &y, std::vector<DefaultVector> &v,
                      DefaultVector &x);

  std::shared_ptr<std::function<Void(DefaultVector &, DefaultVector &)>>
      solveFunctionPtr_;

  std::shared_ptr<std::function<Void(DefaultVector &, DefaultVector &)>>
      preconditioningFunctionPtr_;
  std::shared_ptr<std::function<Void(
      LinearAlgebra::Vector<LinearAlgebra::Impl::DefaultBackend> &,
      LinearAlgebra::Vector<LinearAlgebra::Impl::DefaultBackend> &)>>
      customPreconditioningFunctionPtr_;

  // common linear solver parameter
  LocalIndex maxIteration_;
  Scalar relativeTolerance_;

  // GMRes parameter and required intermidiate variables
  LocalIndex restartIteration_;

public:
  DefaultLinearSolver();

  ~DefaultLinearSolver();

  Void AddLinearSystem(
      std::shared_ptr<DefaultMatrix> matPtr,
      const LinearSolverDescriptor<LinearAlgebra::Impl::DefaultBackend>
          &descriptor);

  Void Solve(DefaultVector &b, DefaultVector &x);
};
} // namespace Impl
} // namespace LinearAlgebra

#endif