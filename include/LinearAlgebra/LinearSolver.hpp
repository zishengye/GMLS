#ifndef _LinearAlgebra_LinearSolver_Hpp_
#define _LinearAlgebra_LinearSolver_Hpp_

#include <memory>

#include "Core/Typedef.hpp"
#include "LinearAlgebra/LinearSolverDescriptor.hpp"
#include "LinearAlgebra/Matrix.hpp"
#include "LinearAlgebra/Vector.hpp"

namespace LinearAlgebra {
template <class LinearAlgebraBackend> class LinearSolver {
protected:
  typename LinearAlgebraBackend::LinearSolverBase solver_;

  typedef typename LinearAlgebraBackend::DefaultInteger Integer;
  typedef typename LinearAlgebraBackend::DefaultScalar Scalar;

public:
  LinearSolver();
  ~LinearSolver();

  virtual Void AddLinearSystem(
      std::shared_ptr<Matrix<LinearAlgebraBackend>> matPtr,
      const LinearSolverDescriptor<LinearAlgebraBackend> &descriptor);
  virtual Void Solve(Vector<LinearAlgebraBackend> &b,
                     Vector<LinearAlgebraBackend> &x);
};
} // namespace LinearAlgebra

#include "LinearAlgebra/Impl/LinearSolver.hpp"

#endif