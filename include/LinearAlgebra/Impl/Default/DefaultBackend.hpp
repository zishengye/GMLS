#ifndef _LinearAlgebra_Impl_Default_DefaultBackend_Hpp_
#define _LinearAlgebra_Impl_Default_DefaultBackend_Hpp_

#include "Core/Typedef.hpp"

#include "LinearAlgebra/Impl/Default/Default.hpp"
#include "LinearAlgebra/Impl/Default/DefaultBlockMatrix.hpp"
#include "LinearAlgebra/Impl/Default/DefaultLinearSolver.hpp"
#include "LinearAlgebra/Impl/Default/DefaultMatrix.hpp"
#include "LinearAlgebra/Impl/Default/DefaultVector.hpp"

#include <string>

namespace LinearAlgebra {
namespace Impl {
class DefaultBackend {
public:
  typedef LinearAlgebra::Impl::DefaultMatrix MatrixBase;
  typedef LinearAlgebra::Impl::DefaultBlockMatrix BlockMatrixBase;
  typedef LinearAlgebra::Impl::DefaultVector VectorBase;
  typedef LinearAlgebra::Impl::DefaultLinearSolver LinearSolverBase;

  typedef GlobalIndex DefaultInteger;
  typedef Scalar DefaultScalar;

  DefaultBackend();
  ~DefaultBackend();

  static Void LinearAlgebraInitialize(int *argc, char ***args,
                                      std::string &fileName);
  static Void LinearAlgebraFinalize();
};
} // namespace Impl
} // namespace LinearAlgebra

#endif