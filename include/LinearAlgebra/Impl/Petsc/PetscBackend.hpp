#ifndef _LinearAlgebra_Impl_Petsc_PetscBackend_Hpp_
#define _LinearAlgebra_Impl_Petsc_PetscBackend_Hpp_

#include "Core/Typedef.hpp"

#include "LinearAlgebra/Impl/Petsc/Petsc.hpp"
#include "LinearAlgebra/Impl/Petsc/PetscBlockMatrix.hpp"
#include "LinearAlgebra/Impl/Petsc/PetscKsp.hpp"
#include "LinearAlgebra/Impl/Petsc/PetscMatrix.hpp"
#include "LinearAlgebra/Impl/Petsc/PetscVector.hpp"

#include <string>

namespace LinearAlgebra {
namespace Impl {
class PetscBackend {
public:
  typedef LinearAlgebra::Impl::PetscMatrix MatrixBase;
  typedef LinearAlgebra::Impl::PetscBlockMatrix BlockMatrixBase;
  typedef LinearAlgebra::Impl::PetscVector VectorBase;
  typedef LinearAlgebra::Impl::PetscKsp LinearSolverBase;

  typedef PetscInt DefaultInteger;
  typedef PetscReal DefaultScalar;

  PetscBackend();
  ~PetscBackend();

  static Void LinearAlgebraInitialize(int *argc, char ***args,
                                      std::string &fileName);
  static Void LinearAlgebraFinalize();
};
} // namespace Impl
} // namespace LinearAlgebra

#endif