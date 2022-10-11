#ifndef _LinearAlgebra_Impl_Petsc_PetscKsp_Hpp_
#define _LinearAlgebra_Impl_Petsc_PetscKsp_Hpp_

#include <memory>

#include <petscksp.h>

#include "Core/Typedef.hpp"
#include "LinearAlgebra/Impl/Petsc/Petsc.hpp"
#include "LinearAlgebra/LinearSolverDescriptor.hpp"

namespace LinearAlgebra {
namespace Impl {
class PetscKsp {
private:
  std::shared_ptr<KSP> kspPtr_;

  Boolean postCheck_;
  Boolean copyConstructed_;

public:
  PetscKsp();

  ~PetscKsp();

  Void AddLinearSystem(
      PetscMatrix &mat,
      const LinearSolverDescriptor<LinearAlgebra::Impl::PetscBackend>
          &descriptor);

  Void Solve(PetscVector &b, PetscVector &x);
};
} // namespace Impl
} // namespace LinearAlgebra

#endif