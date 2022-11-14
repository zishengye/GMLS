#ifndef _LinearAlgebra_LinearAlgebra_Hpp_
#define _LinearAlgebra_LinearAlgebra_Hpp_

#include "Core/Parallel.hpp"
#include "Core/Typedef.hpp"

namespace LinearAlgebra {
template <class LinearAlgebraBackend>
Void LinearAlgebraInitialize(int *argc, char ***args, const char file[],
                             const char help[]) {
  LinearAlgebraBackend::LinearAlgebraInitialize(argc, args, file, help);
}

template <class LinearAlgebraBackend> Void LinearAlgebraFinalize() {
  LinearAlgebraBackend::LinearAlgebraFinalize();
}
} // namespace LinearAlgebra

#include "LinearAlgebra/Impl/Petsc/PetscBackend.hpp"
typedef LinearAlgebra::Impl::PetscBackend DefaultLinearAlgebraBackend;

#include "LinearAlgebra/BlockMatrix.hpp"
#include "LinearAlgebra/LinearSolver.hpp"
#include "LinearAlgebra/Matrix.hpp"
#include "LinearAlgebra/Vector.hpp"
#include "LinearAlgebra/VectorEntry.hpp"

#endif