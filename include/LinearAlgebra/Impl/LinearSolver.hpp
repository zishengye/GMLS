#ifndef _LinearAlgebra_Impl_LinearSolver_Hpp_
#define _LinearAlgebra_Impl_LinearSolver_Hpp_

#include "LinearAlgebra/LinearSolverDescriptor.hpp"
#include <mpi.h>
namespace LinearAlgebra {
template <class LinearAlgebraBackend>
LinearSolver<LinearAlgebraBackend>::LinearSolver() {}

template <class LinearAlgebraBackend>
LinearSolver<LinearAlgebraBackend>::~LinearSolver() {}

template <class LinearAlgebraBackend>
Void LinearSolver<LinearAlgebraBackend>::AddLinearSystem(
    std::shared_ptr<Matrix<LinearAlgebraBackend>> mat,
    const LinearSolverDescriptor<LinearAlgebraBackend> &descriptor) {
  solver_.AddLinearSystem(mat->GetMatrix(), descriptor);

  if (descriptor.outerIteration > 0) {
    int mpiRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    if (mpiRank == 0) {
      printf("Summary of linear system setup\n");
      if (descriptor.outerIteration == 1 && descriptor.spd == -1)
        printf("\tFlexible-GMRES iterative method\n");
      else
        printf("\tGMRES iterative method\n");
    }
  }
}

template <class LinearAlgebraBackend>
Void LinearSolver<LinearAlgebraBackend>::Solve(
    Vector<LinearAlgebraBackend> &b, Vector<LinearAlgebraBackend> &x) {
  solver_.Solve(b.GetVector(), x.GetVector());
}
} // namespace LinearAlgebra

#endif