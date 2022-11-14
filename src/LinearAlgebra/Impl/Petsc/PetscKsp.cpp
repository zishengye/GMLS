#include "LinearAlgebra/Impl/Petsc/PetscKsp.hpp"
#include "LinearAlgebra/Impl/Petsc/PetscBackend.hpp"
#include "LinearAlgebra/Impl/Petsc/PetscMatrix.hpp"
#include "LinearAlgebra/Impl/Petsc/PetscVector.hpp"
#include "LinearAlgebra/LinearSolverDescriptor.hpp"
#include "petscksp.h"
#include "petscpc.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include <memory>
#include <mpi.h>

PetscErrorCode PCShellWrapper(PC pc, Vec x, Vec y) {
  LinearAlgebra::LinearSolverDescriptor<LinearAlgebra::Impl::PetscBackend>
      *descriptor;

  PCShellGetContext(pc, (void **)&descriptor);

  LinearAlgebra::Vector<LinearAlgebra::Impl::PetscBackend> vecX;
  LinearAlgebra::Vector<LinearAlgebra::Impl::PetscBackend> vecY;

  LinearAlgebra::Impl::PetscVector vecPetscX;

  vecPetscX.Create(x);

  vecX.Create(vecPetscX);
  vecY.Create(vecPetscX);

  int mpiRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

  descriptor->preconditioningIteration(vecX, vecY);

  vecY.Copy(vecPetscX);

  vecPetscX.Copy(y);

  return 0;
}

LinearAlgebra::Impl::PetscKsp::PetscKsp()
    : postCheck_(false), copyConstructed_(false) {
  kspPtr_ = std::make_shared<KSP>();

  *kspPtr_ = PETSC_NULL;
}

LinearAlgebra::Impl::PetscKsp::~PetscKsp() {
  if (kspPtr_.use_count() == 1)
    if (*kspPtr_ != PETSC_NULL)
      KSPDestroy(kspPtr_.get());
}

Void LinearAlgebra::Impl::PetscKsp::AddLinearSystem(
    PetscMatrix &mat,
    const LinearSolverDescriptor<LinearAlgebra::Impl::PetscBackend>
        &descriptor) {
  if (*kspPtr_ != PETSC_NULL)
    KSPDestroy(kspPtr_.get());

  KSPCreate(MPI_COMM_WORLD, kspPtr_.get());
  if (descriptor.outerIteration == 1) {
    KSPSetType(*kspPtr_, KSPFGMRES);
  } else if (descriptor.outerIteration == 0) {
    KSPSetType(*kspPtr_, KSPGMRES);
    KSPSetTolerances(*kspPtr_, 1e-1, 1e-50, 1e20, 500);
  } else {
    KSPSetType(*kspPtr_, KSPPREONLY);
  }
  KSPSetOperators(*kspPtr_, *mat.matPtr_, *mat.matPtr_);

  if (descriptor.setFromDatabase)
    KSPSetFromOptions(*kspPtr_);

  if (descriptor.outerIteration > 0) {
    PC pc;
    KSPGetPC(*kspPtr_, &pc);
    PCSetType(pc, PCSHELL);
    PCShellSetContext(pc, (void *)&descriptor);
    PCShellSetApply(pc, PCShellWrapper);

    postCheck_ = true;
  } else {
    if (descriptor.customPreconditioner == false) {
      PC pc;
      KSPGetPC(*kspPtr_, &pc);
      PCSetType(pc, PCSOR);
      PCSetFromOptions(pc);
    } else {
      PC pc;
      KSPGetPC(*kspPtr_, &pc);
      PCSetType(pc, PCSHELL);
      PCShellSetContext(pc, (void *)&descriptor);
      PCShellSetApply(pc, PCShellWrapper);
    }
  }

  KSPSetUp(*kspPtr_);
}

Void LinearAlgebra::Impl::PetscKsp::Solve(PetscVector &b, PetscVector &x) {
  KSPSolve(*kspPtr_, *b.vecPtr_, *x.vecPtr_);

  if (postCheck_) {
    Vec residual;
    Mat op;
    VecDuplicate(*b.vecPtr_, &residual);
    KSPGetOperators(*kspPtr_, &op, PETSC_NULL);
    MatMult(op, *x.vecPtr_, residual);
    VecAXPY(residual, -1.0, *b.vecPtr_);
    PetscReal norm1, norm2;
    VecNorm(residual, NORM_2, &norm2);
    VecNorm(*b.vecPtr_, NORM_2, &norm1);
    PetscPrintf(PETSC_COMM_WORLD, "The residual norm in the post check: %f\n",
                norm2 / norm1);
    VecDestroy(&residual);

    PetscLogDouble mem, maxMem;
    PetscMemoryGetCurrentUsage(&mem);
    MPI_Allreduce(&mem, &maxMem, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD,
                "Current memory usage %.2f GB, maximum memory usage: %.2f GB\n",
                mem / 1e9, maxMem / 1e9);
  }
}