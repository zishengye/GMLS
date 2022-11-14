#include "Equation/Stokes/StokesPreconditioner.hpp"
#include "Equation/MultilevelPreconditioner.hpp"
#include "Equation/Stokes/StokesMatrix.hpp"

#include <Compadre_GMLS.hpp>
#include <Compadre_PointCloudSearch.hpp>
#include <memory>
#include <mpi.h>

Equation::StokesPreconditioner::StokesPreconditioner()
    : MultilevelPreconditioner() {}

Equation::StokesPreconditioner::~StokesPreconditioner() {}

Void Equation::StokesPreconditioner::ConstructInterpolation(
    DefaultParticleManager &particleMgr) {
  double tStart, tEnd;
  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();

  MultilevelPreconditioner::ConstructInterpolation(particleMgr);

  const int currentLevel = linearSystemsPtr_.size() - 1;

  if (currentLevel > 0) {
  }

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD,
              "Duration of building Stokes equation interpolation:%.4fs\n",
              tEnd - tStart);
}

Void Equation::StokesPreconditioner::ConstructRestriction(
    DefaultParticleManager &particleMgr) {
  double tStart, tEnd;
  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();

  MultilevelPreconditioner::ConstructRestriction(particleMgr);

  const int currentLevel = linearSystemsPtr_.size() - 1;

  if (currentLevel > 0) {
  }

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD,
              "Duration of building Stokes equation restriction:%.4fs\n",
              tEnd - tStart);
}

Void Equation::StokesPreconditioner::ConstructSmoother() {
  double tStart, tEnd;
  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();

  if (mpiRank_ == 0)
    printf("Start of constructing Stokes equation smoother\n");
  MultilevelPreconditioner::ConstructSmoother();

  const int currentLevel = linearSystemsPtr_.size() - 1;

  auto stokesPtr =
      std::static_pointer_cast<StokesMatrix>(linearSystemsPtr_[currentLevel]);
  stokesPtr->PrepareSchurComplementPreconditioner();

  descriptorList_.emplace_back();
  descriptorList_[currentLevel].setFromDatabase = false;
  descriptorList_[currentLevel].outerIteration = 0;
  descriptorList_[currentLevel].spd = -1;
  descriptorList_[currentLevel].maxIter = 500;
  descriptorList_[currentLevel].relativeTol = 1e-3;
  descriptorList_[currentLevel].customPreconditioner = true;

  descriptorList_[currentLevel].preconditioningIteration =
      std::function<Void(DefaultVector &, DefaultVector &)>(
          [=](DefaultVector &x, DefaultVector &y) {
            stokesPtr->ApplyPreconditioningIteration(x, y);
          });

  smootherPtr_[currentLevel].AddLinearSystem(linearSystemsPtr_[currentLevel],
                                             descriptorList_[currentLevel]);

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  if (mpiRank_ == 0)
    printf("Duration of building Stokes equation smoother:%.4fs\n",
           tEnd - tStart);
}