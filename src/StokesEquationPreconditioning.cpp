#include "StokesEquationPreconditioning.hpp"

#include <Compadre_GMLS.hpp>
#include <Compadre_PointCloudSearch.hpp>

StokesEquationPreconditioning::StokesEquationPreconditioning() {}

StokesEquationPreconditioning::~StokesEquationPreconditioning() {}

void StokesEquationPreconditioning::ConstructInterpolation(
    std::shared_ptr<HierarchicalParticleManager> particleMgr) {
  double tStart, tEnd;
  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();

  MultilevelPreconditioning::ConstructInterpolation(particleMgr);

  const int currentLevel = linearSystemsPtr_.size() - 1;

  if (currentLevel > 0) {
  }

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD,
              "Duration of building Stokes equation interpolation:%.4fs\n",
              tEnd - tStart);
}

void StokesEquationPreconditioning::ConstructRestriction(
    std::shared_ptr<HierarchicalParticleManager> particleMgr) {
  double tStart, tEnd;
  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();

  MultilevelPreconditioning::ConstructRestriction(particleMgr);

  const int currentLevel = linearSystemsPtr_.size() - 1;

  if (currentLevel > 0) {
  }

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD,
              "Duration of building Stokes equation restriction:%.4fs\n",
              tEnd - tStart);
}

void StokesEquationPreconditioning::ConstructSmoother(
    std::shared_ptr<HierarchicalParticleManager> particleMgr) {
  PetscPrintf(PETSC_COMM_WORLD,
              "Start of constructing Stokes equation smoother\n");
  MultilevelPreconditioning::ConstructSmoother(particleMgr);

  const int currentLevel = linearSystemsPtr_.size() - 1;
  if (currentLevel > 0) {
  }
}