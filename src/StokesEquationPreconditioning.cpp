#include "StokesEquationPreconditioning.hpp"
#include "PetscBlockMatrix.hpp"

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

void StokesEquationPreconditioning::ConstructSmoother() {
  double tStart, tEnd;
  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();

  PetscPrintf(PETSC_COMM_WORLD,
              "Start of constructing Stokes equation smoother\n");
  MultilevelPreconditioning::ConstructSmoother();

  const int currentLevel = linearSystemsPtr_.size() - 1;
  if (currentLevel > 0) {
  } else {
    KSP &ksp = smootherPtr_[currentLevel]->GetReference();
    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetType(ksp, KSPFGMRES);
    KSPSetTolerances(ksp, 1e-3, 1e-50, 1e20, 200);
    KSPSetOperators(ksp, linearSystemsPtr_[currentLevel]->GetReference(),
                    linearSystemsPtr_[currentLevel]->GetReference());

    schurComplementPreconditioningPtr_.push_back(
        std::make_shared<SchurComplementPreconditioning>());
    schurComplementPreconditioningPtr_[currentLevel]->AddLinearSystem(
        std::static_pointer_cast<PetscBlockMatrix>(
            linearSystemsPtr_[currentLevel]));

    PC pc;
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCSHELL);

    PCShellSetApply(pc, SchurComplementIterationWrapper);
    PCShellSetContext(pc,
                      schurComplementPreconditioningPtr_[currentLevel].get());

    KSPSetUp(ksp);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD,
              "Duration of building Stokes equation smoother:%.4fs\n",
              tEnd - tStart);
}