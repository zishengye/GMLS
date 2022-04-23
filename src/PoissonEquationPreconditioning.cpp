#include "PoissonEquationPreconditioning.hpp"

PoissonEquationPreconditioning::PoissonEquationPreconditioning() {}

void PoissonEquationPreconditioning::ConstructInterpolation(
    std::shared_ptr<HierarchicalParticleManager> particleMgr) {
  MultilevelPreconditioning::ConstructInterpolation(particleMgr);
}

void PoissonEquationPreconditioning::ConstructRestriction() {}

void PoissonEquationPreconditioning::ConstructSmoother() {
  PetscPrintf(PETSC_COMM_WORLD, "Construct Poisson equation smoother\n");
  MultilevelPreconditioning::ConstructSmoother();

  const int index = smootherPtr_.size() - 1;
  KSP &ksp = smootherPtr_[index]->GetReference();
  KSPCreate(MPI_COMM_WORLD, &ksp);
  KSPSetType(ksp, KSPPREONLY);

  PC pc;
  KSPGetPC(ksp, &pc);
  PCSetFromOptions(pc);
  PCSetUp(pc);

  KSPSetUp(ksp);
}