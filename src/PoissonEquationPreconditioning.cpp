#include "PoissonEquationPreconditioning.hpp"

PoissonEquationPreconditioning::PoissonEquationPreconditioning() {}

void PoissonEquationPreconditioning::ConstructInterpolation(
    std::shared_ptr<HierarchicalParticleManager> particleMgr) {
  MultilevelPreconditioning::ConstructInterpolation(particleMgr);

  const int currentLevel = linearSystemsPtr_.size() - 1;

  if (currentLevel > 0) {
    HostRealMatrix interpolationSourceParticleCoords;
    interpolationGhost_.ApplyGhost(
        particleMgr->GetParticleCoordsByLevel(currentLevel - 1),
        interpolationSourceParticleCoords);
  }
}

void PoissonEquationPreconditioning::ConstructRestriction(
    std::shared_ptr<HierarchicalParticleManager> particleMgr) {
  MultilevelPreconditioning::ConstructRestriction(particleMgr);

  const int currentLevel = linearSystemsPtr_.size() - 1;

  if (currentLevel > 0) {
    HostRealMatrix restrictionSourceParticleCoords;
    restrictionGhost_.ApplyGhost(
        particleMgr->GetParticleCoordsByLevel(currentLevel),
        restrictionSourceParticleCoords);

    HostIntVector restrictionSourceParticleType;
    restrictionGhost_.ApplyGhost(
        particleMgr->GetParticleTypeByLevel(currentLevel),
        restrictionSourceParticleType);
  }
}

void PoissonEquationPreconditioning::ConstructSmoother() {
  PetscPrintf(PETSC_COMM_WORLD,
              "Start of constructing Poisson equation smoother\n");
  MultilevelPreconditioning::ConstructSmoother();

  const int currentLevel = linearSystemsPtr_.size() - 1;
  KSP &ksp = smootherPtr_[currentLevel]->GetReference();
  KSPCreate(MPI_COMM_WORLD, &ksp);
  KSPSetType(ksp, KSPPREONLY);
  KSPSetOperators(ksp, linearSystemsPtr_[currentLevel]->GetReference(),
                  linearSystemsPtr_[currentLevel]->GetReference());

  PC pc;
  KSPGetPC(ksp, &pc);
  PCSetFromOptions(pc);
  PCSetUp(pc);

  KSPSetUp(ksp);
}