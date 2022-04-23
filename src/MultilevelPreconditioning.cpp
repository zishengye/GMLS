#include "MultilevelPreconditioning.hpp"

PetscErrorCode PreconditioningIterationWrapper(PC pc, Vec x, Vec y) {
  MultilevelPreconditioning *shellPtr;
  PCShellGetContext(pc, (void **)&shellPtr);

  return shellPtr->ApplyPreconditioningIteration(x, y);
}

MultilevelPreconditioning::MultilevelPreconditioning() {}

PetscErrorCode MultilevelPreconditioning::ApplyPreconditioningIteration(Vec x,
                                                                        Vec y) {
  const int numLevel = linearSystemsPtr_.size();
  // sweep down
  VecCopy(x, auxiliaryVectorBPtr_[numLevel - 1]->GetReference());
  for (int i = numLevel - 1; i > 0; i--) {
    // pre-smooth
    KSPSolve(smootherPtr_[i]->GetReference(),
             auxiliaryVectorBPtr_[i]->GetReference(),
             auxiliaryVectorXPtr_[i]->GetReference());

    // get residual
    MatMult(linearSystemsPtr_[i]->GetReference(),
            auxiliaryVectorXPtr_[i]->GetReference(),
            auxiliaryVectorRPtr_[i]->GetReference());
    VecAXPY(auxiliaryVectorRPtr_[i]->GetReference(), -1.0,
            auxiliaryVectorBPtr_[i]->GetReference());
    VecScale(auxiliaryVectorRPtr_[i]->GetReference(), -1.0);

    // restrict
    MatMult(restrictionPtr_[i]->GetReference(),
            auxiliaryVectorRPtr_[i]->GetReference(),
            auxiliaryVectorBPtr_[i - 1]->GetReference());
  }

  // smooth on the base level
  KSPSolve(smootherPtr_[0]->GetReference(),
           auxiliaryVectorBPtr_[0]->GetReference(),
           auxiliaryVectorXPtr_[0]->GetReference());

  // sweep up
  for (int i = 1; i < numLevel; i++) {
    // interpolate
    MatMultAdd(interpolationPtr_[i]->GetReference(),
               auxiliaryVectorXPtr_[i - 1]->GetReference(),
               auxiliaryVectorXPtr_[i]->GetReference(),
               auxiliaryVectorXPtr_[i]->GetReference());

    // get residual
    MatMult(linearSystemsPtr_[i]->GetReference(),
            auxiliaryVectorXPtr_[i]->GetReference(),
            auxiliaryVectorRPtr_[i]->GetReference());
    VecAXPY(auxiliaryVectorRPtr_[i]->GetReference(), -1.0,
            auxiliaryVectorBPtr_[i]->GetReference());
    VecScale(auxiliaryVectorRPtr_[i]->GetReference(), -1.0);

    // post smooth
    KSPSolve(smootherPtr_[i]->GetReference(),
             auxiliaryVectorRPtr_[i]->GetReference(),
             auxiliaryVectorBPtr_[i]->GetReference());
    VecAXPY(auxiliaryVectorXPtr_[i]->GetReference(), 1.0,
            auxiliaryVectorBPtr_[i]->GetReference());
  }
  VecCopy(auxiliaryVectorXPtr_[numLevel - 1]->GetReference(), y);

  return 0;
}

void MultilevelPreconditioning::AddLinearSystem(
    std::shared_ptr<PetscMatrix> &mat) {
  linearSystemsPtr_.push_back(mat);
}

void MultilevelPreconditioning::ConstructInterpolation(
    std::shared_ptr<HierarchicalParticleManager> particleMgr) {
  // build ghost
  Ghost ghost;
}

void MultilevelPreconditioning::ConstructRestriction() {}

void MultilevelPreconditioning::ConstructSmoother() {
  smootherPtr_.push_back(std::make_shared<PetscKsp>());
}