#ifndef _MultilevelPreconditioning_HPP_
#define _MultilevelPreconditioning_HPP_

#include <vector>

#include "Ghost.hpp"
#include "ParticleManager.hpp"
#include "PetscKsp.hpp"
#include "PetscMatrix.hpp"

PetscErrorCode PreconditioningIterationWrapper(PC pc, Vec x, Vec y);

class MultilevelPreconditioning {
protected:
  std::vector<std::shared_ptr<PetscMatrix>> linearSystemsPtr_;
  std::vector<std::shared_ptr<PetscMatrix>> interpolationPtr_;
  std::vector<std::shared_ptr<PetscMatrix>> restrictionPtr_;

  std::vector<std::shared_ptr<PetscVector>> auxiliaryVectorXPtr_;
  std::vector<std::shared_ptr<PetscVector>> auxiliaryVectorRPtr_;
  std::vector<std::shared_ptr<PetscVector>> auxiliaryVectorBPtr_;

  std::vector<std::shared_ptr<PetscKsp>> smootherPtr_;

  Ghost interpolationGhost_, restrictionGhost_;

public:
  MultilevelPreconditioning();

  virtual PetscErrorCode ApplyPreconditioningIteration(Vec x, Vec y);

  friend PetscErrorCode PreconditioningIterationWrapper(PC, Vec, Vec);

  PetscMatrix &GetInterpolation(const int level);
  PetscMatrix &GetRestriction(const int level);
  PetscKsp &GetSmoother(const int level);

  void AddLinearSystem(std::shared_ptr<PetscMatrix> &mat);
  void PrepareVectors(const int localSize);
  virtual void ConstructInterpolation(
      std::shared_ptr<HierarchicalParticleManager> particleMgr);
  virtual void ConstructRestriction(
      std::shared_ptr<HierarchicalParticleManager> particleMgr);
  virtual void ConstructSmoother();
};

#endif