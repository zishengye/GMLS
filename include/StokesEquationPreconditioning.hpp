#ifndef _StokesEquationPreconditioning_Hpp_
#define _StokesEquationPreconditioning_Hpp_

#include "MultilevelPreconditioning.hpp"
#include "SchurComplementPreconditioning.hpp"

class StokesEquationPreconditioning : public MultilevelPreconditioning {
protected:
  std::vector<std::shared_ptr<SchurComplementPreconditioning>>
      schurComplementPreconditioningPtr_;

public:
  StokesEquationPreconditioning();
  ~StokesEquationPreconditioning();

  virtual void ConstructInterpolation(
      std::shared_ptr<HierarchicalParticleManager> particleMgr);
  virtual void ConstructRestriction(
      std::shared_ptr<HierarchicalParticleManager> particleMgr);
  virtual void ConstructSmoother();
};

#endif