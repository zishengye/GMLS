#ifndef _StokesEquationPreconditioning_Hpp_
#define _StokesEquationPreconditioning_Hpp_

#include "MultilevelPreconditioning.hpp"

class StokesEquationPreconditioning : public MultilevelPreconditioning {
public:
  StokesEquationPreconditioning();
  ~StokesEquationPreconditioning();

  virtual void ConstructInterpolation(
      std::shared_ptr<HierarchicalParticleManager> particleMgr);
  virtual void ConstructRestriction(
      std::shared_ptr<HierarchicalParticleManager> particleMgr);
  virtual void
  ConstructSmoother(std::shared_ptr<HierarchicalParticleManager> particleMgr);
};

#endif