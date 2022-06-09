#ifndef _PoissonEquationPreconditioning_Hpp_
#define _PoissonEquationPreconditioning_Hpp_

#include "MultilevelPreconditioning.hpp"

class PoissonEquationPreconditioning : public MultilevelPreconditioning {
public:
  PoissonEquationPreconditioning();

  virtual void ConstructInterpolation(
      std::shared_ptr<HierarchicalParticleManager> particleMgr);
  virtual void ConstructRestriction(
      std::shared_ptr<HierarchicalParticleManager> particleMgr);
  virtual void ConstructSmoother();
};

#endif