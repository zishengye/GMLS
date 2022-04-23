#ifndef _PoissonEquationPreconditioning_HPP_
#define _PoissonEquationPreconditioning_HPP_

#include "MultilevelPreconditioning.hpp"

class PoissonEquationPreconditioning : public MultilevelPreconditioning {

public:
  PoissonEquationPreconditioning();

  virtual void ConstructInterpolation(
      std::shared_ptr<HierarchicalParticleManager> particleMgr);
  virtual void ConstructRestriction();
  virtual void ConstructSmoother();
};

#endif