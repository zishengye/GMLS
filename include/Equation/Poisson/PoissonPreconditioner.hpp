#ifndef _PoissonEquationPreconditioning_Hpp_
#define _PoissonEquationPreconditioning_Hpp_

#include "Equation/MultilevelPreconditioner.hpp"

namespace Equation {
class PoissonPreconditioner : public MultilevelPreconditioner {
public:
  typedef typename MultilevelPreconditioner::DefaultParticleManager
      DefaultParticleManager;

  PoissonPreconditioner();

  ~PoissonPreconditioner();

  virtual Void ConstructInterpolation(DefaultParticleManager &particleMgr);
  virtual Void ConstructRestriction(DefaultParticleManager &particleMgr);
  virtual Void ConstructSmoother();
};
} // namespace Equation

#endif