#ifndef _Equation_Stokes_StokesPreconditioner_Hpp_
#define _Equation_Stokes_StokesPreconditioner_Hpp_

#include <memory>
#include <mpi.h>
#include <vector>

#include "Equation/MultilevelPreconditioner.hpp"

namespace Equation {
class StokesPreconditioner : public MultilevelPreconditioner {
protected:
  std::vector<
      LinearAlgebra::LinearSolverDescriptor<DefaultLinearAlgebraBackend>>
      descriptorList_;

public:
  typedef typename MultilevelPreconditioner::DefaultParticleManager
      DefaultParticleManager;

  StokesPreconditioner();
  ~StokesPreconditioner();

  Void ApplyPreconditioningIteration(DefaultVector &x, DefaultVector &y);

  virtual Void ConstructInterpolation(DefaultParticleManager &particleMgr);
  virtual Void ConstructRestriction(DefaultParticleManager &particleMgr);
  virtual Void ConstructSmoother();
};
} // namespace Equation

#endif