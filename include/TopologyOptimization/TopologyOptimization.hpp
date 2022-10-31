#ifndef _TopologyOptimization_TopologyOptimization_Hpp_
#define _TopologyOptimization_TopologyOptimization_Hpp_

#include "Equation/Equation.hpp"
#include "Geometry/ParticleGeometry.hpp"

namespace TopologyOptimization {
class TopologyOptimization {
public:
  typedef typename Geometry::HierarchicalEulerianParticleManager
      DefaultParticleManager;

protected:
  Equation::Equation *equationPtr_;

  DefaultParticleManager particleMgr_;

  Void CalculateSensitivity();
  virtual Void Output();
  virtual Void Output(String outputFileName);

  HostRealVector sensitivity_, density_, oldDensity_, volume_;

  Scalar volumeFraction_, domainVolume_;

  Size maxIteration_, iteration_;

  int mpiRank_, mpiSize_;

public:
  TopologyOptimization();

  ~TopologyOptimization();

  Void AddEquation(Equation::Equation &equation);

  Void SetDimension(const Size dimension);
  Void SetDomainType(const Geometry::SupportedDomainShape shape);
  Void SetDomainSize(const std::vector<Scalar> &size);
  Void SetInitialDiscretizationResolution(const Scalar spacing);
  Void SetVolumeFraction(const Scalar volumeFraction);
  Void SetMaxIteration(const Size maxIteration);

  virtual Void Init();
  virtual Void Optimize();
};
} // namespace TopologyOptimization

#endif