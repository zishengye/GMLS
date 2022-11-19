#ifndef _TopologyOptimization_GeneralizedBendersDecomposition_Hpp_
#define _TopologyOptimization_GeneralizedBendersDecomposition_Hpp_

#include <gurobi_c++.h>
#include <vector>

#include "Core/Typedef.hpp"
#include "TopologyOptimization/TopologyOptimization.hpp"

namespace TopologyOptimization {
class GeneralizedBendersDecomposition : public TopologyOptimization {
protected:
  virtual Void CalculateSensitivity();
  virtual Void Output();
  virtual Void Output(String outputFileName);

  Void MasterCut();

  std::vector<HostRealVector> sensitivityMatrix_;
  std::vector<double> objFunc_;
  std::vector<double> adjustedObjFunc_;
  std::vector<unsigned int> effectiveIndex_;
  std::vector<int> recvCount_;
  std::vector<int> recvOffset_;

  HostRealVector optimalDensity_;

  unsigned int volumeFractionParticleNum_, deltaParticleNum_,
      volumeConstraintParticleNum_;

  unsigned int innerLoop_, outerLoop_;

  Scalar cutCost_;

  // GRBEnv grbEnv_;

public:
  GeneralizedBendersDecomposition();

  ~GeneralizedBendersDecomposition();

  virtual Void Init();
  virtual Void Optimize();
};
} // namespace TopologyOptimization

#endif