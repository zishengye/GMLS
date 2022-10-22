#ifndef _TopologyOptimization_SolidIsotropicMicrostructurePenalization_Hpp_
#define _TopologyOptimization_SolidIsotropicMicrostructurePenalization_Hpp_

#include "TopologyOptimization/TopologyOptimization.hpp"

namespace TopologyOptimization {
class SolidIsotropicMicrostructurePenalization : public TopologyOptimization {
protected:
  virtual Void Output();

public:
  SolidIsotropicMicrostructurePenalization();

  ~SolidIsotropicMicrostructurePenalization();

  virtual Void Init();
  virtual Void Optimize();
};
} // namespace TopologyOptimization

#endif