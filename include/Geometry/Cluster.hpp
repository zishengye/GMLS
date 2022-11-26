#ifndef _Geometry_Cluster_Hpp_
#define _Geometry_Cluster_Hpp_

#include <vector>

#include "Core/Typedef.hpp"
#include "Geometry/KdTree.hpp"

namespace Geometry {
class Cluster {
protected:
  std::vector<int> clusterIndex_;

public:
  Void ConstructCluster(HostRealMatrix &coords, const int dimension,
                        const int maxLeaf);
  Void ApplyCluster(HostRealMatrix &data);
  Void ApplyCluster(HostRealVector &data);
  Void ApplyCluster(HostIndexVector &data);
  Void ApplyCluster(HostIntVector &data);
};
} // namespace Geometry

#endif