#ifndef _Geometry_Ghost_Hpp_
#define _Geometry_Ghost_Hpp_

#include <vector>

#include "Core/Parallel.hpp"
#include "Core/Typedef.hpp"

namespace Geometry {
class Ghost {
private:
  LocalIndex mpiRank_, mpiSize_;

  GlobalIndex localReserveNum_, remoteInNum_;

  std::vector<GlobalIndex> ghostOutGraph_, ghostInGraph_, ghostOutNum_,
      ghostInNum_;
  std::vector<GlobalIndex> ghostOutOffset_, ghostInOffset_;
  std::vector<GlobalIndex> ghostMap_, reserveMap_;

public:
  Ghost();

  Void Init(const HostRealMatrix &targetCoords,
            const HostRealVector &targetSpacing,
            const HostRealMatrix &sourceCoords, const Scalar multiplier,
            const LocalIndex dimension);
  Void ApplyGhost(const HostRealMatrix &sourceData, HostRealMatrix &ghostData);
  Void ApplyGhost(const HostRealVector &sourceData, HostRealVector &ghostData);
  Void ApplyGhost(const HostIntVector &sourceData, HostIntVector &ghostData);
  Void ApplyGhost(const HostIndexVector &sourceData,
                  HostIndexVector &ghostData);
};
} // namespace Geometry

#endif