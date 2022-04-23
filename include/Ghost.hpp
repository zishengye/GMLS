#ifndef _Ghost_Hpp_
#define _Ghost_Hpp_

#include <vector>

#include "Parallel.hpp"
#include "Typedef.hpp"

class Ghost {
private:
  int mpiRank_, mpiSize_;

  int localReserveNum_, remoteInNum_;

  std::vector<int> ghostOutGraph_, ghostInGraph_, ghostOutNum_, ghostInNum_;
  std::vector<int> ghostOutOffset_, ghostInOffset_;
  std::vector<int> ghostMap_, reserveMap_;

public:
  Ghost();

  void Init(const HostRealMatrix &targetCoords,
            const HostRealVector &targetSpacing,
            const HostRealMatrix &sourceCoords, const double multiplier,
            const int dimension);
  void ApplyGhost(const HostRealMatrix &sourceData, HostRealMatrix &ghostData);
  void ApplyGhost(const HostRealVector &sourceData, HostRealVector &ghostData);
  void ApplyGhost(const HostIntVector &sourceData, HostIntVector &ghostData);
  void ApplyGhost(const HostIndexVector &sourceData,
                  HostIndexVector &ghostData);
};

#endif