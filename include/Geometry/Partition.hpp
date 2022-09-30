#ifndef _Geometry_Partition_Hpp_
#define _Geometry_Partition_Hpp_

#include <algorithm>
#include <vector>

#include <Zoltan2_BasicVectorAdapter.hpp>
#include <Zoltan2_CoordinatePartitioningGraph.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_PartitioningSolution.hpp>

#include "Core/Parallel.hpp"
#include "Core/Typedef.hpp"

typedef Zoltan2::BasicVectorAdapter<
    Zoltan2::BasicUserTypes<Scalar, LocalIndex, GlobalIndex>>
    InputAdapter;

class Partition {
private:
  int mpiRank_, mpiSize_;

  std::vector<std::size_t> migrationInNum_, migrationOutNum_;

  std::vector<std::size_t> migrationInGraph_, migrationInGraphNum_;
  std::vector<std::size_t> migrationOutGraph_, migrationOutGraphNum_;

  std::vector<std::size_t> migrationInOffset_, migrationOutOffset_;

  std::vector<std::size_t> localReserveMap_, localMigrationMap_;

  std::vector<std::size_t> migrationMapIdx_;

public:
  Partition();

  ~Partition();

  void ConstructPartition(const HostRealMatrix &coords,
                          const HostIndexVector &index);
  void ApplyPartition(HostRealMatrix &data);
  void ApplyPartition(HostRealVector &data);
  void ApplyPartition(HostIndexVector &data);
  void ApplyPartition(HostIntVector &data);
};

#endif