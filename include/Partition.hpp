#ifndef _Partition_Hpp_
#define _Partition_Hpp_

#include <algorithm>
#include <vector>

#include <Zoltan2_BasicVectorAdapter.hpp>
#include <Zoltan2_CoordinatePartitioningGraph.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_PartitioningSolution.hpp>

#include "Typedef.hpp"

typedef Zoltan2::BasicVectorAdapter<
    Zoltan2::BasicUserTypes<double, int, long long>>
    InputAdapter;

class Partition {
private:
  int mpiRank_, mpiSize_;

  std::vector<int> migrationInNum_, migrationOutNum_;

  std::vector<int> migrationInGraph_, migrationInGraphNum_;
  std::vector<int> migrationOutGraph_, migrationOutGraphNum_;

  std::vector<int> migrationInOffset_, migrationOutOffset_;

  std::vector<int> localReserveMap_, localMigrationMap_;

  std::vector<int> migrationMapIdx_;

public:
  Partition();

  void ConstructPartition(Kokkos::View<Scalar **> coords,
                          Kokkos::View<GlobalIndex *> index);
  void ApplyPartition(Kokkos::View<Scalar **> data);
  void ApplyPartition(Kokkos::View<Scalar *> data);
  void ApplyPartition(Kokkos::View<LocalIndex *> data);
};

#endif