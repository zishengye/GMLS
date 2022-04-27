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

  std::vector<std::size_t> migrationInNum_, migrationOutNum_;

  std::vector<std::size_t> migrationInGraph_, migrationInGraphNum_;
  std::vector<std::size_t> migrationOutGraph_, migrationOutGraphNum_;

  std::vector<std::size_t> migrationInOffset_, migrationOutOffset_;

  std::vector<std::size_t> localReserveMap_, localMigrationMap_;

  std::vector<std::size_t> migrationMapIdx_;

public:
  Partition();

  ~Partition();

  void ConstructPartition(Kokkos::View<Scalar **> coords,
                          Kokkos::View<GlobalIndex *> index);
  void ApplyPartition(Kokkos::View<Scalar **> data);
  void ApplyPartition(Kokkos::View<Scalar *> data);
  void ApplyPartition(Kokkos::View<std::size_t *> data);
  void ApplyPartition(Kokkos::View<int *> data);
};

#endif