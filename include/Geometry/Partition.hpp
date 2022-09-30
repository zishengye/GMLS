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

namespace Geometry {
/*! \note Type definition of InputAdapter
 *! \date Sept 30, 2022
 *! \author Zisheng Ye <zisheng_ye@outlook.com>
 *
 * The index type seems have to be signed types. Unsigned types fails the
 * instantiation of the adapter.
 */
typedef Zoltan2::BasicVectorAdapter<
    Zoltan2::BasicUserTypes<double, int, long long>>
    InputAdapter;

class Partition {
private:
  int mpiRank_, mpiSize_;

  std::vector<GlobalIndex> migrationInNum_, migrationOutNum_;

  std::vector<GlobalIndex> migrationInGraph_, migrationInGraphNum_;
  std::vector<GlobalIndex> migrationOutGraph_, migrationOutGraphNum_;

  std::vector<GlobalIndex> migrationInOffset_, migrationOutOffset_;

  std::vector<GlobalIndex> localReserveMap_, localMigrationMap_;

  std::vector<GlobalIndex> migrationMapIdx_;

public:
  Partition();

  ~Partition();

  Void ConstructPartition(const HostRealMatrix &coords,
                          const HostIndexVector &index);
  Void ApplyPartition(HostRealMatrix &data);
  Void ApplyPartition(HostRealVector &data);
  Void ApplyPartition(HostIndexVector &data);
  Void ApplyPartition(HostIntVector &data);
};
} // namespace Geometry

#endif