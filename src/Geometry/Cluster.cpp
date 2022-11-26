#include "Geometry/Cluster.hpp"
#include "Core/Typedef.hpp"
#include "Geometry/KdTree.hpp"
#include "Kokkos_UnorderedMap.hpp"
#include <mpi.h>

void Geometry::Cluster::ConstructCluster(HostRealMatrix &coords,
                                         const int dimension,
                                         const int maxLeaf) {
  Geometry::KdTree pointCloud(coords, dimension, maxLeaf);
  pointCloud.generateKDTree();

  pointCloud.getIndex(clusterIndex_);
}

void Geometry::Cluster::ApplyCluster(HostRealMatrix &data) {
  HostRealMatrix copyData;
  Kokkos::resize(copyData, data.extent(0), data.extent(1));
  Kokkos::deep_copy(copyData, data);

  for (int i = 0; i < clusterIndex_.size(); i++)
    for (int j = 0; j < data.extent(1); j++)
      data(i, j) = copyData(clusterIndex_[i], j);
}

void Geometry::Cluster::ApplyCluster(HostRealVector &data) {
  HostRealVector copyData;
  Kokkos::resize(copyData, data.extent(0));
  Kokkos::deep_copy(copyData, data);

  for (int i = 0; i < clusterIndex_.size(); i++)
    data(i) = copyData(clusterIndex_[i]);
}

void Geometry::Cluster::ApplyCluster(HostIndexVector &data) {
  HostIndexVector copyData;
  Kokkos::resize(copyData, data.extent(0));
  Kokkos::deep_copy(copyData, data);

  for (int i = 0; i < clusterIndex_.size(); i++)
    data(i) = copyData(clusterIndex_[i]);
}

void Geometry::Cluster::ApplyCluster(HostIntVector &data) {
  HostIntVector copyData;
  Kokkos::resize(copyData, data.extent(0));
  Kokkos::deep_copy(copyData, data);

  for (int i = 0; i < clusterIndex_.size(); i++)
    data(i) = copyData(clusterIndex_[i]);
}