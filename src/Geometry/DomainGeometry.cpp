#include "Geometry/DomainGeometry.hpp"
#include "Core/Typedef.hpp"

Geometry::DomainGeometry::DomainGeometry() {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);
}

Geometry::DomainGeometry::DomainGeometry(const DomainGeometry &geo) {
  dimension_ = geo.dimension_;
  size_ = geo.size_;

  mpiRank_ = geo.mpiRank_;
  mpiSize_ = geo.mpiSize_;
}

Geometry::DomainGeometry::~DomainGeometry() {}

bool Geometry::DomainGeometry::IsInterior(Scalar x, Scalar y, Scalar z) {
  if (dimension_ == 2) {
    if (shape_ == Box) {
      if (x > -size_[0] / 2.0 && x < size_[0] / 2.0 && y > -size_[1] / 2.0 &&
          y < size_[1] / 2.0)
        return true;
      else
        return false;
    } else if (shape_ == Cylinder) {
      Scalar r = sqrt(x * x + y * y);
      if (r < size_[0])
        return true;
      else
        return false;
    } else {
      return false;
    }
  }

  if (dimension_ == 3) {
    if (shape_ == Box) {
      if (x > -size_[0] / 2.0 && x < size_[0] / 2.0 && y > -size_[1] / 2.0 &&
          y < size_[1] / 2.0 && z > -size_[2] / 2.0 && z < size_[2] / 2.0)
        return true;
      else
        return false;
    } else if (shape_ == Cylinder) {
      Scalar r = sqrt(x * x + y * y);
      if (r < size_[0] && z > -size_[1] / 2.0 && z < size_[1] / 2.0)
        return true;
      else
        return false;
    } else {
      return false;
    }
  }

  return false;
}

Void Geometry::DomainGeometry::IsInterior(HostRealMatrix coords,
                                          HostBooleanVector results) {
  if (dimension_ == 2) {
    if (shape_ == Box) {
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
              0, coords.extent(0)),
          [=](const int i) {
            if (coords(i, 0) > -size_[0] / 2.0 &&
                coords(i, 0) < size_[0] / 2.0 &&
                coords(i, 1) > -size_[1] / 2.0 && coords(i, 1) < size_[1] / 2.0)
              results(i) = true;
            else
              results(i) = false;
          });
      Kokkos::fence();
    }
  }

  if (dimension_ == 3) {
    if (shape_ == Box) {
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
              0, coords.extent(0)),
          [=](const int i) {
            if (coords(i, 0) > -size_[0] / 2.0 &&
                coords(i, 0) < size_[0] / 2.0 &&
                coords(i, 1) > -size_[1] / 2.0 &&
                coords(i, 1) < size_[1] / 2.0 &&
                coords(i, 2) > -size_[2] / 2.0 && coords(i, 2) < size_[2] / 2.0)
              results(i) = true;
            else
              results(i) = false;
          });
      Kokkos::fence();
    }
  }
}

Void Geometry::DomainGeometry::SetType(
    const Geometry::SupportedDomainShape shape) {
  shape_ = shape;
}

Void Geometry::DomainGeometry::SetDimension(const Size dimension) {
  dimension_ = dimension;
}

Void Geometry::DomainGeometry::SetSize(const std::vector<Scalar> &size) {
  size_ = size;
}

Geometry::SupportedDomainShape Geometry::DomainGeometry::GetType() {
  return shape_;
}

Size Geometry::DomainGeometry::GetDimension() { return dimension_; }

Scalar Geometry::DomainGeometry::GetSize(const LocalIndex size_index) {
  return size_[size_index];
}

LocalIndex Geometry::DomainGeometry::EstimateNodeNum(const Scalar spacing) {
  GlobalIndex globalBoundaryNodeNum = 0, globalInteriorNodeNum = 0;

  if (dimension_ == 2) {
    if (shape_ == Box) {
      LocalIndex xNodeNum = ceil(size_[0] / spacing);
      LocalIndex yNodeNum = ceil(size_[1] / spacing);
      globalBoundaryNodeNum = 2 * xNodeNum + 2 * yNodeNum;
      globalInteriorNodeNum = xNodeNum * yNodeNum;
    }
  }

  if (dimension_ == 3) {
    if (shape_ == Box) {
      LocalIndex xNodeNum = ceil(size_[0] / spacing);
      LocalIndex yNodeNum = ceil(size_[1] / spacing);
      LocalIndex zNodeNum = ceil(size_[2] / spacing);
      globalBoundaryNodeNum = 2 * xNodeNum * yNodeNum +
                              2 * yNodeNum * zNodeNum + 2 * zNodeNum * xNodeNum;
      globalInteriorNodeNum = xNodeNum * yNodeNum * zNodeNum;
    }
  }

  LocalIndex localBoundaryNodeNum, localInteriorNodeNum;
  localBoundaryNodeNum =
      globalBoundaryNodeNum / mpiSize_ +
      ((globalBoundaryNodeNum % static_cast<GlobalIndex>(mpiSize_) >
        static_cast<GlobalIndex>(mpiRank_))
           ? 1
           : 0);
  localInteriorNodeNum =
      globalInteriorNodeNum / mpiSize_ +
      ((globalInteriorNodeNum % static_cast<GlobalIndex>(mpiSize_) >
        static_cast<GlobalIndex>(mpiRank_))
           ? 1
           : 0);

  return localBoundaryNodeNum + localInteriorNodeNum;
}

Void Geometry::DomainGeometry::AssignUniformNode(HostRealMatrix nodeCoords,
                                                 HostRealMatrix nodeNormal,
                                                 HostRealVector nodeSize,
                                                 HostIndexVector nodeType,
                                                 const Scalar spacing) {
  if (dimension_ == 2) {
    if (shape_ == Box) {
      LocalIndex xNodeNum = ceil(size_[0] / spacing);
      LocalIndex yNodeNum = ceil(size_[1] / spacing);

      GlobalIndex globalBoundaryNodeNum, globalInteriorNodeNum;
      globalBoundaryNodeNum = 2 * xNodeNum + 2 * yNodeNum;
      globalInteriorNodeNum = xNodeNum * yNodeNum;

      LocalIndex nodeIndex = 0;

      // boundary node
      GlobalIndex boundaryNodeIndex[5];
      boundaryNodeIndex[0] = 0;
      boundaryNodeIndex[1] = xNodeNum;
      boundaryNodeIndex[2] = xNodeNum + yNodeNum;
      boundaryNodeIndex[3] = 2 * xNodeNum + yNodeNum;
      boundaryNodeIndex[4] = 2 * xNodeNum + 2 * yNodeNum;

      std::vector<LocalIndex> rankBoundaryNodeNumList(mpiSize_);
      std::vector<GlobalIndex> rankBoundaryNodeOffsetList(mpiSize_ + 1);
      for (int rank = 0; rank < mpiSize_; rank++) {
        rankBoundaryNodeNumList[rank] =
            globalBoundaryNodeNum / mpiSize_ +
            ((globalBoundaryNodeNum % static_cast<GlobalIndex>(mpiSize_) >
              static_cast<GlobalIndex>(rank))
                 ? 1
                 : 0);
      }
      rankBoundaryNodeOffsetList[0] = 0;
      for (int rank = 0; rank < mpiSize_; rank++) {
        rankBoundaryNodeOffsetList[rank + 1] =
            rankBoundaryNodeOffsetList[rank] + rankBoundaryNodeNumList[rank];
      }

      // line: y = size_[1] / 2.0
      for (int i = 0; i < xNodeNum; i++) {
        GlobalIndex globalIndex = boundaryNodeIndex[0] + i;
        if (globalIndex >= rankBoundaryNodeOffsetList[mpiRank_] &&
            globalIndex < rankBoundaryNodeOffsetList[mpiRank_ + 1]) {
          nodeCoords(nodeIndex, 0) = (i + 0.5) * spacing - size_[0] / 2.0;
          nodeCoords(nodeIndex, 1) = size_[1] / 2.0;
          nodeCoords(nodeIndex, 2) = 0.0;
          nodeNormal(nodeIndex, 0) = 0.0;
          nodeNormal(nodeIndex, 1) = -1.0;
          nodeNormal(nodeIndex, 2) = 0.0;
          nodeSize(nodeIndex) = spacing;
          nodeType(nodeIndex) = 1;
          nodeIndex++;
        }
      }

      // line: x = size_[0] / 2.0
      for (int i = 0; i < yNodeNum; i++) {
        GlobalIndex globalIndex = boundaryNodeIndex[1] + i;
        if (globalIndex >= rankBoundaryNodeOffsetList[mpiRank_] &&
            globalIndex < rankBoundaryNodeOffsetList[mpiRank_ + 1]) {
          nodeCoords(nodeIndex, 0) = size_[0] / 2.0;
          nodeCoords(nodeIndex, 1) = (i + 0.5) * spacing - size_[1] / 2.0;
          nodeCoords(nodeIndex, 2) = 0.0;
          nodeNormal(nodeIndex, 0) = -1.0;
          nodeNormal(nodeIndex, 1) = 0.0;
          nodeNormal(nodeIndex, 2) = 0.0;
          nodeSize(nodeIndex) = spacing;
          nodeType(nodeIndex) = 1;
          nodeIndex++;
        }
      }

      // line: y = -size_[0] / 2.0
      for (int i = 0; i < xNodeNum; i++) {
        GlobalIndex globalIndex = boundaryNodeIndex[2] + i;
        if (globalIndex >= rankBoundaryNodeOffsetList[mpiRank_] &&
            globalIndex < rankBoundaryNodeOffsetList[mpiRank_ + 1]) {
          nodeCoords(nodeIndex, 0) = (i + 0.5) * spacing - size_[0] / 2.0;
          nodeCoords(nodeIndex, 1) = -size_[1] / 2.0;
          nodeCoords(nodeIndex, 2) = 0.0;
          nodeNormal(nodeIndex, 0) = 0.0;
          nodeNormal(nodeIndex, 1) = 1.0;
          nodeNormal(nodeIndex, 2) = 0.0;
          nodeSize(nodeIndex) = spacing;
          nodeType(nodeIndex) = 1;
          nodeIndex++;
        }
      }

      // line: x = -size_[0] / 2.0
      for (int i = 0; i < yNodeNum; i++) {
        GlobalIndex globalIndex = boundaryNodeIndex[3] + i;
        if (globalIndex >= rankBoundaryNodeOffsetList[mpiRank_] &&
            globalIndex < rankBoundaryNodeOffsetList[mpiRank_ + 1]) {
          nodeCoords(nodeIndex, 0) = -size_[0] / 2.0;
          nodeCoords(nodeIndex, 1) = (i + 0.5) * spacing - size_[1] / 2.0;
          nodeCoords(nodeIndex, 2) = 0.0;
          nodeNormal(nodeIndex, 0) = 1.0;
          nodeNormal(nodeIndex, 1) = 0.0;
          nodeNormal(nodeIndex, 2) = 0.0;
          nodeSize(nodeIndex) = spacing;
          nodeType(nodeIndex) = 1;
          nodeIndex++;
        }
      }

      // interior node
      std::vector<LocalIndex> rankInteriorNodeNumList(mpiSize_);
      std::vector<GlobalIndex> rankInteriorNodeOffsetList(mpiSize_ + 1);
      for (int rank = 0; rank < mpiSize_; rank++) {
        rankInteriorNodeNumList[rank] =
            globalInteriorNodeNum / mpiSize_ +
            ((globalInteriorNodeNum % static_cast<GlobalIndex>(mpiSize_) >
              static_cast<GlobalIndex>(rank))
                 ? 1
                 : 0);
      }
      rankInteriorNodeOffsetList[0] = 0;
      for (int rank = 0; rank < mpiSize_; rank++) {
        rankInteriorNodeOffsetList[rank + 1] =
            rankInteriorNodeOffsetList[rank] + rankInteriorNodeNumList[rank];
      }

      for (int i = 0; i < xNodeNum; i++) {
        for (int j = 0; j < yNodeNum; j++) {
          GlobalIndex globalIndex = i * yNodeNum + j;
          if (globalIndex >= rankInteriorNodeOffsetList[mpiRank_] &&
              globalIndex < rankInteriorNodeOffsetList[mpiRank_ + 1]) {
            nodeCoords(nodeIndex, 0) = (i + 0.5) * spacing - size_[0] / 2.0;
            nodeCoords(nodeIndex, 1) = (j + 0.5) * spacing - size_[1] / 2.0;
            nodeCoords(nodeIndex, 2) = 0.0;
            nodeNormal(nodeIndex, 0) = 1.0;
            nodeNormal(nodeIndex, 1) = 0.0;
            nodeNormal(nodeIndex, 2) = 0.0;
            nodeSize(nodeIndex) = spacing;
            nodeType(nodeIndex) = 0;
            nodeIndex++;
          }
        }
      }
    }
  }

  if (dimension_ == 3) {
    if (shape_ == Box) {
      LocalIndex xNodeNum = ceil(size_[0] / spacing);
      LocalIndex yNodeNum = ceil(size_[1] / spacing);
      LocalIndex zNodeNum = ceil(size_[2] / spacing);

      GlobalIndex globalBoundaryNodeNum, globalInteriorNodeNum;
      globalBoundaryNodeNum = 2 * xNodeNum * yNodeNum +
                              2 * yNodeNum * zNodeNum + 2 * zNodeNum * xNodeNum;
      globalInteriorNodeNum = xNodeNum * yNodeNum * zNodeNum;

      LocalIndex nodeIndex = 0;

      // boundary node
      GlobalIndex boundaryNodeIndex[7];
      boundaryNodeIndex[0] = 0;
      boundaryNodeIndex[1] = boundaryNodeIndex[0] + xNodeNum * yNodeNum;
      boundaryNodeIndex[2] = boundaryNodeIndex[1] + xNodeNum * yNodeNum;
      boundaryNodeIndex[3] = boundaryNodeIndex[2] + yNodeNum * zNodeNum;
      boundaryNodeIndex[4] = boundaryNodeIndex[3] + yNodeNum * zNodeNum;
      boundaryNodeIndex[5] = boundaryNodeIndex[4] + zNodeNum * xNodeNum;
      boundaryNodeIndex[6] = boundaryNodeIndex[5] + zNodeNum * xNodeNum;

      std::vector<LocalIndex> rankBoundaryNodeNumList(mpiSize_);
      std::vector<GlobalIndex> rankBoundaryNodeOffsetList(mpiSize_ + 1);
      for (int rank = 0; rank < mpiSize_; rank++) {
        rankBoundaryNodeNumList[rank] =
            globalBoundaryNodeNum / mpiSize_ +
            ((globalBoundaryNodeNum % static_cast<GlobalIndex>(mpiSize_) >
              static_cast<GlobalIndex>(rank))
                 ? 1
                 : 0);
      }
      rankBoundaryNodeOffsetList[0] = 0;
      for (int rank = 0; rank < mpiSize_; rank++) {
        rankBoundaryNodeOffsetList[rank + 1] =
            rankBoundaryNodeOffsetList[rank] + rankBoundaryNodeNumList[rank];
      }

      // plane: z = -size_[2] / 2.0
      for (int i = 0; i < xNodeNum; i++) {
        for (int j = 0; j < yNodeNum; j++) {
          GlobalIndex globalIndex = boundaryNodeIndex[0] + i * yNodeNum + j;
          if (globalIndex >= rankBoundaryNodeOffsetList[mpiRank_] &&
              globalIndex < rankBoundaryNodeOffsetList[mpiRank_ + 1]) {
            nodeCoords(nodeIndex, 0) = (i + 0.5) * spacing - size_[0] / 2.0;
            nodeCoords(nodeIndex, 1) = (j + 0.5) * spacing - size_[1] / 2.0;
            nodeCoords(nodeIndex, 2) = -size_[2] / 2.0;
            nodeNormal(nodeIndex, 0) = 0.0;
            nodeNormal(nodeIndex, 1) = 0.0;
            nodeNormal(nodeIndex, 2) = 1.0;
            nodeSize(nodeIndex) = spacing;
            nodeType(nodeIndex) = 1;
            nodeIndex++;
          }
        }
      }

      // plane: z = size_[2] / 2.0
      for (int i = 0; i < xNodeNum; i++) {
        for (int j = 0; j < yNodeNum; j++) {
          GlobalIndex globalIndex = boundaryNodeIndex[1] + i * yNodeNum + j;
          if (globalIndex >= rankBoundaryNodeOffsetList[mpiRank_] &&
              globalIndex < rankBoundaryNodeOffsetList[mpiRank_ + 1]) {
            nodeCoords(nodeIndex, 0) = (i + 0.5) * spacing - size_[0] / 2.0;
            nodeCoords(nodeIndex, 1) = (j + 0.5) * spacing - size_[1] / 2.0;
            nodeCoords(nodeIndex, 2) = size_[2] / 2.0;
            nodeNormal(nodeIndex, 0) = 0.0;
            nodeNormal(nodeIndex, 1) = 0.0;
            nodeNormal(nodeIndex, 2) = -1.0;
            nodeSize(nodeIndex) = spacing;
            nodeType(nodeIndex) = 1;
            nodeIndex++;
          }
        }
      }

      // plane: x = -size_[2] / 2.0
      for (int i = 0; i < yNodeNum; i++) {
        for (int j = 0; j < zNodeNum; j++) {
          GlobalIndex globalIndex = boundaryNodeIndex[2] + i * zNodeNum + j;
          if (globalIndex >= rankBoundaryNodeOffsetList[mpiRank_] &&
              globalIndex < rankBoundaryNodeOffsetList[mpiRank_ + 1]) {
            nodeCoords(nodeIndex, 0) = -size_[0] / 2.0;
            nodeCoords(nodeIndex, 1) = (i + 0.5) * spacing - size_[1] / 2.0;
            nodeCoords(nodeIndex, 2) = (j + 0.5) * spacing - size_[2] / 2.0;
            nodeNormal(nodeIndex, 0) = 1.0;
            nodeNormal(nodeIndex, 1) = 0.0;
            nodeNormal(nodeIndex, 2) = 0.0;
            nodeSize(nodeIndex) = spacing;
            nodeType(nodeIndex) = 1;
            nodeIndex++;
          }
        }
      }

      // plane: x = size_[2] / 2.0
      for (int i = 0; i < yNodeNum; i++) {
        for (int j = 0; j < zNodeNum; j++) {
          GlobalIndex globalIndex = boundaryNodeIndex[3] + i * zNodeNum + j;
          if (globalIndex >= rankBoundaryNodeOffsetList[mpiRank_] &&
              globalIndex < rankBoundaryNodeOffsetList[mpiRank_ + 1]) {
            nodeCoords(nodeIndex, 0) = size_[0] / 2.0;
            nodeCoords(nodeIndex, 1) = (i + 0.5) * spacing - size_[1] / 2.0;
            nodeCoords(nodeIndex, 2) = (j + 0.5) * spacing - size_[2] / 2.0;
            nodeNormal(nodeIndex, 0) = -1.0;
            nodeNormal(nodeIndex, 1) = 0.0;
            nodeNormal(nodeIndex, 2) = 0.0;
            nodeSize(nodeIndex) = spacing;
            nodeType(nodeIndex) = 1;
            nodeIndex++;
          }
        }
      }

      // plane: y = -size_[2] / 2.0
      for (int i = 0; i < zNodeNum; i++) {
        for (int j = 0; j < xNodeNum; j++) {
          GlobalIndex globalIndex = boundaryNodeIndex[4] + i * xNodeNum + j;
          if (globalIndex >= rankBoundaryNodeOffsetList[mpiRank_] &&
              globalIndex < rankBoundaryNodeOffsetList[mpiRank_ + 1]) {
            nodeCoords(nodeIndex, 0) = (j + 0.5) * spacing - size_[1] / 2.0;
            nodeCoords(nodeIndex, 1) = -size_[1] / 2.0;
            nodeCoords(nodeIndex, 2) = (i + 0.5) * spacing - size_[2] / 2.0;
            nodeNormal(nodeIndex, 0) = 0.0;
            nodeNormal(nodeIndex, 1) = 1.0;
            nodeNormal(nodeIndex, 2) = 0.0;
            nodeSize(nodeIndex) = spacing;
            nodeType(nodeIndex) = 1;
            nodeIndex++;
          }
        }
      }

      // plane: y = size_[2] / 2.0
      for (int i = 0; i < zNodeNum; i++) {
        for (int j = 0; j < xNodeNum; j++) {
          GlobalIndex globalIndex = boundaryNodeIndex[5] + i * xNodeNum + j;
          if (globalIndex >= rankBoundaryNodeOffsetList[mpiRank_] &&
              globalIndex < rankBoundaryNodeOffsetList[mpiRank_ + 1]) {
            nodeCoords(nodeIndex, 0) = (j + 0.5) * spacing - size_[1] / 2.0;
            nodeCoords(nodeIndex, 1) = size_[1] / 2.0;
            nodeCoords(nodeIndex, 2) = (i + 0.5) * spacing - size_[2] / 2.0;
            nodeNormal(nodeIndex, 0) = 0.0;
            nodeNormal(nodeIndex, 1) = -1.0;
            nodeNormal(nodeIndex, 2) = 0.0;
            nodeSize(nodeIndex) = spacing;
            nodeType(nodeIndex) = 1;
            nodeIndex++;
          }
        }
      }

      // interior node
      std::vector<LocalIndex> rankInteriorNodeNumList(mpiSize_);
      std::vector<GlobalIndex> rankInteriorNodeOffsetList(mpiSize_ + 1);
      for (int rank = 0; rank < mpiSize_; rank++) {
        rankInteriorNodeNumList[rank] =
            globalInteriorNodeNum / mpiSize_ +
            ((globalInteriorNodeNum % static_cast<GlobalIndex>(mpiSize_) >
              static_cast<GlobalIndex>(rank))
                 ? 1
                 : 0);
      }
      rankInteriorNodeOffsetList[0] = 0;
      for (int rank = 0; rank < mpiSize_; rank++) {
        rankInteriorNodeOffsetList[rank + 1] =
            rankInteriorNodeOffsetList[rank] + rankInteriorNodeNumList[rank];
      }

      for (GlobalIndex i =
               rankInteriorNodeOffsetList[mpiRank_] / (yNodeNum * zNodeNum);
           i <=
           rankInteriorNodeOffsetList[mpiRank_ + 1] / (yNodeNum * zNodeNum) + 1;
           i++) {
        for (int j = 0; j < yNodeNum; j++) {
          for (int k = 0; k < zNodeNum; k++) {
            GlobalIndex globalIndex =
                i * (yNodeNum * zNodeNum) + j * zNodeNum + k;
            if (globalIndex >= rankInteriorNodeOffsetList[mpiRank_] &&
                globalIndex < rankInteriorNodeOffsetList[mpiRank_ + 1]) {
              nodeCoords(nodeIndex, 0) = (i + 0.5) * spacing - size_[0] / 2.0;
              nodeCoords(nodeIndex, 1) = (j + 0.5) * spacing - size_[1] / 2.0;
              nodeCoords(nodeIndex, 2) = (k + 0.5) * spacing - size_[2] / 2.0;
              nodeNormal(nodeIndex, 0) = 1.0;
              nodeNormal(nodeIndex, 1) = 0.0;
              nodeNormal(nodeIndex, 2) = 0.0;
              nodeSize(nodeIndex) = spacing;
              nodeType(nodeIndex) = 0;
              nodeIndex++;
            }
          }
        }
      }
    }
  }
}