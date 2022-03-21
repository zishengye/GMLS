#include "DomainGeometry.hpp"

DomainGeometry::DomainGeometry() {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);
}

DomainGeometry::DomainGeometry(const DomainGeometry &geo) {
  dimension_ = geo.dimension_;
  size_ = geo.size_;

  mpiRank_ = geo.mpiRank_;
  mpiSize_ = geo.mpiSize_;
}

DomainGeometry::~DomainGeometry() {}

bool DomainGeometry::IsInterior(Scalar x, Scalar y, Scalar z) {
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
}

void DomainGeometry::IsInterior(Kokkos::View<Scalar **> coords,
                                Kokkos::View<bool *> results) {
  if (dimension_ == 2) {
    if (shape_ == Box) {
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,
                                                             coords.extent(0)),
          KOKKOS_LAMBDA(const int i) {
            if (coords(i, 0) > -size_[0] / 2.0 &&
                coords(i, 0) < size_[0] / 2.0 &&
                coords(i, 1) > -size_[1] / 2.0 && coords(i, 1) < size_[1] / 2.0)
              results(i) = true;
            else
              results(i) = false;
          });
    }
  }

  if (dimension_ == 3) {
    if (shape_ == Box) {
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,
                                                             coords.extent(0)),
          KOKKOS_LAMBDA(const int i) {
            if (coords(i, 0) > -size_[0] / 2.0 &&
                coords(i, 0) < size_[0] / 2.0 &&
                coords(i, 1) > -size_[1] / 2.0 &&
                coords(i, 1) < size_[1] / 2.0 &&
                coords(i, 2) > -size_[2] / 2.0 && coords(i, 2) < size_[2] / 2.0)
              results(i) = true;
            else
              results(i) = false;
          });
    }
  }
}

void DomainGeometry::SetType(SimpleDomainShape shape) { shape_ = shape; }

void DomainGeometry::SetDimension(const int dimension) {
  dimension_ = dimension;
}

void DomainGeometry::SetSize(const std::vector<Scalar> &size) { size_ = size; }

SimpleDomainShape DomainGeometry::GetType() { return shape_; }

const int DomainGeometry::GetDimension() { return dimension_; }

Scalar DomainGeometry::GetSize(const int size_index) {
  return size_[size_index];
}

const LocalIndex DomainGeometry::EstimateNodeNum(const Scalar spacing) {
  GlobalIndex globalBoundaryNodeNum, globalInteriorNodeNum;

  if (dimension_ == 2) {
    if (shape_ == Box) {
      LocalIndex xNodeNum = ceil(size_[0] / spacing);
      LocalIndex yNodeNum = ceil(size_[1] / spacing);
      globalBoundaryNodeNum = 2 * xNodeNum + 2 * yNodeNum;
      globalInteriorNodeNum = xNodeNum * yNodeNum;
    }
  }

  LocalIndex localBoundaryNodeNum, localInteriorNodeNum;
  localBoundaryNodeNum =
      globalBoundaryNodeNum / mpiSize_ +
      ((globalBoundaryNodeNum % mpiSize_ > mpiRank_) ? 1 : 0);
  localInteriorNodeNum =
      globalInteriorNodeNum / mpiSize_ +
      ((globalInteriorNodeNum % mpiSize_ > mpiRank_) ? 1 : 0);

  return localBoundaryNodeNum + localInteriorNodeNum;
}

void DomainGeometry::AssignUniformNode(Kokkos::View<Scalar **> nodeCoords,
                                       Kokkos::View<Scalar **> nodeNormal,
                                       Kokkos::View<Scalar *> nodeSize,
                                       Kokkos::View<LocalIndex *> nodeType,
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
            ((globalBoundaryNodeNum % mpiSize_ > rank) ? 1 : 0);
      }
      rankBoundaryNodeOffsetList[0] = 0;
      for (int rank = 0; rank < mpiSize_; rank++) {
        rankBoundaryNodeOffsetList[rank + 1] =
            rankBoundaryNodeOffsetList[rank] + rankBoundaryNodeNumList[rank];
      }

      GlobalIndex startIndex, endIndex;
      int boundaryIterStart, boundaryIterEnd;
      // y = size_[1] / 2.0
      startIndex = std::max<GlobalIndex>(boundaryNodeIndex[0],
                                         rankBoundaryNodeOffsetList[mpiRank_]);
      endIndex = std::min<GlobalIndex>(
          boundaryNodeIndex[1], rankBoundaryNodeOffsetList[mpiRank_ + 1]);
      boundaryIterStart =
          std::max<GlobalIndex>(0, startIndex - boundaryNodeIndex[0]);
      boundaryIterEnd =
          std::min<GlobalIndex>(xNodeNum, endIndex - boundaryNodeIndex[0]);
      for (int i = boundaryIterStart; i < boundaryIterEnd; i++) {
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

      // x = size_[0] / 2.0
      startIndex = std::max<GlobalIndex>(boundaryNodeIndex[1],
                                         rankBoundaryNodeOffsetList[mpiRank_]);
      endIndex = std::min<GlobalIndex>(
          boundaryNodeIndex[2], rankBoundaryNodeOffsetList[mpiRank_ + 1]);
      boundaryIterStart = std::max<int>(0, startIndex - boundaryNodeIndex[1]);
      boundaryIterEnd =
          std::min<int>(yNodeNum, endIndex - boundaryNodeIndex[1]);
      for (int i = boundaryIterStart; i < boundaryIterEnd; i++) {
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

      // y = -size_[0] / 2.0
      startIndex = std::max<GlobalIndex>(boundaryNodeIndex[2],
                                         rankBoundaryNodeOffsetList[mpiRank_]);
      endIndex = std::min<GlobalIndex>(
          boundaryNodeIndex[3], rankBoundaryNodeOffsetList[mpiRank_ + 1]);
      boundaryIterStart = std::max<int>(0, startIndex - boundaryNodeIndex[2]);
      boundaryIterEnd =
          std::min<int>(yNodeNum, endIndex - boundaryNodeIndex[2]);
      for (int i = boundaryIterStart; i < boundaryIterEnd; i++) {
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

      // x = -size_[0] / 2.0
      startIndex = std::max<GlobalIndex>(boundaryNodeIndex[3],
                                         rankBoundaryNodeOffsetList[mpiRank_]);
      endIndex = std::min<GlobalIndex>(
          boundaryNodeIndex[4], rankBoundaryNodeOffsetList[mpiRank_ + 1]);
      boundaryIterStart = std::max<int>(0, startIndex - boundaryNodeIndex[3]);
      boundaryIterEnd =
          std::min<int>(yNodeNum, endIndex - boundaryNodeIndex[3]);
      for (int i = boundaryIterStart; i < boundaryIterEnd; i++) {
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

      // interior node
      std::vector<LocalIndex> rankInteriorNodeNumList(mpiSize_);
      std::vector<GlobalIndex> rankInteriorNodeOffsetList(mpiSize_ + 1);
      for (int rank = 0; rank < mpiSize_; rank++) {
        rankInteriorNodeNumList[rank] =
            globalInteriorNodeNum / mpiSize_ +
            ((globalInteriorNodeNum % mpiSize_ > rank) ? 1 : 0);
      }
      rankInteriorNodeOffsetList[0] = 0;
      for (int rank = 0; rank < mpiSize_; rank++) {
        rankInteriorNodeOffsetList[rank + 1] =
            rankInteriorNodeOffsetList[rank] + rankInteriorNodeNumList[rank];
      }

      for (int i = rankInteriorNodeOffsetList[mpiRank_] / yNodeNum;
           i <= rankInteriorNodeOffsetList[mpiRank_ + 1] / yNodeNum + 1; i++) {
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
}