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

  return false;
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
      Kokkos::fence();
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
      Kokkos::fence();
    }
  }
}

void DomainGeometry::SetType(const SimpleDomainShape shape) { shape_ = shape; }

void DomainGeometry::SetDimension(const int dimension) {
  dimension_ = dimension;
}

void DomainGeometry::SetSize(const std::vector<Scalar> &size) { size_ = size; }

SimpleDomainShape DomainGeometry::GetType() { return shape_; }

int DomainGeometry::GetDimension() { return dimension_; }

Scalar DomainGeometry::GetSize(const int size_index) {
  return size_[size_index];
}

LocalIndex DomainGeometry::EstimateNodeNum(const Scalar spacing) {
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

void DomainGeometry::AssignUniformNode(Kokkos::View<Scalar **> nodeCoords,
                                       Kokkos::View<Scalar **> nodeNormal,
                                       Kokkos::View<Scalar *> nodeSize,
                                       Kokkos::View<std::size_t *> nodeType,
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

      GlobalIndex startIndex, endIndex;
      int boundaryIterStart, boundaryIterEnd;
      // line: y = size_[1] / 2.0
      startIndex = std::max<GlobalIndex>(boundaryNodeIndex[0],
                                         rankBoundaryNodeOffsetList[mpiRank_]);
      endIndex = std::min<GlobalIndex>(
          boundaryNodeIndex[1], rankBoundaryNodeOffsetList[mpiRank_ + 1]);
      boundaryIterStart = std::max<int>(0, startIndex - boundaryNodeIndex[0]);
      boundaryIterEnd =
          std::min<int>(xNodeNum, endIndex - boundaryNodeIndex[0]);
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

      // line: x = size_[0] / 2.0
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

      // line: y = -size_[0] / 2.0
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

      // line: x = -size_[0] / 2.0
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

      for (GlobalIndex i = rankInteriorNodeOffsetList[mpiRank_] / yNodeNum;
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

      GlobalIndex startIndex, endIndex;
      int boundaryIterStart, boundaryIterEnd;
      // plane: z = -size_[2] / 2.0
      startIndex = std::max<GlobalIndex>(boundaryNodeIndex[0],
                                         rankBoundaryNodeOffsetList[mpiRank_]);
      endIndex = std::min<GlobalIndex>(
          boundaryNodeIndex[1], rankBoundaryNodeOffsetList[mpiRank_ + 1]);
      boundaryIterStart = std::max<int>(0, startIndex - boundaryNodeIndex[0]);
      boundaryIterEnd =
          std::min<int>(xNodeNum * yNodeNum, endIndex - boundaryNodeIndex[0]);

      for (int i = boundaryIterStart / yNodeNum; i < boundaryIterEnd / yNodeNum;
           i++) {
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
      startIndex = std::max<GlobalIndex>(boundaryNodeIndex[1],
                                         rankBoundaryNodeOffsetList[mpiRank_]);
      endIndex = std::min<GlobalIndex>(
          boundaryNodeIndex[2], rankBoundaryNodeOffsetList[mpiRank_ + 1]);
      boundaryIterStart = std::max<int>(0, startIndex - boundaryNodeIndex[1]);
      boundaryIterEnd =
          std::min<int>(xNodeNum * yNodeNum, endIndex - boundaryNodeIndex[1]);

      for (int i = boundaryIterStart / yNodeNum; i < boundaryIterEnd / yNodeNum;
           i++) {
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
      startIndex = std::max<GlobalIndex>(boundaryNodeIndex[2],
                                         rankBoundaryNodeOffsetList[mpiRank_]);
      endIndex = std::min<GlobalIndex>(
          boundaryNodeIndex[3], rankBoundaryNodeOffsetList[mpiRank_ + 1]);
      boundaryIterStart = std::max<int>(0, startIndex - boundaryNodeIndex[2]);
      boundaryIterEnd =
          std::min<int>(yNodeNum * zNodeNum, endIndex - boundaryNodeIndex[2]);

      for (int i = boundaryIterStart / zNodeNum; i < boundaryIterEnd / zNodeNum;
           i++) {
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
      startIndex = std::max<GlobalIndex>(boundaryNodeIndex[3],
                                         rankBoundaryNodeOffsetList[mpiRank_]);
      endIndex = std::min<GlobalIndex>(
          boundaryNodeIndex[4], rankBoundaryNodeOffsetList[mpiRank_ + 1]);
      boundaryIterStart = std::max<int>(0, startIndex - boundaryNodeIndex[3]);
      boundaryIterEnd =
          std::min<int>(yNodeNum * zNodeNum, endIndex - boundaryNodeIndex[3]);

      for (int i = boundaryIterStart / zNodeNum; i < boundaryIterEnd / zNodeNum;
           i++) {
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
      startIndex = std::max<GlobalIndex>(boundaryNodeIndex[4],
                                         rankBoundaryNodeOffsetList[mpiRank_]);
      endIndex = std::min<GlobalIndex>(
          boundaryNodeIndex[5], rankBoundaryNodeOffsetList[mpiRank_ + 1]);
      boundaryIterStart = std::max<int>(0, startIndex - boundaryNodeIndex[4]);
      boundaryIterEnd =
          std::min<int>(zNodeNum * xNodeNum, endIndex - boundaryNodeIndex[4]);

      for (int i = boundaryIterStart / xNodeNum; i < boundaryIterEnd / xNodeNum;
           i++) {
        for (int j = 0; j < xNodeNum; j++) {
          GlobalIndex globalIndex = boundaryNodeIndex[4] + i * xNodeNum + j;
          if (globalIndex >= rankBoundaryNodeOffsetList[mpiRank_] &&
              globalIndex < rankBoundaryNodeOffsetList[mpiRank_ + 1]) {
            nodeCoords(nodeIndex, 0) = (i + 0.5) * spacing - size_[1] / 2.0;
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
      startIndex = std::max<GlobalIndex>(boundaryNodeIndex[5],
                                         rankBoundaryNodeOffsetList[mpiRank_]);
      endIndex = std::min<GlobalIndex>(
          boundaryNodeIndex[6], rankBoundaryNodeOffsetList[mpiRank_ + 1]);
      boundaryIterStart = std::max<int>(0, startIndex - boundaryNodeIndex[5]);
      boundaryIterEnd =
          std::min<int>(zNodeNum * xNodeNum, endIndex - boundaryNodeIndex[5]);

      for (int i = boundaryIterStart / xNodeNum; i < boundaryIterEnd / xNodeNum;
           i++) {
        for (int j = 0; j < xNodeNum; j++) {
          GlobalIndex globalIndex = boundaryNodeIndex[5] + i * xNodeNum + j;
          if (globalIndex >= rankBoundaryNodeOffsetList[mpiRank_] &&
              globalIndex < rankBoundaryNodeOffsetList[mpiRank_ + 1]) {
            nodeCoords(nodeIndex, 0) = (i + 0.5) * spacing - size_[1] / 2.0;
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