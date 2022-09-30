#include <fstream>

#include "Core/Typedef.hpp"
#include "Geometry/DomainGeometry.hpp"
#include "Geometry/ParticleGeometry.hpp"

Geometry::ParticleSet::ParticleSet(CoordType coordType) {
  coordType_ = coordType;

  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);
}

Void Geometry::ParticleSet::SetDimension(const Size dimension) {
  dimension_ = dimension;
}

HostRealMatrix &Geometry::ParticleSet::GetParticleCoords() {
  return hostParticleCoords_;
}

HostRealMatrix &Geometry::ParticleSet::GetParticleNormal() {
  return hostParticleNormal_;
}

HostRealVector &Geometry::ParticleSet::GetParticleSize() {
  return hostParticleSize_;
}

HostIndexVector &Geometry::ParticleSet::GetParticleType() {
  return hostParticleType_;
}

HostIndexVector &Geometry::ParticleSet::GetParticleIndex() {
  return hostParticleIndex_;
}

LocalIndex Geometry::ParticleSet::GetLocalParticleNum() {
  return hostParticleCoords_.extent(0);
}

GlobalIndex Geometry::ParticleSet::GetGlobalParticleNum() {
  GlobalIndex globalSize;
  globalSize = hostParticleCoords_.extent(0);
  MPI_Allreduce(MPI_IN_PLACE, &globalSize, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  return globalSize;
}

Scalar Geometry::ParticleSet::GetParticleCoords(const LocalIndex index,
                                                const Size dimension) {
  return hostParticleCoords_(index, dimension);
}

Void Geometry::ParticleSet::Resize(const LocalIndex newLocalSize) {
  Kokkos::resize(hostParticleCoords_, newLocalSize, 3);
  Kokkos::resize(hostParticleNormal_, newLocalSize, 3);
  Kokkos::resize(hostParticleSize_, newLocalSize);
  Kokkos::resize(hostParticleType_, newLocalSize);
  Kokkos::resize(hostParticleIndex_, newLocalSize);
}

Void Geometry::ParticleSet::Balance() {
  partition_.ConstructPartition(hostParticleCoords_, hostParticleIndex_);
  partition_.ApplyPartition(hostParticleCoords_);
  partition_.ApplyPartition(hostParticleNormal_);
  partition_.ApplyPartition(hostParticleSize_);
  partition_.ApplyPartition(hostParticleType_);
}

Void Geometry::ParticleSet::Output(const std::string outputFileName,
                                   bool isBinary) {
  std::size_t globalParticleNum = GetGlobalParticleNum();
  std::ofstream vtkStream;

  // particle positions
  if (mpiRank_ == 0) {
    if (isBinary)
      vtkStream.open(outputFileName,
                     std::ios::out | std::ios::trunc | std::ios::binary);
    else
      vtkStream.open(outputFileName, std::ios::out | std::ios::trunc);

    assert(vtkStream.is_open() == true);

    vtkStream << "# vtk DataFile Version 2.0" << std::endl;

    vtkStream << outputFileName + " output " << std::endl;

    if (isBinary)
      vtkStream << "BINARY" << std::endl << std::endl;
    else
      vtkStream << "ASCII" << std::endl << std::endl;

    vtkStream << "DATASET POLYDATA" << std::endl
              << "POINTS " << globalParticleNum << " float" << std::endl;

    vtkStream.close();
  }

  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      if (isBinary)
        vtkStream.open(outputFileName,
                       std::ios::out | std::ios::app | std::ios::binary);
      else
        vtkStream.open(outputFileName, std::ios::out | std::ios::app);

      for (std::size_t i = 0; i < hostParticleCoords_.extent(0); i++) {
        for (std::size_t j = 0; j < hostParticleCoords_.extent(1); j++) {
          float x = hostParticleCoords_(i, j);
          if (isBinary) {
            SwapEnd(x);
            vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));
          } else {
            vtkStream << x << " ";
          }
        }
        if (!isBinary)
          vtkStream << std::endl;
      }
      vtkStream.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (mpiRank_ == 0) {
    if (isBinary)
      vtkStream.open(outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
    else
      vtkStream.open(outputFileName, std::ios::out | std::ios::app);

    vtkStream << "POINT_DATA " << globalParticleNum << std::endl;

    vtkStream.close();
  }

  // particle normal
  if (mpiRank_ == 0) {
    if (isBinary)
      vtkStream.open(outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
    else
      vtkStream.open(outputFileName, std::ios::out | std::ios::app);

    vtkStream << "SCALARS n float 3" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }

  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      if (isBinary)
        vtkStream.open(outputFileName,
                       std::ios::out | std::ios::app | std::ios::binary);
      else
        vtkStream.open(outputFileName, std::ios::out | std::ios::app);

      for (std::size_t i = 0; i < hostParticleNormal_.extent(0); i++) {
        for (std::size_t j = 0; j < hostParticleNormal_.extent(1); j++) {
          float x = hostParticleNormal_(i, j);
          if (isBinary) {
            SwapEnd(x);
            vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));
          } else {
            vtkStream << x << " ";
          }
        }
        if (!isBinary)
          vtkStream << std::endl;
      }
      vtkStream.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // particle size
  if (mpiRank_ == 0) {
    if (isBinary)
      vtkStream.open(outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
    else
      vtkStream.open(outputFileName, std::ios::out | std::ios::app);

    vtkStream << "SCALARS h float 1" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }

  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      if (isBinary)
        vtkStream.open(outputFileName,
                       std::ios::out | std::ios::app | std::ios::binary);
      else
        vtkStream.open(outputFileName, std::ios::out | std::ios::app);

      for (std::size_t i = 0; i < hostParticleSize_.extent(0); i++) {
        float x = hostParticleSize_(i);
        if (isBinary) {
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));
        } else {
          vtkStream << x << " ";
        }
        if (!isBinary)
          vtkStream << std::endl;
      }
      vtkStream.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // particle type
  if (mpiRank_ == 0) {
    if (isBinary)
      vtkStream.open(outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
    else
      vtkStream.open(outputFileName, std::ios::out | std::ios::app);

    vtkStream << "SCALARS ID int 1" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }

  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      if (isBinary)
        vtkStream.open(outputFileName,
                       std::ios::out | std::ios::app | std::ios::binary);
      else
        vtkStream.open(outputFileName, std::ios::out | std::ios::app);

      for (std::size_t i = 0; i < hostParticleType_.extent(0); i++) {
        int x = hostParticleType_(i);
        if (isBinary) {
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(int));
        } else {
          vtkStream << x << " ";
        }
        if (!isBinary)
          vtkStream << std::endl;
      }
      vtkStream.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

Geometry::Partition &Geometry::ParticleSet::GetPartition() {
  return partition_;
}

Void Geometry::EulerianParticleManager::BalanceAndIndexInternal() {
  // repartition for load balancing
  auto &particleIndex = particleSetPtr_->GetParticleIndex();
  std::vector<GlobalIndex> rankParticleNumOffset(mpiSize_ + 1);
  std::vector<LocalIndex> rankParticleNum(mpiSize_);
  for (int i = 0; i < mpiSize_; i++) {
    rankParticleNum[i] = 0;
  }
  rankParticleNum[mpiRank_] = particleIndex.extent(0);

  MPI_Allreduce(MPI_IN_PLACE, rankParticleNum.data(), mpiSize_, MPI_INT,
                MPI_SUM, MPI_COMM_WORLD);
  rankParticleNumOffset[0] = 0;
  for (int i = 0; i < mpiSize_; i++) {
    rankParticleNumOffset[i + 1] =
        rankParticleNumOffset[i] + rankParticleNum[i];
  }
  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                           0, particleIndex.extent(0)),
                       [&](const int i) {
                         particleIndex(i) = i + rankParticleNumOffset[mpiRank_];
                       });
  Kokkos::fence();

  particleSetPtr_->Balance();

  // reindex
  Kokkos::resize(particleIndex, particleSetPtr_->GetParticleCoords().extent(0));
  for (int i = 0; i < mpiSize_; i++) {
    rankParticleNum[i] = 0;
  }
  rankParticleNum[mpiRank_] = particleIndex.extent(0);

  MPI_Allreduce(MPI_IN_PLACE, rankParticleNum.data(), mpiSize_, MPI_INT,
                MPI_SUM, MPI_COMM_WORLD);

  rankParticleNumOffset[0] = 0;
  for (int i = 0; i < mpiSize_; i++) {
    rankParticleNumOffset[i + 1] =
        rankParticleNumOffset[i] + rankParticleNum[i];
  }

  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                           0, particleIndex.extent(0)),
                       [&](const int i) {
                         particleIndex(i) = i + rankParticleNumOffset[mpiRank_];
                       });
  Kokkos::fence();
}

Geometry::EulerianParticleManager::EulerianParticleManager()
    : isPeriodicBCs_(false) {
  geometryPtr_ = std::make_shared<DomainGeometry>();
  particleSetPtr_ = std::make_shared<ParticleSet>();

  geometryPtr_->SetDimension(0);
  geometryPtr_->SetType(UndefinedDomain);

  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);
}

Void Geometry::EulerianParticleManager::SetDimension(const Size dimension) {
  geometryPtr_->SetDimension(dimension);
  particleSetPtr_->SetDimension(dimension);
}

Void Geometry::EulerianParticleManager::SetDomainType(
    const SupportedDomainShape shape) {
  geometryPtr_->SetType(shape);
}

Void Geometry::EulerianParticleManager::SetSize(
    const std::vector<Scalar> &size) {
  geometryPtr_->SetSize(size);
}

Void Geometry::EulerianParticleManager::SetSpacing(const Scalar spacing) {
  spacing_ = spacing;
}

Size Geometry::EulerianParticleManager::GetDimension() {
  return geometryPtr_->GetDimension();
}

Void Geometry::EulerianParticleManager::Init() {
  assert(geometryPtr_->GetType() != UndefinedDomain);
  LocalIndex localParticleNum = geometryPtr_->EstimateNodeNum(spacing_);

  particleSetPtr_->Resize(localParticleNum);

  geometryPtr_->AssignUniformNode(particleSetPtr_->GetParticleCoords(),
                                  particleSetPtr_->GetParticleNormal(),
                                  particleSetPtr_->GetParticleSize(),
                                  particleSetPtr_->GetParticleType(), spacing_);

  BalanceAndIndexInternal();
}

Void Geometry::EulerianParticleManager::Clear() {}

LocalIndex Geometry::EulerianParticleManager::GetLocalParticleNum() {
  return particleSetPtr_->GetLocalParticleNum();
}

GlobalIndex Geometry::EulerianParticleManager::GetGlobalParticleNum() {
  return particleSetPtr_->GetGlobalParticleNum();
}

HostRealMatrix &Geometry::EulerianParticleManager::GetParticleNormal() {
  return particleSetPtr_->GetParticleNormal();
}

HostRealMatrix &Geometry::EulerianParticleManager::GetParticleCoords() {
  return particleSetPtr_->GetParticleCoords();
}

HostRealVector &Geometry::EulerianParticleManager::GetParticleSize() {
  return particleSetPtr_->GetParticleSize();
}

HostIndexVector &Geometry::EulerianParticleManager::GetParticleType() {
  return particleSetPtr_->GetParticleType();
}

HostIndexVector &Geometry::EulerianParticleManager::GetParticleIndex() {
  return particleSetPtr_->GetParticleIndex();
}

Void Geometry::EulerianParticleManager::Output(const std::string outputFileName,
                                               const bool isBinary) {
  particleSetPtr_->Output(outputFileName, isBinary);
}

Void Geometry::HierarchicalEulerianParticleManager::RefineInternal(
    const HostIndexVector &splitTag) {
  int dimension = GetDimension();
  // estimate new number of particles
  auto &oldParticleCoords = GetParticleCoords();
  auto &oldParticleNormal = GetParticleNormal();
  auto &oldParticleSize = GetParticleSize();
  auto &oldParticleType = GetParticleType();
  Kokkos::View<int *> newLocalParticleNum("new generated particle number",
                                          splitTag.extent(0));
  int newParticleNum = 0;

  int newInteriorParticleNum = pow(2, dimension);
  int newBoundaryParticleNum = pow(2, dimension - 1);

  // esimate number of particles in the next refinement iteration
  Kokkos::parallel_scan(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, splitTag.extent(0)),
      KOKKOS_LAMBDA(const int i, int &tNewLocalParticleNum, bool isFinal) {
        if (isFinal)
          newLocalParticleNum(i) = tNewLocalParticleNum;
        if (splitTag(i) == 0)
          tNewLocalParticleNum++;
        else {
          if (oldParticleType(i) == 0)
            tNewLocalParticleNum += newInteriorParticleNum;
          else
            tNewLocalParticleNum += newBoundaryParticleNum;
        }
      },
      newParticleNum);

  std::size_t globalNewParticleNum = newParticleNum;
  MPI_Allreduce(MPI_IN_PLACE, &globalNewParticleNum, 1, MPI_UNSIGNED_LONG,
                MPI_SUM, MPI_COMM_WORLD);

  if (mpiRank_ == 0)
    printf("Generated %ld particles in the new refinement level\n",
           globalNewParticleNum);

  hierarchicalParticleSetPtr_.push_back(std::make_shared<ParticleSet>());
  currentRefinementLevel_++;
  particleSetPtr_ = hierarchicalParticleSetPtr_[currentRefinementLevel_];

  auto &particleCoords = GetParticleCoords();
  auto &particleNormal = GetParticleNormal();
  auto &particleSize = GetParticleSize();
  auto &particleType = GetParticleType();
  auto &particleIndex = GetParticleIndex();

  Kokkos::resize(particleCoords, newParticleNum, 3);
  Kokkos::resize(particleNormal, newParticleNum, 3);
  Kokkos::resize(particleSize, newParticleNum);
  Kokkos::resize(particleType, newParticleNum);
  Kokkos::resize(particleIndex, newParticleNum);

  HostIndexVector oldParticleRefinementLevel;
  Kokkos::resize(oldParticleRefinementLevel,
                 hostParticleRefinementLevel_.extent(0));
  for (unsigned int i = 0; i < hostParticleRefinementLevel_.extent(0); i++) {
    oldParticleRefinementLevel(i) = hostParticleRefinementLevel_(i);
  }
  Kokkos::resize(hostParticleRefinementLevel_, newParticleNum);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, splitTag.extent(0)),
      [&](const int i) {
        if (splitTag(i) == 0) {
          for (int j = 0; j < 3; j++) {
            particleCoords(newLocalParticleNum(i), j) = oldParticleCoords(i, j);
            particleNormal(newLocalParticleNum(i), j) = oldParticleNormal(i, j);
          }
          particleSize(newLocalParticleNum(i)) = oldParticleSize(i);
          particleType(newLocalParticleNum(i)) = oldParticleType(i);
          hostParticleRefinementLevel_(newLocalParticleNum(i)) =
              oldParticleRefinementLevel(i);
        } else {
          if (oldParticleType(i) == 0) {
            if (dimension == 2) {
              double newSpacing = oldParticleSize(i) / 2.0;
              int counter = 0;
              for (int axes1 = -1; axes1 < 2; axes1 += 2) {
                for (int axes2 = -1; axes2 < 2; axes2 += 2) {
                  particleCoords(newLocalParticleNum(i) + counter, 0) =
                      oldParticleCoords(i, 0) + 0.5 * axes1 * newSpacing;
                  particleCoords(newLocalParticleNum(i) + counter, 1) =
                      oldParticleCoords(i, 1) + 0.5 * axes2 * newSpacing;
                  particleCoords(newLocalParticleNum(i) + counter, 2) = 0.0;

                  for (int j = 0; j < 3; j++) {
                    particleNormal(newLocalParticleNum(i) + counter, j) =
                        oldParticleNormal(i, j);
                  }
                  particleSize(newLocalParticleNum(i) + counter) = newSpacing;
                  particleType(newLocalParticleNum(i) + counter) =
                      oldParticleType(i);
                  hostParticleRefinementLevel_(newLocalParticleNum(i) +
                                               counter) =
                      oldParticleRefinementLevel(i) + 1;
                  counter++;
                }
              }
            }
            if (dimension == 3) {
              double newSpacing = oldParticleSize(i) / 2.0;
              int counter = 0;
              for (int axes1 = -1; axes1 < 2; axes1 += 2) {
                for (int axes2 = -1; axes2 < 2; axes2 += 2) {
                  for (int axes3 = -1; axes3 < 2; axes3 += 2) {
                    particleCoords(newLocalParticleNum(i) + counter, 0) =
                        oldParticleCoords(i, 0) + 0.5 * axes1 * newSpacing;
                    particleCoords(newLocalParticleNum(i) + counter, 1) =
                        oldParticleCoords(i, 1) + 0.5 * axes2 * newSpacing;
                    particleCoords(newLocalParticleNum(i) + counter, 2) =
                        oldParticleCoords(i, 2) + 0.5 * axes3 * newSpacing;

                    for (int j = 0; j < 3; j++) {
                      particleNormal(newLocalParticleNum(i) + counter, j) =
                          oldParticleNormal(i, j);
                    }
                    particleSize(newLocalParticleNum(i) + counter) = newSpacing;
                    particleType(newLocalParticleNum(i) + counter) =
                        oldParticleType(i);
                    hostParticleRefinementLevel_(newLocalParticleNum(i) +
                                                 counter) =
                        oldParticleRefinementLevel(i) + 1;
                    counter++;
                  }
                }
              }
            }
          } else {
            if (dimension == 2) {
              double newSpacing = oldParticleSize(i) / 2.0;
              int counter = 0;
              for (int axes1 = -1; axes1 < 2; axes1 += 2) {
                particleCoords(newLocalParticleNum(i) + counter, 0) =
                    oldParticleCoords(i, 0) +
                    0.5 * axes1 * newSpacing * oldParticleNormal(i, 1);
                particleCoords(newLocalParticleNum(i) + counter, 1) =
                    oldParticleCoords(i, 1) -
                    0.5 * axes1 * newSpacing * oldParticleNormal(i, 0);
                particleCoords(newLocalParticleNum(i) + counter, 2) = 0.0;

                for (int j = 0; j < 3; j++) {
                  particleNormal(newLocalParticleNum(i) + counter, j) =
                      oldParticleNormal(i, j);
                }
                particleSize(newLocalParticleNum(i) + counter) = newSpacing;
                particleType(newLocalParticleNum(i) + counter) =
                    oldParticleType(i);
                hostParticleRefinementLevel_(newLocalParticleNum(i) + counter) =
                    oldParticleRefinementLevel(i) + 1;
                counter++;
              }
            }

            if (dimension == 3) {
              double newSpacing = oldParticleSize(i) / 2.0;
              int counter = 0;
              std::vector<double> normal1(3);
              std::vector<double> normal2(3);
              if (abs(oldParticleNormal(i, 0)) > 1e-3) {
                normal1[0] = 0;
                normal1[1] = 1;
                normal1[2] = 0;

                normal2[0] = 0;
                normal2[1] = 0;
                normal2[2] = 1;
              }

              if (abs(oldParticleNormal(i, 1)) > 1e-3) {
                normal1[0] = 1;
                normal1[1] = 0;
                normal1[2] = 0;

                normal2[0] = 0;
                normal2[1] = 0;
                normal2[2] = 1;
              }

              if (abs(oldParticleNormal(i, 2)) > 1e-3) {
                normal1[0] = 1;
                normal1[1] = 0;
                normal1[2] = 0;

                normal2[0] = 0;
                normal2[1] = 1;
                normal2[2] = 0;
              }

              for (int axes1 = -1; axes1 < 2; axes1 += 2) {
                for (int axes2 = -1; axes2 < 2; axes2 += 2) {
                  particleCoords(newLocalParticleNum(i) + counter, 0) =
                      oldParticleCoords(i, 0) +
                      0.5 * axes1 * newSpacing * normal1[0] +
                      0.5 * axes2 * newSpacing * normal2[0];
                  particleCoords(newLocalParticleNum(i) + counter, 1) =
                      oldParticleCoords(i, 1) +
                      0.5 * axes1 * newSpacing * normal1[1] +
                      0.5 * axes2 * newSpacing * normal2[1];
                  particleCoords(newLocalParticleNum(i) + counter, 2) =
                      oldParticleCoords(i, 2) +
                      0.5 * axes1 * newSpacing * normal1[2] +
                      0.5 * axes2 * newSpacing * normal2[2];

                  for (int j = 0; j < 3; j++) {
                    particleNormal(newLocalParticleNum(i) + counter, j) =
                        oldParticleNormal(i, j);
                  }
                  particleSize(newLocalParticleNum(i) + counter) = newSpacing;
                  particleType(newLocalParticleNum(i) + counter) =
                      oldParticleType(i);
                  hostParticleRefinementLevel_(newLocalParticleNum(i) +
                                               counter) =
                      oldParticleRefinementLevel(i) + 1;
                  counter++;
                }
              }
            }
          }
        }
      });
  Kokkos::fence();

  BalanceAndIndexInternal();
}

Geometry::HierarchicalEulerianParticleManager::
    HierarchicalEulerianParticleManager()
    : EulerianParticleManager(), currentRefinementLevel_(0) {
  hierarchicalParticleSetPtr_.push_back(std::make_shared<ParticleSet>());
  particleSetPtr_ = hierarchicalParticleSetPtr_[0];
}

HostIndexVector &
Geometry::HierarchicalEulerianParticleManager::GetParticleRefinementLevel() {
  return hostParticleRefinementLevel_;
}

Void Geometry::HierarchicalEulerianParticleManager::Init() {
  EulerianParticleManager::Init();

  Kokkos::resize(hostParticleRefinementLevel_, this->GetLocalParticleNum());
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
          0, hostParticleRefinementLevel_.extent(0)),
      KOKKOS_LAMBDA(const int i) { hostParticleRefinementLevel_(i) = 0; });
  Kokkos::fence();
}

Void Geometry::HierarchicalEulerianParticleManager::Clear() {
  EulerianParticleManager::Clear();
}

Void Geometry::HierarchicalEulerianParticleManager::Refine(
    const HostIndexVector &splitTag) {
  RefineInternal(splitTag);

  // repartition objects defined in hierarchical scheme
  auto &partition = particleSetPtr_->GetPartition();
  partition.ApplyPartition(hostParticleRefinementLevel_);
}

LocalIndex
Geometry::HierarchicalEulerianParticleManager::GetLocalParticleNum() {
  return hierarchicalParticleSetPtr_[currentRefinementLevel_]
      ->GetLocalParticleNum();
}

GlobalIndex
Geometry::HierarchicalEulerianParticleManager::GetGlobalParticleNum() {
  return hierarchicalParticleSetPtr_[currentRefinementLevel_]
      ->GetGlobalParticleNum();
}

HostRealMatrix &
Geometry::HierarchicalEulerianParticleManager::GetParticleCoordsByLevel(
    const int level) {
  return hierarchicalParticleSetPtr_[level]->GetParticleCoords();
}

HostRealMatrix &
Geometry::HierarchicalEulerianParticleManager::GetParticleNormalByLevel(
    const int level) {
  return hierarchicalParticleSetPtr_[level]->GetParticleNormal();
}

HostRealVector &
Geometry::HierarchicalEulerianParticleManager::GetParticleSizeByLevel(
    const int level) {
  return hierarchicalParticleSetPtr_[level]->GetParticleSize();
}

HostIndexVector &
Geometry::HierarchicalEulerianParticleManager::GetParticleTypeByLevel(
    const int level) {
  return hierarchicalParticleSetPtr_[level]->GetParticleType();
}

HostIndexVector &
Geometry::HierarchicalEulerianParticleManager::GetParticleIndexByLevel(
    const int level) {
  return hierarchicalParticleSetPtr_[level]->GetParticleIndex();
}

Void Geometry::HierarchicalEulerianParticleManager::Output(
    const std::string outputFileName, const bool isBinary) {
  EulerianParticleManager::Output(outputFileName, isBinary);

  std::ofstream vtkStream;
  // output refinement level
  if (mpiRank_ == 0) {
    vtkStream.open(outputFileName,
                   std::ios::out | std::ios::app | std::ios::binary);

    vtkStream << "SCALARS level int 1" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open(outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
      for (std::size_t i = 0; i < hostParticleRefinementLevel_.extent(0); i++) {
        int x = hostParticleRefinementLevel_(i);
        SwapEnd(x);
        vtkStream.write(reinterpret_cast<char *>(&x), sizeof(int));
      }
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // output domain ID
  if (mpiRank_ == 0) {
    vtkStream.open(outputFileName,
                   std::ios::out | std::ios::app | std::ios::binary);

    vtkStream << "SCALARS domain int 1" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open(outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
      for (std::size_t i = 0; i < hostParticleRefinementLevel_.extent(0); i++) {
        int x = mpiRank_;
        SwapEnd(x);
        vtkStream.write(reinterpret_cast<char *>(&x), sizeof(int));
      }
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}