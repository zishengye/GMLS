#include <fstream>

#include "ParticleManager.hpp"

ParticleSet::ParticleSet(CoordType coordType) {
  coordType_ = coordType;

  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);
}

void ParticleSet::SetDimension(const int dimension) { dimension_ = dimension; }

HostRealMatrix &ParticleSet::GetParticleCoords() { return hostParticleCoords_; }

HostRealMatrix &ParticleSet::GetParticleNormal() { return hostParticleNormal_; }

HostRealVector &ParticleSet::GetParticleSize() { return hostParticleSize_; }

HostIntVector &ParticleSet::GetParticleType() { return hostParticleType_; }

HostIndexVector &ParticleSet::GetParticleIndex() { return hostParticleIndex_; }

const LocalIndex ParticleSet::GetLocalParticleNum() {
  return hostParticleCoords_.extent(0);
}

const GlobalIndex ParticleSet::GetGlobalParticleNum() {
  GlobalIndex globalSize;
  globalSize = hostParticleCoords_.extent(0);
  MPI_Allreduce(MPI_IN_PLACE, &globalSize, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  return globalSize;
}

Scalar ParticleSet::GetParticleCoords(const int index, const int dimension) {
  return hostParticleCoords_(index, dimension);
}

void ParticleSet::Resize(const int newLocalSize) {
  Kokkos::resize(hostParticleCoords_, newLocalSize, 3);
  Kokkos::resize(hostParticleNormal_, newLocalSize, 3);
  Kokkos::resize(hostParticleSize_, newLocalSize);
  Kokkos::resize(hostParticleType_, newLocalSize);
  Kokkos::resize(hostParticleIndex_, newLocalSize);
}

void ParticleSet::Balance() {
  partition_.ConstructPartition(hostParticleCoords_, hostParticleIndex_);
  partition_.ApplyPartition(hostParticleCoords_);
  partition_.ApplyPartition(hostParticleNormal_);
  partition_.ApplyPartition(hostParticleSize_);
  partition_.ApplyPartition(hostParticleType_);
}

void ParticleSet::Output(std::string outputFileName, bool isBinary) {
  std::size_t globalParticleNum = GetGlobalParticleNum();
  std::ofstream vtkStream;

  // particle positions
  if (mpiRank_ == 0) {
    if (isBinary)
      vtkStream.open("vtk/" + outputFileName,
                     std::ios::out | std::ios::trunc | std::ios::binary);
    else
      vtkStream.open("vtk/" + outputFileName, std::ios::out | std::ios::trunc);

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
        vtkStream.open("vtk/" + outputFileName,
                       std::ios::out | std::ios::app | std::ios::binary);
      else
        vtkStream.open("vtk/" + outputFileName, std::ios::out | std::ios::app);

      for (int i = 0; i < hostParticleCoords_.extent(0); i++) {
        for (int j = 0; j < hostParticleCoords_.extent(1); j++) {
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
      vtkStream.open("vtk/" + outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
    else
      vtkStream.open("vtk/" + outputFileName, std::ios::out | std::ios::app);

    vtkStream << "POINT_DATA " << globalParticleNum << std::endl;

    vtkStream.close();
  }

  // particle normal
  if (mpiRank_ == 0) {
    if (isBinary)
      vtkStream.open("vtk/" + outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
    else
      vtkStream.open("vtk/" + outputFileName, std::ios::out | std::ios::app);

    vtkStream << "SCALARS n float 3" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }

  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      if (isBinary)
        vtkStream.open("vtk/" + outputFileName,
                       std::ios::out | std::ios::app | std::ios::binary);
      else
        vtkStream.open("vtk/" + outputFileName, std::ios::out | std::ios::app);

      for (int i = 0; i < hostParticleNormal_.extent(0); i++) {
        for (int j = 0; j < hostParticleNormal_.extent(1); j++) {
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
      vtkStream.open("vtk/" + outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
    else
      vtkStream.open("vtk/" + outputFileName, std::ios::out | std::ios::app);

    vtkStream << "SCALARS h float 1" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }

  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      if (isBinary)
        vtkStream.open("vtk/" + outputFileName,
                       std::ios::out | std::ios::app | std::ios::binary);
      else
        vtkStream.open("vtk/" + outputFileName, std::ios::out | std::ios::app);

      for (int i = 0; i < hostParticleSize_.extent(0); i++) {
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
      vtkStream.open("vtk/" + outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
    else
      vtkStream.open("vtk/" + outputFileName, std::ios::out | std::ios::app);

    vtkStream << "SCALARS ID int 1" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }

  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      if (isBinary)
        vtkStream.open("vtk/" + outputFileName,
                       std::ios::out | std::ios::app | std::ios::binary);
      else
        vtkStream.open("vtk/" + outputFileName, std::ios::out | std::ios::app);

      for (int i = 0; i < hostParticleType_.extent(0); i++) {
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

void ParticleManager::BalanceAndIndexInternal() {
  // repartition for load balancing
  auto particleIndex = particleSetPtr_->GetParticleIndex();
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

  // particleSetPtr_->Balance();

  // reindex
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

ParticleManager::ParticleManager() {
  geometryPtr_ = std::make_shared<DomainGeometry>();
  particleSetPtr_ = std::make_shared<ParticleSet>();

  geometryPtr_->SetDimension(0);
  geometryPtr_->SetType(UndefinedDomain);

  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);
}

void ParticleManager::SetDimension(const int dimension) {
  geometryPtr_->SetDimension(dimension);
  particleSetPtr_->SetDimension(dimension);
}

void ParticleManager::SetDomainType(const SimpleDomainShape shape) {
  geometryPtr_->SetType(shape);
}

void ParticleManager::SetSize(const std::vector<Scalar> &size) {
  geometryPtr_->SetSize(size);
}

void ParticleManager::SetSpacing(const Scalar spacing) { spacing_ = spacing; }

const int ParticleManager::GetDimension() {
  return geometryPtr_->GetDimension();
}

void ParticleManager::Init() {
  assert(geometryPtr_->GetType() != UndefinedDomain);
  LocalIndex localParticleNum = geometryPtr_->EstimateNodeNum(spacing_);

  particleSetPtr_->Resize(localParticleNum);

  geometryPtr_->AssignUniformNode(particleSetPtr_->GetParticleCoords(),
                                  particleSetPtr_->GetParticleNormal(),
                                  particleSetPtr_->GetParticleSize(),
                                  particleSetPtr_->GetParticleType(), spacing_);

  BalanceAndIndexInternal();
}

void ParticleManager::Clear() {}

const LocalIndex ParticleManager::GetLocalParticleNum() {
  return particleSetPtr_->GetLocalParticleNum();
}

const GlobalIndex ParticleManager::GetGlobalParticleNum() {
  return particleSetPtr_->GetGlobalParticleNum();
}

HostRealMatrix &ParticleManager::GetParticleNormal() {
  return particleSetPtr_->GetParticleNormal();
}

HostRealMatrix &ParticleManager::GetParticleCoords() {
  return particleSetPtr_->GetParticleCoords();
}

HostRealVector &ParticleManager::GetParticleSize() {
  return particleSetPtr_->GetParticleSize();
}

HostIntVector &ParticleManager::GetParticleType() {
  return particleSetPtr_->GetParticleType();
}

HostIndexVector &ParticleManager::GetParticleIndex() {
  return particleSetPtr_->GetParticleIndex();
}

void ParticleManager::Output(std::string outputFileName, bool isBinary) {
  particleSetPtr_->Output(outputFileName, isBinary);
}

void HierarchicalParticleManager::RefineInternal(HostIndexVector &splitTag) {
  int dimension = GetDimension();
  // estimate new number of particles
  auto &oldParticleCoords = GetParticleCoords();
  auto &oldParticleNormal = GetParticleNormal();
  auto &oldParticleSize = GetParticleSize();
  auto &oldParticleType = GetParticleType();
  Kokkos::View<int *> newLocalParticleNum("new generated particle number",
                                          splitTag.extent(0));
  int newParticleNum = 0;

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
            tNewLocalParticleNum += 4;
          else
            tNewLocalParticleNum += 2;
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

  HostIntVector oldParticleRefinementLevel;
  Kokkos::resize(oldParticleRefinementLevel,
                 hostParticleRefinementLevel_.extent(0));
  Kokkos::deep_copy(hostParticleRefinementLevel_, oldParticleRefinementLevel);
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
                      oldParticleRefinementLevel(i);
                  counter++;
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
                    oldParticleRefinementLevel(i);
                counter++;
              }
            }
          }
        }
      });
  Kokkos::fence();

  BalanceAndIndexInternal();
}

HierarchicalParticleManager::HierarchicalParticleManager()
    : ParticleManager(), currentRefinementLevel_(0) {
  hierarchicalParticleSetPtr_.push_back(std::make_shared<ParticleSet>());
  particleSetPtr_ = hierarchicalParticleSetPtr_[0];
}

HostIntVector &HierarchicalParticleManager::GetParticleRefinementLevel() {
  return hostParticleRefinementLevel_;
}

void HierarchicalParticleManager::Init() {
  ParticleManager::Init();

  Kokkos::resize(hostParticleRefinementLevel_, this->GetLocalParticleNum());
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
          0, hostParticleRefinementLevel_.extent(0)),
      KOKKOS_LAMBDA(const int i) { hostParticleRefinementLevel_(i) = 0; });
  Kokkos::fence();
}

void HierarchicalParticleManager::Clear() { ParticleManager::Clear(); }

void HierarchicalParticleManager::Refine(HostIndexVector &splitTag) {
  RefineInternal(splitTag);
}

const LocalIndex HierarchicalParticleManager::GetLocalParticleNum() {
  return hierarchicalParticleSetPtr_[currentRefinementLevel_]
      ->GetLocalParticleNum();
}

const GlobalIndex HierarchicalParticleManager::GetGlobalParticleNum() {
  return hierarchicalParticleSetPtr_[currentRefinementLevel_]
      ->GetGlobalParticleNum();
}