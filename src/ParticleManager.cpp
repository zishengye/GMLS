#include <fstream>

#include "ParticleManager.hpp"

template <typename T> void SwapEnd(T &var) {
  char *varArray = reinterpret_cast<char *>(&var);
  for (long i = 0; i < static_cast<long>(sizeof(var) / 2); i++)
    std::swap(varArray[sizeof(var) - 1 - i], varArray[i]);
}

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

void ParticleSet::BuildGhost() {
  ghost_.Init(hostParticleCoords_, hostParticleSize_, hostParticleCoords_, 3.0,
              dimension_);
  ghost_.ApplyGhost(hostParticleCoords_, hostGhostParticleCoords_);
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

ParticleManager::ParticleManager() {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);
}

void ParticleManager::SetDimension(const int dimension) {
  geometry_.SetDimension(dimension);
  particleSet_.SetDimension(dimension);
}

void ParticleManager::SetDomainType(SimpleDomainShape shape) {
  geometry_.SetType(shape);
}

void ParticleManager::SetSize(const std::vector<Scalar> &size) {
  geometry_.SetSize(size);
}

void ParticleManager::SetSpacing(const Scalar spacing) { spacing_ = spacing; }

void ParticleManager::Init() {
  LocalIndex localParticleNum = geometry_.EstimateNodeNum(spacing_);

  particleSet_.Resize(localParticleNum);

  geometry_.AssignUniformNode(
      particleSet_.GetParticleCoords(), particleSet_.GetParticleNormal(),
      particleSet_.GetParticleSize(), particleSet_.GetParticleType(), spacing_);

  // repartition for load balancing
  auto particleIndex = particleSet_.GetParticleIndex();
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
  for (int i = 0; i < particleIndex.extent(0); i++) {
    particleIndex(i) = i + rankParticleNumOffset[mpiRank_];
  }

  particleSet_.Balance();
  particleSet_.BuildGhost();

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
  for (int i = 0; i < particleIndex.extent(0); i++) {
    particleIndex(i) = i + rankParticleNumOffset[mpiRank_];
  }
}

const LocalIndex ParticleManager::GetLocalParticleNum() {
  return particleSet_.GetLocalParticleNum();
}

const GlobalIndex ParticleManager::GetGlobalParticleNum() {
  return particleSet_.GetGlobalParticleNum();
}

void ParticleManager::Output(std::string outputFileName, bool isBinary) {
  particleSet_.Output(outputFileName, isBinary);
}

HierarchicalParticleManager::HierarchicalParticleManager() {}

void HierarchicalParticleManager::Init() { ParticleManager::Init(); }