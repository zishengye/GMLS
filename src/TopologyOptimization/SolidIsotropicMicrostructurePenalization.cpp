#include "TopologyOptimization/SolidIsotropicMicrostructurePenalization.hpp"
#include "Core/Typedef.hpp"
#include "Geometry/Ghost.hpp"
#include "TopologyOptimization/TopologyOptimization.hpp"
#include "petsclog.h"
#include <mpi.h>

Void TopologyOptimization::SolidIsotropicMicrostructurePenalization::Output() {
  std::string outputFileName =
      "vtk/TopologyOptimization" + std::to_string(iteration_) + ".vtk";

  Output(outputFileName);
}

Void TopologyOptimization::SolidIsotropicMicrostructurePenalization::Output(
    String outputFileName) {
  equationPtr_->Output(outputFileName);

  std::ofstream vtkStream;
  // output density
  if (mpiRank_ == 0) {
    vtkStream.open(outputFileName,
                   std::ios::out | std::ios::app | std::ios::binary);

    vtkStream << "SCALARS density float 1" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open(outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
      for (std::size_t i = 0; i < density_.extent(0); i++) {
        float x = density_(i);
        SwapEnd(x);
        vtkStream.write(reinterpret_cast<char *>(&x), sizeof(int));
      }
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // output sensitivity
  if (mpiRank_ == 0) {
    vtkStream.open(outputFileName,
                   std::ios::out | std::ios::app | std::ios::binary);

    vtkStream << "SCALARS sensitivity float 1" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open(outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
      for (std::size_t i = 0; i < sensitivity_.extent(0); i++) {
        float x = sensitivity_(i);
        SwapEnd(x);
        vtkStream.write(reinterpret_cast<char *>(&x), sizeof(int));
      }
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

TopologyOptimization::SolidIsotropicMicrostructurePenalization::
    SolidIsotropicMicrostructurePenalization() {}

TopologyOptimization::SolidIsotropicMicrostructurePenalization::
    ~SolidIsotropicMicrostructurePenalization() {}

Void TopologyOptimization::SolidIsotropicMicrostructurePenalization::Init() {
  TopologyOptimization::Init();

  particleMgr_.Init();

  auto localSourceParticleNum = particleMgr_.GetLocalParticleNum();
  Kokkos::resize(volume_, localSourceParticleNum);
  Kokkos::resize(density_, localSourceParticleNum);
  Kokkos::resize(oldDensity_, localSourceParticleNum);
  Kokkos::resize(sensitivity_, localSourceParticleNum);

  int dimension = particleMgr_.GetDimension();

  auto &spacing = particleMgr_.GetParticleSize();

  for (auto i = 0; i < localSourceParticleNum; i++) {
    density_(i) = volumeFraction_;

    volume_(i) = pow(spacing(i), dimension);
  }

  equationPtr_->SetKappa([=](const HostRealMatrix &coords,
                             const HostRealVector &spacing,
                             HostRealVector &kappa) {
    auto &sourceCoords = particleMgr_.GetParticleCoords();

    auto localTargetParticleNum = coords.extent(0);

    Geometry::Ghost ghost;
    ghost.Init(coords, spacing, sourceCoords, 3.0, particleMgr_.GetDimension());

    HostRealVector ghostDensity;
    HostRealMatrix ghostCoords;
    ghost.ApplyGhost(density_, ghostDensity);
    ghost.ApplyGhost(sourceCoords, ghostCoords);

    HostRealVector epsilon;
    HostIndexMatrix neighborLists;
    Kokkos::resize(epsilon, localTargetParticleNum);
    Kokkos::resize(neighborLists, localTargetParticleNum, 2);
    auto pointCloudSearch(Compadre::CreatePointCloudSearch(
        ghostCoords, particleMgr_.GetDimension()));

    pointCloudSearch.generate2DNeighborListsFromKNNSearch(
        false, coords, neighborLists, epsilon, 1, 1.0);

    for (auto i = 0; i < kappa.extent(0); i++)
      kappa(i) = pow(ghostDensity(neighborLists(i, 1)), 3.0);
  });
}

Void TopologyOptimization::SolidIsotropicMicrostructurePenalization::
    Optimize() {
  iteration_ = 0;
  Scalar change = 1;

  if (mpiRank_ == 0)
    printf("Start of SIMP optimization\n");

  Output();

  Scalar newVolume;
  Scalar localVolume;

  while (iteration_ < maxIteration_ && change > 0.01) {
    iteration_++;

    if (mpiRank_ == 0)
      printf("SIMP iteration: %ld\n", iteration_);
    CalculateSensitivity();

    for (auto i = 0; i < density_.extent(0); i++)
      oldDensity_(i) = density_(i);

    for (auto i = 0; i < sensitivity_.extent(0); i++)
      sensitivity_(i) = 3 * pow(density_(i), 2.0) * sensitivity_(i);

    Scalar lowerBound = 0;
    Scalar upperBound = 1e9;
    Scalar middle;
    Scalar move = 0.2;

    Scalar objFunc = equationPtr_->GetObjFunc();

    HostRealVector resultingDensity;
    Kokkos::resize(resultingDensity, density_.extent(0));

    while (upperBound - lowerBound > 1e-6) {
      middle = 0.5 * (upperBound + lowerBound);

      for (auto i = 0; i < sensitivity_.extent(0); i++)
        resultingDensity(i) = std::max(
            1e-3, std::max(density_(i) - move,
                           std::min(1.0, std::min(density_(i) + move,
                                                  density_(i) *
                                                      sqrt(-sensitivity_(i) /
                                                           middle)))));

      localVolume = 0.0;
      for (auto i = 0; i < density_.extent(0); i++)
        localVolume += volume_(i) * resultingDensity(i);

      MPI_Allreduce(&localVolume, &newVolume, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      if (newVolume > volumeFraction_ * domainVolume_)
        lowerBound = middle;
      else
        upperBound = middle;
    }

    for (auto i = 0; i < density_.extent(0); i++) {
      density_(i) = resultingDensity(i);
    }

    change = 0.0;
    for (auto i = 0; i < density_.extent(0); i++)
      if (std::abs(density_(i) - oldDensity_(i)) > change)
        change = std::abs(density_(i) - oldDensity_(i));

    MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    if (mpiRank_ == 0)
      printf("Obj Func: %f, change: %f, volume: %f, volume constraint: %f\n",
             objFunc, change, newVolume, volumeFraction_ * domainVolume_);

    Output();
  }
}