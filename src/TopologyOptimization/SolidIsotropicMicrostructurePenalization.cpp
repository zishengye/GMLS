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
  TopologyOptimization::TopologyOptimization::Output(outputFileName);
}

TopologyOptimization::SolidIsotropicMicrostructurePenalization::
    SolidIsotropicMicrostructurePenalization() {}

TopologyOptimization::SolidIsotropicMicrostructurePenalization::
    ~SolidIsotropicMicrostructurePenalization() {}

Void TopologyOptimization::SolidIsotropicMicrostructurePenalization::Init() {
  TopologyOptimization::Init();

  particleMgr_.Init();

  auto localSourceParticleNum = particleMgr_.GetLocalParticleNum();
  auto &sourceCoords = particleMgr_.GetParticleCoords();
  Kokkos::resize(volume_, localSourceParticleNum);
  Kokkos::resize(density_, localSourceParticleNum);
  Kokkos::resize(oldDensity_, localSourceParticleNum);
  Kokkos::resize(sensitivity_, localSourceParticleNum);

  int dimension = particleMgr_.GetDimension();

  auto &particleSize = particleMgr_.GetParticleSize();

  for (auto i = 0; i < localSourceParticleNum; i++) {
    density_(i) = volumeFraction_;

    volume_(i) = pow(particleSize(i), dimension);
  }

  equationPtr_->SetKappa([=](const HostRealMatrix &coords,
                             const HostRealVector &spacing,
                             HostRealVector &kappa) {
    auto localTargetParticleNum = coords.extent(0);

    Geometry::Ghost ghost;
    ghost.Init(coords, spacing, sourceCoords, 8.0, particleMgr_.GetDimension());

    HostRealVector ghostDensity;
    HostRealMatrix ghostCoords;
    ghost.ApplyGhost(density_, ghostDensity);
    ghost.ApplyGhost(sourceCoords, ghostCoords);

    HostRealVector epsilon;
    HostIndexMatrix neighborLists;
    Kokkos::resize(epsilon, localTargetParticleNum);
    Kokkos::resize(neighborLists, localTargetParticleNum, 3);
    auto pointCloudSearch(Compadre::CreatePointCloudSearch(
        ghostCoords, particleMgr_.GetDimension()));

    pointCloudSearch.generate2DNeighborListsFromKNNSearch(
        false, coords, neighborLists, epsilon, 2, 1.0);

    for (int i = 0; i < localTargetParticleNum; i++) {
      int scaling = floor(epsilon(i) / spacing(i) * 1000 + 0.5) + 1;
      epsilon(i) = scaling * 1e-3 * spacing(i);
    }

    unsigned int minNeighborLists =
        1 + pointCloudSearch.generate2DNeighborListsFromRadiusSearch(
                true, coords, neighborLists, epsilon, 0.0, 0.0);
    if (minNeighborLists > neighborLists.extent(1))
      Kokkos::resize(neighborLists, localTargetParticleNum, minNeighborLists);
    pointCloudSearch.generate2DNeighborListsFromRadiusSearch(
        false, coords, neighborLists, epsilon, 0.0, 0.0);

    for (auto i = 0; i < kappa.extent(0); i++)
      kappa(i) = ghostDensity(neighborLists(i, 1));
  });
}

Void TopologyOptimization::SolidIsotropicMicrostructurePenalization::
    Optimize() {
  iteration_ = 0;
  Scalar change = 1;

  auto &particleType = particleMgr_.GetParticleType();

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
      sensitivity_(i) = density_(i) * sensitivity_(i);

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
        if (particleType(i) == 0)
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