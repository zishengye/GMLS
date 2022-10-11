#include "Equation/Poisson/PoissonPreconditioner.hpp"
#include "Equation/MultilevelPreconditioner.hpp"

#include <Compadre_GMLS.hpp>
#include <Compadre_PointCloudSearch.hpp>

Equation::PoissonPreconditioner::PoissonPreconditioner()
    : MultilevelPreconditioner() {}

Equation::PoissonPreconditioner::~PoissonPreconditioner() {}

void Equation::PoissonPreconditioner::ConstructInterpolation(
    DefaultParticleManager &particleMgr) {
  double tStart, tEnd;
  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();

  MultilevelPreconditioner::ConstructInterpolation(particleMgr);

  const int currentLevel = linearSystemsPtr_.size() - 1;

  if (currentLevel > 0) {
    HostRealMatrix interpolationSourceParticleCoords;
    HostIndexVector interpolationSourceParticleIndex;
    interpolationGhost_.ApplyGhost(
        particleMgr.GetParticleCoordsByLevel(currentLevel - 1),
        interpolationSourceParticleCoords);
    interpolationGhost_.ApplyGhost(
        particleMgr.GetParticleIndexByLevel(currentLevel - 1),
        interpolationSourceParticleIndex);

    HostRealMatrix &sourceParticleCoordsHost =
        interpolationSourceParticleCoords;
    HostIndexVector &sourceParticleIndexHost = interpolationSourceParticleIndex;
    HostRealMatrix &targetParticleCoordsHost =
        particleMgr.GetParticleCoordsByLevel(currentLevel);
    HostRealVector &targetParticleSizeHost =
        particleMgr.GetParticleSizeByLevel(currentLevel);

    const unsigned int dimension = particleMgr.GetDimension();

    auto pointCloudSearch(
        Compadre::CreatePointCloudSearch(sourceParticleCoordsHost, dimension));

    const unsigned int satisfiedNumNeighbor =
        pow(sqrt(2), dimension) * Compadre::GMLS::getNP(2, dimension);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        sourceParticleCoordsDevice("interpolation source particle coords",
                                   sourceParticleCoordsHost.extent(0),
                                   sourceParticleCoordsHost.extent(1));
    Kokkos::deep_copy(sourceParticleCoordsDevice, sourceParticleCoordsHost);

    Kokkos::View<std::size_t **, Kokkos::DefaultHostExecutionSpace>
        neighborListsHost("interpolation neighbor lists",
                          targetParticleCoordsHost.extent(0),
                          satisfiedNumNeighbor + 1);
    Kokkos::View<double *, Kokkos::DefaultHostExecutionSpace> epsilonHost(
        "interpolation epsilon", targetParticleCoordsHost.extent(0));

    const unsigned int localTargetParticleNum =
        targetParticleCoordsHost.extent(0);
    const unsigned int localSourceParticleNum =
        particleMgr.GetParticleCoordsByLevel(currentLevel - 1).extent(0);
    unsigned long globalTargetParticleNum = localTargetParticleNum;
    MPI_Allreduce(MPI_IN_PLACE, &globalTargetParticleNum, 1, MPI_UNSIGNED_LONG,
                  MPI_SUM, MPI_COMM_WORLD);

    // initialize epsilon
    pointCloudSearch.generate2DNeighborListsFromKNNSearch(
        true, targetParticleCoordsHost, neighborListsHost, epsilonHost,
        satisfiedNumNeighbor, 1.0);

    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                             0, localTargetParticleNum),
                         [&](const int i) {
                           double minEpsilon = 1.50 * targetParticleSizeHost(i);
                           double minSpacing = 0.25 * targetParticleSizeHost(i);
                           epsilonHost(i) =
                               std::max(minEpsilon, epsilonHost(i));
                           unsigned int scaling = std::ceil(
                               (epsilonHost(i) - minEpsilon) / minSpacing);
                           epsilonHost(i) = minEpsilon + scaling * minSpacing;
                         });
    Kokkos::fence();

    double maxRatio, meanNeighbor;
    unsigned int minNeighbor, maxNeighbor;
    unsigned int minNeighborLists =
        1 + pointCloudSearch.generate2DNeighborListsFromRadiusSearch(
                true, targetParticleCoordsHost, neighborListsHost, epsilonHost,
                0.0, 0.0);
    if (minNeighborLists > neighborListsHost.extent(1))
      Kokkos::resize(neighborListsHost, localTargetParticleNum,
                     minNeighborLists);
    pointCloudSearch.generate2DNeighborListsFromRadiusSearch(
        false, targetParticleCoordsHost, neighborListsHost, epsilonHost, 0.0,
        0.0);

    maxRatio = 0.0;
    minNeighbor = 1000;
    maxNeighbor = 0;
    meanNeighbor = 0;
    for (unsigned int i = 0; i < localTargetParticleNum; i++) {
      if (neighborListsHost(i, 0) < minNeighbor)
        minNeighbor = neighborListsHost(i, 0);
      if (neighborListsHost(i, 0) > maxNeighbor)
        maxNeighbor = neighborListsHost(i, 0);
      meanNeighbor += neighborListsHost(i, 0);

      if (maxRatio < epsilonHost(i) / targetParticleSizeHost(i)) {
        maxRatio = epsilonHost(i) / targetParticleSizeHost(i);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &minNeighbor, 1, MPI_UNSIGNED, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxNeighbor, 1, MPI_UNSIGNED, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &meanNeighbor, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxRatio, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (mpiRank_ == 0)
      printf(
          "\nAfter satisfying least number of neighbors when building "
          "interpolation operator\nmin neighbor: %d, max neighbor: %d , mean "
          "neighbor: %.2f, max ratio: %.2f\n",
          minNeighbor, maxNeighbor,
          meanNeighbor / (double)globalTargetParticleNum, maxRatio);

    int refinedParticleNum = 0;
    std::vector<bool> refinedParticle(localTargetParticleNum);
    Kokkos::parallel_reduce(
        "find refined particles",
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
            0, localTargetParticleNum),
        [&](const std::size_t i, int &tRefinedParticleNum) {
          double x, y, z;
          x = targetParticleCoordsHost(i, 0) -
              sourceParticleCoordsHost(neighborListsHost(i, 1), 0);
          y = targetParticleCoordsHost(i, 1) -
              sourceParticleCoordsHost(neighborListsHost(i, 1), 1);
          z = targetParticleCoordsHost(i, 2) -
              sourceParticleCoordsHost(neighborListsHost(i, 1), 2);
          double distance = sqrt(x * x + y * y + z * z);
          if (distance < 1e-3 * targetParticleSizeHost(i))
            refinedParticle[i] = false;
          else {
            refinedParticle[i] = true;
            tRefinedParticleNum++;
          }
        },
        Kokkos::Sum<int>(refinedParticleNum));
    Kokkos::fence();

    DefaultMatrix &I = interpolationPtr_[currentLevel];
    I.Resize(localTargetParticleNum, localSourceParticleNum);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            0, localTargetParticleNum),
        [&](const int i) {
          std::vector<PetscInt> index;
          const PetscInt currentParticleIndex = i;
          if (refinedParticle[i]) {
            index.resize(neighborListsHost(i, 0));
            for (unsigned int j = 0; j < neighborListsHost(i, 0); j++) {
              index[j] = sourceParticleIndexHost(neighborListsHost(i, j + 1));
            }
          } else {
            index.resize(1);
            index[0] = sourceParticleIndexHost(neighborListsHost(i, 1));
          }
          std::sort(index.begin(), index.end());
          I.SetColIndex(currentParticleIndex, index);
        });

    I.GraphAssemble();

    Kokkos::View<std::size_t **, Kokkos::DefaultExecutionSpace>
        refinedNeighborListsDevice("refined particle neighbor lists",
                                   refinedParticleNum,
                                   neighborListsHost.extent(1));
    Kokkos::View<std::size_t **>::HostMirror refinedNeighborListsHost =
        Kokkos::create_mirror_view(refinedNeighborListsDevice);

    Kokkos::View<double *, Kokkos::DefaultExecutionSpace> refinedEpsilonDevice(
        "refined particle epsilon", refinedParticleNum);
    Kokkos::View<double *>::HostMirror refinedEpsilonHost =
        Kokkos::create_mirror_view(refinedEpsilonDevice);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        refinedParticleCoordsDevice("refined particle coord",
                                    refinedParticleNum, 3);
    Kokkos::View<double **>::HostMirror refinedParticleCoordsHost =
        Kokkos::create_mirror_view(refinedParticleCoordsDevice);

    unsigned int refinedCounter;
    refinedCounter = 0;
    for (unsigned int i = 0; i < localTargetParticleNum; i++) {
      if (refinedParticle[i]) {
        refinedEpsilonHost(refinedCounter) = epsilonHost(i);
        for (unsigned int j = 0; j <= neighborListsHost(i, 0); j++) {
          refinedNeighborListsHost(refinedCounter, j) = neighborListsHost(i, j);
        }
        for (unsigned int j = 0; j < 3; j++) {
          refinedParticleCoordsHost(refinedCounter, j) =
              targetParticleCoordsHost(i, j);
        }

        refinedCounter++;
      }
    }

    Kokkos::deep_copy(refinedParticleCoordsDevice, refinedParticleCoordsHost);
    Kokkos::deep_copy(refinedEpsilonDevice, refinedEpsilonHost);
    Kokkos::deep_copy(refinedNeighborListsDevice, refinedNeighborListsHost);

    Compadre::GMLS interpolationBasis =
        Compadre::GMLS(Compadre::ScalarTaylorPolynomial, Compadre::PointSample,
                       2, dimension, "LU", "STANDARD");

    interpolationBasis.setProblemData(
        refinedNeighborListsDevice, sourceParticleCoordsDevice,
        refinedParticleCoordsDevice, refinedEpsilonDevice);

    interpolationBasis.addTargets(Compadre::ScalarPointEvaluation);

    interpolationBasis.setWeightingType(Compadre::WeightingFunctionType::Power);
    interpolationBasis.setWeightingParameter(4);
    interpolationBasis.setOrderOfQuadraturePoints(2);
    interpolationBasis.setDimensionOfQuadraturePoints(1);
    interpolationBasis.setQuadratureType("LINE");

    interpolationBasis.generateAlphas(1, false);

    auto solutionSet = interpolationBasis.getSolutionSetHost();
    auto interpolationAlpha = solutionSet->getAlphas();

    const unsigned int scalarIndex = solutionSet->getAlphaColumnOffset(
        Compadre::ScalarPointEvaluation, 0, 0, 0, 0);

    std::vector<PetscInt> index;
    std::vector<PetscReal> value;
    refinedCounter = 0;
    for (unsigned int i = 0; i < localTargetParticleNum; i++) {
      if (refinedParticle[i]) {
        index.resize(neighborListsHost(i, 0));
        value.resize(neighborListsHost(i, 0));
        for (unsigned int j = 0; j < neighborListsHost(i, 0); j++) {
          index[j] = sourceParticleIndexHost(neighborListsHost(i, j + 1));

          auto alphaIndex =
              solutionSet->getAlphaIndex(refinedCounter, scalarIndex);
          value[j] = interpolationAlpha(alphaIndex + j);
        }

        refinedCounter++;
      } else {
        index.resize(1);
        value.resize(1);
        index[0] = sourceParticleIndexHost(neighborListsHost(i, 1));
        value[0] = 1.0;
      }

      I.Increment(i, index, value);
    }

    I.Assemble();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  if (mpiRank_ == 0)
    printf("Duration of building Poisson equation interpolation:%.4fs\n",
           tEnd - tStart);
}

void Equation::PoissonPreconditioner::ConstructRestriction(
    DefaultParticleManager &particleMgr) {
  double tStart, tEnd;
  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();

  MultilevelPreconditioner::ConstructRestriction(particleMgr);

  const int currentLevel = linearSystemsPtr_.size() - 1;

  if (currentLevel > 0) {
    HostRealMatrix restrictionSourceParticleCoords;
    HostIndexVector restrictionSourceParticleIndex;
    HostIndexVector restrictionSourceParticleType;
    restrictionGhost_.ApplyGhost(
        particleMgr.GetParticleCoordsByLevel(currentLevel),
        restrictionSourceParticleCoords);
    restrictionGhost_.ApplyGhost(
        particleMgr.GetParticleIndexByLevel(currentLevel),
        restrictionSourceParticleIndex);
    restrictionGhost_.ApplyGhost(
        particleMgr.GetParticleTypeByLevel(currentLevel),
        restrictionSourceParticleType);

    // split source particles according to their particle type
    const std::size_t sourceParticleNum =
        restrictionSourceParticleCoords.extent(0);
    std::size_t interiorSourceParticleNum = 0;
    std::size_t boundarySourceParticleNum = 0;
    for (std::size_t i = 0; i < sourceParticleNum; i++) {
      if (restrictionSourceParticleType(i) == 0)
        interiorSourceParticleNum++;
      else
        boundarySourceParticleNum++;
    }

    HostRealMatrix interiorSourceParticleCoordsHost(
        "interior source particle coords", interiorSourceParticleNum, 3);
    HostIndexVector interiorSourceParticleIndexHost(
        "interior source particle index", interiorSourceParticleNum);
    HostRealMatrix boundarySourceParticleCoordsHost(
        "boundary source particle coords", boundarySourceParticleNum, 3);
    HostIndexVector boundarySourceParticleIndexHost(
        "boundary source particle index", boundarySourceParticleNum);

    std::size_t interiorCounter, boundaryCounter;
    interiorCounter = 0;
    boundaryCounter = 0;
    for (std::size_t i = 0; i < sourceParticleNum; i++) {
      if (restrictionSourceParticleType(i) == 0) {
        for (unsigned int j = 0; j < 3; j++)
          interiorSourceParticleCoordsHost(interiorCounter, j) =
              restrictionSourceParticleCoords(i, j);
        interiorSourceParticleIndexHost(interiorCounter) =
            restrictionSourceParticleIndex(i);
        interiorCounter++;
      } else {
        for (unsigned int j = 0; j < 3; j++)
          boundarySourceParticleCoordsHost(boundaryCounter, j) =
              restrictionSourceParticleCoords(i, j);
        boundarySourceParticleIndexHost(boundaryCounter) =
            restrictionSourceParticleIndex(i);
        boundaryCounter++;
      }
    }

    const unsigned int dimension = particleMgr.GetDimension();

    // search neighbor based on particle type
    std::size_t localInteriorParticleNum = 0;
    std::size_t localBoundaryParticleNum = 0;

    auto &targetParticleCoordsHost =
        particleMgr.GetParticleCoordsByLevel(currentLevel - 1);
    auto &targetParticleType =
        particleMgr.GetParticleTypeByLevel(currentLevel - 1);
    auto &targetParticleSizeHost =
        particleMgr.GetParticleSizeByLevel(currentLevel - 1);

    const std::size_t localTargetParticleNum =
        targetParticleCoordsHost.extent(0);
    const std::size_t localSourceParticleNum =
        particleMgr.GetParticleCoordsByLevel(currentLevel).extent(0);

    for (std::size_t i = 0; i < localTargetParticleNum; i++) {
      if (targetParticleType(i) == 0)
        localInteriorParticleNum++;
      else
        localBoundaryParticleNum++;
    }

    std::size_t globalInteriorParticleNum, globalBoundaryParticleNum;
    MPI_Allreduce(&localInteriorParticleNum, &globalInteriorParticleNum, 1,
                  MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localBoundaryParticleNum, &globalBoundaryParticleNum, 1,
                  MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);

    auto interiorPointCloudSearch(Compadre::CreatePointCloudSearch(
        interiorSourceParticleCoordsHost, dimension));
    auto boundaryPointCloudSearch(Compadre::CreatePointCloudSearch(
        boundarySourceParticleCoordsHost, dimension));

    DeviceRealMatrix interiorTargetParticleCoordsDevice(
        "interior target particle coords", localInteriorParticleNum, 3);
    Kokkos::View<double **>::HostMirror interiorTargetParticleCoordsHost =
        Kokkos::create_mirror_view(interiorTargetParticleCoordsDevice);
    DeviceRealMatrix boundaryTargetParticleCoordsDevice(
        "boundary target particle coords", localBoundaryParticleNum, 3);
    Kokkos::View<double **>::HostMirror boundaryTargetParticleCoordsHost =
        Kokkos::create_mirror_view(boundaryTargetParticleCoordsDevice);

    HostRealVector interiorTargetParticleSize("interior particle size",
                                              localInteriorParticleNum);
    HostRealVector boundaryTargetParticleSize("boundary particle size",
                                              localBoundaryParticleNum);

    interiorCounter = 0;
    boundaryCounter = 0;
    for (std::size_t i = 0; i < localTargetParticleNum; i++) {
      if (targetParticleType(i) == 0) {
        for (unsigned int j = 0; j < 3; j++)
          interiorTargetParticleCoordsHost(interiorCounter, j) =
              targetParticleCoordsHost(i, j);
        interiorTargetParticleSize(interiorCounter) = targetParticleSizeHost(i);
        interiorCounter++;
      } else {
        for (unsigned int j = 0; j < 3; j++)
          boundaryTargetParticleCoordsHost(boundaryCounter, j) =
              targetParticleCoordsHost(i, j);
        boundaryTargetParticleSize(boundaryCounter) = targetParticleSizeHost(i);
        boundaryCounter++;
      }
    }

    unsigned int minNeighborLists;
    unsigned int satisfiedNumNeighbor =
        pow(sqrt(2), dimension) * Compadre::GMLS::getNP(2, dimension);

    DeviceIndexMatrix interiorNeighborListsDevice(
        "interior particle neighbor lists", localInteriorParticleNum,
        satisfiedNumNeighbor + 1);
    Kokkos::View<std::size_t **>::HostMirror interiorNeighborListsHost =
        Kokkos::create_mirror_view(interiorNeighborListsDevice);

    DeviceRealVector interiorEpsilonDevice("interior epsilon",
                                           localInteriorParticleNum);
    Kokkos::View<double *>::HostMirror interiorEpsilonHost =
        Kokkos::create_mirror_view(interiorEpsilonDevice);

    // initialize epsilon
    interiorPointCloudSearch.generate2DNeighborListsFromKNNSearch(
        true, interiorTargetParticleCoordsHost, interiorNeighborListsHost,
        interiorEpsilonHost, satisfiedNumNeighbor, 1.0);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            0, localInteriorParticleNum),
        [&](const int i) {
          double minEpsilon = 0.25 * interiorTargetParticleSize(i);
          double minSpacing = 0.05 * interiorTargetParticleSize(i);
          interiorEpsilonHost(i) = std::max(minEpsilon, interiorEpsilonHost(i));
          unsigned int scaling =
              std::ceil((interiorEpsilonHost(i) - minEpsilon) / minSpacing);
          interiorEpsilonHost(i) = minEpsilon + scaling * minSpacing;
        });
    Kokkos::fence();

    double maxRatio, meanNeighbor;
    unsigned int minNeighbor, maxNeighbor;

    minNeighborLists =
        1 + interiorPointCloudSearch.generate2DNeighborListsFromRadiusSearch(
                true, interiorTargetParticleCoordsHost,
                interiorNeighborListsHost, interiorEpsilonHost, 0.0, 0.0);
    if (minNeighborLists > interiorNeighborListsHost.extent(1))
      Kokkos::resize(interiorNeighborListsHost, localInteriorParticleNum,
                     minNeighborLists);
    interiorPointCloudSearch.generate2DNeighborListsFromRadiusSearch(
        false, interiorTargetParticleCoordsHost, interiorNeighborListsHost,
        interiorEpsilonHost, 0.0, 0.0);

    minNeighbor = 1000;
    maxNeighbor = 0;
    maxRatio = 0;
    meanNeighbor = 0;
    for (unsigned int i = 0; i < localInteriorParticleNum; i++) {
      const std::size_t numNeighbor = interiorNeighborListsHost(i, 0);
      if (numNeighbor < minNeighbor)
        minNeighbor = numNeighbor;
      if (numNeighbor > maxNeighbor)
        maxNeighbor = numNeighbor;
      meanNeighbor += numNeighbor;

      if (maxRatio < interiorEpsilonHost(i) / interiorTargetParticleSize(i))
        maxRatio = interiorEpsilonHost(i) / interiorTargetParticleSize(i);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &minNeighbor, 1, MPI_UNSIGNED, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxNeighbor, 1, MPI_UNSIGNED, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &meanNeighbor, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxRatio, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (mpiRank_ == 0)
      printf("\nAfter satisfying least number of neighbors when building "
             "restriction operator for interior particles\nmin neighbor: "
             "%d, max neighbor: %d , mean "
             "neighbor: %.2f, max ratio: %.2f\n",
             minNeighbor, maxNeighbor,
             meanNeighbor / (double)globalInteriorParticleNum, maxRatio);

    satisfiedNumNeighbor =
        pow(sqrt(2), dimension - 1) * Compadre::GMLS::getNP(2, dimension - 1);

    DeviceIndexMatrix boundaryNeighborListsDevice(
        "boundary particle neighbor list", localBoundaryParticleNum,
        satisfiedNumNeighbor + 1);
    Kokkos::View<std::size_t **>::HostMirror boundaryNeighborListsHost =
        Kokkos::create_mirror_view(boundaryNeighborListsDevice);

    DeviceRealVector boundaryEpsilonDevice("boundary epsilon",
                                           localBoundaryParticleNum);
    Kokkos::View<double *>::HostMirror boundaryEpsilonHost =
        Kokkos::create_mirror_view(boundaryEpsilonDevice);

    boundaryPointCloudSearch.generate2DNeighborListsFromKNNSearch(
        true, boundaryTargetParticleCoordsHost, boundaryNeighborListsHost,
        boundaryEpsilonHost, satisfiedNumNeighbor, 1.0);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            0, localBoundaryParticleNum),
        [&](const int i) {
          double minEpsilon = 0.250005 * boundaryTargetParticleSize(i);
          double minSpacing = 0.05 * boundaryTargetParticleSize(i);
          boundaryEpsilonHost(i) = std::max(minEpsilon, boundaryEpsilonHost(i));
          unsigned int scaling =
              std::ceil((boundaryEpsilonHost(i) - minEpsilon) / minSpacing);
          boundaryEpsilonHost(i) = minEpsilon + scaling * minSpacing;
        });
    Kokkos::fence();

    minNeighborLists =
        1 + boundaryPointCloudSearch.generate2DNeighborListsFromRadiusSearch(
                true, boundaryTargetParticleCoordsHost,
                boundaryNeighborListsHost, boundaryEpsilonHost, 0.0, 0.0);
    if (minNeighborLists > boundaryNeighborListsHost.extent(1))
      Kokkos::resize(boundaryNeighborListsHost, localBoundaryParticleNum,
                     minNeighborLists);
    boundaryPointCloudSearch.generate2DNeighborListsFromRadiusSearch(
        false, boundaryTargetParticleCoordsHost, boundaryNeighborListsHost,
        boundaryEpsilonHost, 0.0, 0.0);

    minNeighbor = 1000;
    maxNeighbor = 0;
    maxRatio = 0;
    meanNeighbor = 0;
    for (unsigned int i = 0; i < localBoundaryParticleNum; i++) {
      const std::size_t numNeighbor = boundaryNeighborListsHost(i, 0);
      if (numNeighbor < minNeighbor)
        minNeighbor = numNeighbor;
      if (numNeighbor > maxNeighbor)
        maxNeighbor = numNeighbor;
      meanNeighbor += numNeighbor;

      if (maxRatio < boundaryEpsilonHost(i) / boundaryTargetParticleSize(i))
        maxRatio = boundaryEpsilonHost(i) / boundaryTargetParticleSize(i);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &minNeighbor, 1, MPI_UNSIGNED, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxNeighbor, 1, MPI_UNSIGNED, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &meanNeighbor, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxRatio, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (mpiRank_ == 0)
      printf("\nAfter satisfying least number of neighbors when building "
             "restriction operator for boundary particles\nmin neighbor: "
             "%d, max neighbor: %d , mean "
             "neighbor: %.2f, max ratio: %.2f\n",
             minNeighbor, maxNeighbor,
             meanNeighbor / (double)globalBoundaryParticleNum, maxRatio);

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<bool> restrictedInteriorParticle(localInteriorParticleNum);
    std::vector<bool> restrictedBoundaryParticle(localBoundaryParticleNum);
    unsigned int numRestrictedInteriorParticle = 0;
    unsigned int numRestrictedBoundaryParticle = 0;
    for (unsigned int i = 0; i < localInteriorParticleNum; i++) {
      std::size_t neighborIndex = interiorNeighborListsHost(i, 1);
      double x, y, z;
      x = interiorTargetParticleCoordsHost(i, 0) -
          interiorSourceParticleCoordsHost(neighborIndex, 0);
      y = interiorTargetParticleCoordsHost(i, 1) -
          interiorSourceParticleCoordsHost(neighborIndex, 1);
      z = interiorTargetParticleCoordsHost(i, 2) -
          interiorSourceParticleCoordsHost(neighborIndex, 2);
      double distance = sqrt(x * x + y * y + z * z);
      if (distance < 1e-3 * interiorTargetParticleSize(i))
        restrictedInteriorParticle[i] = false;
      else {
        restrictedInteriorParticle[i] = true;
        numRestrictedInteriorParticle++;
      }
    }

    for (unsigned int i = 0; i < localBoundaryParticleNum; i++) {
      std::size_t neighborIndex = boundaryNeighborListsHost(i, 1);
      double x, y, z;
      x = boundaryTargetParticleCoordsHost(i, 0) -
          boundarySourceParticleCoordsHost(neighborIndex, 0);
      y = boundaryTargetParticleCoordsHost(i, 1) -
          boundarySourceParticleCoordsHost(neighborIndex, 1);
      z = boundaryTargetParticleCoordsHost(i, 2) -
          boundarySourceParticleCoordsHost(neighborIndex, 2);
      double distance = sqrt(x * x + y * y + z * z);
      if (distance < 1e-3 * boundaryTargetParticleSize(i))
        restrictedBoundaryParticle[i] = false;
      else {
        restrictedBoundaryParticle[i] = true;
        numRestrictedBoundaryParticle++;
      }
    }

    // TODO: change to GMLS basis (Currently, restriction is done by average)
    DefaultMatrix &R = restrictionPtr_[currentLevel];
    R.Resize(localTargetParticleNum, localSourceParticleNum);
    std::vector<PetscInt> index;
    std::vector<PetscReal> value;
    interiorCounter = 0;
    boundaryCounter = 0;
    for (unsigned int i = 0; i < localTargetParticleNum; i++) {
      if (targetParticleType(i) == 0) {
        if (restrictedInteriorParticle[interiorCounter]) {
          unsigned int numNeighbor =
              interiorNeighborListsHost(interiorCounter, 0);
          index.resize(numNeighbor);
          for (unsigned int j = 0; j < numNeighbor; j++) {
            index[j] = interiorSourceParticleIndexHost(
                interiorNeighborListsHost(interiorCounter, j + 1));
          }
        } else {
          index.resize(1);
          index[0] = interiorSourceParticleIndexHost(
              interiorNeighborListsHost(interiorCounter, 1));
        }
        interiorCounter++;
      } else {
        if (restrictedBoundaryParticle[boundaryCounter]) {
          unsigned int numNeighbor =
              boundaryNeighborListsHost(boundaryCounter, 0);
          index.resize(numNeighbor);
          for (unsigned int j = 0; j < numNeighbor; j++) {
            index[j] = boundarySourceParticleIndexHost(
                boundaryNeighborListsHost(boundaryCounter, j + 1));
          }
        } else {
          index.resize(1);
          index[0] = boundarySourceParticleIndexHost(
              boundaryNeighborListsHost(boundaryCounter, 1));
        }
        boundaryCounter++;
      }
      std::sort(index.begin(), index.end());
      R.SetColIndex(i, index);
    }

    R.GraphAssemble();

    // Kokkos::View<std::size_t **, Kokkos::DefaultExecutionSpace>
    //     restrictedInteriorNeighborListsDevice(
    //         "restricted interior particle neighbor lists",
    //         numRestrictedInteriorParticle,
    //         interiorNeighborListsHost.extent(1));
    // Kokkos::View<std::size_t **>::HostMirror
    //     restrictedInteriorNeighborListsHost =
    //         Kokkos::create_mirror_view(restrictedInteriorNeighborListsDevice);

    // Kokkos::View<double *, Kokkos::DefaultExecutionSpace>
    //     restrictedInteriorEpsilonDevice("restricted interior particle
    //     epsilon",
    //                                     numRestrictedInteriorParticle);
    // Kokkos::View<double *>::HostMirror restrictedInteriorEpsilonHost =
    //     Kokkos::create_mirror_view(restrictedInteriorEpsilonDevice);

    // Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
    //     restrictedInteriorParticleCoordsDevice(
    //         "restricted interior particle coord",
    //         numRestrictedInteriorParticle, dimension);
    // Kokkos::View<double **>::HostMirror restrictedInteriorParticleCoordsHost
    // =
    //     Kokkos::create_mirror_view(restrictedInteriorParticleCoordsDevice);

    // interiorCounter = 0;
    // unsigned int restrictedInteriorCounter = 0;
    // for (unsigned int i = 0; i < localTargetParticleNum; i++) {
    //   if (targetParticleType(i) == 0) {
    //     if (restrictedInteriorParticle[interiorCounter]) {
    //       restrictedInteriorEpsilonHost(restrictedInteriorCounter) =
    //           interiorEpsilonHost(interiorCounter);
    //       for (unsigned int j = 0;
    //            j <= interiorNeighborListsHost(interiorCounter, 0); j++) {
    //         restrictedInteriorNeighborListsHost(restrictedInteriorCounter, j)
    //         =
    //             interiorNeighborListsHost(interiorCounter, j);
    //       }
    //       for (unsigned int j = 0; j < dimension; j++) {
    //         restrictedInteriorParticleCoordsHost(restrictedInteriorCounter,
    //         j) =
    //             interiorTargetParticleCoordsHost(interiorCounter, j);
    //       }
    //       restrictedInteriorCounter++;
    //     }
    //     interiorCounter++;
    //   }
    // }

    // Kokkos::deep_copy(restrictedInteriorNeighborListsDevice,
    //                   restrictedInteriorNeighborListsHost);
    // Kokkos::deep_copy(restrictedInteriorEpsilonDevice,
    //                   restrictedInteriorEpsilonHost);
    // Kokkos::deep_copy(restrictedInteriorParticleCoordsDevice,
    //                   restrictedInteriorParticleCoordsHost);

    // Compadre::GMLS restrictionInteriorBasis =
    //     Compadre::GMLS(Compadre::ScalarTaylorPolynomial,
    //     Compadre::PointSample,
    //                    2, dimension, "LU", "STANDARD");

    // restrictionInteriorBasis.setProblemData(
    //     restrictedInteriorNeighborListsDevice,
    //     interiorSourceParticleCoordsHost,
    //     restrictedInteriorParticleCoordsDevice,
    //     restrictedInteriorEpsilonDevice);

    // restrictionInteriorBasis.addTargets(Compadre::ScalarPointEvaluation);

    // restrictionInteriorBasis.setWeightingType(
    //     Compadre::WeightingFunctionType::Power);
    // restrictionInteriorBasis.setWeightingParameter(4);
    // restrictionInteriorBasis.setOrderOfQuadraturePoints(2);
    // restrictionInteriorBasis.setDimensionOfQuadraturePoints(1);
    // restrictionInteriorBasis.setQuadratureType("LINE");

    // restrictionInteriorBasis.generateAlphas(1, false);

    // auto solutionSet = restrictionInteriorBasis.getSolutionSetHost();
    // auto restrictionInteriorAlpha = solutionSet->getAlphas();

    // const unsigned int scalarIndex = solutionSet->getAlphaColumnOffset(
    //     Compadre::ScalarPointEvaluation, 0, 0, 0, 0);

    // build interior restriction basis
    interiorCounter = 0;
    boundaryCounter = 0;
    // restrictedInteriorCounter = 0;
    for (unsigned int i = 0; i < localTargetParticleNum; i++) {
      if (targetParticleType(i) == 0) {
        if (restrictedInteriorParticle[interiorCounter]) {
          unsigned int numNeighbor =
              interiorNeighborListsHost(interiorCounter, 0);
          index.resize(numNeighbor);
          value.resize(numNeighbor);
          for (unsigned int j = 0; j < numNeighbor; j++) {
            index[j] = interiorSourceParticleIndexHost(
                interiorNeighborListsHost(interiorCounter, j + 1));

            // auto alphaIndex = solutionSet->getAlphaIndex(
            //     restrictedInteriorCounter, scalarIndex);
            // value[j] = restrictionInteriorAlpha(alphaIndex + j);
            value[j] = 1.0 / (double)numNeighbor;
          }

          // restrictedInteriorCounter++;
        } else {
          index.resize(1);
          value.resize(1);
          index[0] = interiorSourceParticleIndexHost(
              interiorNeighborListsHost(interiorCounter, 1));
          value[0] = 1.0;
        }
        interiorCounter++;
      } else {
        if (restrictedBoundaryParticle[boundaryCounter]) {
          unsigned int numNeighbor =
              boundaryNeighborListsHost(boundaryCounter, 0);
          index.resize(numNeighbor);
          value.resize(numNeighbor);
          for (unsigned int j = 0; j < numNeighbor; j++) {
            index[j] = boundarySourceParticleIndexHost(
                boundaryNeighborListsHost(boundaryCounter, j + 1));
            value[j] = 1.0 / (double)numNeighbor;
          }
        } else {
          index.resize(1);
          value.resize(1);
          index[0] = boundarySourceParticleIndexHost(
              boundaryNeighborListsHost(boundaryCounter, 1));
          value[0] = 1.0;
        }
        boundaryCounter++;
      }
      R.Increment(i, index, value);
    }

    R.Assemble();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  if (mpiRank_ == 0)
    printf("Duration of building Poisson equation restriction:%.4fs\n",
           tEnd - tStart);
}

void Equation::PoissonPreconditioner::ConstructSmoother() {
  double tStart, tEnd;
  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();

  if (mpiRank_ == 0)
    printf("Start of constructing Poisson equation smoother\n");
  MultilevelPreconditioner::ConstructSmoother();

  const int currentLevel = linearSystemsPtr_.size() - 1;

  LinearAlgebra::LinearSolverDescriptor<DefaultLinearAlgebraBackend> descriptor;
  if (currentLevel > 0)
    descriptor.outerIteration = -1;
  else
    descriptor.outerIteration = 0;
  descriptor.spd = -1;
  descriptor.setFromDatabase = false;

  smootherPtr_[currentLevel].AddLinearSystem(linearSystemsPtr_[currentLevel],
                                             descriptor);

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  if (mpiRank_ == 0)
    printf("Duration of building Poisson equation smoother:%.4fs\n",
           tEnd - tStart);
}