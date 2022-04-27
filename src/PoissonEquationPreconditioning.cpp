#include "PoissonEquationPreconditioning.hpp"

#include <Compadre_GMLS.hpp>
#include <Compadre_PointCloudSearch.hpp>

PoissonEquationPreconditioning::PoissonEquationPreconditioning() {}

void PoissonEquationPreconditioning::ConstructInterpolation(
    std::shared_ptr<HierarchicalParticleManager> particleMgr) {
  /*! \brief construct interpolation operator for multilevel preconditioning of
   *         Poisson equation
   *! \date Apr 27, 2022
   *! \author Zisheng Ye <zisheng_ye@outlook.com>
   *
   * Interpolation operator is constructed based on GMLS discretization.  The
   * source particle sites are the coarse level particles and the target
   * particle sites are the fine level particles.  Since the interpolation
   * operator works on the field values, there is no need to distinguish
   * boundary or interior particles.  2nd order polynomials are enough for
   * discretization.
   *
   * The general workflow of the function is as follows:
   * 1. Do neighbor search for target particle sites in the source particle site
   *    set.
   * 2. Calculate the distance between the first element in the target
   *    particle's neighbor list and the target particle itself.  If these two
   *    particles are at the same site, no discretization is needed at this site
   *    and the particle is marked as false in the vector refinedParticle.
   * 3. Do GMLS discretization for all particles marked as true in the vector
   *    refinedParticle.
   * 4. Assemble the interpolation operator into a matrix.
   */
  double tStart, tEnd;
  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();

  MultilevelPreconditioning::ConstructInterpolation(particleMgr);

  const int currentLevel = linearSystemsPtr_.size() - 1;

  if (currentLevel > 0) {
    HostRealMatrix interpolationSourceParticleCoords;
    HostIndexVector interpolationSourceParticleIndex;
    interpolationGhost_.ApplyGhost(
        particleMgr->GetParticleCoordsByLevel(currentLevel - 1),
        interpolationSourceParticleCoords);
    interpolationGhost_.ApplyGhost(
        particleMgr->GetParticleIndexByLevel(currentLevel - 1),
        interpolationSourceParticleIndex);

    HostRealMatrix &sourceParticleCoordsHost =
        interpolationSourceParticleCoords;
    HostIndexVector &sourceParticleIndexHost = interpolationSourceParticleIndex;
    HostRealMatrix &targetParticleCoordsHost =
        particleMgr->GetParticleCoordsByLevel(currentLevel);
    HostRealVector &targetParticleSizeHost =
        particleMgr->GetParticleSizeByLevel(currentLevel);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        sourceParticleCoordsDevice("interpolation source particle coords",
                                   sourceParticleCoordsHost.extent(0),
                                   sourceParticleCoordsHost.extent(1));
    Kokkos::deep_copy(sourceParticleCoordsDevice, sourceParticleCoordsHost);

    Kokkos::View<int **, Kokkos::DefaultHostExecutionSpace> neighborListsHost(
        "interpolation neighborlists", targetParticleCoordsHost.extent(0), 1);
    Kokkos::View<double *, Kokkos::DefaultHostExecutionSpace> epsilonHost(
        "interpolation epsilon", targetParticleCoordsHost.extent(0));

    const int dimension = particleMgr->GetDimension();

    const int localTargetParticleNum = targetParticleCoordsHost.extent(0);
    const int localSourceParticleNum =
        particleMgr->GetParticleCoordsByLevel(currentLevel - 1).extent(0);
    unsigned long globalTargetParticleNum = localTargetParticleNum;
    MPI_Allreduce(MPI_IN_PLACE, &globalTargetParticleNum, 1, MPI_UNSIGNED_LONG,
                  MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < localTargetParticleNum; i++) {
      epsilonHost(i) = 1.50005 * targetParticleSizeHost(i);
    }

    auto pointCloudSearch(
        Compadre::CreatePointCloudSearch(sourceParticleCoordsHost, dimension));
    bool isNeighborSearchPassed = false;

    const int satisfiedNumNeighbor = 2 * Compadre::GMLS::getNP(2, dimension);

    double maxRatio, meanNeighbor;
    int minNeighbor, maxNeighbor, iteCounter;
    iteCounter = 0;
    while (!isNeighborSearchPassed) {
      iteCounter++;
      unsigned int minNeighborLists =
          1 + pointCloudSearch.generate2DNeighborListsFromRadiusSearch(
                  true, targetParticleCoordsHost, neighborListsHost,
                  epsilonHost, 0.0, 0.0);
      if (minNeighborLists > neighborListsHost.extent(1))
        Kokkos::resize(neighborListsHost, localTargetParticleNum,
                       minNeighborLists);
      pointCloudSearch.generate2DNeighborListsFromRadiusSearch(
          false, targetParticleCoordsHost, neighborListsHost, epsilonHost, 0.0,
          0.0);

      bool passNeighborNumCheck = true;

      maxRatio = 0.0;
      minNeighbor = 1000;
      maxNeighbor = 0;
      meanNeighbor = 0;
      for (int i = 0; i < localTargetParticleNum; i++) {
        if (neighborListsHost(i, 0) <= satisfiedNumNeighbor) {
          epsilonHost(i) += 0.25 * targetParticleSizeHost(i);
          passNeighborNumCheck = false;
        }
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
      MPI_Allreduce(MPI_IN_PLACE, &minNeighbor, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &maxNeighbor, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &meanNeighbor, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &maxRatio, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);

      int corePassCheck = 0;
      if (!passNeighborNumCheck)
        corePassCheck = 1;

      MPI_Allreduce(MPI_IN_PLACE, &corePassCheck, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
      if (corePassCheck == 0)
        isNeighborSearchPassed = true;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD,
                "\nAfter satisfying least number of neighbors when building "
                "interpolation operator\niteration count: %d "
                "min neighbor: %d, max neighbor: %d , mean "
                "neighbor: %.2f, max ratio: %.2f\n",
                iteCounter, minNeighbor, maxNeighbor,
                meanNeighbor / (double)globalTargetParticleNum, maxRatio);

    int refinedParticleNum = 0;
    std::vector<bool> refinedParticle(localTargetParticleNum);
    for (int i = 0; i < localTargetParticleNum; i++) {
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
        refinedParticleNum++;
      }
    }

    PetscMatrix &I = *(interpolationPtr_[currentLevel]);
    I.Resize(localTargetParticleNum, localSourceParticleNum);
    std::vector<PetscInt> index;
    std::vector<PetscReal> value;
    for (int i = 0; i < localTargetParticleNum; i++) {
      const PetscInt currentParticleIndex = i;
      if (refinedParticle[i]) {
        index.resize(neighborListsHost(i, 0));
        for (int j = 0; j < neighborListsHost(i, 0); j++) {
          index[j] = sourceParticleIndexHost(neighborListsHost(i, j + 1));
        }
      } else {
        index.resize(1);
        index[0] = sourceParticleIndexHost(neighborListsHost(i, 1));
      }
      std::sort(index.begin(), index.end());
      I.SetColIndex(currentParticleIndex, index);
    }

    I.GraphAssemble();

    Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
        refinedNeighborListsDevice("refined particle neighborlists",
                                   refinedParticleNum,
                                   neighborListsHost.extent(1));
    Kokkos::View<int **>::HostMirror refinedNeighborListsHost =
        Kokkos::create_mirror_view(refinedNeighborListsDevice);

    Kokkos::View<double *, Kokkos::DefaultExecutionSpace> refinedEpsilonDevice(
        "refined particle epsilon", refinedParticleNum);
    Kokkos::View<double *>::HostMirror refinedEpsilonHost =
        Kokkos::create_mirror_view(refinedEpsilonDevice);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        refinedParticleCoordsDevice("refined particle coord",
                                    refinedParticleNum, dimension);
    Kokkos::View<double **>::HostMirror refinedParticleCoordsHost =
        Kokkos::create_mirror_view(refinedParticleCoordsDevice);

    int refinedCounter;
    refinedCounter = 0;
    for (int i = 0; i < localTargetParticleNum; i++) {
      if (refinedParticle[i]) {
        refinedEpsilonHost(refinedCounter) = epsilonHost(i);
        for (int j = 0; j <= neighborListsHost(i, 0); j++) {
          refinedNeighborListsHost(refinedCounter, j) = neighborListsHost(i, j);
        }
        for (int j = 0; j < dimension; j++) {
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

    const int scalarIndex = solutionSet->getAlphaColumnOffset(
        Compadre::ScalarPointEvaluation, 0, 0, 0, 0);

    refinedCounter = 0;
    for (int i = 0; i < localTargetParticleNum; i++) {
      if (refinedParticle[i]) {
        index.resize(neighborListsHost(i, 0));
        value.resize(neighborListsHost(i, 0));
        for (int j = 0; j < neighborListsHost(i, 0); j++) {
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

    const unsigned long nnz = I.Assemble();
    unsigned long nnzMax, nnzMin;
    MPI_Allreduce(&nnz, &nnzMax, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&nnz, &nnzMin, 1, MPI_UNSIGNED_LONG, MPI_MIN, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    tEnd = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,
                "End of building Poisson equation interpolation\nMax nonzeros: "
                "%lu, Min nonzeros: %lu\n",
                nnzMax, nnzMin);
    PetscPrintf(PETSC_COMM_WORLD,
                "Duration of building Poisson equation interpolation:%.4fs\n",
                tEnd - tStart);
  }
}

void PoissonEquationPreconditioning::ConstructRestriction(
    std::shared_ptr<HierarchicalParticleManager> particleMgr) {
  /*! \brief construct restriction operator for multilevel preconditioning of
   *         Poisson equations
   *! \date Apr 27, 2022
   *! \author Zisheng Ye <zisheng_ye@outlook.com>
   *
   * Restriction operator is constructed based on GMLS discretization.  The
   * source particle sites are the fine level particles and the target
   * particle sites are the coarse level particles.  However the restriction
   * operator deals with the rhs (adjoint operator), it is necessary to
   * distinguish boundary and interior particles.  GMLS discretization is done
   * separately for these particles.  2nd order polynomials are enough for
   * discretization.
   *
   * The general workflow for the function is as follows:
   * 1. Split the target and source targets into interior and boundary
   *    particles.
   * 2. Do neighbor search for both boundary and interior target particle sites
   *    in the boundary and interior source particle site set, respectively.
   * 3. Calculate the distance between the first element in the target
   *    particle's neighbor list and the target particle itself.  If these two
   *    particles are at the same site, no discretization is needed at this site
   *    and the particle is marked as false in the vector restrictedParticle.
   * 4. Do GMLS discretization for all particles marked as true in the vector
   *    restricedParticle.
   * 5. Assemble the restriction operator into a matrix.
   */
  MultilevelPreconditioning::ConstructRestriction(particleMgr);

  const int currentLevel = linearSystemsPtr_.size() - 1;

  if (currentLevel > 0) {
    HostRealMatrix restrictionSourceParticleCoords;
    HostIndexVector restrictionSourceParticleIndex;
    HostIndexVector restrictionSourceParticleType;
    restrictionGhost_.ApplyGhost(
        particleMgr->GetParticleCoordsByLevel(currentLevel),
        restrictionSourceParticleCoords);
    restrictionGhost_.ApplyGhost(
        particleMgr->GetParticleIndexByLevel(currentLevel),
        restrictionSourceParticleIndex);
    restrictionGhost_.ApplyGhost(
        particleMgr->GetParticleTypeByLevel(currentLevel),
        restrictionSourceParticleType);

    const unsigned int dimension = particleMgr->GetDimension();

    // search neighbor based on particle type
    std::size_t localInteriorParticleNum = 0;
    std::size_t localBoundaryParticleNum = 0;

    auto &targetParticleCoordsHost =
        particleMgr->GetParticleCoordsByLevel(currentLevel - 1);
    auto &targetParticleType =
        particleMgr->GetParticleTypeByLevel(currentLevel - 1);

    const std::size_t localTargetParticleNum =
        targetParticleCoordsHost.extent(0);
    // const std::size_t localSourceParticleNum =
    //     GetParticleCoordsByLevel(currentLevel).extent(0);

    for (std::size_t i = 0; i < localTargetParticleNum; i++) {
      if (targetParticleType(i) == 0)
        localInteriorParticleNum++;
      else
        localBoundaryParticleNum++;
    }

    auto pointCloudSearch(Compadre::CreatePointCloudSearch(
        restrictionSourceParticleCoords, dimension));

    DeviceIndexMatrix interiorNeighborListsDevice(
        "interior particle neighborlists", localInteriorParticleNum, 1);
    Kokkos::View<std::size_t **>::HostMirror interiorNeighborListsHost =
        Kokkos::create_mirror_view(interiorNeighborListsDevice);
    DeviceIndexMatrix boundaryNeighborListsDevice(
        "boundary particle neighborlist", localBoundaryParticleNum, 1);
    Kokkos::View<std::size_t **>::HostMirror boundaryNeighborListsHost =
        Kokkos::create_mirror_view(boundaryNeighborListsDevice);

    unsigned int minNeighborLists;
    minNeighborLists =
        1 + pointCloudSearch.generate2DNeighborListsFromRadiusSearch(
                true, coords, interiorNeighborListsHost, epsilon_, 0.0, 0.0);
    if (minNeighborLists > interiorNeighborListsHost.extent(1))
      Kokkos::resize(neighborLists_, localParticleNum, minNeighborLists);
    pointCloudSearch.generate2DNeighborListsFromRadiusSearch(
        false, coords, neighborLists_, epsilon_, 0.0, 0.0);
  }
}

void PoissonEquationPreconditioning::ConstructSmoother() {
  PetscPrintf(PETSC_COMM_WORLD,
              "Start of constructing Poisson equation smoother\n");
  MultilevelPreconditioning::ConstructSmoother();

  const int currentLevel = linearSystemsPtr_.size() - 1;
  KSP &ksp = smootherPtr_[currentLevel]->GetReference();
  KSPCreate(MPI_COMM_WORLD, &ksp);
  KSPSetType(ksp, KSPPREONLY);
  KSPSetOperators(ksp, linearSystemsPtr_[currentLevel]->GetReference(),
                  linearSystemsPtr_[currentLevel]->GetReference());

  PC pc;
  KSPGetPC(ksp, &pc);
  PCSetFromOptions(pc);
  PCSetUp(pc);

  KSPSetUp(ksp);
}