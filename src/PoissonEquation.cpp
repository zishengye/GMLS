#include "PoissonEquation.hpp"
#include "PolyBasis.hpp"

#include <Compadre_Evaluator.hpp>
#include <Compadre_GMLS.hpp>
#include <Compadre_PointCloudSearch.hpp>

void PoissonEquation::InitLinearSystem() {
  /*! \brief initialize the operating matrix of Poisson equation
   *! \date Apr 27, 2022
   *! \author Zisheng Ye <zisheng_ye@outlook.com>
   *
   * The initialization of the linear system prepares the assembly of the final
   * operator.  An important part in the function is to construct a neighbor
   * list for each particle site.  Such neighbor list has to satisfy two
   * requirements.  One is the minimum number of neighbors depending on the
   * order of polynomials used.  Another one is the quality of discretization
   * required by the preconditioning.  The positivity of the diagonal entry of
   * the discretized operator is necessary for a good convergence for the linear
   * system solving stage.  Once all neighbor lists for each particle have
   * satisfied the two requirements, a graph assembly of the final linear system
   * is performed at the end of the function.  The final graph depends on the
   * exact boundary conditions for each particles.
   *
   * The general workflow of the function is as follows:
   * 1. Construct a neighbor list for each particle site with the given minimum
   *    number of neighbors requirement.
   * 2. Enlarge the neighbor radius for each particle site until it satisfies
   *    the quality requirement.
   * 3. Do the graph assembly.
   */

  /*!
   * TODO: add support for Neumann BCs.
   */
  double tStart, tEnd;
  tStart = MPI_Wtime();
  Equation::InitLinearSystem();
  if (mpiRank_ == 0)
    printf("Start of initializing physics: Poisson\n");

  auto &sourceCoords = hostGhostParticleCoords_;
  auto &sourceIndex = hostGhostParticleIndex_;
  auto &coords = particleMgr_.GetParticleCoords();
  auto &spacing = particleMgr_.GetParticleSize();
  auto &particleType = particleMgr_.GetParticleType();
  auto &normal = particleMgr_.GetParticleNormal();

  const unsigned int localParticleNum = coords.extent(0);
  const unsigned long globalParticleNum = particleMgr_.GetGlobalParticleNum();

  const unsigned int dimension = particleMgr_.GetDimension();

  unsigned int interiorParticleNum = 0, boundaryParticleNum = 0;
  for (std::size_t i = 0; i < localParticleNum; i++) {
    if (particleType(i) == 0)
      interiorParticleNum++;
    else
      boundaryParticleNum++;
  }

  const unsigned satisfiedNumNeighbor =
      2 * Compadre::GMLS::getNP(polyOrder_, dimension);

  Equation::ConstructNeighborLists(satisfiedNumNeighbor);

  std::vector<bool> discretizationCheck(localParticleNum);
  for (std::size_t i = 0; i < localParticleNum; i++) {
    discretizationCheck[i] = false;
  }

  auto pointCloudSearch(
      Compadre::CreatePointCloudSearch(sourceCoords, dimension));
  bool isNeighborSearchPassed = false;

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
      "source coords", sourceCoords.extent(0), sourceCoords.extent(1));
  Kokkos::deep_copy(sourceCoordsDevice, sourceCoords);

  double maxRatio, meanNeighbor;
  unsigned int minNeighbor, maxNeighbor, iteCounter;
  iteCounter = 0;
  while (!isNeighborSearchPassed) {
    iteCounter++;

    Kokkos::View<std::size_t **, Kokkos::DefaultExecutionSpace>
        interiorNeighborListsDevice("interior particle neighborlists",
                                    interiorParticleNum,
                                    neighborLists_.extent(1));
    Kokkos::View<std::size_t **>::HostMirror interiorNeighborListsHost =
        Kokkos::create_mirror_view(interiorNeighborListsDevice);
    Kokkos::View<std::size_t **, Kokkos::DefaultExecutionSpace>
        boundaryNeighborListsDevice("boundary particle neighborlists",
                                    boundaryParticleNum,
                                    neighborLists_.extent(1));
    Kokkos::View<std::size_t **>::HostMirror boundaryNeighborListsHost =
        Kokkos::create_mirror_view(boundaryNeighborListsDevice);

    Kokkos::View<double *, Kokkos::DefaultExecutionSpace> interiorEpsilonDevice(
        "interior particle epsilon", interiorParticleNum);
    Kokkos::View<double *>::HostMirror interiorEpsilonHost =
        Kokkos::create_mirror_view(interiorEpsilonDevice);
    Kokkos::View<double *, Kokkos::DefaultExecutionSpace> boundaryEpsilonDevice(
        "boundary particle epsilon", boundaryParticleNum);
    Kokkos::View<double *>::HostMirror boundaryEpsilonHost =
        Kokkos::create_mirror_view(boundaryEpsilonDevice);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        interiorParticleCoordsDevice("interior particle coord",
                                     interiorParticleNum, dimension);
    Kokkos::View<double **>::HostMirror interiorParticleCoordsHost =
        Kokkos::create_mirror_view(interiorParticleCoordsDevice);
    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        boundaryParticleCoordsDevice("boundary particle coord",
                                     boundaryParticleNum, dimension);
    Kokkos::View<double **>::HostMirror boundaryParticleCoordsHost =
        Kokkos::create_mirror_view(boundaryParticleCoordsDevice);

    unsigned int interiorCounter = 0;
    unsigned int boundaryCounter = 0;
    for (std::size_t i = 0; i < localParticleNum; i++) {
      if (!discretizationCheck[i]) {
        if (particleType(i) == 0) {
          interiorEpsilonHost(interiorCounter) = epsilon_(i);
          for (std::size_t j = 0; j <= neighborLists_(i, 0); j++) {
            interiorNeighborListsHost(interiorCounter, j) =
                neighborLists_(i, j);
          }
          for (unsigned int j = 0; j < dimension; j++) {
            interiorParticleCoordsHost(interiorCounter, j) = coords(i, j);
          }

          interiorCounter++;
        } else {
          boundaryEpsilonHost(boundaryCounter) = epsilon_(i);
          for (std::size_t j = 0; j <= neighborLists_(i, 0); j++) {
            boundaryNeighborListsHost(boundaryCounter, j) =
                neighborLists_(i, j);
          }
          for (unsigned int j = 0; j < dimension; j++) {
            boundaryParticleCoordsHost(boundaryCounter, j) = coords(i, j);
          }

          boundaryCounter++;
        }
      }
    }

    Kokkos::deep_copy(interiorNeighborListsDevice, interiorNeighborListsHost);
    Kokkos::deep_copy(boundaryNeighborListsDevice, boundaryNeighborListsHost);
    Kokkos::deep_copy(interiorEpsilonDevice, interiorEpsilonHost);
    Kokkos::deep_copy(boundaryEpsilonDevice, boundaryEpsilonHost);
    Kokkos::deep_copy(interiorParticleCoordsDevice, interiorParticleCoordsHost);
    Kokkos::deep_copy(boundaryParticleCoordsDevice, boundaryParticleCoordsHost);

    {
      Compadre::GMLS interiorBasis =
          Compadre::GMLS(Compadre::ScalarTaylorPolynomial,
                         Compadre::StaggeredEdgeAnalyticGradientIntegralSample,
                         polyOrder_, dimension, "LU", "STANDARD");

      interiorBasis.setProblemData(
          interiorNeighborListsDevice, sourceCoordsDevice,
          interiorParticleCoordsDevice, interiorEpsilonDevice);

      interiorBasis.addTargets(Compadre::LaplacianOfScalarPointEvaluation);

      interiorBasis.setWeightingType(Compadre::WeightingFunctionType::Power);
      interiorBasis.setWeightingParameter(4);
      interiorBasis.setOrderOfQuadraturePoints(2);
      interiorBasis.setDimensionOfQuadraturePoints(dimension - 1);
      interiorBasis.setQuadratureType("LINE");

      interiorBasis.generateAlphas(1, false);

      auto solutionSet = interiorBasis.getSolutionSetHost();
      auto interiorAlpha = solutionSet->getAlphas();

      const unsigned int interiorLaplacianIndex =
          solutionSet->getAlphaColumnOffset(
              Compadre::LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);

      interiorCounter = 0;
      for (std::size_t i = 0; i < localParticleNum; i++) {
        if (particleType(i) == 0 && discretizationCheck[i] == false) {
          double Aij = 0.0;
          for (std::size_t j = 0;
               j < interiorNeighborListsHost(interiorCounter, 0); j++) {
            auto alphaIndex = solutionSet->getAlphaIndex(
                interiorCounter, interiorLaplacianIndex);
            Aij -= interiorAlpha(alphaIndex + j);
          }
          if (Aij > 1e-3)
            discretizationCheck[i] = true;

          interiorCounter++;
        }
      }
    }

    interiorParticleNum = 0;
    for (std::size_t i = 0; i < localParticleNum; i++) {
      if (particleType(i) == 0 && discretizationCheck[i] == false) {
        interiorParticleNum++;
        epsilon_(i) += 0.25 * spacing(i);
      }
    }

    Kokkos::View<double ***, Kokkos::DefaultExecutionSpace> tangentBundleDevice(
        "tangent bundles", boundaryParticleNum, dimension, dimension);
    Kokkos::View<double ***>::HostMirror tangentBundleHost =
        Kokkos::create_mirror_view(tangentBundleDevice);

    boundaryCounter = 0;
    for (std::size_t i = 0; i < localParticleNum; i++) {
      if (particleType(i) != 0 && discretizationCheck[i] == false) {
        if (dimension == 3) {
          tangentBundleHost(boundaryCounter, 0, 0) = 0.0;
          tangentBundleHost(boundaryCounter, 0, 1) = 0.0;
          tangentBundleHost(boundaryCounter, 0, 2) = 0.0;
          tangentBundleHost(boundaryCounter, 1, 0) = 0.0;
          tangentBundleHost(boundaryCounter, 1, 1) = 0.0;
          tangentBundleHost(boundaryCounter, 1, 2) = 0.0;
          tangentBundleHost(boundaryCounter, 2, 0) = normal(i, 0);
          tangentBundleHost(boundaryCounter, 2, 1) = normal(i, 1);
          tangentBundleHost(boundaryCounter, 2, 2) = normal(i, 2);
        }
        if (dimension == 2) {
          tangentBundleHost(boundaryCounter, 0, 0) = 0.0;
          tangentBundleHost(boundaryCounter, 0, 1) = 0.0;
          tangentBundleHost(boundaryCounter, 1, 0) = normal(i, 0);
          tangentBundleHost(boundaryCounter, 1, 1) = normal(i, 1);
        }
        boundaryCounter++;
      }
    }

    Kokkos::deep_copy(tangentBundleDevice, tangentBundleHost);

    {
      Compadre::GMLS boundaryBasis = Compadre::GMLS(
          Compadre::ScalarTaylorPolynomial,
          Compadre::StaggeredEdgeAnalyticGradientIntegralSample, polyOrder_,
          dimension, "LU", "STANDARD", "NEUMANN_GRAD_SCALAR");

      boundaryBasis.setProblemData(
          boundaryNeighborListsDevice, sourceCoordsDevice,
          boundaryParticleCoordsDevice, boundaryEpsilonDevice);

      boundaryBasis.setTangentBundle(tangentBundleDevice);

      boundaryBasis.addTargets(Compadre::LaplacianOfScalarPointEvaluation);

      boundaryBasis.setWeightingType(Compadre::WeightingFunctionType::Power);
      boundaryBasis.setWeightingParameter(4);
      boundaryBasis.setOrderOfQuadraturePoints(2);
      boundaryBasis.setDimensionOfQuadraturePoints(dimension - 1);
      boundaryBasis.setQuadratureType("LINE");

      boundaryBasis.generateAlphas(1, false);

      auto solutionSet = boundaryBasis.getSolutionSetHost();
      auto boundaryAlpha = solutionSet->getAlphas();

      const unsigned int boundaryLaplacianIndex =
          solutionSet->getAlphaColumnOffset(
              Compadre::LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);

      boundaryCounter = 0;
      for (std::size_t i = 0; i < localParticleNum; i++) {
        if (particleType(i) != 0 && discretizationCheck[i] == false) {
          double Aij = 0.0;
          for (std::size_t j = 0;
               j < boundaryNeighborListsHost(boundaryCounter, 0); j++) {
            auto alphaIndex = solutionSet->getAlphaIndex(
                boundaryCounter, boundaryLaplacianIndex);
            Aij -= boundaryAlpha(alphaIndex + j);
          }
          if (Aij > 1e-3)
            discretizationCheck[i] = true;

          boundaryCounter++;
        }
      }
    }

    boundaryParticleNum = 0;
    for (std::size_t i = 0; i < localParticleNum; i++) {
      if (particleType(i) != 0 && discretizationCheck[i] == false) {
        boundaryParticleNum++;
        epsilon_(i) += 0.25 * spacing(i);
      }
    }

    int corePassCheck = 0;
    if (boundaryParticleNum != 0 || interiorParticleNum != 0)
      corePassCheck = 1;

    MPI_Allreduce(MPI_IN_PLACE, &corePassCheck, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    if (corePassCheck == 0)
      isNeighborSearchPassed = true;
    else {
      unsigned int minNeighborLists =
          1 + pointCloudSearch.generate2DNeighborListsFromRadiusSearch(
                  true, coords, neighborLists_, epsilon_, 0.0, 0.0);
      if (minNeighborLists > neighborLists_.extent(1))
        Kokkos::resize(neighborLists_, localParticleNum, minNeighborLists);
      pointCloudSearch.generate2DNeighborListsFromRadiusSearch(
          false, coords, neighborLists_, epsilon_, 0.0, 0.0);
    }
  }

  // check if there are any duplicate particles
  for (std::size_t i = 0; i < localParticleNum; i++) {
    for (std::size_t j = 1; j < neighborLists_(i, 0); j++) {
      std::size_t neighborIndex = neighborLists_(i, j + 1);
      double x = coords(i, 0) - sourceCoords(neighborIndex, 0);
      double y = coords(i, 1) - sourceCoords(neighborIndex, 1);
      double z = coords(i, 2) - sourceCoords(neighborIndex, 2);
      double dist = sqrt(x * x + y * y + z * z);
      if (dist < 1e-6) {
        if (dimension == 2)
          printf("mpi rank: %d, coord %ld: (%f, %f), coord "
                 "%ld, %ld, %ld, %ld: (%f, %f)\n",
                 mpiRank_, i, coords(i, 0), coords(i, 1), neighborIndex,
                 neighborLists_(i, 1), sourceIndex(i),
                 sourceIndex(neighborIndex), sourceCoords(neighborIndex, 0),
                 sourceCoords(neighborIndex, 1));
        if (dimension == 3)
          printf("mpi rank: %d, coord %ld: (%f, %f, %f), coord "
                 "%ld: (%f, %f, %f), %d\n",
                 mpiRank_, i, coords(i, 0), coords(i, 1), coords(i, 2),
                 neighborLists_(i, 1), sourceCoords(neighborIndex, 0),
                 sourceCoords(neighborIndex, 1), sourceCoords(neighborIndex, 2),
                 neighborIndex > localParticleNum);
        break;
      }
    }
  }

  maxRatio = 0.0;
  minNeighbor = 1000;
  maxNeighbor = 0;
  meanNeighbor = 0;
  for (std::size_t i = 0; i < localParticleNum; i++) {
    if (neighborLists_(i, 0) < minNeighbor)
      minNeighbor = neighborLists_(i, 0);
    if (neighborLists_(i, 0) > maxNeighbor)
      maxNeighbor = neighborLists_(i, 0);
    meanNeighbor += neighborLists_(i, 0);

    if (maxRatio < epsilon_(i) / spacing(i)) {
      maxRatio = epsilon_(i) / spacing(i);
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

  auto &A = *(linearSystemsPtr_[refinementIteration_]);
  A.Resize(localParticleNum, localParticleNum);
  std::vector<PetscInt> index;
  for (std::size_t i = 0; i < localParticleNum; i++) {
    const PetscInt currentParticleIndex = i;
    if (particleType[i] == 0) {
      index.resize(neighborLists_(i, 0));
      for (std::size_t j = 0; j < neighborLists_(i, 0); j++) {
        const PetscInt neighborParticleIndex =
            sourceIndex(neighborLists_(i, j + 1));
        index[j] = neighborParticleIndex;
      }
    } else {
      index.resize(1);
      index[0] = sourceIndex(i);
    }
    std::sort(index.begin(), index.end());
    A.SetColIndex(currentParticleIndex, index);
  }

  A.GraphAssemble();

  tEnd = MPI_Wtime();
  if (mpiRank_ == 0) {
    printf("\nAfter satisfying conditioning of Laplacian operator\niteration "
           "count: %d min neighbor: %d, max neighbor: %d , mean "
           "neighbor: %.2f, max ratio: %.2f\n",
           iteCounter, minNeighbor, maxNeighbor,
           meanNeighbor / (double)globalParticleNum, maxRatio);
    printf("Duration of initializing linear system:%.4fs\n", tEnd - tStart);
  }
}

void PoissonEquation::ConstructLinearSystem() {
  /*! \brief construct the linear system
   *! \date Apr 27, 2022
   *! \author Zisheng Ye <zisheng_ye@outlook.com>
   *
   * Perform the GMLS discretization for each particle and assemble the linear
   * system.  In order to reduce the peak memory usage and take advantage of
   * thread-level parallism, the GMLS discretization and assembly operation is
   * split into several batches.  Each batch contains several particles.  In
   * each batch, GMLS discretization is performed first and the generated
   * coefficients are assembled into the linear system then.
   *
   * The general workflow of the function is as follows:
   * 1. Loop over all batches:
   *    a. Perform the GMLS discretization;
   *    b. Insert the generated coefficients into the final linear system.
   * 2. Assemble the linear system.
   */
  double tStart, tEnd;
  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();
  if (mpiRank_ == 0)
    printf("Start of building linear system of physics: Poisson\n");

  auto &sourceCoords = hostGhostParticleCoords_;
  auto &sourceIndex = hostGhostParticleIndex_;
  auto &coords = particleMgr_.GetParticleCoords();
  auto &particleType = particleMgr_.GetParticleType();
  // auto &normal = particleMgr_.GetParticleNormal();

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
      "source coords", sourceCoords.extent(0), sourceCoords.extent(1));
  Kokkos::deep_copy(sourceCoordsDevice, sourceCoords);

  const unsigned int localParticleNum = coords.extent(0);

  const unsigned int dimension = particleMgr_.GetDimension();

  auto &A = *(linearSystemsPtr_[refinementIteration_]);

  const unsigned int batchSize = (dimension == 2) ? 5000 : 1000;
  const unsigned int batchNum = localParticleNum / batchSize +
                                ((localParticleNum % batchSize > 0) ? 1 : 0);
  for (unsigned int batch = 0; batch < batchNum; batch++) {
    const unsigned int startParticle = batch * batchSize;
    const unsigned int endParticle =
        std::min((batch + 1) * batchSize, localParticleNum);
    unsigned int interiorParticleNum, boundaryParticleNum;
    interiorParticleNum = 0;
    boundaryParticleNum = 0;
    for (unsigned int i = startParticle; i < endParticle; i++) {
      if (particleType(i) == 0)
        interiorParticleNum++;
      else
        boundaryParticleNum++;
    }

    Kokkos::View<std::size_t **, Kokkos::DefaultExecutionSpace>
        interiorNeighborListsDevice("interior particle neighborlist",
                                    interiorParticleNum,
                                    neighborLists_.extent(1));
    Kokkos::View<std::size_t **>::HostMirror interiorNeighborListsHost =
        Kokkos::create_mirror_view(interiorNeighborListsDevice);
    Kokkos::View<std::size_t **, Kokkos::DefaultExecutionSpace>
        boundaryNeighborListsDevice("boundary particle neighborlist",
                                    boundaryParticleNum,
                                    neighborLists_.extent(1));
    Kokkos::View<std::size_t **>::HostMirror boundaryNeighborListsHost =
        Kokkos::create_mirror_view(boundaryNeighborListsDevice);

    Kokkos::View<double *, Kokkos::DefaultExecutionSpace> interiorEpsilonDevice(
        "interior particle epsilon", interiorParticleNum);
    Kokkos::View<double *>::HostMirror interiorEpsilonHost =
        Kokkos::create_mirror_view(interiorEpsilonDevice);
    Kokkos::View<double *, Kokkos::DefaultExecutionSpace> boundaryEpsilonDevice(
        "boundary particle epsilon", boundaryParticleNum);
    Kokkos::View<double *>::HostMirror boundaryEpsilonHost =
        Kokkos::create_mirror_view(boundaryEpsilonDevice);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        interiorParticleCoordsDevice("interior particle coord",
                                     interiorParticleNum, dimension);
    Kokkos::View<double **>::HostMirror interiorParticleCoordsHost =
        Kokkos::create_mirror_view(interiorParticleCoordsDevice);
    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        boundaryParticleCoordsDevice("boundary particle coord",
                                     boundaryParticleNum, dimension);
    Kokkos::View<double **>::HostMirror boundaryParticleCoordsHost =
        Kokkos::create_mirror_view(boundaryParticleCoordsDevice);

    unsigned int boundaryCounter, interiorCounter;

    boundaryCounter = 0;
    interiorCounter = 0;
    for (unsigned int i = startParticle; i < endParticle; i++) {
      if (particleType(i) == 0) {
        interiorEpsilonHost(interiorCounter) = epsilon_(i);
        for (std::size_t j = 0; j <= neighborLists_(i, 0); j++) {
          interiorNeighborListsHost(interiorCounter, j) = neighborLists_(i, j);
        }
        for (unsigned int j = 0; j < dimension; j++) {
          interiorParticleCoordsHost(interiorCounter, j) = coords(i, j);
        }

        interiorCounter++;
      } else {
        boundaryEpsilonHost(boundaryCounter) = epsilon_(i);
        for (std::size_t j = 0; j <= neighborLists_(i, 0); j++) {
          boundaryNeighborListsHost(boundaryCounter, j) = neighborLists_(i, j);
        }
        for (unsigned int j = 0; j < dimension; j++) {
          boundaryParticleCoordsHost(boundaryCounter, j) = coords(i, j);
        }

        boundaryCounter++;
      }
    }

    Kokkos::deep_copy(interiorNeighborListsDevice, interiorNeighborListsHost);
    Kokkos::deep_copy(boundaryNeighborListsDevice, boundaryNeighborListsHost);
    Kokkos::deep_copy(interiorEpsilonDevice, interiorEpsilonHost);
    Kokkos::deep_copy(boundaryEpsilonDevice, boundaryEpsilonHost);
    Kokkos::deep_copy(interiorParticleCoordsDevice, interiorParticleCoordsHost);
    Kokkos::deep_copy(boundaryParticleCoordsDevice, boundaryParticleCoordsHost);

    {
      Compadre::GMLS interiorBasis =
          Compadre::GMLS(Compadre::ScalarTaylorPolynomial,
                         Compadre::StaggeredEdgeAnalyticGradientIntegralSample,
                         polyOrder_, dimension, "LU", "STANDARD");

      interiorBasis.setProblemData(
          interiorNeighborListsDevice, sourceCoordsDevice,
          interiorParticleCoordsDevice, interiorEpsilonDevice);

      interiorBasis.addTargets(Compadre::LaplacianOfScalarPointEvaluation);

      interiorBasis.setWeightingType(Compadre::WeightingFunctionType::Power);
      interiorBasis.setWeightingParameter(4);
      interiorBasis.setOrderOfQuadraturePoints(2);
      interiorBasis.setDimensionOfQuadraturePoints(1);
      interiorBasis.setQuadratureType("LINE");

      interiorBasis.generateAlphas(1, false);

      auto solutionSet = interiorBasis.getSolutionSetHost();
      auto interiorAlpha = solutionSet->getAlphas();

      const unsigned int interiorLaplacianIndex =
          solutionSet->getAlphaColumnOffset(
              Compadre::LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);

      interiorCounter = 0;
      std::vector<PetscInt> index;
      std::vector<PetscReal> value;
      for (unsigned int i = startParticle; i < endParticle; i++) {
        const PetscInt currentParticleIndex = i;
        if (particleType(i) == 0) {
          double Aij = 0.0;
          index.resize(interiorNeighborListsHost(interiorCounter, 0));
          value.resize(interiorNeighborListsHost(interiorCounter, 0));
          for (std::size_t j = 0;
               j < interiorNeighborListsHost(interiorCounter, 0); j++) {
            const PetscInt neighborParticleIndex =
                sourceIndex(neighborLists_(i, j + 1));
            auto alphaIndex = solutionSet->getAlphaIndex(
                interiorCounter, interiorLaplacianIndex);
            index[j] = neighborParticleIndex;
            value[j] = interiorAlpha(alphaIndex + j);
            Aij -= interiorAlpha(alphaIndex + j);
          }
          value[0] = Aij;

          A.Increment(currentParticleIndex, index, value);

          interiorCounter++;
        }
      }
    }

    {
      // Dirichlet boundary condition
      for (unsigned int i = startParticle; i < endParticle; i++) {
        if (particleType(i) != 0) {
          A.Increment(i, sourceIndex(i), 1.0);
        }
      }
    }
  }

  const unsigned long nnz = A.Assemble();
  unsigned long nnzMax, nnzMin;
  MPI_Allreduce(&nnz, &nnzMax, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&nnz, &nnzMin, 1, MPI_UNSIGNED_LONG, MPI_MIN, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  if (mpiRank_ == 0) {
    printf("End of building linear system of physics: Poisson\nMax nonzeros: "
           "%lu, Min nonzeros: %lu\n",
           nnzMax, nnzMin);
    printf("Duration of building linear system is:%.4fs\n", tEnd - tStart);
  }
}

void PoissonEquation::ConstructRhs() {
  double tStart, tEnd;
  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();
  if (mpiRank_ == 0) {
    printf("Start of building right hand side\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  auto &coords = particleMgr_.GetParticleCoords();
  auto &particleType = particleMgr_.GetParticleType();

  const unsigned int localParticleNum = coords.extent(0);

  // copy old field value
  Kokkos::resize(oldField_, field_.extent(0));
  for (unsigned int i = 0; i < field_.extent(0); i++) {
    oldField_(i) = field_(i);
  }

  Kokkos::resize(field_, localParticleNum);
  Kokkos::resize(error_, localParticleNum);

  std::vector<double> rhs(localParticleNum);
  for (std::size_t i = 0; i < localParticleNum; i++) {
    if (particleType(i) != 0) {
      rhs[i] = boundaryRhs_(coords(i, 0), coords(i, 1), coords(i, 2));
    } else {
      rhs[i] = interiorRhs_(coords(i, 0), coords(i, 1), coords(i, 2));
    }
  }
  b_.Create(rhs);
  x_.Create(field_);
  tEnd = MPI_Wtime();
  if (mpiRank_ == 0) {
    printf("Duration of building right hand side:%.4fs\n", tEnd - tStart);
  }
}

void PoissonEquation::SolveEquation() {
  unsigned int currentRefinementLevel = linearSystemsPtr_.size() - 1;
  const unsigned int localParticleNum =
      particleMgr_.GetParticleCoords().extent(0);
  // interpolation previous result
  if (currentRefinementLevel == 0) {
    for (unsigned int i = 0; i < localParticleNum; i++) {
      field_(i) = 0.0;
    }
    x_.Copy(field_);
  } else {
    PetscVector y;
    y.Create(oldField_);
    MatMult(preconditionerPtr_->GetInterpolation(currentRefinementLevel)
                .GetReference(),
            y.GetReference(), x_.GetReference());
  }
  Equation::SolveEquation();
  x_.Copy(field_);
}

void PoissonEquation::CalculateError() {
  /*! \brief calculate the recovered error for each particle
   *! \date Apr 27, 2022
   *! Last modified: \date May 17, 2022
   *! \author Zisheng Ye <zisheng_ye@outlook.com>
   *
   * The recovered error for each particle is calculated based on the recovered
   * gradient and reconstructed gradient.  In order to reduce the peak usage of
   * memory, the discretization coefficients generated in the linear system
   * construction stage has been dropped.  Those coefficients have to be
   * regenerated in this stage.  For the same reason, the entire workload has
   * been split into several batches when working with GMLS discretization.  In
   * each batch, the direct gradient is evaluated for each particle and the
   * polynomial coefficients at each particle site is evaluated as well.  Once
   * all polynomial coefficients have been generated, the recovered and
   * reconstructed gradient is evaluated based these coefficients and the
   * recovered error is estimated based on the difference between the recovered
   * and reconstructed gradients.  Finally, if the analytical solution is
   * provided, the error compared to analytical solution is calculated.
   *
   * The general workflow of the function is as follows:
   * 1. Loop over all batches:
   *    a. Evaluate the direct gradient for each particle;
   *    b. Evaluate the polynomial coefficients for each particle.
   * 2. Evaluate the recovered gradient for each particle.
   * 3. Evaluate the reconstructed gradient and the recovered error for each
   *    particle.
   * 4. Evaluate the error compared to the analytical solution is provided.
   */
  Equation::CalculateError();

  auto &sourceCoords = hostGhostParticleCoords_;
  auto &coords = particleMgr_.GetParticleCoords();
  auto &spacing = particleMgr_.GetParticleSize();

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
      "source coords", sourceCoords.extent(0), sourceCoords.extent(1));
  Kokkos::deep_copy(sourceCoordsDevice, sourceCoords);

  const unsigned int localParticleNum = particleMgr_.GetLocalParticleNum();

  const unsigned int dimension = particleMgr_.GetDimension();

  HostRealVector ghostField, ghostEpsilon, ghostSpacing;
  ghost_.ApplyGhost(field_, ghostField);
  ghost_.ApplyGhost(epsilon_, ghostEpsilon);
  ghost_.ApplyGhost(spacing, ghostSpacing);

  double localDirectGradientNorm = 0.0;
  double globalDirectGradientNorm;

  unsigned int coefficientSize;
  {
    Compadre::GMLS testBasis =
        Compadre::GMLS(Compadre::ScalarTaylorPolynomial, Compadre::PointSample,
                       polyOrder_, dimension, "LU", "STANDARD");
    coefficientSize = testBasis.getPolynomialCoefficientsSize();
  }

  HostRealMatrix coefficientChunk("coefficient chunk", localParticleNum,
                                  coefficientSize);
  HostRealMatrix ghostCoefficientChunk;

  // get polynomial coefficients and direct gradients
  const unsigned int batchSize = (dimension == 2) ? 5000 : 1000;
  const unsigned int batchNum = localParticleNum / batchSize +
                                ((localParticleNum % batchSize > 0) ? 1 : 0);
  for (unsigned int batch = 0; batch < batchNum; batch++) {
    const unsigned int startParticle = batch * batchSize;
    const unsigned int endParticle =
        std::min((batch + 1) * batchSize, localParticleNum);
    const unsigned int batchParticleNum = endParticle - startParticle;

    Kokkos::View<std::size_t **, Kokkos::DefaultExecutionSpace>
        batchNeighborListsDevice("batch particle neighborlist",
                                 batchParticleNum, neighborLists_.extent(1));
    Kokkos::View<std::size_t **>::HostMirror batchNeighborListsHost =
        Kokkos::create_mirror_view(batchNeighborListsDevice);

    Kokkos::View<double *, Kokkos::DefaultExecutionSpace> batchEpsilonDevice(
        "batch particle epsilon", batchParticleNum);
    Kokkos::View<double *>::HostMirror batchEpsilonHost =
        Kokkos::create_mirror_view(batchEpsilonDevice);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        batchParticleCoordsDevice("batch particle coord", batchParticleNum, 3);
    Kokkos::View<double **>::HostMirror batchParticleCoordsHost =
        Kokkos::create_mirror_view(batchParticleCoordsDevice);

    std::size_t particleCounter = 0;
    for (std::size_t i = startParticle; i < endParticle; i++) {
      batchEpsilonHost(particleCounter) = epsilon_(i);
      for (std::size_t j = 0; j <= neighborLists_(i, 0); j++) {
        batchNeighborListsHost(particleCounter, j) = neighborLists_(i, j);
      }
      for (unsigned int j = 0; j < 3; j++) {
        batchParticleCoordsHost(particleCounter, j) = coords(i, j);
      }

      particleCounter++;
    }

    Kokkos::deep_copy(batchNeighborListsDevice, batchNeighborListsHost);
    Kokkos::deep_copy(batchEpsilonDevice, batchEpsilonHost);
    Kokkos::deep_copy(batchParticleCoordsDevice, batchParticleCoordsHost);

    {
      Compadre::GMLS batchBasis = Compadre::GMLS(
          Compadre::ScalarTaylorPolynomial, Compadre::PointSample, polyOrder_,
          dimension, "LU", "STANDARD");

      batchBasis.setProblemData(batchNeighborListsDevice, sourceCoordsDevice,
                                batchParticleCoordsDevice, batchEpsilonDevice);

      batchBasis.addTargets(Compadre::GradientOfScalarPointEvaluation);

      batchBasis.setWeightingType(Compadre::WeightingFunctionType::Power);
      batchBasis.setWeightingParameter(4);
      batchBasis.setOrderOfQuadraturePoints(2);
      batchBasis.setDimensionOfQuadraturePoints(1);
      batchBasis.setQuadratureType("LINE");

      batchBasis.generateAlphas(1, true);

      Compadre::Evaluator batchEvaluator(&batchBasis);

      auto batchCoefficient =
          batchEvaluator
              .applyFullPolynomialCoefficientsBasisToDataAllComponents<
                  double **, Kokkos::HostSpace>(ghostField);
      // duplicate coefficients
      particleCounter = 0;
      for (std::size_t i = startParticle; i < endParticle; i++) {
        for (unsigned int j = 0; j < coefficientSize; j++)
          coefficientChunk(i, j) = batchCoefficient(particleCounter, j);
        particleCounter++;
      }

      auto batchGradient =
          batchEvaluator.applyAlphasToDataAllComponentsAllTargetSites<
              double **, Kokkos::HostSpace>(
              ghostField, Compadre::GradientOfScalarPointEvaluation);

      particleCounter = 0;
      for (std::size_t i = startParticle; i < endParticle; i++) {
        double localVolume = pow(spacing(i), dimension);
        for (unsigned int j = 0; j < dimension; j++)
          localDirectGradientNorm +=
              pow(batchGradient(particleCounter, j), 2) * localVolume;
        particleCounter++;
      }
    }
  }

  MPI_Allreduce(&localDirectGradientNorm, &globalDirectGradientNorm, 1,
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  globalDirectGradientNorm = std::sqrt(globalDirectGradientNorm);

  ghost_.ApplyGhost(coefficientChunk, ghostCoefficientChunk);

  // estimate recovered gradient
  Kokkos::resize(recoveredGradientChunk_, localParticleNum, dimension);
  HostRealMatrix ghostRecoveredGradientChunk;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,
                                                             coords.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        for (unsigned int j = 0; j < dimension; j++)
          recoveredGradientChunk_(i, j) = 0.0;
        for (std::size_t j = 0; j < neighborLists_(i, 0); j++) {
          const std::size_t neighborParticleIndex = neighborLists_(i, j + 1);
          auto coeffView = Kokkos::subview(ghostCoefficientChunk,
                                           neighborParticleIndex, Kokkos::ALL);

          for (unsigned int axes = 0; axes < dimension; axes++)
            recoveredGradientChunk_(i, axes) += CalScalarGrad(
                axes, dimension,
                coords(i, 0) - sourceCoords(neighborParticleIndex, 0),
                coords(i, 1) - sourceCoords(neighborParticleIndex, 1),
                coords(i, 2) - sourceCoords(neighborParticleIndex, 2),
                polyOrder_, ghostEpsilon(neighborParticleIndex), coeffView);
        }

        for (unsigned int j = 0; j < dimension; j++)
          recoveredGradientChunk_(i, j) /= neighborLists_(i, 0);
      });
  Kokkos::fence();

  ghost_.ApplyGhost(recoveredGradientChunk_, ghostRecoveredGradientChunk);

  // estimate the reconstructed gradient and the recovered error
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,
                                                             coords.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        const double localVolume = pow(spacing(i), dimension);
        double totalNeighborVolume = 0.0;
        error_(i) = 0.0;
        double reconstructedGradient;
        for (std::size_t j = 0; j < neighborLists_(i, 0); j++) {
          const std::size_t neighborParticleIndex = neighborLists_(i, j + 1);

          double sourceVolume =
              pow(ghostSpacing(neighborParticleIndex), dimension);
          totalNeighborVolume += sourceVolume;

          auto coeffView = Kokkos::subview(coefficientChunk, i, Kokkos::ALL);

          for (unsigned int axes = 0; axes < dimension; axes++) {
            reconstructedGradient = CalScalarGrad(
                axes, dimension,
                sourceCoords(neighborParticleIndex, 0) - coords(i, 0),
                sourceCoords(neighborParticleIndex, 1) - coords(i, 1),
                sourceCoords(neighborParticleIndex, 2) - coords(i, 2),
                polyOrder_, epsilon_(i), coeffView);

            error_(i) +=
                pow(reconstructedGradient - ghostRecoveredGradientChunk(
                                                neighborParticleIndex, axes),
                    2) *
                sourceVolume;
          }
        }

        error_(i) /= totalNeighborVolume;
        error_(i) = sqrt(error_(i) * localVolume);
      });
  Kokkos::fence();
  MPI_Barrier(MPI_COMM_WORLD);

  double localError = 0.0;
  double globalError;
  Kokkos::parallel_reduce(
      "get local error",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,
                                                             coords.extent(0)),
      [&](const std::size_t i, double &tLocalError) {
        tLocalError += pow(error_(i), 2);
      },
      Kokkos::Sum<double>(localError));
  Kokkos::fence();
  MPI_Allreduce(&localError, &globalError, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  globalError_ = sqrt(globalError);
  globalNormalizedError_ = globalError_ / globalDirectGradientNorm;

  if (isFieldAnalyticalSolutionSet_) {
    double fieldAnalyticalError = 0.0;
    double fieldNorm = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
            0, field_.extent(0)),
        [&](const int i, double &tFieldAnalyticalError) {
          const double x = coords(i, 0);
          const double y = coords(i, 1);
          const double z = coords(i, 2);

          double difference = analyticalFieldSolution_(x, y, z) - field_(i);
          tFieldAnalyticalError += pow(difference, 2);
        },
        Kokkos::Sum<double>(fieldAnalyticalError));
    Kokkos::fence();

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
            0, field_.extent(0)),
        [&](const int i, double &tFieldNorm) {
          const double x = coords(i, 0);
          const double y = coords(i, 1);
          const double z = coords(i, 2);

          tFieldNorm += pow(analyticalFieldSolution_(x, y, z), 2);
        },
        Kokkos::Sum<double>(fieldNorm));
    Kokkos::fence();

    MPI_Allreduce(MPI_IN_PLACE, &fieldAnalyticalError, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &fieldNorm, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    fieldAnalyticalError = sqrt(fieldAnalyticalError / fieldNorm);
    PetscPrintf(PETSC_COMM_WORLD, "Field analytical solution error: %.6f\n",
                fieldAnalyticalError);
  }

  if (isFieldGradientAnalyticalSolutionSet_) {
    double fieldGradientAnalyticalError = 0.0;
    double fieldGradientNorm = 0.0;

    for (unsigned int axes = 0; axes < dimension; axes++) {
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
              0, field_.extent(0)),
          [&](const int i, double &tFieldGradientAnalyticalError) {
            const double x = coords(i, 0);
            const double y = coords(i, 1);
            const double z = coords(i, 2);

            double difference =
                analyticalFieldGradientSolution_(x, y, z, axes) -
                recoveredGradientChunk_(i, axes);
            tFieldGradientAnalyticalError += pow(difference, 2);
          },
          Kokkos::Sum<double>(fieldGradientAnalyticalError));
      Kokkos::fence();

      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
              0, field_.extent(0)),
          [&](const int i, double &tFieldGradientNorm) {
            const double x = coords(i, 0);
            const double y = coords(i, 1);
            const double z = coords(i, 2);

            tFieldGradientNorm +=
                pow(analyticalFieldGradientSolution_(x, y, z, axes), 2);
          },
          Kokkos::Sum<double>(fieldGradientNorm));
      Kokkos::fence();
    }

    MPI_Allreduce(MPI_IN_PLACE, &fieldGradientAnalyticalError, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &fieldGradientNorm, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    fieldGradientAnalyticalError =
        sqrt(fieldGradientAnalyticalError / fieldGradientNorm);
    PetscPrintf(PETSC_COMM_WORLD,
                "Field gradient analytical solution error: %.6f\n",
                fieldGradientAnalyticalError);
  }
}

void PoissonEquation::Output() {
  Equation::Output();

  const unsigned int dimension = particleMgr_.GetDimension();

  std::ofstream vtkStream;
  // output field value
  if (mpiRank_ == 0) {
    vtkStream.open("vtk/AdaptiveStep" + std::to_string(refinementIteration_) +
                       ".vtk",
                   std::ios::out | std::ios::app | std::ios::binary);

    vtkStream << "SCALARS f float 1" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open("vtk/AdaptiveStep" + std::to_string(refinementIteration_) +
                         ".vtk",
                     std::ios::out | std::ios::app | std::ios::binary);
      for (std::size_t i = 0; i < field_.extent(0); i++) {
        float x = field_(i);
        SwapEnd(x);
        vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));
      }
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // output recovered gradient
  if (mpiRank_ == 0) {
    vtkStream.open("vtk/AdaptiveStep" + std::to_string(refinementIteration_) +
                       ".vtk",
                   std::ios::out | std::ios::app | std::ios::binary);

    if (dimension == 2)
      vtkStream << "SCALARS grad float 2" << std::endl
                << "LOOKUP_TABLE default" << std::endl;
    if (dimension == 3)
      vtkStream << "SCALARS grad float 3" << std::endl
                << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open("vtk/AdaptiveStep" + std::to_string(refinementIteration_) +
                         ".vtk",
                     std::ios::out | std::ios::app | std::ios::binary);
      for (std::size_t i = 0; i < recoveredGradientChunk_.extent(0); i++) {
        for (std::size_t j = 0; j < recoveredGradientChunk_.extent(1); j++) {
          float x = recoveredGradientChunk_(i, j);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));
        }
      }
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // output error
  if (mpiRank_ == 0) {
    vtkStream.open("vtk/AdaptiveStep" + std::to_string(refinementIteration_) +
                       ".vtk",
                   std::ios::out | std::ios::app | std::ios::binary);

    vtkStream << "SCALARS error float 1" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open("vtk/AdaptiveStep" + std::to_string(refinementIteration_) +
                         ".vtk",
                     std::ios::out | std::ios::app | std::ios::binary);
      for (std::size_t i = 0; i < error_.extent(0); i++) {
        float x = error_(i);
        SwapEnd(x);
        vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));
      }
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

PoissonEquation::PoissonEquation()
    : Equation(), isFieldAnalyticalSolutionSet_(false),
      isFieldGradientAnalyticalSolutionSet_(false) {}

PoissonEquation::~PoissonEquation() {}

void PoissonEquation::Init() {
  Equation::Init();

  preconditionerPtr_ = std::make_shared<PoissonEquationPreconditioning>();
}

HostRealVector &PoissonEquation::GetField() { return field_; }

void PoissonEquation::SetInteriorRhs(
    const std::function<double(const double, const double, const double)>
        &func) {
  interiorRhs_ = func;
}

void PoissonEquation::SetBoundaryRhs(
    const std::function<double(const double, const double, const double)>
        &func) {
  boundaryRhs_ = func;
}

void PoissonEquation::SetAnalyticalFieldSolution(
    const std::function<double(const double, const double, const double)>
        &func) {
  analyticalFieldSolution_ = func;
  isFieldAnalyticalSolutionSet_ = true;
}

void PoissonEquation::SetAnalyticalFieldGradientSolution(
    const std::function<double(const double, const double, const double,
                               const unsigned int)> &func) {
  analyticalFieldGradientSolution_ = func;
  isFieldGradientAnalyticalSolutionSet_ = true;
}