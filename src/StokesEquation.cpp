#include "StokesEquation.hpp"
#include "PetscBlockMatrix.hpp"
#include "PolyBasis.hpp"

#include <Compadre_Evaluator.hpp>
#include <Compadre_GMLS.hpp>
#include <Compadre_PointCloudSearch.hpp>

void StokesEquation::InitLinearSystem() {
  double tStart, tEnd;
  tStart = MPI_Wtime();
  Equation::InitLinearSystem();
  if (mpiRank_ == 0)
    printf("Start of initializing physics: Stokes\n");

  auto newMat = std::make_shared<PetscBlockMatrix>();
  newMat->Resize(2, 2);

  newMat->SetSubMat(0, 0,
                    std::static_pointer_cast<PetscMatrixBase>(
                        std::make_shared<PetscMatrix>()));
  newMat->SetSubMat(0, 1,
                    std::static_pointer_cast<PetscMatrixBase>(
                        std::make_shared<PetscMatrix>()));
  newMat->SetSubMat(1, 0,
                    std::static_pointer_cast<PetscMatrixBase>(
                        std::make_shared<PetscMatrix>()));
  newMat->SetSubMat(1, 1,
                    std::static_pointer_cast<PetscMatrixBase>(
                        std::make_shared<PetscMatrix>()));

  AddLinearSystem(std::static_pointer_cast<PetscMatrixBase>(newMat));

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
      pow(sqrt(2), dimension) *
      std::max(Compadre::GMLS::getNP(polyOrder_, dimension,
                                     Compadre::ReconstructionSpace::
                                         DivergenceFreeVectorTaylorPolynomial),
               Compadre::GMLS::getNP(polyOrder_ + 1, dimension));

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
        interiorNeighborListsDevice("interior particle neighbor lists",
                                    interiorParticleNum,
                                    neighborLists_.extent(1));
    Kokkos::View<std::size_t **>::HostMirror interiorNeighborListsHost =
        Kokkos::create_mirror_view(interiorNeighborListsDevice);
    Kokkos::View<std::size_t **, Kokkos::DefaultExecutionSpace>
        boundaryNeighborListsDevice("boundary particle neighbor lists",
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
      Compadre::GMLS interiorVelocityBasis =
          Compadre::GMLS(Compadre::ScalarTaylorPolynomial,
                         Compadre::StaggeredEdgeAnalyticGradientIntegralSample,
                         polyOrder_, dimension, "LU", "STANDARD");

      interiorVelocityBasis.setProblemData(
          interiorNeighborListsDevice, sourceCoordsDevice,
          interiorParticleCoordsDevice, interiorEpsilonDevice);

      interiorVelocityBasis.addTargets(
          Compadre::LaplacianOfScalarPointEvaluation);

      interiorVelocityBasis.setWeightingType(
          Compadre::WeightingFunctionType::Power);
      interiorVelocityBasis.setWeightingParameter(4);
      interiorVelocityBasis.setOrderOfQuadraturePoints(2);
      interiorVelocityBasis.setDimensionOfQuadraturePoints(dimension - 1);
      interiorVelocityBasis.setQuadratureType("LINE");

      interiorVelocityBasis.generateAlphas(1, false);

      auto solutionSet = interiorVelocityBasis.getSolutionSetHost();
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

  auto &uu = *(std::static_pointer_cast<PetscMatrix>(
      std::static_pointer_cast<PetscBlockMatrix>(
          linearSystemsPtr_[refinementIteration_])
          ->GetSubMat(0, 0)));

  uu.Resize(localParticleNum, localParticleNum, dimension);

  auto &up = *(std::static_pointer_cast<PetscMatrix>(
      std::static_pointer_cast<PetscBlockMatrix>(
          linearSystemsPtr_[refinementIteration_])
          ->GetSubMat(0, 1)));

  up.Resize(dimension * localParticleNum, localParticleNum);

  auto &pu = *(std::static_pointer_cast<PetscMatrix>(
      std::static_pointer_cast<PetscBlockMatrix>(
          linearSystemsPtr_[refinementIteration_])
          ->GetSubMat(1, 0)));

  pu.Resize(localParticleNum, localParticleNum * dimension);

  auto &pp = *(std::static_pointer_cast<PetscMatrix>(
      std::static_pointer_cast<PetscBlockMatrix>(
          linearSystemsPtr_[refinementIteration_])
          ->GetSubMat(1, 1)));
  pp.Resize(localParticleNum, localParticleNum);

  std::vector<PetscInt> index;
  for (std::size_t i = 0; i < localParticleNum; i++) {
    const PetscInt currentParticleIndex = i;
    if (particleType(i) == 0) {
      index.resize(neighborLists_(i, 0));
      for (std::size_t j = 0; j < neighborLists_(i, 0); j++) {
        const PetscInt neighborParticleIndex =
            sourceIndex(neighborLists_(i, j + 1));
        index[j] = neighborParticleIndex;
      }
      std::sort(index.begin(), index.end());
      uu.SetColIndex(currentParticleIndex, index);
      pp.SetColIndex(currentParticleIndex, index);
      for (unsigned int j = 0; j < dimension; j++) {
        up.SetColIndex(currentParticleIndex * dimension + j, index);
      }
    } else {
      index.resize(1);
      index[0] = sourceIndex(i);
      uu.SetColIndex(currentParticleIndex, index);

      index.resize(neighborLists_(i, 0));
      for (std::size_t j = 0; j < neighborLists_(i, 0); j++) {
        const PetscInt neighborParticleIndex =
            sourceIndex(neighborLists_(i, j + 1));
        index[j] = neighborParticleIndex;
      }
      std::sort(index.begin(), index.end());
      pp.SetColIndex(currentParticleIndex, index);

      index.resize(dimension * neighborLists_(i, 0));
      for (std::size_t j = 0; j < neighborLists_(i, 0); j++) {
        const PetscInt neighborParticleIndex =
            sourceIndex(neighborLists_(i, j + 1));
        for (unsigned int k = 0; k < dimension; k++) {
          index[j * dimension + k] = neighborParticleIndex * dimension + k;
        }
      }
      std::sort(index.begin(), index.end());
      pu.SetColIndex(currentParticleIndex, index);
    }
  }

  uu.GraphAssemble();
  up.GraphAssemble();
  pu.GraphAssemble();
  pp.GraphAssemble();

  tEnd = MPI_Wtime();
  if (mpiRank_ == 0) {
    printf("\nAfter satisfying conditioning of Laplacian operator\niteration"
           "count: %d min neighbor: %d, max neighbor: %d , mean "
           "neighbor: %.2f, max ratio: %.2f\n",
           iteCounter, minNeighbor, maxNeighbor,
           meanNeighbor / (double)globalParticleNum, maxRatio);
    printf("Duration of initializing linear system:%.4fs\n", tEnd - tStart);
  }
}

void StokesEquation::ConstructLinearSystem() {
  double tStart, tEnd;
  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();
  if (mpiRank_ == 0)
    printf("Start of building linear system of physics: Stokes\n");

  auto &sourceCoords = hostGhostParticleCoords_;
  auto &sourceIndex = hostGhostParticleIndex_;
  auto &coords = particleMgr_.GetParticleCoords();
  auto &particleType = particleMgr_.GetParticleType();
  auto &normal = particleMgr_.GetParticleNormal();

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
      "source coords", sourceCoords.extent(0), sourceCoords.extent(1));
  Kokkos::deep_copy(sourceCoordsDevice, sourceCoords);

  const unsigned int localParticleNum = coords.extent(0);

  const unsigned int dimension = particleMgr_.GetDimension();

  auto &A = *(std::static_pointer_cast<PetscBlockMatrix>(
      linearSystemsPtr_[refinementIteration_]));

  Kokkos::resize(bi_, localParticleNum);

  auto &uu = *(std::static_pointer_cast<PetscMatrix>(A.GetSubMat(0, 0)));
  auto &up = *(std::static_pointer_cast<PetscMatrix>(A.GetSubMat(0, 1)));
  auto &pu = *(std::static_pointer_cast<PetscMatrix>(A.GetSubMat(1, 0)));
  auto &pp = *(std::static_pointer_cast<PetscMatrix>(A.GetSubMat(1, 1)));

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
        interiorNeighborListsDevice("interior particle neighbor list",
                                    interiorParticleNum,
                                    neighborLists_.extent(1));
    Kokkos::View<std::size_t **>::HostMirror interiorNeighborListsHost =
        Kokkos::create_mirror_view(interiorNeighborListsDevice);
    Kokkos::View<std::size_t **, Kokkos::DefaultExecutionSpace>
        boundaryNeighborListsDevice("boundary particle neighbor list",
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

    Kokkos::View<double ***, Kokkos::DefaultExecutionSpace> tangentBundleDevice(
        "tangent bundles", boundaryParticleNum, dimension, dimension);
    Kokkos::View<double ***>::HostMirror tangentBundleHost =
        Kokkos::create_mirror_view(tangentBundleDevice);

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

    Kokkos::deep_copy(interiorNeighborListsDevice, interiorNeighborListsHost);
    Kokkos::deep_copy(boundaryNeighborListsDevice, boundaryNeighborListsHost);
    Kokkos::deep_copy(interiorEpsilonDevice, interiorEpsilonHost);
    Kokkos::deep_copy(boundaryEpsilonDevice, boundaryEpsilonHost);
    Kokkos::deep_copy(interiorParticleCoordsDevice, interiorParticleCoordsHost);
    Kokkos::deep_copy(boundaryParticleCoordsDevice, boundaryParticleCoordsHost);

    // velocity-velocity block
    {
      Compadre::GMLS interiorVelocityBasis = Compadre::GMLS(
          Compadre::DivergenceFreeVectorTaylorPolynomial,
          Compadre::VectorPointSample, polyOrder_, dimension, "LU", "STANDARD");

      interiorVelocityBasis.setProblemData(
          interiorNeighborListsDevice, sourceCoordsDevice,
          interiorParticleCoordsDevice, interiorEpsilonDevice);

      interiorVelocityBasis.addTargets(
          Compadre::CurlCurlOfVectorPointEvaluation);

      interiorVelocityBasis.setWeightingType(
          Compadre::WeightingFunctionType::Power);
      interiorVelocityBasis.setWeightingParameter(4);
      interiorVelocityBasis.setOrderOfQuadraturePoints(2);
      interiorVelocityBasis.setDimensionOfQuadraturePoints(1);
      interiorVelocityBasis.setQuadratureType("LINE");

      interiorVelocityBasis.generateAlphas(1, false);

      auto velocitySolutionSet = interiorVelocityBasis.getSolutionSetHost();
      auto interiorVelocityAlpha = velocitySolutionSet->getAlphas();

      std::vector<unsigned int> interiorCurlCurlIndex(pow(dimension, 2));
      for (unsigned int i = 0; i < dimension; i++) {
        for (unsigned int j = 0; j < dimension; j++) {
          interiorCurlCurlIndex[i * dimension + j] =
              velocitySolutionSet->getAlphaColumnOffset(
                  Compadre::CurlCurlOfVectorPointEvaluation, i, 0, j, 0);
        }
      }

      unsigned int blockStorageSize = dimension * dimension;

      interiorCounter = 0;
      std::vector<PetscInt> index;
      std::vector<PetscReal> value;
      for (unsigned int i = startParticle; i < endParticle; i++) {
        if (particleType(i) == 0) {
          unsigned int numNeighbor = neighborLists_(i, 0);
          unsigned int singleRowSize = numNeighbor * dimension;
          index.resize(numNeighbor);
          value.resize(numNeighbor * blockStorageSize);
          for (std::size_t j = 0; j < numNeighbor; j++) {
            const PetscInt neighborParticleIndex =
                sourceIndex(neighborLists_(i, j + 1));
            index[j] = neighborParticleIndex;
            for (unsigned int axes1 = 0; axes1 < dimension; axes1++) {
              for (unsigned int axes2 = 0; axes2 < dimension; axes2++) {
                auto alphaIndex = velocitySolutionSet->getAlphaIndex(
                    interiorCounter,
                    interiorCurlCurlIndex[axes1 * dimension + axes2]);
                value[axes1 * singleRowSize + j * dimension + axes2] =
                    interiorVelocityAlpha(alphaIndex + j);
              }
            }
          }

          uu.Increment(i, index, value);

          interiorCounter++;
        } else {
          index.resize(1);
          index[0] = sourceIndex(i);
          value.resize(blockStorageSize);
          for (unsigned int j = 0; j < blockStorageSize; j++) {
            value[j] = 0.0;
          }
          for (unsigned int j = 0; j < dimension; j++) {
            value[j * dimension + j] = 1.0;
          }
          uu.Increment(i, index, value);
        }
      }
    }

    // pressure-pressure block and velocity-pressure block for interior
    // particles
    {
      Compadre::GMLS interiorPressureBasis =
          Compadre::GMLS(Compadre::ScalarTaylorPolynomial,
                         Compadre::StaggeredEdgeAnalyticGradientIntegralSample,
                         polyOrder_, dimension, "LU", "STANDARD");

      interiorPressureBasis.setProblemData(
          interiorNeighborListsDevice, sourceCoordsDevice,
          interiorParticleCoordsDevice, interiorEpsilonDevice);

      std::vector<Compadre::TargetOperation> interiorPressureOptions(2);
      interiorPressureOptions[0] = Compadre::LaplacianOfScalarPointEvaluation;
      interiorPressureOptions[1] = Compadre::GradientOfScalarPointEvaluation;
      interiorPressureBasis.addTargets(interiorPressureOptions);

      interiorPressureBasis.setWeightingType(
          Compadre::WeightingFunctionType::Power);
      interiorPressureBasis.setWeightingParameter(4);
      interiorPressureBasis.setOrderOfQuadraturePoints(2);
      interiorPressureBasis.setDimensionOfQuadraturePoints(1);
      interiorPressureBasis.setQuadratureType("LINE");

      interiorPressureBasis.generateAlphas(1, false);

      auto pressureSolutionSet = interiorPressureBasis.getSolutionSetHost();
      auto interiorPressureAlpha = pressureSolutionSet->getAlphas();

      const unsigned int interiorPressureLaplacianIndex =
          pressureSolutionSet->getAlphaColumnOffset(
              Compadre::LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);
      std::vector<unsigned int> interiorPressureGradientIndex(dimension);
      for (unsigned int i = 0; i < dimension; i++) {
        interiorPressureGradientIndex[i] =
            pressureSolutionSet->getAlphaColumnOffset(
                Compadre::GradientOfScalarPointEvaluation, i, 0, 0, 0);
      }

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
            auto alphaIndex = pressureSolutionSet->getAlphaIndex(
                interiorCounter, interiorPressureLaplacianIndex);
            index[j] = neighborParticleIndex;
            value[j] = interiorPressureAlpha(alphaIndex + j);
            Aij -= interiorPressureAlpha(alphaIndex + j);
          }
          value[0] = Aij;

          pp.Increment(currentParticleIndex, index, value);

          for (unsigned int k = 0; k < dimension; k++) {
            Aij = 0.0;
            for (std::size_t j = 0;
                 j < interiorNeighborListsHost(interiorCounter, 0); j++) {
              auto alphaIndex = pressureSolutionSet->getAlphaIndex(
                  interiorCounter, interiorPressureGradientIndex[k]);
              value[j] = -interiorPressureAlpha(alphaIndex + j);
              Aij += interiorPressureAlpha(alphaIndex + j);
            }
            value[0] = Aij;
            up.Increment(currentParticleIndex * dimension + k, index, value);
          }

          interiorCounter++;
        }
      }
    }

    // pressure-pressure block and pressure-velocity block for boundary
    // particles
    {
      Compadre::GMLS boundaryPressureBasis = Compadre::GMLS(
          Compadre::ScalarTaylorPolynomial,
          Compadre::StaggeredEdgeAnalyticGradientIntegralSample, polyOrder_,
          dimension, "LU", "STANDARD", "NEUMANN_GRAD_SCALAR");

      boundaryPressureBasis.setProblemData(
          boundaryNeighborListsDevice, sourceCoordsDevice,
          boundaryParticleCoordsDevice, boundaryEpsilonDevice);

      boundaryPressureBasis.setTangentBundle(tangentBundleDevice);

      std::vector<Compadre::TargetOperation> boundaryPressureOptions(2);
      boundaryPressureOptions[0] = Compadre::LaplacianOfScalarPointEvaluation;
      boundaryPressureOptions[1] = Compadre::GradientOfScalarPointEvaluation;
      boundaryPressureBasis.addTargets(boundaryPressureOptions);

      boundaryPressureBasis.setWeightingType(
          Compadre::WeightingFunctionType::Power);
      boundaryPressureBasis.setWeightingParameter(4);
      boundaryPressureBasis.setOrderOfQuadraturePoints(2);
      boundaryPressureBasis.setDimensionOfQuadraturePoints(dimension - 1);
      boundaryPressureBasis.setQuadratureType("LINE");

      boundaryPressureBasis.generateAlphas(1, false);

      auto pressureSolutionSet = boundaryPressureBasis.getSolutionSetHost();
      auto boundaryPressureAlpha = pressureSolutionSet->getAlphas();

      const unsigned int boundaryPressureLaplacianIndex =
          pressureSolutionSet->getAlphaColumnOffset(
              Compadre::LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);
      std::vector<unsigned int> boundaryPressureGradientIndex(dimension);
      for (unsigned int i = 0; i < dimension; i++) {
        boundaryPressureGradientIndex[i] =
            pressureSolutionSet->getAlphaColumnOffset(
                Compadre::GradientOfScalarPointEvaluation, i, 0, 0, 0);
      }

      Compadre::GMLS boundaryVelocityBasis = Compadre::GMLS(
          Compadre::DivergenceFreeVectorTaylorPolynomial,
          Compadre::VectorPointSample, polyOrder_, dimension, "LU", "STANDARD");

      boundaryVelocityBasis.setProblemData(
          boundaryNeighborListsDevice, sourceCoordsDevice,
          boundaryParticleCoordsDevice, boundaryEpsilonDevice);

      boundaryVelocityBasis.addTargets(
          Compadre::CurlCurlOfVectorPointEvaluation);

      boundaryVelocityBasis.setWeightingType(
          Compadre::WeightingFunctionType::Power);
      boundaryVelocityBasis.setWeightingParameter(4);
      boundaryVelocityBasis.setOrderOfQuadraturePoints(2);
      boundaryVelocityBasis.setDimensionOfQuadraturePoints(1);
      boundaryVelocityBasis.setQuadratureType("LINE");

      boundaryVelocityBasis.generateAlphas(1, false);

      auto velocitySolutionSet = boundaryVelocityBasis.getSolutionSetHost();
      auto boundaryVelocityAlpha = velocitySolutionSet->getAlphas();

      std::vector<unsigned int> boundaryCurlCurlIndex(pow(dimension, 2));
      for (unsigned int i = 0; i < dimension; i++) {
        for (unsigned int j = 0; j < dimension; j++) {
          boundaryCurlCurlIndex[i * dimension + j] =
              velocitySolutionSet->getAlphaColumnOffset(
                  Compadre::CurlCurlOfVectorPointEvaluation, i, 0, j, 0);
        }
      }

      boundaryCounter = 0;
      std::vector<PetscInt> index;
      std::vector<PetscReal> value;
      for (unsigned int i = startParticle; i < endParticle; i++) {
        const PetscInt currentParticleIndex = i;
        if (particleType(i) != 0) {
          double Aij = 0.0;
          const unsigned int numNeighbor = neighborLists_(i, 0);

          index.resize(numNeighbor);
          value.resize(numNeighbor);

          bi_(i) = pressureSolutionSet->getAlpha0TensorTo0Tensor(
              Compadre::LaplacianOfScalarPointEvaluation, boundaryCounter,
              numNeighbor);

          for (std::size_t j = 0; j < numNeighbor; j++) {
            const PetscInt neighborParticleIndex =
                sourceIndex(neighborLists_(i, j + 1));
            auto alphaIndex = pressureSolutionSet->getAlphaIndex(
                boundaryCounter, boundaryPressureLaplacianIndex);
            index[j] = neighborParticleIndex;
            value[j] = boundaryPressureAlpha(alphaIndex + j);
            Aij -= boundaryPressureAlpha(alphaIndex + j);
          }
          value[0] = Aij;

          pp.Increment(currentParticleIndex, index, value);

          index.resize(dimension * numNeighbor);
          value.resize(dimension * numNeighbor);

          for (std::size_t j = 0; j < numNeighbor; j++) {
            const PetscInt neighborParticleIndex =
                sourceIndex(neighborLists_(i, j + 1));
            for (unsigned int axes2 = 0; axes2 < dimension; axes2++) {
              index[j * dimension + axes2] =
                  dimension * neighborParticleIndex + axes2;
              value[j * dimension + axes2] = 0.0;
              for (unsigned int axes1 = 0; axes1 < dimension; axes1++) {
                auto alphaIndex = velocitySolutionSet->getAlphaIndex(
                    boundaryCounter,
                    boundaryCurlCurlIndex[dimension * axes1 + axes2]);
                value[j * dimension + axes2] +=
                    normal(i, axes1) * boundaryVelocityAlpha(alphaIndex + j);
              }
              value[j * dimension + axes2] *= bi_(i);
            }
          }

          pu.Increment(currentParticleIndex, index, value);

          boundaryCounter++;
        }
      }
    }
  }

  unsigned long nnz = A.Assemble();
  unsigned long nnzMax, nnzMin;
  MPI_Allreduce(&nnz, &nnzMax, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&nnz, &nnzMin, 1, MPI_UNSIGNED_LONG, MPI_MIN, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  if (mpiRank_ == 0) {
    printf("End of building linear system of physics: Stokes\nMax nonzeros: "
           "%lu, Min nonzeros: %lu\n",
           nnzMax, nnzMin);
    printf("Duration of building linear system is:%.4fs\n", tEnd - tStart);
  }
}

void StokesEquation::ConstructRhs() {
  double tStart, tEnd;
  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();
  if (mpiRank_ == 0) {
    printf("Start of building right hand side\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  auto &coords = particleMgr_.GetParticleCoords();
  auto &particleType = particleMgr_.GetParticleType();
  auto &normal = particleMgr_.GetParticleNormal();

  const unsigned int dimension = particleMgr_.GetDimension();
  const unsigned int localParticleNum = coords.extent(0);
  const unsigned int velocityDof = dimension;
  const unsigned int fieldDof = dimension + 1;

  // copy old field value
  Kokkos::resize(oldVelocity_, velocity_.extent(0), dimension);
  Kokkos::resize(oldPressure_, pressure_.extent(0));
  for (unsigned int i = 0; i < pressure_.extent(0); i++)
    oldPressure_(i) = pressure_(i);

  for (unsigned int i = 0; i < velocity_.extent(0); i++)
    for (unsigned int j = 0; j < dimension; j++)
      oldVelocity_(i, j) = velocity_(i, j);

  Kokkos::resize(velocity_, localParticleNum, dimension);
  Kokkos::resize(pressure_, localParticleNum);
  Kokkos::resize(error_, localParticleNum);

  std::vector<double> rhs(localParticleNum * fieldDof);
  for (std::size_t i = 0; i < localParticleNum; i++) {
    if (particleType(i) != 0) {
      rhs[localParticleNum * velocityDof + i] =
          interiorPressureRhs_(coords(i, 0), coords(i, 1), coords(i, 2));
      for (unsigned int j = 0; j < dimension; j++) {
        rhs[i * velocityDof + j] =
            boundaryVelocityRhs_(coords(i, 0), coords(i, 1), coords(i, 2), j);
        rhs[localParticleNum * velocityDof + i] +=
            bi_(i) * normal(i, j) *
            interiorVelocityRhs_(coords(i, 0), coords(i, 1), coords(i, 2), j);
      }
    } else {
      rhs[localParticleNum * velocityDof + i] =
          interiorPressureRhs_(coords(i, 0), coords(i, 1), coords(i, 2));
      for (unsigned int j = 0; j < dimension; j++)
        rhs[i * velocityDof + j] =
            interiorVelocityRhs_(coords(i, 0), coords(i, 1), coords(i, 2), j);
    }
  }

  // ensure rhs on pressure has zero sum
  double sum = 0.0;
  for (std::size_t i = 0; i < localParticleNum; i++) {
    sum += rhs[localParticleNum * velocityDof + i];
  }
  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double average = sum / particleMgr_.GetGlobalParticleNum();
  for (std::size_t i = 0; i < localParticleNum; i++) {
    rhs[localParticleNum * velocityDof + i] -= average;
  }

  b_.Create(rhs);
  x_.Create(rhs);

  tEnd = MPI_Wtime();
  if (mpiRank_ == 0) {
    printf("Duration of building right hand side:%.4fs\n", tEnd - tStart);
  }
}

void StokesEquation::SolveEquation() {
  unsigned int currentRefinementLevel = linearSystemsPtr_.size() - 1;
  // interpolation previous result
  if (currentRefinementLevel == 0) {
    VecSet(x_.GetReference(), 0.0);
  } else {
  }
  Equation::SolveEquation();

  // copy data
  const unsigned int dimension = particleMgr_.GetDimension();
  const unsigned int localParticleNum =
      particleMgr_.GetParticleCoords().extent(0);
  const unsigned int velocityDof = dimension;

  PetscScalar *a;
  VecGetArray(x_.GetReference(), &a);
  for (unsigned int i = 0; i < localParticleNum; i++) {
    for (unsigned int j = 0; j < dimension; j++) {
      velocity_(i, j) = a[i * velocityDof + j];
    }
    pressure_(i) = a[localParticleNum * velocityDof + i];
  }
  VecRestoreArray(x_.GetReference(), &a);
}

void StokesEquation::CalculateError() { Equation::CalculateError(); }

void StokesEquation::Output() {
  Equation::Output();

  if (outputLevel_ == 0)
    return;

  const unsigned int dimension = particleMgr_.GetDimension();

  std::ofstream vtkStream;
  // output velocity value
  if (mpiRank_ == 0) {
    vtkStream.open("vtk/AdaptiveStep" + std::to_string(refinementIteration_) +
                       ".vtk",
                   std::ios::out | std::ios::app | std::ios::binary);

    if (dimension == 2)
      vtkStream << "SCALARS u float 2" << std::endl
                << "LOOKUP_TABLE default" << std::endl;

    if (dimension == 3)
      vtkStream << "SCALARS u float 3" << std::endl
                << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open("vtk/AdaptiveStep" + std::to_string(refinementIteration_) +
                         ".vtk",
                     std::ios::out | std::ios::app | std::ios::binary);
      for (std::size_t i = 0; i < velocity_.extent(0); i++) {
        for (std::size_t j = 0; j < velocity_.extent(1); j++) {
          float x = velocity_(i, j);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));
        }
      }
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // output pressure value
  if (mpiRank_ == 0) {
    vtkStream.open("vtk/AdaptiveStep" + std::to_string(refinementIteration_) +
                       ".vtk",
                   std::ios::out | std::ios::app | std::ios::binary);

    vtkStream << "SCALARS p float 1" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open("vtk/AdaptiveStep" + std::to_string(refinementIteration_) +
                         ".vtk",
                     std::ios::out | std::ios::app | std::ios::binary);
      for (std::size_t i = 0; i < pressure_.extent(0); i++) {
        float x = pressure_(i);
        SwapEnd(x);
        vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));
      }
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

StokesEquation::StokesEquation() : Equation() {}

StokesEquation::~StokesEquation() {}

void StokesEquation::Init() {
  Equation::Init();

  preconditionerPtr_ = std::make_shared<StokesEquationPreconditioning>();
}

HostRealMatrix &StokesEquation::GetVelocity() { return velocity_; }

HostRealVector &StokesEquation::GetPressure() { return pressure_; }

void StokesEquation::SetVelocityInteriorRhs(
    const std::function<double(const double, const double, const double,
                               const unsigned int)> &func) {
  interiorVelocityRhs_ = func;
}

void StokesEquation::SetVelocityBoundaryRhs(
    const std::function<double(const double, const double, const double,
                               const unsigned int)> &func) {
  boundaryVelocityRhs_ = func;
}

void StokesEquation::SetPressureInteriorRhs(
    const std::function<double(const double, const double, const double)>
        &func) {
  interiorPressureRhs_ = func;
}