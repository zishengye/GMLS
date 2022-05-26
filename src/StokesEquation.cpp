#include "StokesEquation.hpp"
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

  auto newMat = std::make_shared<PetscMatrix>();
  AddLinearSystem(newMat);

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
      2 * Compadre::GMLS::getNP(polyOrder_ + 1, dimension);

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

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  if (mpiRank_ == 0) {
    // printf("End of building linear system of physics: Poisson\nMax nonzeros:
    // "
    //        "%lu, Min nonzeros: %lu\n",
    //        nnzMax, nnzMin);
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

  tEnd = MPI_Wtime();
  if (mpiRank_ == 0) {
    printf("Duration of building right hand side:%.4fs\n", tEnd - tStart);
  }
}

void StokesEquation::SolveEquation() { Equation::SolveEquation(); }

void StokesEquation::CalculateError() {}

void StokesEquation::Output() {}

StokesEquation::StokesEquation() : Equation() {}

StokesEquation::~StokesEquation() {}

void StokesEquation::Init() { Equation::Init(); }

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

void StokesEquation::SetPressureBoundaryRhs(
    const std::function<double(const double, const double, const double)>
        &func) {
  boundaryPressureRhs_ = func;
}