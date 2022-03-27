#include "PoissonEquation.hpp"

#include <Compadre_GMLS.hpp>
#include <Compadre_PointCloudSearch.hpp>

void PoissonEquation::InitLinearSystem() {
  double tStart, tEnd;
  tStart = MPI_Wtime();
  Equation::InitLinearSystem();
  if (mpiRank_ == 0)
    printf("Start of initializing physics: Poisson\n");

  auto &sourceCoords = particleMgr_.GetGhostParticleCoords();
  auto &sourceIndex = particleMgr_.GetGhostParticleIndex();
  auto &coords = particleMgr_.GetParticleCoords();
  auto &spacing = particleMgr_.GetParticleSize();
  auto &particleType = particleMgr_.GetParticleType();
  auto &normal = particleMgr_.GetParticleNormal();

  const unsigned int localParticleNum = coords.extent(0);
  const unsigned long globalParticleNum = particleMgr_.GetGlobalParticleNum();

  const int dimension = particleMgr_.GetDimension();

  unsigned int interiorParticleNum, boundaryParticleNum;
  for (int i = 0; i < localParticleNum; i++) {
    if (particleType(i) == 0)
      interiorParticleNum++;
    else
      boundaryParticleNum++;
  }

  const int satisfiedNumNeighbor =
      2 * Compadre::GMLS::getNP(polyOrder_, dimension);

  Equation::ConstructNeighborLists(satisfiedNumNeighbor);

  std::vector<bool> discretizationCheck(localParticleNum);
  for (int i = 0; i < localParticleNum; i++) {
    discretizationCheck[i] = false;
  }

  auto pointCloudSearch(
      Compadre::CreatePointCloudSearch(sourceCoords, dimension));
  bool isNeighborSearchPassed = false;

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
      "source coords", sourceCoords.extent(0), sourceCoords.extent(1));
  Kokkos::deep_copy(sourceCoordsDevice, sourceCoords);

  double maxRatio, meanNeighbor;
  int minNeighbor, maxNeighbor, iteCounter;
  iteCounter = 0;
  while (!isNeighborSearchPassed) {
    iteCounter++;

    Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
        interiorNeighborListsDevice("interior particle neighborlist",
                                    interiorParticleNum,
                                    neighborLists_.extent(1));
    Kokkos::View<int **>::HostMirror interiorNeighborListsHost =
        Kokkos::create_mirror_view(interiorNeighborListsDevice);
    Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
        boundaryNeighborListsDevice("boundary particle neighborlist",
                                    boundaryParticleNum,
                                    neighborLists_.extent(1));
    Kokkos::View<int **>::HostMirror boundaryNeighborListsHost =
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

    int interiorCounter = 0;
    int boundaryCounter = 0;
    for (int i = 0; i < localParticleNum; i++) {
      if (!discretizationCheck[i]) {
        if (particleType(i) == 0) {
          interiorEpsilonHost(interiorCounter) = epsilon_(i);
          for (int j = 0; j <= neighborLists_(i, 0); j++) {
            interiorNeighborListsHost(interiorCounter, j) =
                neighborLists_(i, j);
          }
          for (int j = 0; j < dimension; j++) {
            interiorParticleCoordsHost(interiorCounter, j) = coords(i, j);
          }

          interiorCounter++;
        } else {
          boundaryEpsilonHost(boundaryCounter) = epsilon_(i);
          for (int j = 0; j <= neighborLists_(i, 0); j++) {
            boundaryNeighborListsHost(boundaryCounter, j) =
                neighborLists_(i, j);
          }
          for (int j = 0; j < dimension; j++) {
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
      interiorBasis.setDimensionOfQuadraturePoints(1);
      interiorBasis.setQuadratureType("LINE");

      interiorBasis.generateAlphas(1, false);

      auto solutionSet = interiorBasis.getSolutionSetHost();
      auto interiorAlpha = solutionSet->getAlphas();

      const int interiorLaplacianIndex = solutionSet->getAlphaColumnOffset(
          Compadre::LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);

      interiorCounter = 0;
      for (int i = 0; i < localParticleNum; i++) {
        if (particleType(i) == 0 && discretizationCheck[i] == false) {
          double Aij = 0.0;
          for (int j = 0; j < interiorNeighborListsHost(interiorCounter, 0);
               j++) {
            auto alphaIndex = solutionSet->getAlphaIndex(
                interiorCounter, interiorLaplacianIndex);
            Aij -= interiorAlpha(alphaIndex + j);
          }
          if (Aij > 0.0)
            discretizationCheck[i] = true;

          interiorCounter++;
        }
      }
    }

    interiorParticleNum = 0;
    for (int i = 0; i < localParticleNum; i++) {
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
    for (int i = 0; i < localParticleNum; i++) {
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
      boundaryBasis.setDimensionOfQuadraturePoints(1);
      boundaryBasis.setQuadratureType("LINE");

      boundaryBasis.generateAlphas(1, false);

      auto solutionSet = boundaryBasis.getSolutionSetHost();
      auto boundaryAlpha = solutionSet->getAlphas();

      const int boundaryLaplacianIndex = solutionSet->getAlphaColumnOffset(
          Compadre::LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);

      boundaryCounter = 0;
      for (int i = 0; i < localParticleNum; i++) {
        if (particleType(i) != 0 && discretizationCheck[i] == false) {
          double Aij = 0.0;
          for (int j = 0; j < boundaryNeighborListsHost(boundaryCounter, 0);
               j++) {
            auto alphaIndex = solutionSet->getAlphaIndex(
                boundaryCounter, boundaryLaplacianIndex);
            Aij -= boundaryAlpha(alphaIndex + j);
          }
          if (Aij > 0.0)
            discretizationCheck[i] = true;

          boundaryCounter++;
        }
      }
    }

    boundaryParticleNum = 0;
    for (int i = 0; i < localParticleNum; i++) {
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
      int minNeighborLists =
          1 + pointCloudSearch.generate2DNeighborListsFromRadiusSearch(
                  true, coords, neighborLists_, epsilon_, 0.0, 0.0);
      if (minNeighborLists > neighborLists_.extent(1))
        Kokkos::resize(neighborLists_, localParticleNum, minNeighborLists);
      pointCloudSearch.generate2DNeighborListsFromRadiusSearch(
          false, coords, neighborLists_, epsilon_, 0.0, 0.0);
    }
  }

  maxRatio = 0.0;
  minNeighbor = 1000;
  maxNeighbor = 0;
  meanNeighbor = 0;
  for (int i = 0; i < localParticleNum; i++) {
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
  MPI_Allreduce(MPI_IN_PLACE, &minNeighbor, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &maxNeighbor, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &meanNeighbor, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &maxRatio, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  auto &A = linearSystems_[refinementIteration_];
  A.Resize(localParticleNum, localParticleNum);
  std::vector<PetscInt> index;
  for (int i = 0; i < localParticleNum; i++) {
    const PetscInt currentParticleIndex = i;
    if (particleType[i] == 0) {
      index.resize(neighborLists_(i, 0));
      for (int j = 0; j < neighborLists_(i, 0); j++) {
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
    printf("Duration of initializing linear system:%fs\n", tEnd - tStart);
  }
}

void PoissonEquation::ConstructLinearSystem() {
  double tStart, tEnd;
  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();
  if (mpiRank_ == 0)
    printf("Start of building linear system of physics: Poisson\n");

  auto &sourceCoords = particleMgr_.GetGhostParticleCoords();
  auto &sourceIndex = particleMgr_.GetGhostParticleIndex();
  auto &coords = particleMgr_.GetParticleCoords();
  auto &spacing = particleMgr_.GetParticleSize();
  auto &particleType = particleMgr_.GetParticleType();
  auto &normal = particleMgr_.GetParticleNormal();

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
      "source coords", sourceCoords.extent(0), sourceCoords.extent(1));
  Kokkos::deep_copy(sourceCoordsDevice, sourceCoords);

  const unsigned int localParticleNum = coords.extent(0);
  const unsigned long globalParticleNum = particleMgr_.GetGlobalParticleNum();

  const int dimension = particleMgr_.GetDimension();

  auto &A = linearSystems_[refinementIteration_];

  const unsigned int batchSize = (dimension == 2) ? 5000 : 1000;
  const unsigned int batchNum =
      localParticleNum / batchSize + (localParticleNum % batchSize > 0) ? 1 : 0;
  for (int batch = 0; batch < batchNum; batch++) {
    const unsigned int startParticle = batch * batchSize;
    const unsigned int endParticle =
        std::min((batch + 1) * batchSize, localParticleNum);
    unsigned int interiorParticleNum, boundaryParticleNum;
    for (int i = startParticle; i < endParticle; i++) {
      if (particleType(i) == 0)
        interiorParticleNum++;
      else
        boundaryParticleNum++;
    }

    Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
        interiorNeighborListsDevice("interior particle neighborlist",
                                    interiorParticleNum,
                                    neighborLists_.extent(1));
    Kokkos::View<int **>::HostMirror interiorNeighborListsHost =
        Kokkos::create_mirror_view(interiorNeighborListsDevice);
    Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
        boundaryNeighborListsDevice("boundary particle neighborlist",
                                    boundaryParticleNum,
                                    neighborLists_.extent(1));
    Kokkos::View<int **>::HostMirror boundaryNeighborListsHost =
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

    int boundaryCounter, interiorCounter;

    boundaryCounter = 0;
    interiorCounter = 0;
    for (int i = startParticle; i < endParticle; i++) {
      if (particleType(i) == 0) {
        interiorEpsilonHost(interiorCounter) = epsilon_(i);
        for (int j = 0; j <= neighborLists_(i, 0); j++) {
          interiorNeighborListsHost(interiorCounter, j) = neighborLists_(i, j);
        }
        for (int j = 0; j < dimension; j++) {
          interiorParticleCoordsHost(interiorCounter, j) = coords(i, j);
        }

        interiorCounter++;
      } else {
        boundaryEpsilonHost(boundaryCounter) = epsilon_(i);
        for (int j = 0; j <= neighborLists_(i, 0); j++) {
          boundaryNeighborListsHost(boundaryCounter, j) = neighborLists_(i, j);
        }
        for (int j = 0; j < dimension; j++) {
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

      const int interiorLaplacianIndex = solutionSet->getAlphaColumnOffset(
          Compadre::LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);

      interiorCounter = 0;
      std::vector<PetscInt> index;
      std::vector<PetscReal> value;
      for (int i = startParticle; i < endParticle; i++) {
        const PetscInt currentParticleIndex = i;
        if (particleType(i) == 0) {
          double Aij = 0.0;
          index.resize(interiorNeighborListsHost(interiorCounter, 0));
          value.resize(interiorNeighborListsHost(interiorCounter, 0));
          for (int j = 0; j < interiorNeighborListsHost(interiorCounter, 0);
               j++) {
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
      for (int i = startParticle; i < endParticle; i++) {
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
    printf("Duration of building linear system is:%fs\n", tEnd - tStart);
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

  const unsigned int localParticleNum = coords.extent(0);
  for (int i = 0; i < localParticleNum; i++) {
  }
  tEnd = MPI_Wtime();
  if (mpiRank_ == 0) {
    printf("Duration of building right hand side:%fs\n", tEnd - tStart);
  }
}

PoissonEquation::PoissonEquation() {}

PoissonEquation::~PoissonEquation() {}

void PoissonEquation::Init() {
  Equation::Init();
  field_ = HostRealVector("Host field", particleMgr_.GetLocalParticleNum());
}

HostRealVector &PoissonEquation::GetField() { return field_; }