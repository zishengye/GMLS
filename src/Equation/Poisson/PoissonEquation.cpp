#include "Equation/Poisson/PoissonEquation.hpp"
#include "Compadre_Operators.hpp"
#include "Core/Typedef.hpp"
#include "Discretization/PolyBasis.hpp"
#include "LinearAlgebra/LinearAlgebra.hpp"
#include "Math/Vec3.hpp"

#include <Compadre_Evaluator.hpp>
#include <Compadre_GMLS.hpp>
#include <Compadre_PointCloudSearch.hpp>
#include <Kokkos_CopyViews.hpp>
#include <Kokkos_Core_fwd.hpp>

void Equation::PoissonEquation::InitLinearSystem() {
  double tStart, tEnd;
  tStart = MPI_Wtime();
  Equation::InitLinearSystem();
  if (mpiRank_ == 0)
    printf("Start of initializing physics: Poisson\n");

  auto newMat = std::make_shared<DefaultMatrix>();
  AddLinearSystem(std::static_pointer_cast<DefaultMatrix>(newMat));

  auto &sourceCoords = hostGhostParticleCoords_;
  auto &sourceIndex = hostGhostParticleIndex_;
  auto &coords = particleMgr_.GetParticleCoords();
  auto &spacing = particleMgr_.GetParticleSize();
  auto &particleType = particleMgr_.GetParticleType();
  auto &normal = particleMgr_.GetParticleNormal();

  const unsigned int localParticleNum = coords.extent(0);
  const unsigned long globalParticleNum = particleMgr_.GetGlobalParticleNum();

  const unsigned int dimension = particleMgr_.GetDimension();

  const unsigned satisfiedNumNeighbor =
      pow(sqrt(2), dimension) *
      (Compadre::GMLS::getNP(polyOrder_ + 1, dimension) + 1);

  Equation::ConstructNeighborLists(satisfiedNumNeighbor);

  std::vector<bool> discretizationCheck(localParticleNum);
  for (std::size_t i = 0; i < localParticleNum; i++) {
    discretizationCheck[i] = false;
    if (particleType(i) != 0)
      discretizationCheck[i] = true;
  }

  auto pointCloudSearch(
      Compadre::CreatePointCloudSearch(sourceCoords, dimension));
  bool isNeighborSearchPassed = false;

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
      "source coords", sourceCoords.extent(0), sourceCoords.extent(1));
  Kokkos::deep_copy(sourceCoordsDevice, sourceCoords);

  const unsigned int batchSize = ((dimension == 2) ? 500 : 100);
  const unsigned int batchNum = localParticleNum / batchSize +
                                ((localParticleNum % batchSize > 0) ? 1 : 0);

  double maxRatio, meanNeighbor;
  unsigned int minNeighbor, maxNeighbor, iteCounter;
  iteCounter = 0;
  while (!isNeighborSearchPassed) {
    iteCounter++;
    unsigned int nonPassedBoundaryParticleNum = 0;
    unsigned int nonPassedInteriorParticleNum = 0;
    for (unsigned int batch = 0; batch < batchNum; batch++) {
      const unsigned int startParticle = batch * batchSize;
      const unsigned int endParticle =
          std::min((batch + 1) * batchSize, localParticleNum);
      unsigned int interiorParticleNum = 0, boundaryParticleNum = 0;

      for (unsigned int i = startParticle; i < endParticle; i++) {
        if (!discretizationCheck[i]) {
          if (particleType(i) == 0)
            interiorParticleNum++;
          else
            boundaryParticleNum++;
        }
      }

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

      Kokkos::View<double *, Kokkos::DefaultExecutionSpace>
          interiorEpsilonDevice("interior particle epsilon",
                                interiorParticleNum);
      Kokkos::View<double *>::HostMirror interiorEpsilonHost =
          Kokkos::create_mirror_view(interiorEpsilonDevice);
      Kokkos::View<double *, Kokkos::DefaultExecutionSpace>
          boundaryEpsilonDevice("boundary particle epsilon",
                                boundaryParticleNum);
      Kokkos::View<double *>::HostMirror boundaryEpsilonHost =
          Kokkos::create_mirror_view(boundaryEpsilonDevice);

      Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
          interiorParticleCoordsDevice("interior particle coord",
                                       interiorParticleNum, coords.extent(1));
      Kokkos::View<double **>::HostMirror interiorParticleCoordsHost =
          Kokkos::create_mirror_view(interiorParticleCoordsDevice);
      Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
          boundaryParticleCoordsDevice("boundary particle coord",
                                       boundaryParticleNum, coords.extent(1));
      Kokkos::View<double **>::HostMirror boundaryParticleCoordsHost =
          Kokkos::create_mirror_view(boundaryParticleCoordsDevice);

      unsigned int interiorCounter = 0;
      unsigned int boundaryCounter = 0;
      for (unsigned int i = startParticle; i < endParticle; i++) {
        if (!discretizationCheck[i]) {
          if (particleType(i) == 0) {
            interiorEpsilonHost(interiorCounter) = epsilon_(i);
            for (std::size_t j = 0; j <= neighborLists_(i, 0); j++) {
              interiorNeighborListsHost(interiorCounter, j) =
                  neighborLists_(i, j);
            }
            for (unsigned int j = 0; j < coords.extent(1); j++) {
              interiorParticleCoordsHost(interiorCounter, j) = coords(i, j);
            }

            interiorCounter++;
          } else {
            boundaryEpsilonHost(boundaryCounter) = epsilon_(i);
            for (std::size_t j = 0; j <= neighborLists_(i, 0); j++) {
              boundaryNeighborListsHost(boundaryCounter, j) =
                  neighborLists_(i, j);
            }
            for (unsigned int j = 0; j < coords.extent(1); j++) {
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
      Kokkos::deep_copy(interiorParticleCoordsDevice,
                        interiorParticleCoordsHost);
      Kokkos::deep_copy(boundaryParticleCoordsDevice,
                        boundaryParticleCoordsHost);

      {
        Compadre::GMLS interiorBasis = Compadre::GMLS(
            Compadre::ScalarTaylorPolynomial,
            Compadre::StaggeredEdgeAnalyticGradientIntegralSample, polyOrder_,
            dimension, "LU", "STANDARD");

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
        for (unsigned int i = startParticle; i < endParticle; i++) {
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

      for (unsigned int i = startParticle; i < endParticle; i++) {
        if (particleType(i) == 0 && discretizationCheck[i] == false) {
          nonPassedInteriorParticleNum++;
          epsilon_(i) += 0.25 * spacing(i);
        }
      }

      Kokkos::View<double ***, Kokkos::DefaultExecutionSpace>
          tangentBundleDevice("tangent bundles", boundaryParticleNum, dimension,
                              dimension);
      Kokkos::View<double ***>::HostMirror tangentBundleHost =
          Kokkos::create_mirror_view(tangentBundleDevice);

      boundaryCounter = 0;
      for (unsigned int i = startParticle; i < endParticle; i++) {
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
        for (unsigned int i = startParticle; i < endParticle; i++) {
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

      for (unsigned int i = startParticle; i < endParticle; i++) {
        if (particleType(i) != 0 && discretizationCheck[i] == false) {
          nonPassedBoundaryParticleNum++;
          epsilon_(i) += 0.25 * spacing(i);
        }
      }
    }

    int corePassCheck =
        nonPassedBoundaryParticleNum + nonPassedInteriorParticleNum;

    MPI_Allreduce(MPI_IN_PLACE, &corePassCheck, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    if (corePassCheck == 0)
      isNeighborSearchPassed = true;
    else {
      if (mpiRank_ == 0)
        printf("iteration count: %d non passed particle num: %d\n", iteCounter,
               corePassCheck);
      unsigned int minNeighborLists =
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

  DefaultMatrix &A = *(std::static_pointer_cast<DefaultMatrix>(
      linearSystemsPtr_[refinementIteration_]));
  A.Resize(localParticleNum, localParticleNum);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localParticleNum),
      [&](const int i) {
        std::vector<PetscInt> index;
        const PetscInt currentParticleIndex = i;
        if (particleType(i) == 0) {
          index.resize(neighborLists_(i, 0));
          for (std::size_t j = 0; j < neighborLists_(i, 0); j++) {
            const PetscInt neighborParticleIndex =
                sourceIndex(neighborLists_(i, j + 1));
            index[j] = neighborParticleIndex;
          }
        } else {
          if (boundaryType_(coords(i, 0), coords(i, 1), coords(i, 2))) {
            index.resize(1);
            index[0] = sourceIndex(i);
          } else {
            index.resize(neighborLists_(i, 0));
            for (std::size_t j = 0; j < neighborLists_(i, 0); j++) {
              const PetscInt neighborParticleIndex =
                  sourceIndex(neighborLists_(i, j + 1));
              index[j] = neighborParticleIndex;
            }
          }
        }
        std::sort(index.begin(), index.end());
        A.SetColIndex(currentParticleIndex, index);
      });
  Kokkos::fence();

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

void Equation::PoissonEquation::ConstructLinearSystem() {
  double tStart, tEnd;
  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();
  if (mpiRank_ == 0)
    printf("Start of building linear system of physics: Poisson\n");

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

  Kokkos::resize(kappa_, localParticleNum);
  Kokkos::resize(bi_, localParticleNum);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localParticleNum),
      KOKKOS_LAMBDA(const int i) {
        kappa_(i) = kappaFunc_(coords(i, 0), coords(i, 1), coords(i, 2));
      });
  Kokkos::fence();
  HostRealVector sourceKappa;
  ghost_.ApplyGhost(kappa_, sourceKappa);

  DefaultMatrix &A = *(std::static_pointer_cast<DefaultMatrix>(
      linearSystemsPtr_[refinementIteration_]));

  const unsigned int batchSize = ((dimension == 2) ? 500 : 100);
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
                                     interiorParticleNum, coords.extent(1));
    Kokkos::View<double **>::HostMirror interiorParticleCoordsHost =
        Kokkos::create_mirror_view(interiorParticleCoordsDevice);
    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        boundaryParticleCoordsDevice("boundary particle coord",
                                     boundaryParticleNum, coords.extent(1));
    Kokkos::View<double **>::HostMirror boundaryParticleCoordsHost =
        Kokkos::create_mirror_view(boundaryParticleCoordsDevice);

    Kokkos::View<double ***, Kokkos::DefaultExecutionSpace> tangentBundleDevice(
        "tangent bundles", boundaryParticleNum, dimension, dimension);
    Kokkos::View<double ***>::HostMirror tangentBundleHost =
        Kokkos::create_mirror_view(tangentBundleDevice);

    std::vector<unsigned int> batchMap;
    batchMap.resize(endParticle - startParticle);

    {
      unsigned int boundaryCounter, interiorCounter;

      boundaryCounter = 0;
      interiorCounter = 0;

      for (unsigned int i = startParticle; i < endParticle; i++) {
        if (particleType(i) == 0) {
          batchMap[i - startParticle] = interiorCounter;

          interiorCounter++;
        } else {
          batchMap[i - startParticle] = boundaryCounter;

          boundaryCounter++;
        }
      }
    }

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(startParticle,
                                                               endParticle),
        [&](const int i) {
          if (particleType(i) == 0) {
            const unsigned int interiorCounter = batchMap[i - startParticle];
            interiorEpsilonHost(interiorCounter) = epsilon_(i);
            for (std::size_t j = 0; j <= neighborLists_(i, 0); j++) {
              interiorNeighborListsHost(interiorCounter, j) =
                  neighborLists_(i, j);
            }
            for (unsigned int j = 0; j < coords.extent(1); j++) {
              interiorParticleCoordsHost(interiorCounter, j) = coords(i, j);
            }
          } else {
            unsigned int boundaryCounter = batchMap[i - startParticle];
            boundaryEpsilonHost(boundaryCounter) = epsilon_(i);
            for (std::size_t j = 0; j <= neighborLists_(i, 0); j++) {
              boundaryNeighborListsHost(boundaryCounter, j) =
                  neighborLists_(i, j);
            }
            for (unsigned int j = 0; j < coords.extent(1); j++) {
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
          }
        });
    Kokkos::fence();

    Kokkos::deep_copy(interiorNeighborListsDevice, interiorNeighborListsHost);
    Kokkos::deep_copy(boundaryNeighborListsDevice, boundaryNeighborListsHost);
    Kokkos::deep_copy(interiorEpsilonDevice, interiorEpsilonHost);
    Kokkos::deep_copy(boundaryEpsilonDevice, boundaryEpsilonHost);
    Kokkos::deep_copy(interiorParticleCoordsDevice, interiorParticleCoordsHost);
    Kokkos::deep_copy(boundaryParticleCoordsDevice, boundaryParticleCoordsHost);
    Kokkos::deep_copy(tangentBundleDevice, tangentBundleHost);

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

      auto interiorSolutionSet = interiorBasis.getSolutionSetHost();
      auto interiorAlpha = interiorSolutionSet->getAlphas();

      const unsigned int interiorLaplacianIndex =
          interiorSolutionSet->getAlphaColumnOffset(
              Compadre::LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);

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

      auto boundarySolutionSet = boundaryBasis.getSolutionSetHost();
      auto boundaryAlpha = boundarySolutionSet->getAlphas();

      const unsigned int boundaryLaplacianIndex =
          boundarySolutionSet->getAlphaColumnOffset(
              Compadre::LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);

      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(startParticle,
                                                                 endParticle),
          [&](const int i) {
            std::vector<PetscInt> index;
            std::vector<PetscReal> value;
            const unsigned int interiorCounter = batchMap[i - startParticle];
            const unsigned int boundaryCounter = batchMap[i - startParticle];
            const PetscInt currentParticleIndex = i;
            if (particleType(i) == 0) {
              double Aij = 0.0;
              const unsigned int numNeighbor =
                  interiorNeighborListsHost(interiorCounter, 0);
              index.resize(numNeighbor);
              value.resize(numNeighbor);
              for (std::size_t j = 0; j < numNeighbor; j++) {
                const PetscInt neighborParticleIndex =
                    sourceIndex(neighborLists_(i, j + 1));
                const int neighborIndex = neighborLists_(i, j + 1);
                auto alphaIndex = interiorSolutionSet->getAlphaIndex(
                    interiorCounter, interiorLaplacianIndex);
                double kappaIJ = kappaFunc_(
                    0.5 * (coords(i, 0) + sourceCoords(neighborIndex, 0)),
                    0.5 * (coords(i, 1) + sourceCoords(neighborIndex, 1)),
                    0.5 * (coords(i, 2) + sourceCoords(neighborIndex, 2)));
                index[j] = neighborParticleIndex;
                value[j] = kappaIJ * interiorAlpha(alphaIndex + j);
                Aij -= kappaIJ * interiorAlpha(alphaIndex + j);
              }
              value[0] = Aij;

              A.Increment(currentParticleIndex, index, value);
            } else {
              if (boundaryType_(coords(i, 0), coords(i, 1), coords(i, 2))) {
                A.Increment(i, sourceIndex(i), 1.0);
              } else {
                double Aij = 0.0;
                const unsigned int numNeighbor =
                    boundaryNeighborListsHost(boundaryCounter, 0);
                index.resize(numNeighbor);
                value.resize(numNeighbor);

                bi_(i) = boundarySolutionSet->getAlpha0TensorTo0Tensor(
                    Compadre::LaplacianOfScalarPointEvaluation, boundaryCounter,
                    numNeighbor);

                for (std::size_t j = 0; j < numNeighbor; j++) {
                  const PetscInt neighborParticleIndex =
                      sourceIndex(neighborLists_(i, j + 1));
                  const int neighborIndex = neighborLists_(i, j + 1);
                  auto alphaIndex = boundarySolutionSet->getAlphaIndex(
                      boundaryCounter, boundaryLaplacianIndex);
                  double kappaIJ = kappaFunc_(
                      0.5 * (coords(i, 0) + sourceCoords(neighborIndex, 0)),
                      0.5 * (coords(i, 1) + sourceCoords(neighborIndex, 1)),
                      0.5 * (coords(i, 2) + sourceCoords(neighborIndex, 2)));
                  index[j] = neighborParticleIndex;
                  value[j] = kappaIJ * boundaryAlpha(alphaIndex + j);
                  Aij -= kappaIJ * boundaryAlpha(alphaIndex + j);
                }
                value[0] = Aij;

                A.Increment(currentParticleIndex, index, value);
              }
            }
          });
      Kokkos::fence();
    }
  }

  A.Assemble();

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  if (mpiRank_ == 0) {
    printf("End of building linear system of physics: Poisson\n");
    printf("Duration of building linear system is:%.4fs\n", tEnd - tStart);
  }
}

void Equation::PoissonEquation::ConstructRhs() {
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
      if (boundaryType_(coords(i, 0), coords(i, 1), coords(i, 2)))
        rhs[i] = boundaryRhs_(coords(i, 0), coords(i, 1), coords(i, 2));
      else
        rhs[i] =
            bi_(i) * boundaryRhs_(coords(i, 0), coords(i, 1), coords(i, 2)) +
            interiorRhs_(coords(i, 0), coords(i, 1), coords(i, 2));
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

void Equation::PoissonEquation::SolveEquation() {
  unsigned int currentRefinementLevel = linearSystemsPtr_.size() - 1;
  // interpolation previous result
  if (currentRefinementLevel > 0) {
    LinearAlgebra::Vector<DefaultLinearAlgebraBackend> y;
    y.Create(oldField_);
    x_ = preconditionerPtr_->GetInterpolation(currentRefinementLevel) * y;
  }

  Equation::SolveEquation();
  x_.Copy(field_);
}

void Equation::PoissonEquation::CalculateError() {
  double tStart, tEnd;
  tStart = MPI_Wtime();
  Equation::CalculateError();

  auto &sourceCoords = hostGhostParticleCoords_;
  auto &coords = particleMgr_.GetParticleCoords();
  auto &spacing = particleMgr_.GetParticleSize();

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
      "source coords", sourceCoords.extent(0), sourceCoords.extent(1));
  Kokkos::deep_copy(sourceCoordsDevice, sourceCoords);

  const unsigned int localParticleNum = particleMgr_.GetLocalParticleNum();

  const unsigned int dimension = particleMgr_.GetDimension();

  HostRealVector ghostKappa, ghostField, ghostEpsilon, ghostSpacing;
  ghost_.ApplyGhost(kappa_, ghostKappa);
  ghost_.ApplyGhost(field_, ghostField);
  ghost_.ApplyGhost(epsilon_, ghostEpsilon);
  ghost_.ApplyGhost(spacing, ghostSpacing);

  double localDirectGradientNorm = 0.0;
  double globalDirectGradientNorm;

  unsigned int coefficientSize;
  {
    Compadre::GMLS testBasis =
        Compadre::GMLS(Compadre::ScalarTaylorPolynomial,
                       Compadre::StaggeredEdgeAnalyticGradientIntegralSample,
                       polyOrder_ + 1, dimension, "LU", "STANDARD");
    coefficientSize = testBasis.getPolynomialCoefficientsSize();
  }

  HostRealMatrix coefficientChunk("coefficient chunk", localParticleNum,
                                  coefficientSize);
  HostRealMatrix ghostCoefficientChunk;

  Kokkos::resize(gradientChunk_, localParticleNum, dimension);

  // get polynomial coefficients and direct gradients
  const unsigned int batchSize = ((dimension == 2) ? 500 : 100);
  const unsigned int batchNum = localParticleNum / batchSize +
                                ((localParticleNum % batchSize > 0) ? 1 : 0);
  for (unsigned int batch = 0; batch < batchNum; batch++) {
    const unsigned int startParticle = batch * batchSize;
    const unsigned int endParticle =
        std::min((batch + 1) * batchSize, localParticleNum);
    const unsigned int batchParticleNum = endParticle - startParticle;

    Kokkos::View<std::size_t **, Kokkos::DefaultExecutionSpace>
        batchNeighborListsDevice("batch particle neighbor list",
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
      Compadre::GMLS batchBasis =
          Compadre::GMLS(Compadre::ScalarTaylorPolynomial,
                         Compadre::StaggeredEdgeAnalyticGradientIntegralSample,
                         polyOrder_, dimension, "LU", "STANDARD");

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

      // default Compadre implementation can't deal with different kappa in the
      // field, the resulting coeff are manually calculated here.
      auto batchCoefficientDevice =
          batchBasis.getFullPolynomialCoefficientsBasis();
      decltype(batchCoefficientDevice) batchCoefficientHost;
      Kokkos::resize(batchCoefficientHost, batchCoefficientDevice.extent(0));
      Kokkos::deep_copy(batchCoefficientHost, batchCoefficientDevice);

      auto coeffMatrixDims =
          batchBasis.getPolynomialCoefficientsDomainRangeSize();
      auto coeffMemoryLayoutDims =
          batchBasis.getPolynomialCoefficientsMemorySize();

      // duplicate coefficients
      particleCounter = 0;
      for (std::size_t i = startParticle; i < endParticle; i++) {
        const unsigned int numNeighbor =
            batchNeighborListsHost(particleCounter, 0);
        for (unsigned int j = 0; j < coefficientSize; j++) {
          coefficientChunk(i, j) = 0.0;
          for (unsigned int k = 0; k < numNeighbor; k++) {
            const int particleIndex =
                batchNeighborListsHost(particleCounter, 1);
            const int neighborIndex =
                batchNeighborListsHost(particleCounter, k + 1);
            coefficientChunk(i, j) +=
                batchCoefficientHost(particleCounter *
                                         coeffMemoryLayoutDims(0) *
                                         coeffMemoryLayoutDims(1) +
                                     j * coeffMemoryLayoutDims(1) + k) *
                (ghostField(particleIndex) - ghostField(neighborIndex));
          }
        }

        auto coeffView = Kokkos::subview(coefficientChunk, i, Kokkos::ALL);
        for (unsigned int axes = 0; axes < dimension; axes++)
          gradientChunk_(i, axes) =
              Discretization::CalScalarGrad(axes, dimension, 0.0, 0.0, 0.0,
                                            polyOrder_, epsilon_(i), coeffView);
        particleCounter++;
      }

      for (std::size_t i = startParticle; i < endParticle; i++) {
        double localVolume = pow(spacing(i), dimension);
        for (unsigned int j = 0; j < dimension; j++)
          localDirectGradientNorm += pow(gradientChunk_(i, j), 2) * localVolume;
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
            recoveredGradientChunk_(i, axes) += Discretization::CalScalarGrad(
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
            reconstructedGradient = Discretization::CalScalarGrad(
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
    if (mpiRank_ == 0)
      printf("Field analytical solution error: %.6f\n", fieldAnalyticalError);
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
                gradientChunk_(i, axes);
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
    if (mpiRank_ == 0)
      printf("Field gradient analytical solution error: %.6f\n",
             fieldGradientAnalyticalError);
  }

  // // smooth stage
  // for (std::size_t i = 0; i < localParticleNum; i++) {
  //   const double localVolume = pow(spacing(i), dimension);
  //   error_(i) = pow(error_(i), 2.0) / localVolume;
  // }

  // for (int ite = 0; ite < 1; ite++) {
  //   HostRealVector ghostError;
  //   ghost_.ApplyGhost(error_, ghostError);

  //   for (std::size_t i = 0; i < localParticleNum; i++) {
  //     error_(i) = 0.0;
  //     double totalNeighborVolume = 0.0;
  //     for (std::size_t j = 0; j < neighborLists_(i, 0); j++) {
  //       const int neighborIndex = neighborLists_(i, j + 1);

  //       double dX[3];
  //       dX[0] = sourceCoords(neighborIndex, 0) - coords(i, 0);
  //       dX[1] = sourceCoords(neighborIndex, 1) - coords(i, 1);
  //       dX[2] = sourceCoords(neighborIndex, 2) - coords(i, 2);

  //       const double r = sqrt(dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2]);
  //       const double h = epsilon_(i);
  //       const int p = 4;

  //       double WabIJ =
  //           pow(1.0 - abs(r / h), p) * double((1.0 - abs(r / h)) > 0.0);

  //       double sourceVolume = pow(ghostSpacing(neighborIndex), dimension);

  //       error_(i) += pow(ghostError(neighborIndex), 2.0) * sourceVolume *
  //       WabIJ; totalNeighborVolume += sourceVolume * WabIJ;
  //     }
  //     error_(i) /= totalNeighborVolume;
  //   }
  // }

  // for (std::size_t i = 0; i < localParticleNum; i++) {
  //   const double localVolume = pow(spacing(i), dimension);
  //   error_(i) = sqrt(error_(i) * localVolume);
  // }

  tEnd = MPI_Wtime();
  if (mpiRank_ == 0) {
    printf("Duration of calculating error:%.4fs\n", tEnd - tStart);
  }
}

void Equation::PoissonEquation::Output() {
  Equation::Output();

  if (outputLevel_ == 0)
    return;

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
      for (std::size_t i = 0; i < gradientChunk_.extent(0); i++) {
        for (std::size_t j = 0; j < gradientChunk_.extent(1); j++) {
          float x = gradientChunk_(i, j);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));
        }
      }
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // output kappa
  if (mpiRank_ == 0) {
    vtkStream.open("vtk/AdaptiveStep" + std::to_string(refinementIteration_) +
                       ".vtk",
                   std::ios::out | std::ios::app | std::ios::binary);

    vtkStream << "SCALARS kappa float 1" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open("vtk/AdaptiveStep" + std::to_string(refinementIteration_) +
                         ".vtk",
                     std::ios::out | std::ios::app | std::ios::binary);
      for (std::size_t i = 0; i < kappa_.extent(0); i++) {
        float x = kappa_(i);
        SwapEnd(x);
        vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));
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

Equation::PoissonEquation::PoissonEquation()
    : Equation(), isFieldAnalyticalSolutionSet_(false),
      isFieldGradientAnalyticalSolutionSet_(false) {}

Equation::PoissonEquation::~PoissonEquation() {}

void Equation::PoissonEquation::Init() {
  Equation::Init();

  preconditionerPtr_ = std::make_shared<PoissonPreconditioner>();
}

HostRealVector &Equation::PoissonEquation::GetField() { return field_; }

void Equation::PoissonEquation::SetBoundaryType(
    const std::function<bool(const double, const double, const double)> &func) {
  boundaryType_ = func;
}

void Equation::PoissonEquation::SetInteriorRhs(
    const std::function<double(const double, const double, const double)>
        &func) {
  interiorRhs_ = func;
}

void Equation::PoissonEquation::SetBoundaryRhs(
    const std::function<double(const double, const double, const double)>
        &func) {
  boundaryRhs_ = func;
}

void Equation::PoissonEquation::SetAnalyticalFieldSolution(
    const std::function<double(const double, const double, const double)>
        &func) {
  analyticalFieldSolution_ = func;
  isFieldAnalyticalSolutionSet_ = true;
}

void Equation::PoissonEquation::SetAnalyticalFieldGradientSolution(
    const std::function<double(const double, const double, const double,
                               const unsigned int)> &func) {
  analyticalFieldGradientSolution_ = func;
  isFieldGradientAnalyticalSolutionSet_ = true;
}

void Equation::PoissonEquation::SetKappa(
    const std::function<double(const double, const double, const double)>
        &func) {
  kappaFunc_ = func;
}