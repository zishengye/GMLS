#include "Equation/Stokes/StokesEquation.hpp"
#include "Core/Typedef.hpp"
#include "Discretization/PolyBasis.hpp"
#include "Equation/Equation.hpp"
#include "LinearAlgebra/LinearAlgebra.hpp"
#include "Math/Vec3.hpp"

#include <Compadre_Evaluator.hpp>
#include <Compadre_GMLS.hpp>
#include <Compadre_Operators.hpp>
#include <Compadre_PointCloudSearch.hpp>
#include <Kokkos_CopyViews.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <cstdio>
#include <mpi.h>

Void Equation::StokesEquation::InitLinearSystem() {
  double tStart, tEnd;
  tStart = MPI_Wtime();
  Equation::InitLinearSystem();
  if (mpiRank_ == 0)
    printf("Start of initializing physics: Stokes\n");

  auto newMat = std::make_shared<StokesMatrix>();
  newMat->Resize(2, 2);

  newMat->SetSubMat(0, 0,
                    std::static_pointer_cast<DefaultMatrix>(
                        std::make_shared<DefaultMatrix>()));
  newMat->SetSubMat(0, 1,
                    std::static_pointer_cast<DefaultMatrix>(
                        std::make_shared<DefaultMatrix>()));
  newMat->SetSubMat(1, 0,
                    std::static_pointer_cast<DefaultMatrix>(
                        std::make_shared<DefaultMatrix>()));
  newMat->SetSubMat(1, 1,
                    std::static_pointer_cast<DefaultMatrix>(
                        std::make_shared<DefaultMatrix>()));

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

  auto &uu = *(std::static_pointer_cast<DefaultMatrix>(
      std::static_pointer_cast<StokesMatrix>(
          linearSystemsPtr_[refinementIteration_])
          ->GetSubMat(0, 0)));

  uu.Resize(localParticleNum, localParticleNum, dimension);

  auto &up = *(std::static_pointer_cast<DefaultMatrix>(
      std::static_pointer_cast<StokesMatrix>(
          linearSystemsPtr_[refinementIteration_])
          ->GetSubMat(0, 1)));

  up.Resize(dimension * localParticleNum, localParticleNum);

  auto &pu = *(std::static_pointer_cast<DefaultMatrix>(
      std::static_pointer_cast<StokesMatrix>(
          linearSystemsPtr_[refinementIteration_])
          ->GetSubMat(1, 0)));

  pu.Resize(localParticleNum, localParticleNum * dimension);

  auto &pp = *(std::static_pointer_cast<DefaultMatrix>(
      std::static_pointer_cast<StokesMatrix>(
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
    printf("\nAfter satisfying conditioning of Laplacian  operator\niteration "
           "count: %d min neighbor: %d, max neighbor: %d , mean "
           "neighbor: %.2f, max ratio: %.2f\n",
           iteCounter, minNeighbor, maxNeighbor,
           meanNeighbor / (double)globalParticleNum, maxRatio);
    printf("Duration of initializing linear system:%.4fs\n", tEnd - tStart);
  }
}

Void Equation::StokesEquation::ConstructLinearSystem() {
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

  auto &A = *(std::static_pointer_cast<StokesMatrix>(
      linearSystemsPtr_[refinementIteration_]));

  Kokkos::resize(bi_, localParticleNum);

  auto &uu = *(std::static_pointer_cast<DefaultMatrix>(A.GetSubMat(0, 0)));
  auto &up = *(std::static_pointer_cast<DefaultMatrix>(A.GetSubMat(0, 1)));
  auto &pu = *(std::static_pointer_cast<DefaultMatrix>(A.GetSubMat(1, 0)));
  auto &pp = *(std::static_pointer_cast<DefaultMatrix>(A.GetSubMat(1, 1)));

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

  A.Assemble();

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  if (mpiRank_ == 0) {
    printf("End of building linear system of physics: Stokes\n");
    printf("Duration of building linear system is:%.4fs\n", tEnd - tStart);
  }
}

Void Equation::StokesEquation::ConstructRhs() {
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

  Kokkos::resize(oldField_, field_.extent(0));
  for (unsigned int i = 0; i < field_.extent(0); i++) {
    oldField_(i) = field_(i);
  }

  Kokkos::resize(velocity_, localParticleNum, dimension);
  Kokkos::resize(pressure_, localParticleNum);
  Kokkos::resize(error_, localParticleNum);

  std::vector<double> rhs(localParticleNum * fieldDof);
  for (std::size_t i = 0; i < localParticleNum; i++) {
    if (particleType(i) != 0) {
      rhs[localParticleNum * velocityDof + i] =
          pressureInteriorRhs_(coords(i, 0), coords(i, 1), coords(i, 2));
      for (unsigned int j = 0; j < dimension; j++) {
        rhs[i * velocityDof + j] =
            velocityBoundaryRhs_(coords(i, 0), coords(i, 1), coords(i, 2), j);
        rhs[localParticleNum * velocityDof + i] +=
            bi_(i) * normal(i, j) *
            velocityInteriorRhs_(coords(i, 0), coords(i, 1), coords(i, 2), j);
      }
    } else {
      rhs[localParticleNum * velocityDof + i] =
          pressureInteriorRhs_(coords(i, 0), coords(i, 1), coords(i, 2));
      for (unsigned int j = 0; j < dimension; j++)
        rhs[i * velocityDof + j] =
            velocityInteriorRhs_(coords(i, 0), coords(i, 1), coords(i, 2), j);
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

Void Equation::StokesEquation::SolveEquation() {
  unsigned int currentRefinementLevel = linearSystemsPtr_.size() - 1;
  // interpolation previous result
  if (currentRefinementLevel > 0) {
    LinearAlgebra::Vector<DefaultLinearAlgebraBackend> y;
    y.Create(oldField_);
    x_ = preconditionerPtr_->GetInterpolation(currentRefinementLevel) * y;
  }

  Equation::SolveEquation();
  x_.Copy(field_);

  const unsigned int localParticleNum = particleMgr_.GetLocalParticleNum();
  const unsigned int dimension = particleMgr_.GetDimension();
  const unsigned int velocityDof = dimension;
  const unsigned int pressureDofOffset = dimension * localParticleNum;
  for (std::size_t i = 0; i < localParticleNum; i++) {
    for (unsigned int j = 0; j < velocityDof; j++)
      velocity_(i, j) = field_(velocityDof * i + j);
  }
  for (std::size_t i = 0; i < localParticleNum; i++) {
    pressure_(i) = field_(pressureDofOffset + i);
  }
}

Void Equation::StokesEquation::CalculateError() {
  double tStart, tEnd;
  tStart = MPI_Wtime();
  Equation::CalculateError();

  if (mpiRank_ == 0)
    printf("Start of calculating error of Stokes equation\n");

  auto &sourceCoords = hostGhostParticleCoords_;
  auto &coords = particleMgr_.GetParticleCoords();
  auto &spacing = particleMgr_.GetParticleSize();

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
      "source coords", sourceCoords.extent(0), sourceCoords.extent(1));
  Kokkos::deep_copy(sourceCoordsDevice, sourceCoords);

  const unsigned int localParticleNum = particleMgr_.GetLocalParticleNum();

  const unsigned int dimension = particleMgr_.GetDimension();
  const unsigned int gradientComponentNum = dimension * dimension;

  HostRealVector ghostEpsilon, ghostSpacing;
  HostRealMatrix ghostVelocity;
  ghost_.ApplyGhost(velocity_, ghostVelocity);
  ghost_.ApplyGhost(epsilon_, ghostEpsilon);
  ghost_.ApplyGhost(spacing, ghostSpacing);

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> ghostVelocityDevice(
      "source velocity", ghostVelocity.extent(0), ghostVelocity.extent(1));

  Kokkos::deep_copy(ghostVelocityDevice, ghostVelocity);

  double localDirectGradientNorm = 0.0;
  double globalDirectGradientNorm;

  unsigned int coefficientSize;
  {
    Compadre::GMLS testBasis = Compadre::GMLS(
        Compadre::DivergenceFreeVectorTaylorPolynomial,
        Compadre::VectorPointSample, polyOrder_, dimension, "LU", "STANDARD");
    coefficientSize = testBasis.getPolynomialCoefficientsSize();
  }

  HostRealMatrix coefficientChunk("coefficient chunk", localParticleNum,
                                  coefficientSize);
  HostRealMatrix ghostCoefficientChunk;

  Kokkos::resize(gradientChunk_, localParticleNum, gradientComponentNum);

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
      Compadre::GMLS batchBasis = Compadre::GMLS(
          Compadre::DivergenceFreeVectorTaylorPolynomial,
          Compadre::VectorPointSample, polyOrder_, dimension, "LU", "STANDARD");

      batchBasis.setProblemData(batchNeighborListsDevice, sourceCoordsDevice,
                                batchParticleCoordsDevice, batchEpsilonDevice);

      batchBasis.addTargets(Compadre::GradientOfVectorPointEvaluation);

      batchBasis.setWeightingType(Compadre::WeightingFunctionType::Power);
      batchBasis.setWeightingParameter(4);
      batchBasis.setOrderOfQuadraturePoints(2);
      batchBasis.setDimensionOfQuadraturePoints(1);
      batchBasis.setQuadratureType("LINE");

      batchBasis.generateAlphas(1, true);

      Compadre::Evaluator batchEvaluator(&batchBasis);

      auto batchCoefficients =
          batchEvaluator
              .applyFullPolynomialCoefficientsBasisToDataAllComponents<
                  double **, Kokkos::HostSpace>(ghostVelocityDevice);

      auto batchGradient =
          batchEvaluator.applyAlphasToDataAllComponentsAllTargetSites<
              double **, Kokkos::HostSpace>(
              ghostVelocityDevice, Compadre::GradientOfVectorPointEvaluation);

      // duplicate coefficients
      particleCounter = 0;
      for (std::size_t i = startParticle; i < endParticle; i++) {
        for (unsigned int j = 0; j < coefficientSize; j++)
          coefficientChunk(i, j) = batchCoefficients(particleCounter, j);

        particleCounter++;
      }

      // duplicate gradients
      particleCounter = 0;
      for (std::size_t i = startParticle; i < endParticle; i++) {
        for (unsigned int axes = 0; axes < gradientComponentNum; axes++)
          gradientChunk_(i, axes) = batchGradient(particleCounter, axes);

        particleCounter++;
      }

      for (std::size_t i = startParticle; i < endParticle; i++) {
        double localVolume = pow(spacing(i), dimension);
        for (unsigned int j = 0; j < gradientComponentNum; j++)
          localDirectGradientNorm += pow(gradientChunk_(i, j), 2) * localVolume;
      }
    }
  }

  MPI_Allreduce(&localDirectGradientNorm, &globalDirectGradientNorm, 1,
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  globalDirectGradientNorm = std::sqrt(globalDirectGradientNorm);

  ghost_.ApplyGhost(coefficientChunk, ghostCoefficientChunk);

  // estimate recovered gradient
  Kokkos::resize(recoveredGradientChunk_, localParticleNum,
                 gradientComponentNum);
  HostRealMatrix ghostRecoveredGradientChunk;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,
                                                             coords.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        for (unsigned int j = 0; j < gradientComponentNum; j++)
          recoveredGradientChunk_(i, j) = 0.0;
        for (std::size_t j = 0; j < neighborLists_(i, 0); j++) {
          const std::size_t neighborParticleIndex = neighborLists_(i, j + 1);
          auto coeffView = Kokkos::subview(ghostCoefficientChunk,
                                           neighborParticleIndex, Kokkos::ALL);

          for (unsigned int axes1 = 0; axes1 < dimension; axes1++)
            for (unsigned int axes2 = 0; axes2 < dimension; axes2++)
              recoveredGradientChunk_(i, axes1 * dimension + axes2) +=
                  Discretization::CalDivFreeGrad(
                      axes1, axes2, dimension,
                      coords(i, 0) - sourceCoords(neighborParticleIndex, 0),
                      coords(i, 1) - sourceCoords(neighborParticleIndex, 1),
                      coords(i, 2) - sourceCoords(neighborParticleIndex, 2),
                      polyOrder_, ghostEpsilon(neighborParticleIndex),
                      coeffView);
        }

        for (unsigned int j = 0; j < gradientComponentNum; j++)
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

          for (unsigned int axes1 = 0; axes1 < dimension; axes1++)
            for (unsigned int axes2 = 0; axes2 < dimension; axes2++) {
              reconstructedGradient = Discretization::CalDivFreeGrad(
                  axes1, axes2, dimension,
                  sourceCoords(neighborParticleIndex, 0) - coords(i, 0),
                  sourceCoords(neighborParticleIndex, 1) - coords(i, 1),
                  sourceCoords(neighborParticleIndex, 2) - coords(i, 2),
                  polyOrder_, epsilon_(i), coeffView);

              error_(i) +=
                  pow(reconstructedGradient -
                          ghostRecoveredGradientChunk(
                              neighborParticleIndex, axes1 * dimension + axes2),
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

  if (isVelocityAnalyticalSolutionSet_) {
    double velocityAnalyticalError = 0.0;
    double velocityNorm = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
            0, velocity_.extent(0)),
        [&](const int i, double &tVelocityAnalyticalError) {
          const double x = coords(i, 0);
          const double y = coords(i, 1);
          const double z = coords(i, 2);

          for (unsigned int j = 0; j < dimension; j++) {
            double difference =
                analyticalVelocitySolution_(x, y, z, j) - velocity_(i, j);
            tVelocityAnalyticalError += pow(difference, 2);
          }
        },
        Kokkos::Sum<double>(velocityAnalyticalError));
    Kokkos::fence();

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
            0, velocity_.extent(0)),
        [&](const int i, double &tVelocityNorm) {
          const double x = coords(i, 0);
          const double y = coords(i, 1);
          const double z = coords(i, 2);

          for (unsigned int j = 0; j < dimension; j++)
            tVelocityNorm += pow(analyticalVelocitySolution_(x, y, z, j), 2);
        },
        Kokkos::Sum<double>(velocityNorm));
    Kokkos::fence();

    MPI_Allreduce(MPI_IN_PLACE, &velocityAnalyticalError, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &velocityNorm, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    velocityAnalyticalError = sqrt(velocityAnalyticalError / velocityNorm);
    if (mpiRank_ == 0)
      printf("Velocity analytical solution error: %.6f\n",
             velocityAnalyticalError);
  }

  if (isPressureAnalyticalSolutionSet_) {
    double pressureAnalyticalError = 0.0;
    double pressureNorm = 0.0;
    double pressureOffset = 0.0;

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
            0, pressure_.extent(0)),
        [&](const int i, double &tPressureOffset) {
          const double x = coords(i, 0);
          const double y = coords(i, 1);
          const double z = coords(i, 2);

          tPressureOffset += analyticalPressureSolution_(x, y, z);
        },
        Kokkos::Sum<double>(pressureOffset));
    Kokkos::fence();
    MPI_Allreduce(MPI_IN_PLACE, &pressureOffset, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    unsigned int globalParticleNum = particleMgr_.GetGlobalParticleNum();
    pressureOffset /= globalParticleNum;

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
            0, pressure_.extent(0)),
        [&](const int i, double &tPressureAnalyticalError) {
          const double x = coords(i, 0);
          const double y = coords(i, 1);
          const double z = coords(i, 2);

          double difference =
              (analyticalPressureSolution_(x, y, z) - pressureOffset) -
              pressure_(i);
          tPressureAnalyticalError += pow(difference, 2);
        },
        Kokkos::Sum<double>(pressureAnalyticalError));
    Kokkos::fence();

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
            0, pressure_.extent(0)),
        [&](const int i, double &tPressureNorm) {
          const double x = coords(i, 0);
          const double y = coords(i, 1);
          const double z = coords(i, 2);

          tPressureNorm +=
              pow(analyticalPressureSolution_(x, y, z) - pressureOffset, 2);
        },
        Kokkos::Sum<double>(pressureNorm));
    Kokkos::fence();

    MPI_Allreduce(MPI_IN_PLACE, &pressureAnalyticalError, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &pressureNorm, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    pressureAnalyticalError = sqrt(pressureAnalyticalError / pressureNorm);
    if (mpiRank_ == 0)
      printf("Pressure analytical solution error: %.6f\n",
             pressureAnalyticalError);
  }

  tEnd = MPI_Wtime();
  if (mpiRank_ == 0)
    printf("Duration of calculating error:%.4fs\n", tEnd - tStart);
}

Equation::StokesEquation::StokesEquation()
    : Equation(), isVelocityAnalyticalSolutionSet_(false),
      isPressureAnalyticalSolutionSet_(false),
      isVelocityGradientAnalyticalSolutionSet_(false),
      isPressureGradientAnalyticalSolutionSet_(false) {}

Equation::StokesEquation::~StokesEquation() {}

Void Equation::StokesEquation::Output() {
  if (outputLevel_ == 0)
    return;

  std::string outputFileName =
      "vtk/AdaptiveStep" + std::to_string(refinementIteration_) + ".vtk";

  Output(outputFileName);
}

Void Equation::StokesEquation::Output(String &outputFileName) {
  Equation::Output(outputFileName);

  const unsigned int dimension = particleMgr_.GetDimension();

  std::ofstream vtkStream;
  // output velocity value
  if (mpiRank_ == 0) {
    vtkStream.open(outputFileName,
                   std::ios::out | std::ios::app | std::ios::binary);

    if (dimension == 2)
      vtkStream << "SCALARS u float 2" << std::endl
                << "LOOKUP_TABLE default" << std::endl;
    else
      vtkStream << "SCALARS u float 3" << std::endl
                << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open(outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
      if (dimension == 2)
        for (std::size_t i = 0; i < velocity_.extent(0); i++) {
          float x = velocity_(i, 0);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));

          x = velocity_(i, 1);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));
        }
      else
        for (std::size_t i = 0; i < velocity_.extent(0); i++) {
          float x = velocity_(i, 0);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));

          x = velocity_(i, 1);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));

          x = velocity_(i, 2);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));
        };
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // output pressure value
  if (mpiRank_ == 0) {
    vtkStream.open(outputFileName,
                   std::ios::out | std::ios::app | std::ios::binary);

    vtkStream << "SCALARS p float 1" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open(outputFileName,
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

  // output velocity gradient value
  if (mpiRank_ == 0) {
    vtkStream.open(outputFileName,
                   std::ios::out | std::ios::app | std::ios::binary);

    if (dimension == 2)
      vtkStream << "SCALARS gradU float 2" << std::endl
                << "LOOKUP_TABLE default" << std::endl;
    else
      vtkStream << "SCALARS gradU float 3" << std::endl
                << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open(outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
      if (dimension == 2)
        for (std::size_t i = 0; i < gradientChunk_.extent(0); i++) {
          float x = gradientChunk_(i, 0);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));

          x = gradientChunk_(i, 1);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));
        }
      else
        for (std::size_t i = 0; i < gradientChunk_.extent(0); i++) {
          float x = gradientChunk_(i, 0);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));

          x = gradientChunk_(i, 1);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));

          x = gradientChunk_(i, 2);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));
        };
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (mpiRank_ == 0) {
    vtkStream.open(outputFileName,
                   std::ios::out | std::ios::app | std::ios::binary);

    if (dimension == 2)
      vtkStream << "SCALARS gradV float 2" << std::endl
                << "LOOKUP_TABLE default" << std::endl;
    else
      vtkStream << "SCALARS gradV float 3" << std::endl
                << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open(outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
      if (dimension == 2)
        for (std::size_t i = 0; i < gradientChunk_.extent(0); i++) {
          float x = gradientChunk_(i, 2);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));

          x = gradientChunk_(i, 3);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));
        }
      else
        for (std::size_t i = 0; i < gradientChunk_.extent(0); i++) {
          float x = gradientChunk_(i, 3);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));

          x = gradientChunk_(i, 4);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));

          x = gradientChunk_(i, 5);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));
        };
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (mpiRank_ == 0) {
    vtkStream.open(outputFileName,
                   std::ios::out | std::ios::app | std::ios::binary);

    if (dimension == 3)
      vtkStream << "SCALARS gradW float 3" << std::endl
                << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open(outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
      if (dimension == 3)
        for (std::size_t i = 0; i < gradientChunk_.extent(0); i++) {
          float x = gradientChunk_(i, 6);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));

          x = gradientChunk_(i, 7);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));

          x = gradientChunk_(i, 8);
          SwapEnd(x);
          vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));
        };
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

Void Equation::StokesEquation::Init() {
  Equation::Init();

  preconditionerPtr_ = std::make_shared<StokesPreconditioner>();
}

Void Equation::StokesEquation::CalculateSensitivity(
    DefaultParticleManager &particleMgr, HostRealVector &sensitivity) {}

Scalar Equation::StokesEquation::GetObjFunc() {
  auto &spacing = particleMgr_.GetParticleSize();
  auto &particleType = particleMgr_.GetParticleType();
  const unsigned int dimension = particleMgr_.GetDimension();

  Scalar result = 0.0;

  return result;
}

Void Equation::StokesEquation::SetVelocityBoundaryType(
    const std::function<bool(const double, const double, const double)> &func) {
  velocityBoundaryType_ = func;
}

Void Equation::StokesEquation::SetPressureBoundaryType(
    const std::function<int(const double, const double, const double)> &func) {
  pressureBoundaryType_ = func;
}

Void Equation::StokesEquation::SetVelocityInteriorRhs(
    const std::function<double(const double, const double, const double,
                               const unsigned int)> &func) {
  velocityInteriorRhs_ = func;
}

Void Equation::StokesEquation::SetVelocityBoundaryRhs(
    const std::function<double(const double, const double, const double,
                               const unsigned int)> &func) {
  velocityBoundaryRhs_ = func;
}

Void Equation::StokesEquation::SetPressureInteriorRhs(
    const std::function<double(const double, const double, const double)>
        &func) {
  pressureInteriorRhs_ = func;
}

Void Equation::StokesEquation::SetPressureBoundaryRhs(
    const std::function<double(const double, const double, const double)>
        &func) {
  pressureBoundaryRhs_ = func;
}

Void Equation::StokesEquation::SetAnalyticalVelocitySolution(
    const std::function<double(const double, const double, const double,
                               const unsigned int)> &func) {
  analyticalVelocitySolution_ = func;

  isVelocityAnalyticalSolutionSet_ = true;
}

Void Equation::StokesEquation::SetAnalyticalPressureSolution(
    const std::function<double(const double, const double, const double)>
        &func) {
  analyticalPressureSolution_ = func;

  isPressureAnalyticalSolutionSet_ = true;
}

Void Equation::StokesEquation::SetAnalyticalVelocityGradientSolution(
    const std::function<double(const double, const double, const double,
                               const unsigned int)> &func) {
  analyticalVelocityGradientSolution_ = func;

  isVelocityGradientAnalyticalSolutionSet_ = true;
}

Void Equation::StokesEquation::SetAnalyticalPressureGradientSolution(
    const std::function<double(const double, const double, const double,
                               const unsigned int)> &func) {
  analyticalPressureGradientSolution_ = func;

  isPressureGradientAnalyticalSolutionSet_ = true;
}