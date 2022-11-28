#include "Equation/Stokes/StokesEquation.hpp"
#include "Core/Typedef.hpp"
#include "Cuda/Kokkos_Cuda_Team.hpp"
#include "Discretization/PolyBasis.hpp"
#include "Equation/Equation.hpp"
#include "Equation/Stokes/StokesMatrix.hpp"
#include "Kokkos_ExecPolicy.hpp"
#include "Kokkos_Parallel_Reduce.hpp"
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
#ifdef DIAGNOSE_DISCRETIZATION
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
#endif

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
    printf("\nAfter satisfying conditioning of Laplacian operator\niteration "
           "count: %d min neighbor: %d, max neighbor: %d , mean "
           "neighbor: %.2f, max ratio: %.2f\n",
           iteCounter, minNeighbor, maxNeighbor,
           meanNeighbor / (double)globalParticleNum, maxRatio);
    printf("Duration of initializing linear system: %.4fs\n", tEnd - tStart);
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

  auto &uu =
      *(std::static_pointer_cast<Equation::DefaultMatrix>(A.GetSubMat(0, 0)));
  auto &up =
      *(std::static_pointer_cast<Equation::DefaultMatrix>(A.GetSubMat(0, 1)));
  auto &pu =
      *(std::static_pointer_cast<Equation::DefaultMatrix>(A.GetSubMat(1, 0)));
  auto &pp =
      *(std::static_pointer_cast<Equation::DefaultMatrix>(A.GetSubMat(1, 1)));

#ifdef KOKKOS_ENABLE_CUDA
  const unsigned int batchSize = (dimension == 2) ? 200000 : 50000;
#else
  const unsigned int batchSize = (dimension == 2) ? 500 : 100;
#endif

  std::vector<unsigned int> interiorParticleIndex;
  std::vector<unsigned int> boundaryParticleIndex;
  for (unsigned int i = 0; i < localParticleNum; i++) {
    if (particleType(i) == 0)
      interiorParticleIndex.push_back(i);
    else
      boundaryParticleIndex.push_back(i);
  }

  const unsigned int interiorParticleNum = interiorParticleIndex.size();
  const unsigned int boundaryParticleNum = boundaryParticleIndex.size();

  const unsigned int interiorBatchNum =
      interiorParticleNum / batchSize +
      ((localParticleNum % batchSize > 0) ? 1 : 0);
  const unsigned int boundaryBatchNum =
      boundaryParticleNum / batchSize +
      ((localParticleNum % batchSize > 0) ? 1 : 0);

  double generateAlphasDuration = 0.0;
  for (unsigned int batch = 0; batch < interiorBatchNum; batch++) {
    const unsigned int startParticle = batch * batchSize;
    const unsigned int endParticle =
        std::min((batch + 1) * batchSize, interiorParticleNum);

    const unsigned int batchParticleNum = endParticle - startParticle;

    Kokkos::View<std::size_t **, Kokkos::DefaultExecutionSpace>
        interiorNeighborListsDevice("interior particle neighbor list",
                                    batchParticleNum, neighborLists_.extent(1));
    Kokkos::View<std::size_t **>::HostMirror interiorNeighborListsHost =
        Kokkos::create_mirror_view(interiorNeighborListsDevice);

    Kokkos::View<double *, Kokkos::DefaultExecutionSpace> interiorEpsilonDevice(
        "interior particle epsilon", batchParticleNum);
    Kokkos::View<double *>::HostMirror interiorEpsilonHost =
        Kokkos::create_mirror_view(interiorEpsilonDevice);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        interiorParticleCoordsDevice("interior particle coord",
                                     batchParticleNum, coords.extent(1));
    Kokkos::View<double **>::HostMirror interiorParticleCoordsHost =
        Kokkos::create_mirror_view(interiorParticleCoordsDevice);

    for (unsigned int i = startParticle; i < endParticle; i++) {
      interiorEpsilonHost(i - startParticle) =
          epsilon_(interiorParticleIndex[i]);
      for (std::size_t j = 0; j <= neighborLists_(interiorParticleIndex[i], 0);
           j++) {
        interiorNeighborListsHost(i - startParticle, j) =
            neighborLists_(interiorParticleIndex[i], j);
      }
      for (unsigned int j = 0; j < dimension; j++) {
        interiorParticleCoordsHost(i - startParticle, j) =
            coords(interiorParticleIndex[i], j);
      }
    }

    Kokkos::deep_copy(interiorNeighborListsDevice, interiorNeighborListsHost);
    Kokkos::deep_copy(interiorEpsilonDevice, interiorEpsilonHost);
    Kokkos::deep_copy(interiorParticleCoordsDevice, interiorParticleCoordsHost);

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

      double generateAlphasTimer1 = MPI_Wtime();
      interiorVelocityBasis.generateAlphas(1, false);

      auto velocitySolutionSet = interiorVelocityBasis.getSolutionSetHost();
      auto interiorVelocityAlpha = velocitySolutionSet->getAlphas();
      double generateAlphasTimer2 = MPI_Wtime();
      generateAlphasDuration += generateAlphasTimer2 - generateAlphasTimer1;

      std::vector<unsigned int> interiorCurlCurlIndex(pow(dimension, 2));
      for (unsigned int i = 0; i < dimension; i++) {
        for (unsigned int j = 0; j < dimension; j++) {
          interiorCurlCurlIndex[i * dimension + j] =
              velocitySolutionSet->getAlphaColumnOffset(
                  Compadre::CurlCurlOfVectorPointEvaluation, i, 0, j, 0);
        }
      }

      unsigned int blockStorageSize = dimension * dimension;

      for (unsigned int i = startParticle; i < endParticle; i++) {
        const unsigned int currentParticleIndex = interiorParticleIndex[i];
        std::vector<DefaultLinearAlgebraBackend::DefaultInteger> index;
        std::vector<DefaultLinearAlgebraBackend::DefaultScalar> value;
        unsigned int numNeighbor =
            interiorNeighborListsHost(i - startParticle, 0);
        unsigned int singleRowSize = numNeighbor * dimension;
        index.resize(numNeighbor);
        value.resize(numNeighbor * blockStorageSize);
        for (std::size_t j = 0; j < numNeighbor; j++) {
          const DefaultLinearAlgebraBackend::DefaultInteger
              neighborParticleIndex = sourceIndex(
                  interiorNeighborListsHost(i - startParticle, j + 1));
          index[j] = neighborParticleIndex;
          for (unsigned int axes1 = 0; axes1 < dimension; axes1++) {
            for (unsigned int axes2 = 0; axes2 < dimension; axes2++) {
              auto alphaIndex = velocitySolutionSet->getAlphaIndex(
                  i - startParticle,
                  interiorCurlCurlIndex[axes1 * dimension + axes2]);
              value[axes1 * singleRowSize + j * dimension + axes2] =
                  interiorVelocityAlpha(alphaIndex + j);
            }
          }
        }

        uu.Increment(currentParticleIndex, index, value);
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

      double generateAlphasTimer1 = MPI_Wtime();
      interiorPressureBasis.generateAlphas(1, false);

      auto pressureSolutionSet = interiorPressureBasis.getSolutionSetHost();
      auto interiorPressureAlpha = pressureSolutionSet->getAlphas();
      double generateAlphasTimer2 = MPI_Wtime();
      generateAlphasDuration += generateAlphasTimer2 - generateAlphasTimer1;

      const unsigned int interiorPressureLaplacianIndex =
          pressureSolutionSet->getAlphaColumnOffset(
              Compadre::LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);
      std::vector<unsigned int> interiorPressureGradientIndex(dimension);
      for (unsigned int i = 0; i < dimension; i++) {
        interiorPressureGradientIndex[i] =
            pressureSolutionSet->getAlphaColumnOffset(
                Compadre::GradientOfScalarPointEvaluation, i, 0, 0, 0);
      }

      for (unsigned int i = startParticle; i < endParticle; i++) {
        std::vector<PetscInt> index;
        std::vector<PetscReal> value;
        const PetscInt currentParticleIndex = interiorParticleIndex[i];
        const unsigned int numNeighbor =
            interiorNeighborListsHost(i - startParticle, 0);
        double Aij = 0.0;
        index.resize(numNeighbor);
        value.resize(numNeighbor);
        for (std::size_t j = 0; j < numNeighbor; j++) {
          const PetscInt neighborParticleIndex =
              sourceIndex(interiorNeighborListsHost(i - startParticle, j + 1));
          auto alphaIndex = pressureSolutionSet->getAlphaIndex(
              i - startParticle, interiorPressureLaplacianIndex);
          index[j] = neighborParticleIndex;
          value[j] = interiorPressureAlpha(alphaIndex + j);
          Aij -= interiorPressureAlpha(alphaIndex + j);
        }
        value[0] = Aij;

        pp.Increment(currentParticleIndex, index, value);

        for (unsigned int k = 0; k < dimension; k++) {
          Aij = 0.0;
          for (std::size_t j = 0; j < numNeighbor; j++) {
            auto alphaIndex = pressureSolutionSet->getAlphaIndex(
                i - startParticle, interiorPressureGradientIndex[k]);
            value[j] = -interiorPressureAlpha(alphaIndex + j);
            Aij += interiorPressureAlpha(alphaIndex + j);
          }
          value[0] = Aij;
          up.Increment(currentParticleIndex * dimension + k, index, value);
        }
      }
    }
  }

  for (unsigned int batch = 0; batch < boundaryBatchNum; batch++) {
    const unsigned int startParticle = batch * batchSize;
    const unsigned int endParticle =
        std::min((batch + 1) * batchSize, boundaryParticleNum);

    const unsigned int batchParticleNum = endParticle - startParticle;

    Kokkos::View<std::size_t **, Kokkos::DefaultExecutionSpace>
        boundaryNeighborListsDevice("boundary particle neighbor list",
                                    batchParticleNum, neighborLists_.extent(1));
    Kokkos::View<std::size_t **>::HostMirror boundaryNeighborListsHost =
        Kokkos::create_mirror_view(boundaryNeighborListsDevice);

    Kokkos::View<double *, Kokkos::DefaultExecutionSpace> boundaryEpsilonDevice(
        "boundary particle epsilon", batchParticleNum);
    Kokkos::View<double *>::HostMirror boundaryEpsilonHost =
        Kokkos::create_mirror_view(boundaryEpsilonDevice);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        boundaryParticleCoordsDevice("boundary particle coord",
                                     batchParticleNum, coords.extent(1));
    Kokkos::View<double **>::HostMirror boundaryParticleCoordsHost =
        Kokkos::create_mirror_view(boundaryParticleCoordsDevice);

    Kokkos::View<double ***, Kokkos::DefaultExecutionSpace> tangentBundleDevice(
        "tangent bundles", batchParticleNum, dimension, dimension);
    Kokkos::View<double ***>::HostMirror tangentBundleHost =
        Kokkos::create_mirror_view(tangentBundleDevice);

    for (unsigned int i = startParticle; i < endParticle; i++) {
      boundaryEpsilonHost(i - startParticle) =
          epsilon_(boundaryParticleIndex[i]);
      for (std::size_t j = 0; j <= neighborLists_(boundaryParticleIndex[i], 0);
           j++) {
        boundaryNeighborListsHost(i - startParticle, j) =
            neighborLists_(boundaryParticleIndex[i], j);
      }
      for (unsigned int j = 0; j < dimension; j++) {
        boundaryParticleCoordsHost(i - startParticle, j) =
            coords(boundaryParticleIndex[i], j);
      }
      if (dimension == 3) {
        tangentBundleHost(i - startParticle, 0, 0) = 0.0;
        tangentBundleHost(i - startParticle, 0, 1) = 0.0;
        tangentBundleHost(i - startParticle, 0, 2) = 0.0;
        tangentBundleHost(i - startParticle, 1, 0) = 0.0;
        tangentBundleHost(i - startParticle, 1, 1) = 0.0;
        tangentBundleHost(i - startParticle, 1, 2) = 0.0;
        tangentBundleHost(i - startParticle, 2, 0) =
            normal(boundaryParticleIndex[i], 0);
        tangentBundleHost(i - startParticle, 2, 1) =
            normal(boundaryParticleIndex[i], 1);
        tangentBundleHost(i - startParticle, 2, 2) =
            normal(boundaryParticleIndex[i], 2);
      }
      if (dimension == 2) {
        tangentBundleHost(i - startParticle, 0, 0) = 0.0;
        tangentBundleHost(i - startParticle, 0, 1) = 0.0;
        tangentBundleHost(i - startParticle, 1, 0) =
            normal(boundaryParticleIndex[i], 0);
        tangentBundleHost(i - startParticle, 1, 1) =
            normal(boundaryParticleIndex[i], 1);
      }
    }

    Kokkos::deep_copy(boundaryNeighborListsDevice, boundaryNeighborListsHost);
    Kokkos::deep_copy(boundaryEpsilonDevice, boundaryEpsilonHost);
    Kokkos::deep_copy(boundaryParticleCoordsDevice, boundaryParticleCoordsHost);
    Kokkos::deep_copy(tangentBundleDevice, tangentBundleHost);

    // velocity-velocity block
    {
      unsigned int blockStorageSize = dimension * dimension;
      for (unsigned int i = startParticle; i < endParticle; i++) {
        std::vector<DefaultLinearAlgebraBackend::DefaultInteger> index;
        std::vector<DefaultLinearAlgebraBackend::DefaultScalar> value;
        index.resize(1);
        index[0] = sourceIndex(boundaryParticleIndex[i]);
        value.resize(blockStorageSize);
        for (unsigned int j = 0; j < blockStorageSize; j++) {
          value[j] = 0.0;
        }
        for (unsigned int j = 0; j < dimension; j++) {
          value[j * dimension + j] = 1.0;
        }
        uu.Increment(boundaryParticleIndex[i], index, value);
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

      double generateAlphasTimer1 = MPI_Wtime();
      boundaryPressureBasis.generateAlphas(1, false);

      auto pressureSolutionSet = boundaryPressureBasis.getSolutionSetHost();
      auto boundaryPressureAlpha = pressureSolutionSet->getAlphas();
      double generateAlphasTimer2 = MPI_Wtime();
      generateAlphasDuration += generateAlphasTimer2 - generateAlphasTimer1;

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

      generateAlphasTimer1 = MPI_Wtime();
      boundaryVelocityBasis.generateAlphas(1, false);

      auto velocitySolutionSet = boundaryVelocityBasis.getSolutionSetHost();
      auto boundaryVelocityAlpha = velocitySolutionSet->getAlphas();
      generateAlphasTimer2 = MPI_Wtime();
      generateAlphasDuration += generateAlphasTimer2 - generateAlphasTimer1;

      std::vector<unsigned int> boundaryCurlCurlIndex(pow(dimension, 2));
      for (unsigned int i = 0; i < dimension; i++) {
        for (unsigned int j = 0; j < dimension; j++) {
          boundaryCurlCurlIndex[i * dimension + j] =
              velocitySolutionSet->getAlphaColumnOffset(
                  Compadre::CurlCurlOfVectorPointEvaluation, i, 0, j, 0);
        }
      }

      for (unsigned int i = startParticle; i < endParticle; i++) {
        std::vector<PetscInt> index;
        std::vector<PetscReal> value;
        const PetscInt currentParticleIndex = boundaryParticleIndex[i];
        double Aij = 0.0;
        const unsigned int numNeighbor =
            boundaryNeighborListsHost(i - startParticle, 0);

        index.resize(numNeighbor);
        value.resize(numNeighbor);

        bi_(currentParticleIndex) =
            pressureSolutionSet->getAlpha0TensorTo0Tensor(
                Compadre::LaplacianOfScalarPointEvaluation, i - startParticle,
                numNeighbor);

        for (std::size_t j = 0; j < numNeighbor; j++) {
          const PetscInt neighborParticleIndex =
              sourceIndex(boundaryNeighborListsHost(i - startParticle, j + 1));
          auto alphaIndex = pressureSolutionSet->getAlphaIndex(
              i - startParticle, boundaryPressureLaplacianIndex);
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
              sourceIndex(boundaryNeighborListsHost(i - startParticle, j + 1));
          for (unsigned int axes2 = 0; axes2 < dimension; axes2++) {
            index[j * dimension + axes2] =
                dimension * neighborParticleIndex + axes2;
            value[j * dimension + axes2] = 0.0;
            for (unsigned int axes1 = 0; axes1 < dimension; axes1++) {
              auto alphaIndex = velocitySolutionSet->getAlphaIndex(
                  i - startParticle,
                  boundaryCurlCurlIndex[dimension * axes1 + axes2]);
              value[j * dimension + axes2] +=
                  normal(currentParticleIndex, axes1) *
                  boundaryVelocityAlpha(alphaIndex + j);
            }
            value[j * dimension + axes2] *= bi_(currentParticleIndex);
          }
        }

        pu.Increment(currentParticleIndex, index, value);
      }
    }
  }

  A.Assemble();

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  if (mpiRank_ == 0) {
    printf("End of building linear system of physics: Stokes\n");
    printf("Duration of building linear system is: %.4fs\n", tEnd - tStart);
    printf("Duration of generating alphas: %.4fs\n", generateAlphasDuration);
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

  // ensure rhs on pressure has zero sum
  std::vector<double> rhs(localParticleNum * fieldDof);
  double sum = 0.0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,
                                                             localParticleNum),
      [&](const std::size_t i, double &tSum) {
        if (particleType(i) != 0) {
          rhs[localParticleNum * velocityDof + i] =
              pressureInteriorRhs_(coords(i, 0), coords(i, 1), coords(i, 2));
          for (unsigned int j = 0; j < dimension; j++) {
            rhs[i * velocityDof + j] = velocityBoundaryRhs_(
                coords(i, 0), coords(i, 1), coords(i, 2), j);
            rhs[localParticleNum * velocityDof + i] +=
                bi_(i) * normal(i, j) *
                velocityInteriorRhs_(coords(i, 0), coords(i, 1), coords(i, 2),
                                     j);
          }
        } else {
          rhs[localParticleNum * velocityDof + i] =
              pressureInteriorRhs_(coords(i, 0), coords(i, 1), coords(i, 2));
          for (unsigned int j = 0; j < dimension; j++)
            rhs[i * velocityDof + j] = velocityInteriorRhs_(
                coords(i, 0), coords(i, 1), coords(i, 2), j);
        }

        tSum += rhs[localParticleNum * velocityDof + i];
      },
      Kokkos::Sum<double>(sum));
  Kokkos::fence();
  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double average = sum / particleMgr_.GetGlobalParticleNum();
  for (std::size_t i = 0; i < localParticleNum; i++) {
    rhs[localParticleNum * velocityDof + i] -= average;
  }

  b_.Create(rhs);
  x_.Create(rhs);

  tEnd = MPI_Wtime();
  if (mpiRank_ == 0) {
    printf("Duration of building right hand side: %.4fs\n", tEnd - tStart);
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

  for (int i = 0; i < linearSystemsPtr_.size(); i++)
    std::static_pointer_cast<StokesMatrix>(linearSystemsPtr_[i])->ClearTimer();

  Equation::SolveEquation();

  if (mpiRank_ == 0) {
    for (int i = linearSystemsPtr_.size() - 1; i >= 0; i--) {
      for (unsigned int j = 0; j < linearSystemsPtr_.size() - i; j++)
        printf("  ");
      printf("Level: %4d, stokes matrix multiplication duration: %.4fs\n", i,
             std::static_pointer_cast<StokesMatrix>(linearSystemsPtr_[i])
                 ->GetMatrixMultiplicationDuration());

      for (unsigned int j = 0; j < linearSystemsPtr_.size() - i; j++)
        printf("  ");
      printf("             stokes matrix smoothing: %.4fs, velocity smoothing: "
             "%.4fs, pressure smoothing: %.4fs\n",
             std::static_pointer_cast<StokesMatrix>(linearSystemsPtr_[i])
                 ->GetMatrixSmoothingDuration(),
             std::static_pointer_cast<StokesMatrix>(linearSystemsPtr_[i])
                 ->GetVelocitySmoothingDuration(),
             std::static_pointer_cast<StokesMatrix>(linearSystemsPtr_[i])
                 ->GetPressureSmoothingDuration());
    }
  }

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

  double localDirectGradientNorm = 0.0, batchLocalDirectGradientNorm;
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
#ifdef KOKKOS_ENABLE_CUDA
  const unsigned int batchSize = (dimension == 2) ? 50000 : 10000;
#else
  const unsigned int batchSize = (dimension == 2) ? 500 : 100;
#endif
  const unsigned int batchNum = localParticleNum / batchSize +
                                ((localParticleNum % batchSize > 0) ? 1 : 0);

  double recoveredGradientDuration, reconstructedGradientDuration,
      coefficientChunkDuration, generateAlphasDuration;
  double timer1, timer2, subTimer1, subTimer2;

  generateAlphasDuration = 0.0;

  MPI_Barrier(MPI_COMM_WORLD);
  timer1 = MPI_Wtime();
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

      subTimer1 = MPI_Wtime();
      batchBasis.generateAlphas(1, true);

      Compadre::Evaluator batchEvaluator(&batchBasis);

      auto batchCoefficients =
          batchEvaluator
              .applyFullPolynomialCoefficientsBasisToDataAllComponents<
                  double **, Kokkos::DefaultHostExecutionSpace>(
                  ghostVelocityDevice);

      auto batchGradient =
          batchEvaluator.applyAlphasToDataAllComponentsAllTargetSites<
              double **, Kokkos::DefaultHostExecutionSpace>(
              ghostVelocityDevice, Compadre::GradientOfVectorPointEvaluation);
      subTimer2 = MPI_Wtime();
      generateAlphasDuration += (subTimer2 - subTimer1);

      // duplicate coefficients and gradients
      Kokkos::parallel_for(
          Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>(
              batchParticleNum, Kokkos::AUTO()),
          [&](const Kokkos::TeamPolicy<
              Kokkos::DefaultHostExecutionSpace>::member_type &teamMember) {
            const int i = teamMember.league_rank();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(teamMember, coefficientSize),
                [&](const int j) {
                  coefficientChunk(i + startParticle, j) =
                      batchCoefficients(i, j);
                });

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(teamMember, gradientComponentNum),
                [&](const int j) {
                  gradientChunk_(i + startParticle, j) = batchGradient(i, j);
                });
          });
    }
  }
  Kokkos::fence();
  MPI_Barrier(MPI_COMM_WORLD);
  timer2 = MPI_Wtime();
  coefficientChunkDuration = timer2 - timer1;

  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,
                                                             localParticleNum),
      [&](const unsigned int i, double &tLocalDirectGradientNorm) {
        double localVolume = pow(spacing(i), dimension);
        for (unsigned int axes = 0; axes < gradientComponentNum; axes++) {
          tLocalDirectGradientNorm +=
              pow(gradientChunk_(i, axes), 2) * localVolume;
        }
      },
      Kokkos::Sum<double>(localDirectGradientNorm));
  Kokkos::fence();

  MPI_Allreduce(&localDirectGradientNorm, &globalDirectGradientNorm, 1,
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  globalDirectGradientNorm = std::sqrt(globalDirectGradientNorm);

  ghost_.ApplyGhost(coefficientChunk, ghostCoefficientChunk);

  MPI_Barrier(MPI_COMM_WORLD);
  timer1 = MPI_Wtime();
  // estimate recovered gradient
  Kokkos::resize(recoveredGradientChunk_, localParticleNum,
                 gradientComponentNum);
  HostRealMatrix ghostRecoveredGradientChunk;

  Kokkos::parallel_for(
      Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>(localParticleNum,
                                                            Kokkos::AUTO()),
      [&](const Kokkos::TeamPolicy<
          Kokkos::DefaultHostExecutionSpace>::member_type &teamMember) {
        const unsigned int i = teamMember.league_rank();
        std::vector<double> recoveredGradient(gradientComponentNum);
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, gradientComponentNum),
            [&](const int j) { recoveredGradient[j] = 0.0; });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, neighborLists_(i, 0)),
            [&](const int j) {
              const std::size_t neighborParticleIndex =
                  neighborLists_(i, j + 1);

              double dX = coords(i, 0) - sourceCoords(neighborParticleIndex, 0);
              double dY = coords(i, 1) - sourceCoords(neighborParticleIndex, 1);
              double dZ = coords(i, 2) - sourceCoords(neighborParticleIndex, 2);

              double epsilon = ghostEpsilon(neighborParticleIndex);

              std::vector<double> scratchSpace(dimension * (polyOrder_ + 1));

              Discretization::PrepareScratchSpace(dimension, dX, dY, dZ,
                                                  epsilon, polyOrder_,
                                                  scratchSpace.data());

              for (unsigned int axes1 = 0; axes1 < dimension; axes1++)
                for (unsigned int axes2 = 0; axes2 < dimension; axes2++)
                  recoveredGradient[axes1 * dimension + axes2] +=
                      Discretization::CalDivFreeGrad(
                          axes1, axes2, dimension, polyOrder_, epsilon,
                          scratchSpace.data(), ghostCoefficientChunk,
                          neighborParticleIndex);
            });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, gradientComponentNum),
            [&](const int j) {
              recoveredGradientChunk_(i, j) =
                  recoveredGradient[j] / neighborLists_(i, 0);
            });
      });
  Kokkos::fence();
  MPI_Barrier(MPI_COMM_WORLD);
  timer2 = MPI_Wtime();
  recoveredGradientDuration = timer2 - timer1;

  ghost_.ApplyGhost(recoveredGradientChunk_, ghostRecoveredGradientChunk);

  // estimate the reconstructed gradient and the recovered error
  MPI_Barrier(MPI_COMM_WORLD);
  timer1 = MPI_Wtime();
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>(localParticleNum,
                                                            Kokkos::AUTO()),
      [&](const Kokkos::TeamPolicy<
          Kokkos::DefaultHostExecutionSpace>::member_type &teamMember) {
        const unsigned int i = teamMember.league_rank();
        error_(i) = 0.0;
        double localVolume = pow(spacing(i), dimension);
        double totalNeighborVolume;

        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(teamMember, neighborLists_(i, 0)),
            [&](const int j, double &tTotalNeighborVolume) {
              const std::size_t neighborParticleIndex =
                  neighborLists_(i, j + 1);
              double sourceVolume =
                  pow(ghostSpacing(neighborParticleIndex), dimension);
              tTotalNeighborVolume += sourceVolume;

              double epsilon = epsilon_(i);

              double dX = sourceCoords(neighborParticleIndex, 0) - coords(i, 0);
              double dY = sourceCoords(neighborParticleIndex, 1) - coords(i, 1);
              double dZ = sourceCoords(neighborParticleIndex, 2) - coords(i, 2);

              std::vector<double> scratchSpace(dimension * (polyOrder_ + 1));

              Discretization::PrepareScratchSpace(dimension, dX, dY, dZ,
                                                  epsilon, polyOrder_,
                                                  scratchSpace.data());

              for (unsigned int axes1 = 0; axes1 < dimension; axes1++)
                for (unsigned int axes2 = 0; axes2 < dimension; axes2++) {
                  double reconstructedGradient = Discretization::CalDivFreeGrad(
                      axes1, axes2, dimension, polyOrder_, epsilon,
                      scratchSpace.data(), coefficientChunk, i);

                  error_(i) += pow(reconstructedGradient -
                                       ghostRecoveredGradientChunk(
                                           neighborParticleIndex,
                                           axes1 * dimension + axes2),
                                   2) *
                               sourceVolume;
                }
            },
            Kokkos::Sum<double>(totalNeighborVolume));
        teamMember.team_barrier();

        error_(i) /= totalNeighborVolume;
        error_(i) = sqrt(error_(i) * localVolume);
      });
  Kokkos::fence();
  MPI_Barrier(MPI_COMM_WORLD);
  timer2 = MPI_Wtime();
  reconstructedGradientDuration = timer2 - timer1;

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
  if (mpiRank_ == 0) {
    printf("Duration of calculating error: %.4fs\n", tEnd - tStart);
    printf("Profiling:\n");
    printf("\tCoefficient duration: %.4fs\n", coefficientChunkDuration);
    printf("\t\tGenerate alphas duration: %.4fs\n", generateAlphasDuration);
    printf("\tRecovered duration: %.4fs\n", recoveredGradientDuration);
    printf("\tReconstructed duration: %.4fs\n", reconstructedGradientDuration);
  }
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