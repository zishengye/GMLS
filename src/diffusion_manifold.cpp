#include "GMLS_solver.h"
#include "manifold.h"
#include "sparse_matrix.h"

#define PI 3.1415926

using namespace std;
using namespace Compadre;

void GMLS_Solver::DiffusionEquationManifold() {
  // create source and target coords
  int numSourceCoords = __backgroundParticle.coord.size();
  int numTargetCoords = __fluid.X.size();
  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
      "source coordinates", numSourceCoords, 3);
  Kokkos::View<double **>::HostMirror sourceCoords =
      Kokkos::create_mirror_view(sourceCoordsDevice);

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> targetCoordsDevice(
      "target coordinates", numTargetCoords, 3);
  Kokkos::View<double **>::HostMirror targetCoords =
      Kokkos::create_mirror_view(targetCoordsDevice);

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> normalDevice(
      "target coordinates", numTargetCoords, 3);
  Kokkos::View<double **>::HostMirror normal =
      Kokkos::create_mirror_view(normalDevice);

  for (size_t i = 0; i < __backgroundParticle.coord.size(); i++) {
    for (int j = 0; j < 3; j++) {
      sourceCoords(i, j) = __backgroundParticle.coord[i][j];
    }
  }

  for (int i = 0; i < __fluid.localParticleNum; i++) {
    for (int j = 0; j < 3; j++) {
      targetCoords(i, j) = __fluid.X[i][j];
    }
  }

  for (int i = 0; i < __fluid.localParticleNum; i++) {
    for (int j = 0; j < 2; j++) {
      normal(i, j) = __fluid.X[i][j];
    }
    normal(i, 2) = 0.0;
  }

  Kokkos::deep_copy(sourceCoordsDevice, sourceCoords);
  Kokkos::deep_copy(targetCoordsDevice, targetCoords);
  Kokkos::deep_copy(normalDevice, normal);

  // neighbor search
  auto pointCloudSearch(CreatePointCloudSearch(sourceCoords));

  const int minNeighbors = Compadre::GMLS::getNP(__polynomialOrder, __dim);

  double epsilonMultiplier;
  epsilonMultiplier = 2.2;
  // int estimatedUpperBoundNumberNeighbors =
  //     PI * pow(__cutoffDistance / __particleSize0[1], __dim) + 1;
  int estimatedUpperBoundNumberNeighbors = 200;

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> neighborListsDevice(
      "neighbor lists", numTargetCoords, estimatedUpperBoundNumberNeighbors);
  Kokkos::View<int **>::HostMirror neighborLists =
      Kokkos::create_mirror_view(neighborListsDevice);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> epsilonDevice(
      "h supports", numTargetCoords);
  Kokkos::View<double *>::HostMirror epsilon =
      Kokkos::create_mirror_view(epsilonDevice);

  pointCloudSearch.generateNeighborListsFromKNNSearch(
      targetCoords, neighborLists, epsilon, minNeighbors, __dim,
      epsilonMultiplier, NULL);

  Kokkos::deep_copy(neighborListsDevice, neighborLists);
  Kokkos::deep_copy(epsilonDevice, epsilon);

  PetscPrintf(PETSC_COMM_WORLD, "\nSetup neighbor lists\n");

  __eq.rhs.resize(__fluid.localParticleNum);
  __eq.x.resize(__fluid.localParticleNum);
  __fluid.pressure.resize(__fluid.localParticleNum);

  if (__fluid.scalarBasis != nullptr) {
    delete __fluid.scalarBasis;
  }
  __fluid.scalarBasis = new GMLS(
      ScalarTaylorPolynomial, StaggeredEdgeAnalyticGradientIntegralSample,
      __polynomialOrder, "MANIFOLD", __manifoldOrder, 3);
  // __fluid.scalarBasis =
  //     new GMLS(__manifoldOrder, "MANIFOLD", __manifoldOrder, 3);

  GMLS &pressureBasis = *__fluid.scalarBasis;

  pressureBasis.setProblemData(neighborListsDevice, sourceCoordsDevice,
                               targetCoordsDevice, epsilonDevice);

  pressureBasis.setReferenceOutwardNormalDirection(normalDevice, true);

  std::vector<TargetOperation> pressureOperation(1);
  pressureOperation[0] = ChainedStaggeredLaplacianOfScalarPointEvaluation;

  pressureBasis.addTargets(pressureOperation);

  pressureBasis.setCurvatureWeightingType(WeightingFunctionType::Power);
  pressureBasis.setCurvatureWeightingPower(2);
  pressureBasis.setWeightingType(WeightingFunctionType::Power);
  pressureBasis.setWeightingPower(2);

  pressureBasis.generateAlphas();

  auto pressureAlphas = pressureBasis.getAlphas();

  const int pressureLaplacianIndex =
      pressureBasis.getAlphaColumnOffset(pressureOperation[0], 0, 0, 0, 0);

  auto pressureNeighborListsLengths = pressureBasis.getNeighborListsLengths();

  PetscPrintf(PETSC_COMM_WORLD, "\nGenerating Diffusion Matrix...\n");

  PetscSparseMatrix A(__fluid.localParticleNum, __fluid.globalParticleNum);
  for (int i = 0; i < __fluid.localParticleNum; i++) {
    const int currentParticleLocalIndex = i;
    const int currentParticleGlobalIndex = __fluid.globalIndex[i];
    double x = __fluid.X[i][0];
    double zi = __fluid.X[i][2];
    double kappa_i, kappa_j;
    if (zi < 0.4) {
      kappa_i = 16;
    } else if (zi < 0.8) {
      kappa_i = 6;
    } else if (zi < 1.2) {
      kappa_i = 1;
    } else if (zi < 1.6) {
      kappa_i = 10;
    } else {
      kappa_i = 2;
    }
    if (x > 0) {
      A.increment(currentParticleLocalIndex, currentParticleGlobalIndex, 1.0);
    } else {
      for (int j = 1; j < pressureNeighborListsLengths(i); j++) {
        const int neighborParticleIndex =
            __backgroundParticle.index[neighborLists(i, j + 1)];

        double zj = __fluid.X[neighborLists(i, j + 1)][2];
        if (zj < 0.4) {
          kappa_j = 16;
        } else if (zj < 0.8) {
          kappa_j = 6;
        } else if (zj < 1.2) {
          kappa_j = 1;
        } else if (zj < 1.6) {
          kappa_j = 10;
        } else {
          kappa_j = 2;
        }
        A.increment(currentParticleLocalIndex, neighborParticleIndex,
                    0.5 * (kappa_i + kappa_j) *
                        pressureAlphas(i, pressureLaplacianIndex, j));
        A.increment(currentParticleLocalIndex, currentParticleGlobalIndex,
                    -0.5 * (kappa_i + kappa_j) *
                        pressureAlphas(i, pressureLaplacianIndex, j));
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  A.FinalAssemble();

  PetscPrintf(PETSC_COMM_WORLD, "\nDiffusion Matrix Assembled\n");

  __eq.rhs.resize(__fluid.localParticleNum);
  __eq.x.resize(__fluid.localParticleNum);
  __fluid.us.resize(__fluid.localParticleNum);
  __fluid.flux.resize(__fluid.localParticleNum);

  for (int i = 0; i < __fluid.localParticleNum; i++) {
    double x = __fluid.X[i][0];
    double y = __fluid.X[i][1];
    if (x > 0) {
      if (y > 0) {
        __eq.rhs[i] = asin(y);
      } else {
        __eq.rhs[i] = asin(y) + 2 * PI;
      }
    } else {
      __eq.rhs[i] = 0.0;
    }
  }

  A.Solve(__eq.rhs, __eq.x);

  for (int i = 0; i < __fluid.localParticleNum; i++) {
    __fluid.us[i] = __eq.x[i];
  }

  // post-processing
  GMLS postProcessingBasis(__polynomialOrder, "MANIFOLD", __manifoldOrder, 3);
  postProcessingBasis.setProblemData(neighborListsDevice, sourceCoordsDevice,
                                     targetCoordsDevice, epsilonDevice);

  postProcessingBasis.setReferenceOutwardNormalDirection(normalDevice, true);

  std::vector<TargetOperation> postProcessingOperation(1);
  postProcessingOperation[0] = GradientOfScalarPointEvaluation;

  postProcessingBasis.addTargets(postProcessingOperation);

  postProcessingBasis.setCurvatureWeightingType(WeightingFunctionType::Power);
  postProcessingBasis.setCurvatureWeightingPower(2);
  postProcessingBasis.setWeightingType(WeightingFunctionType::Power);
  postProcessingBasis.setWeightingPower(2);

  postProcessingBasis.generateAlphas();

  auto postProcessingAlpha = postProcessingBasis.getAlphas();

  int postProcessingGradientIndex[3];
  for (int i = 0; i < 3; i++)
    postProcessingGradientIndex[i] = postProcessingBasis.getAlphaColumnOffset(
        postProcessingOperation[0], i, 0, 0, 0);

  auto postProcessingNeighborListsLengths =
      postProcessingBasis.getNeighborListsLengths();
  for (int i = 0; i < __fluid.localParticleNum; i++) {
    for (int j = 0; j < 3; j++) {
      __fluid.flux[i][j] = 0.0;
    }

    double zi = __fluid.X[i][2];
    double kappa_i, kappa_j;
    if (zi < 0.4) {
      kappa_i = 16;
    } else if (zi < 0.8) {
      kappa_i = 6;
    } else if (zi < 1.2) {
      kappa_i = 1;
    } else if (zi < 1.6) {
      kappa_i = 10;
    } else {
      kappa_i = 2;
    }

    for (int j = 0; j < postProcessingNeighborListsLengths(i); j++) {
      const int neighborParticleIndex =
          __backgroundParticle.index[neighborLists(i, j + 1)];

      for (int k = 0; k < 3; k++) {
        __fluid.flux[i][k] +=
            kappa_i *
            postProcessingAlpha(i, postProcessingGradientIndex[k], j) *
            (__fluid.us[neighborParticleIndex] - __fluid.us[i]);
      }
    }
  }
}