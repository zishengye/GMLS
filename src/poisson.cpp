#include "GMLS_solver.h"
#include "sparse_matrix.h"

#define PI 3.1415926

using namespace std;
using namespace Compadre;

void GMLS_Solver::PoissonEquation() {
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

  Kokkos::deep_copy(sourceCoordsDevice, sourceCoords);
  Kokkos::deep_copy(targetCoordsDevice, targetCoords);

  // neighbor search
  auto pointCloudSearch(CreatePointCloudSearch(sourceCoords));

  const int minNeighbors = Compadre::GMLS::getNP(__polynomialOrder, __dim);

  double epsilonMultiplier;
  epsilonMultiplier = 1.5;
  int estimatedUpperBoundNumberNeighbors =
      PI * pow(__cutoffDistance / __particleSize0[0], __dim) + 1;

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
      epsilonMultiplier, NULL, __cutoffDistance);

  Kokkos::deep_copy(neighborListsDevice, neighborLists);
  Kokkos::deep_copy(epsilonDevice, epsilon);

  PetscPrintf(PETSC_COMM_WORLD, "\nSetup neighbor lists\n");

  if (__fluid.scalarBasis != nullptr) {
    delete __fluid.scalarBasis;
  }
  __fluid.scalarBasis = new GMLS(ScalarTaylorPolynomial,
                                 StaggeredEdgeAnalyticGradientIntegralSample,
                                 __polynomialOrder, "QR", 0, __dim);

  GMLS &pressureBasis = *__fluid.scalarBasis;

  pressureBasis.setProblemData(neighborListsDevice, sourceCoordsDevice,
                               targetCoordsDevice, epsilonDevice);

  std::vector<TargetOperation> pressureOperation(1);
  pressureOperation[0] = LaplacianOfScalarPointEvaluation;

  pressureBasis.addTargets(pressureOperation);

  pressureBasis.setWeightingType(WeightingFunctionType::Power);
  pressureBasis.setWeightingPower(2);

  pressureBasis.generateAlphas();

  auto pressureAlphas = pressureBasis.getAlphas();

  const int pressureLaplacianIndex = pressureBasis.getAlphaColumnOffset(
      LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);

  auto pressureNeighborListsLengths = pressureBasis.getNeighborListsLengths();

  PetscPrintf(PETSC_COMM_WORLD, "\nGenerating Poisson Matrix...\n");

  PetscSparseMatrix A(__fluid.localParticleNum, __fluid.globalParticleNum);
  for (int i = 0; i < __fluid.localParticleNum; i++) {
    const int currentParticleLocalIndex = i;
    const int currentParticleGlobalIndex = __fluid.globalIndex[i];
    if (__fluid.particleType[i] != 0) {
      A.increment(currentParticleLocalIndex, currentParticleGlobalIndex, 1.0);
    } else {
      for (int j = 1; j < pressureNeighborListsLengths(i); j++) {
        const int neighborParticleIndex =
            __backgroundParticle.index[neighborLists(i, j + 1)];
        A.increment(currentParticleLocalIndex, neighborParticleIndex,
                    -pressureAlphas(i, pressureLaplacianIndex, j));
        A.increment(currentParticleLocalIndex, currentParticleGlobalIndex,
                    pressureAlphas(i, pressureLaplacianIndex, j));
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  A.FinalAssemble();

  PetscPrintf(PETSC_COMM_WORLD, "\nPoisson Matrix Assembled\n");

  __eq.rhs.resize(__fluid.localParticleNum);
  __eq.x.resize(__fluid.localParticleNum);
  __fluid.pressure.resize(__fluid.localParticleNum);

  for (int i = 0; i < __fluid.localParticleNum; i++) {
    if (__fluid.particleType[i] == 0) {
      __eq.rhs[i] = 1.0;
    } else {
      __eq.rhs[i] = 0.0;
    }
  }

  A.Solve(__eq.rhs, __eq.x);

  for (int i = 0; i < __fluid.localParticleNum; i++) {
    __fluid.pressure[i] = __eq.x[i];
  }
}