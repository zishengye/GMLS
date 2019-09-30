#include "GMLS_solver.h"
#include "sparse_matrix.h"

using namespace std;
using namespace Compadre;

void GMLS_Solver::StokesEquation() {
  // create source and target coords
  int numSourceCoords = __backgroundParticle.coord.size();
  int numTargetCoords = __fluid.X.size();
  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
      "source coordinates", numSourceCoords, 3);
  Kokkos::View<double **>::HostMirror sourceCoords =
      Kokkos::create_mirror_view(sourceCoordsDevice);

  int neumanBoundarynumTargetCoords = 0;
  for (int i = 0; i < __fluid.localParticleNum; i++) {
    if (__fluid.particleType[i] != 0) {
      neumanBoundarynumTargetCoords++;
    }
  }

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> targetCoordsDevice(
      "target coordinates", numTargetCoords, 3);
  Kokkos::View<double **>::HostMirror targetCoords =
      Kokkos::create_mirror_view(targetCoordsDevice);

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
      neumanBoundaryTargetCoordsDevice("target coordinates",
                                       neumanBoundarynumTargetCoords, 3);
  Kokkos::View<double **>::HostMirror neumanBoundaryTargetCoords =
      Kokkos::create_mirror_view(neumanBoundaryTargetCoordsDevice);

  for (size_t i = 0; i < __backgroundParticle.coord.size(); i++) {
    for (int j = 0; j < 3; j++) {
      sourceCoords(i, j) = __backgroundParticle.coord[i][j];
    }
  }

  vector<int> fluid2NeumanBoundary;
  int iNeumanBoundary = 0;
  for (int i = 0; i < __fluid.localParticleNum; i++) {
    for (int j = 0; j < 3; j++) {
      targetCoords(i, j) = __fluid.X[i][j];
    }
    fluid2NeumanBoundary.push_back(iNeumanBoundary);
    if (__fluid.particleType[i] != 0) {
      for (int j = 0; j < 3; j++) {
        neumanBoundaryTargetCoords(iNeumanBoundary, j) = __fluid.X[i][j];
      }
      iNeumanBoundary++;
    }
  }

  Kokkos::deep_copy(sourceCoordsDevice, sourceCoords);
  Kokkos::deep_copy(targetCoordsDevice, targetCoords);
  Kokkos::deep_copy(neumanBoundaryTargetCoordsDevice,
                    neumanBoundaryTargetCoords);

  // neighbor search
  auto pointCloudSearch(CreatePointCloudSearch(sourceCoords));

  const int minNeighbors = Compadre::GMLS::getNP(__polynomialOrder, __dim);

  double epsilonMultiplier = 1.6;
  int estimatedUpperBoundNumberNeighbors =
      pointCloudSearch.getEstimatedNumberNeighborsUpperBound(
          minNeighbors, __dim, epsilonMultiplier);

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> neighborListsDevice(
      "neighbor lists", numTargetCoords, estimatedUpperBoundNumberNeighbors);
  Kokkos::View<int **>::HostMirror neighborLists =
      Kokkos::create_mirror_view(neighborListsDevice);

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
      neumanBoundaryNeighborListsDevice("neuman boundary neighbor lists",
                                        neumanBoundarynumTargetCoords,
                                        estimatedUpperBoundNumberNeighbors);
  Kokkos::View<int **>::HostMirror neumanBoundaryNeighborLists =
      Kokkos::create_mirror_view(neumanBoundaryNeighborListsDevice);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> epsilonDevice(
      "h supports", numTargetCoords);
  Kokkos::View<double *>::HostMirror epsilon =
      Kokkos::create_mirror_view(epsilonDevice);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace>
      neumanBoundaryEpsilonDevice("neuman boundary h supports",
                                  neumanBoundarynumTargetCoords);
  Kokkos::View<double *>::HostMirror neumanBoundaryEpsilon =
      Kokkos::create_mirror_view(neumanBoundaryEpsilonDevice);

  pointCloudSearch.generateNeighborListsFromKNNSearch(
      targetCoords, neighborLists, epsilon, minNeighbors, __dim,
      epsilonMultiplier, NULL, __cutoffDistance);

  pointCloudSearch.generateNeighborListsFromKNNSearch(
      neumanBoundaryTargetCoords, neumanBoundaryNeighborLists,
      neumanBoundaryEpsilon, minNeighbors, __dim, epsilonMultiplier, NULL,
      __cutoffDistance);

  Kokkos::deep_copy(neighborListsDevice, neighborLists);
  Kokkos::deep_copy(epsilonDevice, epsilon);
  Kokkos::deep_copy(neumanBoundaryNeighborListsDevice,
                    neumanBoundaryNeighborLists);
  Kokkos::deep_copy(neumanBoundaryEpsilonDevice, neumanBoundaryEpsilon);

  GMLS pressureBasis(ScalarTaylorPolynomial, PointSample, __polynomialOrder,
                     "SVD", 0, __dim);

  pressureBasis.setProblemData(neighborListsDevice, sourceCoordsDevice,
                               targetCoordsDevice, epsilonDevice);

  vector<TargetOperation> pressureOperation(2);
  pressureOperation[0] = LaplacianOfScalarPointEvaluation;
  pressureOperation[1] = GradientOfScalarPointEvaluation;

  pressureBasis.addTargets(pressureOperation);

  pressureBasis.setWeightingType(WeightingFunctionType::Power);
  pressureBasis.setWeightingPower(2);

  pressureBasis.generateAlphas();

  auto pressureAlphas = pressureBasis.getAlphas();

  const int pressureLaplacianIndex = pressureBasis.getAlphaColumnOffset(
      LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);
  const int pressureGradientIndex = pressureBasis.getAlphaColumnOffset(
      GradientOfScalarPointEvaluation, 0, 0, 0, 0);

  auto pressureNeighborListsLengths = pressureBasis.getNeighborListsLengths();

  // Neuman boundary condition
  GMLS pressureNeumanBoundaryBasis(ScalarTaylorPolynomial,
                                   FaceNormalPointSample, __polynomialOrder,
                                   "SVD", 0, __dim);

  pressureNeumanBoundaryBasis.setProblemData(
      neumanBoundaryNeighborListsDevice, sourceCoordsDevice,
      neumanBoundaryTargetCoordsDevice, neumanBoundaryEpsilonDevice);

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
      neumanBoundaryNormalDevice("Neuman boundary particle normal",
                                 neumanBoundarynumTargetCoords, __dim * 4);
  Kokkos::View<double **>::HostMirror neumanBoundaryNormal =
      Kokkos::create_mirror_view(neumanBoundaryNormalDevice);
  iNeumanBoundary = 0;
  for (int i = 0; i < __fluid.localParticleNum; i++) {
    if (__fluid.particleType[i] != 0) {
      for (int j = 0; j < __dim; j++) {
        neumanBoundaryNormal(iNeumanBoundary, j + __dim * 2) =
            __fluid.normal[i][j];
      }
      iNeumanBoundary++;
    }
  }
  Kokkos::deep_copy(neumanBoundaryNormalDevice, neumanBoundaryNormal);
  pressureNeumanBoundaryBasis.setExtraData(neumanBoundaryNormalDevice);

  vector<TargetOperation> pressureNeumanBoundaryOperation(2);
  pressureNeumanBoundaryOperation[0] = LaplacianOfScalarPointEvaluation;
  pressureNeumanBoundaryOperation[1] = GradientOfScalarPointEvaluation;

  pressureNeumanBoundaryBasis.addTargets(pressureNeumanBoundaryOperation);

  pressureNeumanBoundaryBasis.setWeightingType(WeightingFunctionType::Power);
  pressureNeumanBoundaryBasis.setWeightingPower(2);

  pressureNeumanBoundaryBasis.generateAlphas();

  auto pressureNeumanBoundaryAlphas = pressureNeumanBoundaryBasis.getAlphas();

  const int pressureNeumanBoundaryLaplacianIndex =
      pressureNeumanBoundaryBasis.getAlphaColumnOffset(
          LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);
  const int pressureNeumanBoundaryGradientIndex =
      pressureNeumanBoundaryBasis.getAlphaColumnOffset(
          GradientOfScalarPointEvaluation, 0, 0, 0, 0);

  auto pressureNeumanBoundaryNeighborListsLengths =
      pressureNeumanBoundaryBasis.getNeighborListsLengths();

  GMLS velocityBasis(DivergenceFreeVectorTaylorPolynomial, VectorPointSample,
                     __polynomialOrder, "SVD", 0, __dim);

  velocityBasis.setProblemData(neighborListsDevice, sourceCoordsDevice,
                               targetCoordsDevice, epsilonDevice);

  vector<TargetOperation> veclocityOperation(1);
  veclocityOperation[0] = CurlCurlOfVectorPointEvaluation;

  velocityBasis.addTargets(veclocityOperation);

  velocityBasis.setWeightingType(WeightingFunctionType::Power);
  velocityBasis.setWeightingPower(2);

  velocityBasis.generateAlphas();

  auto velocityAlphas = velocityBasis.getAlphas();

  vector<int> velocityCurCurlIndex;
  if (__dim == 2) {
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        velocityCurCurlIndex.push_back(velocityBasis.getAlphaColumnOffset(
            CurlCurlOfVectorPointEvaluation, i, 0, j, 0));
      }
    }
  }
  if (__dim == 3) {
  }

  auto velocityNeighborListsLengths = velocityBasis.getNeighborListsLengths();

  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "\nGenerating Stokes Matrix...\n");

  PetscSparseMatrix A;

  if (__dim == 2) {
    A.resize(__fluid.localParticleNum * 3, __fluid.globalParticleNum * 3);

#define uOffset(x) 3 * x
#define vOffset(x) 3 * x + 1
#define pOffset(x) 3 * x + 2

    for (int i = 0; i < __fluid.localParticleNum; i++) {
      const int currentParticleLocalIndex = i;
      const int currentParticleGlobalIndex = __fluid.globalIndex[i];
      for (int j = 1; j < velocityNeighborListsLengths(i); j++) {
        const int neighborParticleIndex =
            __backgroundParticle.index[neighborLists(i, j + 1)];

        if (__fluid.particleType[i] == 0) {
          // uu
          A.increment(uOffset(currentParticleLocalIndex),
                      uOffset(neighborParticleIndex),
                      __eta * velocityAlphas(i, velocityCurCurlIndex[0], j));
          A.increment(uOffset(currentParticleLocalIndex),
                      uOffset(currentParticleGlobalIndex),
                      -__eta * velocityAlphas(i, velocityCurCurlIndex[0], j));
          // uv
          A.increment(uOffset(currentParticleLocalIndex),
                      vOffset(neighborParticleIndex),
                      __eta * velocityAlphas(i, velocityCurCurlIndex[1], j));
          A.increment(uOffset(currentParticleLocalIndex),
                      vOffset(currentParticleGlobalIndex),
                      -__eta * velocityAlphas(i, velocityCurCurlIndex[1], j));
          // vu
          A.increment(uOffset(currentParticleLocalIndex),
                      vOffset(neighborParticleIndex),
                      __eta * velocityAlphas(i, velocityCurCurlIndex[2], j));
          A.increment(uOffset(currentParticleLocalIndex),
                      vOffset(currentParticleGlobalIndex),
                      -__eta * velocityAlphas(i, velocityCurCurlIndex[2], j));
          // vv
          A.increment(uOffset(currentParticleLocalIndex),
                      vOffset(neighborParticleIndex),
                      __eta * velocityAlphas(i, velocityCurCurlIndex[3], j));
          A.increment(uOffset(currentParticleLocalIndex),
                      vOffset(currentParticleGlobalIndex),
                      -__eta * velocityAlphas(i, velocityCurCurlIndex[3], j));
        } else {
          A.increment(uOffset(currentParticleLocalIndex),
                      uOffset(currentParticleGlobalIndex), 1.0);
          A.increment(vOffset(currentParticleLocalIndex),
                      vOffset(currentParticleGlobalIndex), 1.0);
        }
      }

      // laplacian p
      if (__fluid.particleType[i] == 0) {
        for (int j = 1; j < pressureNeighborListsLengths(i); j++) {
          const int neighborParticleIndex =
              __backgroundParticle.index[neighborLists(i, j + 1)];

          A.increment(pOffset(currentParticleLocalIndex),
                      pOffset(neighborParticleIndex),
                      pressureAlphas(i, pressureLaplacianIndex, j));
          A.increment(pOffset(currentParticleLocalIndex),
                      pOffset(currentParticleGlobalIndex),
                      -pressureAlphas(i, pressureLaplacianIndex, j));
        }
      } else {
        for (int j = 1; j < pressureNeumanBoundaryNeighborListsLengths(
                                fluid2NeumanBoundary[i]);
             j++) {
          const int neighborParticleIndex =
              __backgroundParticle.index[neumanBoundaryNeighborLists(i, j + 1)];

          A.increment(pOffset(currentParticleLocalIndex),
                      pOffset(neighborParticleIndex),
                      pressureNeumanBoundaryAlphas(
                          fluid2NeumanBoundary[i],
                          pressureNeumanBoundaryLaplacianIndex, j));
          A.increment(pOffset(currentParticleLocalIndex),
                      pOffset(currentParticleGlobalIndex),
                      -pressureNeumanBoundaryAlphas(
                          fluid2NeumanBoundary[i],
                          pressureNeumanBoundaryLaplacianIndex, j));
        }
      }
    }
  }

  A.FinalAssemble();

  PetscPrintf(PETSC_COMM_WORLD, "\nStokes Matrix Assembled\n");

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

  //   A.Solve(__eq.rhs, __eq.x);

  for (int i = 0; i < __fluid.localParticleNum; i++) {
    __fluid.pressure[i] = __eq.x[i];
  }
}