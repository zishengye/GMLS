#include "GMLS_solver.h"
#include "sparse_matrix.h"

using namespace std;
using namespace Compadre;

void GMLS_Solver::StokesEquation() {
  // create source and target coords
  int numSourceCoords = __backgroundParticle.coord.size();
  int numTargetCoords = __particle.X.size();
  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
      "source coordinates", numSourceCoords, 3);
  Kokkos::View<double **>::HostMirror sourceCoords =
      Kokkos::create_mirror_view(sourceCoordsDevice);

  int neumanBoundarynumTargetCoords = 0;
  for (int i = 0; i < __particle.localParticleNum; i++) {
    if (__particle.particleType[i] != 0) {
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
  for (int i = 0; i < __particle.localParticleNum; i++) {
    for (int j = 0; j < 3; j++) {
      targetCoords(i, j) = __particle.X[i][j];
    }
    fluid2NeumanBoundary.push_back(iNeumanBoundary);
    if (__particle.particleType[i] != 0) {
      for (int j = 0; j < 3; j++) {
        neumanBoundaryTargetCoords(iNeumanBoundary, j) = __particle.X[i][j];
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
                     __dim, "SVD", "STANDARD", "NO_CONSTRAINT");

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

  GMLS velocityBasis(DivergenceFreeVectorTaylorPolynomial, VectorPointSample,
                     __polynomialOrder, __dim, "SVD", "STANDARD");

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
    A.resize(__particle.localParticleNum * 3, __particle.globalParticleNum * 3);

#define uOffset(x) 3 * x
#define vOffset(x) 3 * x + 1
#define pOffset(x) 3 * x + 2

    for (int i = 0; i < __particle.localParticleNum; i++) {
      const int currentParticleLocalIndex = i;
      const int currentParticleGlobalIndex = __particle.globalIndex[i];
      for (int j = 1; j < velocityNeighborListsLengths(i); j++) {
        const int neighborParticleIndex =
            __backgroundParticle.index[neighborLists(i, j + 1)];

        if (__particle.particleType[i] == 0) {
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
      if (__particle.particleType[i] == 0) {
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
      }
    }
  }

  A.FinalAssemble();

  PetscPrintf(PETSC_COMM_WORLD, "\nStokes Matrix Assembled\n");

  __eq.rhs.resize(__particle.localParticleNum);
  __eq.x.resize(__particle.localParticleNum);
  __particle.pressure.resize(__particle.localParticleNum);

  for (int i = 0; i < __particle.localParticleNum; i++) {
    if (__particle.particleType[i] == 0) {
      __eq.rhs[i] = 1.0;
    } else {
      __eq.rhs[i] = 0.0;
    }
  }

  //   A.Solve(__eq.rhs, __eq.x);

  for (int i = 0; i < __particle.localParticleNum; i++) {
    __particle.pressure[i] = __eq.x[i];
  }
}