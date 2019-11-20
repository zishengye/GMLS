#include "GMLS_solver.h"
#include "sparse_matrix.h"

using namespace std;
using namespace Compadre;

void GMLS_Solver::StokesEquation() {
  // create source and target coords (full particle set)
  int numSourceCoords = __backgroundParticle.coord.size();
  int numTargetCoords = __particle.X.size();
  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
      "source coordinates", numSourceCoords, 3);
  Kokkos::View<double **>::HostMirror sourceCoords =
      Kokkos::create_mirror_view(sourceCoordsDevice);

  int neumannBoundarynumTargetCoords = 0;
  for (int i = 0; i < __particle.localParticleNum; i++) {
    if (__particle.particleType[i] != 0) {
      neumannBoundarynumTargetCoords++;
    }
  }

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> targetCoordsDevice(
      "target coordinates", numTargetCoords, 3);
  Kokkos::View<double **>::HostMirror targetCoords =
      Kokkos::create_mirror_view(targetCoordsDevice);

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
      neumannBoundaryTargetCoordsDevice("target coordinates",
                                        neumannBoundarynumTargetCoords, 3);
  Kokkos::View<double **>::HostMirror neumannBoundaryTargetCoords =
      Kokkos::create_mirror_view(neumannBoundaryTargetCoordsDevice);

  for (size_t i = 0; i < __backgroundParticle.coord.size(); i++) {
    for (int j = 0; j < 3; j++) {
      sourceCoords(i, j) = __backgroundParticle.coord[i][j];
    }
  }

  // create target coords (neumann boundary particle set)
  vector<int> fluid2NeumannBoundary;
  int iNeumanBoundary = 0;
  for (int i = 0; i < __particle.localParticleNum; i++) {
    for (int j = 0; j < 3; j++) {
      targetCoords(i, j) = __particle.X[i][j];
    }
    fluid2NeumannBoundary.push_back(iNeumanBoundary);
    if (__particle.particleType[i] != 0) {
      for (int j = 0; j < 3; j++) {
        neumannBoundaryTargetCoords(iNeumanBoundary, j) = __particle.X[i][j];
      }
      iNeumanBoundary++;
    }
  }

  Kokkos::deep_copy(sourceCoordsDevice, sourceCoords);
  Kokkos::deep_copy(targetCoordsDevice, targetCoords);
  Kokkos::deep_copy(neumannBoundaryTargetCoordsDevice,
                    neumannBoundaryTargetCoords);

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
      neumannBoundaryNeighborListsDevice("neumann boundary neighbor lists",
                                         neumannBoundarynumTargetCoords,
                                         estimatedUpperBoundNumberNeighbors);
  Kokkos::View<int **>::HostMirror neumannBoundaryNeighborLists =
      Kokkos::create_mirror_view(neumannBoundaryNeighborListsDevice);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> epsilonDevice(
      "h supports", numTargetCoords);
  Kokkos::View<double *>::HostMirror epsilon =
      Kokkos::create_mirror_view(epsilonDevice);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace>
      neumannBoundaryEpsilonDevice("neumann boundary h supports",
                                   neumannBoundarynumTargetCoords);
  Kokkos::View<double *>::HostMirror neumannBoundaryEpsilon =
      Kokkos::create_mirror_view(neumannBoundaryEpsilonDevice);

  pointCloudSearch.generateNeighborListsFromKNNSearch(
      targetCoords, neighborLists, epsilon, minNeighbors, __dim,
      epsilonMultiplier, NULL, __cutoffDistance);

  pointCloudSearch.generateNeighborListsFromKNNSearch(
      neumannBoundaryTargetCoords, neumannBoundaryNeighborLists,
      neumannBoundaryEpsilon, minNeighbors, __dim, epsilonMultiplier, NULL,
      __cutoffDistance);

  Kokkos::deep_copy(neighborListsDevice, neighborLists);
  Kokkos::deep_copy(epsilonDevice, epsilon);
  Kokkos::deep_copy(neumannBoundaryNeighborListsDevice,
                    neumannBoundaryNeighborLists);
  Kokkos::deep_copy(neumannBoundaryEpsilonDevice, neumannBoundaryEpsilon);

  // pressure basis
  if (__particle.scalarBasis == nullptr)
    __particle.scalarBasis =
        new GMLS(ScalarTaylorPolynomial, PointSample, __polynomialOrder, __dim,
                 "QR", "STANDARD", "NO_CONSTRAINT");
  GMLS &pressureBasis = *__particle.scalarBasis;

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
  vector<int> pressureGradientIndex;
  if (__dim == 2)
    for (int i = 0; i < 2; i++)
      pressureGradientIndex.push_back(pressureBasis.getAlphaColumnOffset(
          GradientOfScalarPointEvaluation, i, 0, 0, 0));

  auto pressureNeighborListsLengths = pressureBasis.getNeighborListsLengths();

  // pressure Neumann boundary basis
  if (__particle.scalarNeumannBoundaryBasis == nullptr) {
    __particle.scalarNeumannBoundaryBasis =
        new GMLS(ScalarTaylorPolynomial, PointSample, __polynomialOrder, __dim,
                 "QR", "STANDARD", "NEUMANN_GRAD_SCALAR");
  }
  GMLS &pressureNeumannBoundaryBasis = *__particle.scalarNeumannBoundaryBasis;

  pressureNeumannBoundaryBasis.setProblemData(
      neumannBoundaryNeighborListsDevice, sourceCoordsDevice,
      neumannBoundaryTargetCoordsDevice, neumannBoundaryEpsilonDevice);

  vector<TargetOperation> pressureNeumannBoundaryOperations(2);
  pressureNeumannBoundaryOperations[0] = LaplacianOfScalarPointEvaluation;
  pressureNeumannBoundaryOperations[1] = GradientOfScalarPointEvaluation;

  pressureNeumannBoundaryBasis.addTargets(pressureNeumannBoundaryOperations);

  pressureNeumannBoundaryBasis.setWeightingType(WeightingFunctionType::Power);
  pressureNeumannBoundaryBasis.setWeightingPower(2);

  pressureNeumannBoundaryBasis.generateAlphas();

  auto pressureNeumannBoundaryAlphas = pressureNeumannBoundaryBasis.getAlphas();

  const int pressureNeumannBoundaryLaplacianIndex =
      pressureNeumannBoundaryBasis.getAlphaColumnOffset(
          LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);
  vector<int> pressureNeumannBoundaryGradientIndex;
  if (__dim == 2)
    for (int i = 0; i < 2; i++)
      pressureNeumannBoundaryGradientIndex.push_back(
          pressureNeumannBoundaryBasis.getAlphaColumnOffset(
              GradientOfScalarPointEvaluation, i, 0, 0, 0));

  auto pressureNeumannBoundaryNeighborListsLengths =
      pressureNeumannBoundaryBasis.getNeighborListsLengths();

  // velocity basis
  if (__particle.vectorBasis == nullptr)
    __particle.vectorBasis =
        new GMLS(DivergenceFreeVectorTaylorPolynomial, VectorPointSample,
                 __polynomialOrder, __dim, "SVD", "STANDARD");
  GMLS &velocityBasis = *__particle.vectorBasis;

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

  PetscSparseMatrix LUV, GXY, DXY, PI;

  if (__dim == 2) {
    LUV.resize(__particle.localParticleNum * 2,
               __particle.globalParticleNum * 2);
    GXY.resize(__particle.localParticleNum * 2,
               __particle.globalParticleNum + 1);
    // put the Lagrangian multipler in the last process
    if (__myID == __MPISize - 1) {
      DXY.resize(__particle.localParticleNum + 1,
                 __particle.globalParticleNum * 2);
      PI.resize(__particle.localParticleNum + 1,
                __particle.globalParticleNum + 1);
    } else {
      DXY.resize(__particle.localParticleNum, __particle.globalParticleNum * 2);
      PI.resize(__particle.localParticleNum, __particle.globalParticleNum + 1);
    }

    for (int i = 0; i < __particle.localParticleNum; i++) {
      const int currentParticleLocalIndex = i;
      const int currentParticleGlobalIndex = __particle.globalIndex[i];
      // velocity block
      for (int j = 1; j < velocityNeighborListsLengths(i); j++) {
        const int neighborParticleIndex =
            __backgroundParticle.index[neighborLists(i, j + 1)];

        if (__particle.particleType[i] == 0) {
          // inner fluid particle

          // curl curl u
          // uu
          const int uiLocal = 2 * currentParticleLocalIndex;
          const int viLocal = 2 * currentParticleLocalIndex + 1;
          const int uiGlobal = 2 * currentParticleGlobalIndex;
          const int viGlobal = 2 * currentParticleGlobalIndex + 1;
          const int ujGlobal = 2 * neighborParticleIndex;
          const int vjGlobal = 2 * neighborParticleIndex + 1;

          const double Lijuu =
              __eta * velocityAlphas(i, velocityCurCurlIndex[0], j);
          const double Lijuv =
              __eta * velocityAlphas(i, velocityCurCurlIndex[1], j);
          const double Lijvu =
              __eta * velocityAlphas(i, velocityCurCurlIndex[2], j);
          const double Lijvv =
              __eta * velocityAlphas(i, velocityCurCurlIndex[3], j);

          LUV.increment(uiLocal, ujGlobal, Lijuu);
          LUV.increment(uiLocal, uiGlobal, -Lijuu);
          LUV.increment(uiLocal, vjGlobal, Lijuv);
          LUV.increment(uiLocal, viGlobal, -Lijuv);
          LUV.increment(viLocal, ujGlobal, Lijvu);
          LUV.increment(viLocal, uiGlobal, -Lijvu);
          LUV.increment(viLocal, vjGlobal, Lijvv);
          LUV.increment(viLocal, viGlobal, -Lijvv);
        } else {
          // wall boundary
          const int uiLocal = 2 * currentParticleLocalIndex;
          const int viLocal = 2 * currentParticleLocalIndex + 1;
          const int uiGlobal = 2 * currentParticleGlobalIndex;
          const int viGlobal = 2 * currentParticleGlobalIndex + 1;

          LUV.increment(uiLocal, uiGlobal, 1.0);
          LUV.increment(viLocal, viGlobal, 1.0);
        }
      }

      // n \cdot grad p
      if (__particle.particleType[i] != 0) {
        const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];
        for (int j = 1; j < pressureNeumannBoundaryNeighborListsLengths(
                                fluid2NeumannBoundary[i]);
             j++) {
          const int neighborParticleIndex =
              __backgroundParticle.index[neighborLists(i, j + 1)];

          const int piLocal = currentParticleLocalIndex;
          const int uiGlobal = 2 * currentParticleGlobalIndex;
          const int viGlobal = 2 * currentParticleGlobalIndex + 1;
          const int ujGlobal = 2 * neighborParticleIndex;
          const int vjGlobal = 2 * neighborParticleIndex + 1;

          const double Lijuu =
              __eta * velocityAlphas(i, velocityCurCurlIndex[0], j);
          const double Lijuv =
              __eta * velocityAlphas(i, velocityCurCurlIndex[1], j);
          const double Lijvu =
              __eta * velocityAlphas(i, velocityCurCurlIndex[2], j);
          const double Lijvv =
              __eta * velocityAlphas(i, velocityCurCurlIndex[3], j);

          const double dpdni_x = pressureNeumannBoundaryAlphas(
              neumannBoudnaryIndex, pressureNeumannBoundaryGradientIndex[0], j);
          const double dpdni_y = pressureNeumannBoundaryAlphas(
              neumannBoudnaryIndex, pressureNeumannBoundaryGradientIndex[1], j);

          DXY.increment(piLocal, ujGlobal, Lijuu * dpdni_x);
          DXY.increment(piLocal, uiGlobal, -Lijuu * dpdni_x);
          DXY.increment(piLocal, vjGlobal, Lijuv * dpdni_x);
          DXY.increment(piLocal, viGlobal, -Lijuv * dpdni_x);
          DXY.increment(piLocal, ujGlobal, Lijvu * dpdni_y);
          DXY.increment(piLocal, uiGlobal, -Lijvu * dpdni_y);
          DXY.increment(piLocal, vjGlobal, Lijvv * dpdni_y);
          DXY.increment(piLocal, viGlobal, -Lijvv * dpdni_y);
        }
      }
      // end of velocity block

      // pressure block
      if (__particle.particleType[i] == 0) {
        for (int j = 1; j < pressureNeighborListsLengths(i); j++) {
          const int neighborParticleIndex =
              __backgroundParticle.index[neighborLists(i, j + 1)];

          const int piLocal = currentParticleLocalIndex;
          const int piGlobal = currentParticleGlobalIndex;
          const int pjGlobal = neighborParticleIndex;
          const int uiLocal = 2 * currentParticleLocalIndex;
          const int viLocal = 2 * currentParticleLocalIndex + 1;

          const double Aij = pressureAlphas(i, pressureLaplacianIndex, j);

          const double Dijx = pressureAlphas(i, pressureGradientIndex[0], j);
          const double Dijy = pressureAlphas(i, pressureGradientIndex[1], j);

          // laplacian p
          PI.increment(piLocal, pjGlobal, Aij);
          PI.increment(piLocal, piGlobal, -Aij);

          // grad p
          GXY.increment(uiLocal, pjGlobal, Dijx);
          GXY.increment(uiLocal, piGlobal, -Dijx);
          GXY.increment(viLocal, pjGlobal, Dijy);
          GXY.increment(viLocal, piGlobal, -Dijy);
        }
      } else {
        const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];
        for (int j = 1; j < pressureNeumannBoundaryNeighborListsLengths(
                                fluid2NeumannBoundary[i]);
             j++) {
          const int neighborParticleIndex =
              __backgroundParticle.index[neighborLists(i, j + 1)];

          const int piLocal = currentParticleLocalIndex;
          const int piGlobal = currentParticleGlobalIndex;
          const int pjGlobal = neighborParticleIndex;

          const double Aij = pressureNeumannBoundaryAlphas(
              neumannBoudnaryIndex, pressureNeumannBoundaryLaplacianIndex, j);

          // laplacian p
          PI.increment(piLocal, pjGlobal, Aij);
          PI.increment(piLocal, piGlobal, -Aij);
        }
      } // end of pressure block
    }   // end of fluid particle loop
  }

  if (__dim == 3) {
  }

  LUV.FinalAssemble();
  DXY.FinalAssemble();
  GXY.FinalAssemble();
  PI.FinalAssemble();

  PetscPrintf(PETSC_COMM_WORLD, "\nStokes Matrix Assembled\n");

  vector<double> &rhsPressure = __eq.rhsScalar;
  vector<double> &rhsVelocity = __eq.rhsVector;
  vector<double> &xPressure = __eq.xScalar;
  vector<double> &xVelocity = __eq.rhsVector;

  if (__myID == __MPISize - 1) {
    rhsPressure.resize(__particle.localParticleNum + 1);
    xPressure.resize(__particle.localParticleNum + 1);
  } else {
    rhsPressure.resize(__particle.localParticleNum);
    xPressure.resize(__particle.localParticleNum);
  }

  for (int i = 0; i < __particle.localParticleNum; i++) {
    if (__particle.particleType[i] == 0) {
      __eq.rhsScalar[i] = 1.0;
    } else {
      __eq.rhsScalar[i] = 0.0;
    }
  }

  Solve(LUV, GXY, DXY, PI, rhsVelocity, rhsPressure, xVelocity, xPressure);

  // copy data
  __particle.pressure.resize(__particle.localParticleNum);
  if (__dim == 2)
    __particle.velocity.resize(__particle.localParticleNum * 2);
  if (__dim == 3)
    __particle.velocity.resize(__particle.localParticleNum * 3);

  if (__dim == 2)
    for (int i = 0; i < __particle.localParticleNum; i++) {
      __particle.pressure[i] = __eq.xScalar[i];
      __particle.velocity[2 * i] = __eq.xVector[2 * i];
      __particle.velocity[2 * i + 1] = __eq.xVector[2 * i + 1];
    }
  if (__dim == 3)
    for (int i = 0; i < __particle.localParticleNum; i++) {
      __particle.pressure[i] = __eq.xScalar[i];
      __particle.velocity[3 * i] = __eq.xVector[3 * i];
      __particle.velocity[3 * i + 1] = __eq.xVector[3 * i + 1];
      __particle.velocity[3 * i + 2] = __eq.xVector[3 * i + 2];
    }
}