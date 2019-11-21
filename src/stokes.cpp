#include "GMLS_solver.h"
#include "sparse_matrix.h"

using namespace std;
using namespace Compadre;

void GMLS_Solver::StokesEquation() {
  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "\nSolving GMLS subproblems...\n");

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

  // tangent bundle for neumann boundary particles
  Kokkos::View<double ***, Kokkos::DefaultExecutionSpace> tangentBundlesDevice(
      "tangent bundles", neumannBoundarynumTargetCoords, 3, 3);
  Kokkos::View<double ***>::HostMirror tangentBundles =
      Kokkos::create_mirror_view(tangentBundlesDevice);

  int counter = 0;
  for (int i = 0; i < __particle.localParticleNum; i++) {
    if (__particle.particleType[i] != 0) {
      tangentBundles(counter, 0, 0) = 0.0;
      tangentBundles(counter, 0, 1) = __particle.normal[i][2];
      tangentBundles(counter, 0, 2) = -__particle.normal[i][1];
      tangentBundles(counter, 1, 0) = -__particle.normal[i][2];
      tangentBundles(counter, 1, 1) = 0.0;
      tangentBundles(counter, 1, 2) = __particle.normal[i][0];
      tangentBundles(counter, 2, 0) = __particle.normal[i][2];
      tangentBundles(counter, 2, 1) = -__particle.normal[i][0];
      tangentBundles(counter, 2, 2) = 0.0;
      counter++;
    }
  }

  Kokkos::deep_copy(tangentBundlesDevice, tangentBundles);

  // pressure basis
  if (__particle.scalarBasis == nullptr)
    __particle.scalarBasis =
        new GMLS(ScalarTaylorPolynomial, PointSample, __polynomialOrder, __dim,
                 "LU", "STANDARD", "NO_CONSTRAINT");
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
  for (int i = 0; i < __dim; i++)
    pressureGradientIndex.push_back(pressureBasis.getAlphaColumnOffset(
        GradientOfScalarPointEvaluation, i, 0, 0, 0));

  auto pressureNeighborListsLengths = pressureBasis.getNeighborListsLengths();

  // pressure Neumann boundary basis
  if (__particle.scalarNeumannBoundaryBasis == nullptr) {
    __particle.scalarNeumannBoundaryBasis =
        new GMLS(ScalarTaylorPolynomial, PointSample, __polynomialOrder, __dim,
                 "LU", "STANDARD", "NEUMANN_GRAD_SCALAR");
  }
  GMLS &pressureNeumannBoundaryBasis = *__particle.scalarNeumannBoundaryBasis;

  pressureNeumannBoundaryBasis.setProblemData(
      neumannBoundaryNeighborListsDevice, sourceCoordsDevice,
      neumannBoundaryTargetCoordsDevice, neumannBoundaryEpsilonDevice);
  pressureNeumannBoundaryBasis.setTangentBundle(tangentBundlesDevice);

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
  for (int i = 0; i < __dim; i++)
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
  for (int i = 0; i < __dim; i++) {
    for (int j = 0; j < __dim; j++) {
      velocityCurCurlIndex.push_back(velocityBasis.getAlphaColumnOffset(
          CurlCurlOfVectorPointEvaluation, i, 0, j, 0));
    }
  }

  auto velocityNeighborListsLengths = velocityBasis.getNeighborListsLengths();

  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "\nGenerating Stokes Matrix...\n");

  int localVelocityDOF = __particle.localParticleNum * __dim;
  int globalVelocityDOF = __particle.globalParticleNum * __dim;
  int localPressureDOF = __particle.localParticleNum;
  int globalPressureDOF = __particle.globalParticleNum + 1;

  if (__myID == __MPISize - 1)
    localPressureDOF++;

  PetscSparseMatrix LUV(localVelocityDOF, localVelocityDOF, globalVelocityDOF);
  PetscSparseMatrix GXY(localVelocityDOF, localPressureDOF, globalPressureDOF);
  PetscSparseMatrix DXY(localPressureDOF, localVelocityDOF, globalVelocityDOF);
  PetscSparseMatrix PI(localPressureDOF, localPressureDOF, globalPressureDOF);

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
        for (int axes1 = 0; axes1 < __dim; axes1++) {
          const int iVelocityLocal = __dim * currentParticleLocalIndex + axes1;
          for (int axes2 = 0; axes2 < __dim; axes2++) {
            const int iVelocityGlobal =
                __dim * currentParticleGlobalIndex + axes2;
            const int jVelocityGlobal = __dim * neighborParticleIndex + axes2;

            const double Lij =
                __eta * velocityAlphas(
                            i, velocityCurCurlIndex[axes1 * __dim + axes2], j);

            LUV.increment(iVelocityLocal, jVelocityGlobal, Lij);
            LUV.increment(iVelocityLocal, iVelocityGlobal, -Lij);
          }
        }
      } else {
        // wall boundary
        for (int axes1 = 0; axes1 < __dim; axes1++) {
          const int iVelocityLocal = __dim * currentParticleLocalIndex + axes1;
          const int iVelocityGlobal =
              __dim * currentParticleGlobalIndex + axes1;

          LUV.increment(iVelocityLocal, iVelocityGlobal, 1.0);
        }
      }
    }

    // n \cdot grad p
    if (__particle.particleType[i] != 0) {
      const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];
      for (int j = 1; j < pressureNeumannBoundaryNeighborListsLengths(
                              neumannBoudnaryIndex);
           j++) {
        const int neighborParticleIndex =
            __backgroundParticle.index[neighborLists(i, j + 1)];

        const int iPressureLocal = currentParticleLocalIndex;
        for (int axes1 = 0; axes1 < __dim; axes1++) {
          const double dpdni =
              pressureNeumannBoundaryAlphas(
                  neumannBoudnaryIndex,
                  pressureNeumannBoundaryGradientIndex[axes1], j) *
              __particle.normal[i][axes1];

          for (int axes2 = 0; axes2 < __dim; axes2++) {
            const int iVelocityGlobal =
                __dim * currentParticleGlobalIndex + axes2;
            const int jVelocityGlobal = __dim * neighborParticleIndex + axes2;

            const double Lij =
                __eta * velocityAlphas(
                            i, velocityCurCurlIndex[axes1 * __dim + axes2], j);

            DXY.increment(iPressureLocal, jVelocityGlobal, Lij * dpdni);
            DXY.increment(iPressureLocal, iVelocityGlobal, -Lij * dpdni);
          }
        }
      }
    } // end of velocity block

    // pressure block
    const int iPressureLocal = currentParticleLocalIndex;
    const int iPressureGlobal = currentParticleGlobalIndex;

    if (__particle.particleType[i] == 0) {
      for (int j = 1; j < pressureNeighborListsLengths(i); j++) {
        const int neighborParticleIndex =
            __backgroundParticle.index[neighborLists(i, j + 1)];

        const int jPressureGlobal = neighborParticleIndex;

        const double Aij = pressureAlphas(i, pressureLaplacianIndex, j);

        // laplacian p
        PI.increment(iPressureLocal, jPressureGlobal, Aij);
        PI.increment(iPressureLocal, iPressureGlobal, -Aij);

        for (int axes1 = 0; axes1 < __dim; axes1++) {
          const int iVelocityLocal = __dim * currentParticleLocalIndex + axes1;

          const double Dijx =
              pressureAlphas(i, pressureGradientIndex[axes1], j);

          // grad p
          GXY.increment(iVelocityLocal, jPressureGlobal, Dijx);
          GXY.increment(iVelocityLocal, iPressureGlobal, -Dijx);
        }
      }

      // Lagrangian multipler
      PI.increment(iPressureLocal, __particle.globalParticleNum, 1.0);
    } else {
      const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];
      for (int j = 1; j < pressureNeumannBoundaryNeighborListsLengths(
                              fluid2NeumannBoundary[i]);
           j++) {
        const int neighborParticleIndex =
            __backgroundParticle.index[neighborLists(i, j + 1)];

        const int jPressureGlobal = neighborParticleIndex;

        const double Aij = pressureNeumannBoundaryAlphas(
            neumannBoudnaryIndex, pressureNeumannBoundaryLaplacianIndex, j);

        // laplacian p
        PI.increment(iPressureLocal, jPressureGlobal, Aij);
        PI.increment(iPressureLocal, iPressureGlobal, -Aij);
      }
    } // end of pressure block
  }   // end of fluid particle loop

  // Lagrangian multipler for pressure
  if (__myID == __MPISize - 1) {
    for (int i = 0; i < __particle.globalParticleNum; i++) {
      if (__particle.particleType[i] == 0)
        PI.increment(__particle.localParticleNum, i, 1.0);
    }

    PI.increment(__particle.localParticleNum, __particle.globalParticleNum,
                 1.0);
  }

  LUV.FinalAssemble();
  DXY.FinalAssemble();
  GXY.FinalAssemble();
  PI.FinalAssemble();

  PetscPrintf(PETSC_COMM_WORLD, "\nStokes Matrix Assembled\n");

  vector<double> &rhsPressure = __eq.rhsScalar;
  vector<double> &rhsVelocity = __eq.rhsVector;
  vector<double> &xPressure = __eq.xScalar;
  vector<double> &xVelocity = __eq.xVector;

  rhsPressure.resize(localPressureDOF);
  rhsVelocity.resize(localVelocityDOF);
  xPressure.resize(localPressureDOF);
  xVelocity.resize(localVelocityDOF);

  // boundary condition
  for (int i = 0; i < __particle.localParticleNum; i++) {
    if (__particle.particleType[i] != 0) {
      for (int axes = 0; axes < __dim; axes++) {
        rhsVelocity[__dim * i + axes] =
            __particle.X[i][1] * 1.0 * double(axes == 0);
      }
    }
  }

  Solve(LUV, GXY, DXY, PI, rhsVelocity, rhsPressure, xVelocity, xPressure);

  cout << xPressure[__particle.localParticleNum] << endl;
  // copy data
  __particle.pressure.resize(__particle.localParticleNum);
  __particle.velocity.resize(__particle.localParticleNum * __dim);

  for (int i = 0; i < __particle.localParticleNum; i++) {
    __particle.pressure[i] = xPressure[i];
    for (int axes1 = 0; axes1 < __dim; axes1++)
      __particle.velocity[__dim * i + axes1] = xVelocity[__dim * i + axes1];
  }
}