#include "GMLS_solver.h"
#include "sparse_matrix.h"

using namespace std;
using namespace Compadre;

void GMLS_Solver::StokesEquation() {
  int number_of_batches = 1;
  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "\nSolving GMLS subproblems...\n");

  // create source coords (full particle set)
  int numSourceCoords = __backgroundParticle.coord.size();
  int numTargetCoords = __particle.X.size();
  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
      "source coordinates", numSourceCoords, 3);
  Kokkos::View<double **>::HostMirror sourceCoords =
      Kokkos::create_mirror_view(sourceCoordsDevice);
  Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
      neumannBoundarySourceCoordsDevice("neumann boundary source coordinates",
                                        numSourceCoords, 3);
  Kokkos::View<double **>::HostMirror neumannBoundarySourceCoords =
      Kokkos::create_mirror_view(neumannBoundarySourceCoordsDevice);

  for (size_t i = 0; i < __backgroundParticle.coord.size(); i++) {
    for (int j = 0; j < 3; j++) {
      sourceCoords(i, j) = __backgroundParticle.coord[i][j];
      neumannBoundarySourceCoords(i, j) = __backgroundParticle.coord[i][j];
    }
  }

  int neumannBoundaryNumTargetCoords = 0;
  for (int i = 0; i < __particle.localParticleNum; i++) {
    if (__particle.particleType[i] != 0) {
      neumannBoundaryNumTargetCoords++;
    }
  }

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> targetCoordsDevice(
      "target coordinates", numTargetCoords, 3);
  Kokkos::View<double **>::HostMirror targetCoords =
      Kokkos::create_mirror_view(targetCoordsDevice);

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
      neumannBoundaryTargetCoordsDevice("target coordinates",
                                        neumannBoundaryNumTargetCoords, 3);
  Kokkos::View<double **>::HostMirror neumannBoundaryTargetCoords =
      Kokkos::create_mirror_view(neumannBoundaryTargetCoordsDevice);

  // create target coords
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
  Kokkos::deep_copy(neumannBoundarySourceCoordsDevice,
                    neumannBoundarySourceCoords);
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
                                         neumannBoundaryNumTargetCoords,
                                         estimatedUpperBoundNumberNeighbors);
  Kokkos::View<int **>::HostMirror neumannBoundaryNeighborLists =
      Kokkos::create_mirror_view(neumannBoundaryNeighborListsDevice);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> epsilonDevice(
      "h supports", numTargetCoords);
  Kokkos::View<double *>::HostMirror epsilon =
      Kokkos::create_mirror_view(epsilonDevice);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace>
      neumannBoundaryEpsilonDevice("neumann boundary h supports",
                                   neumannBoundaryNumTargetCoords);
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
      "tangent bundles", neumannBoundaryNumTargetCoords, __dim, __dim);
  Kokkos::View<double ***>::HostMirror tangentBundles =
      Kokkos::create_mirror_view(tangentBundlesDevice);

  int counter = 0;
  for (int i = 0; i < __particle.localParticleNum; i++) {
    if (__particle.particleType[i] != 0) {
      if (__dim == 3) {
        tangentBundles(counter, 0, 0) = 0.0;
        tangentBundles(counter, 0, 1) = 0.0;
        tangentBundles(counter, 0, 2) = 0.0;
        tangentBundles(counter, 1, 0) = 0.0;
        tangentBundles(counter, 1, 1) = 0.0;
        tangentBundles(counter, 1, 2) = 0.0;
        tangentBundles(counter, 2, 0) = __particle.normal[i][0];
        tangentBundles(counter, 2, 1) = __particle.normal[i][1];
        tangentBundles(counter, 2, 2) = __particle.normal[i][2];
      }
      if (__dim == 2) {
        tangentBundles(counter, 0, 0) = 0.0;
        tangentBundles(counter, 0, 1) = 0.0;
        tangentBundles(counter, 1, 0) = __particle.normal[i][0];
        tangentBundles(counter, 1, 1) = __particle.normal[i][1];
      }
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

  pressureBasis.generateAlphas(number_of_batches);

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

  vector<TargetOperation> pressureNeumannBoundaryOperations(1);
  pressureNeumannBoundaryOperations[0] = LaplacianOfScalarPointEvaluation;

  pressureNeumannBoundaryBasis.addTargets(pressureNeumannBoundaryOperations);

  pressureNeumannBoundaryBasis.setWeightingType(WeightingFunctionType::Power);
  pressureNeumannBoundaryBasis.setWeightingPower(2);

  pressureNeumannBoundaryBasis.generateAlphas(number_of_batches);

  auto pressureNeumannBoundaryAlphas = pressureNeumannBoundaryBasis.getAlphas();

  const int pressureNeumannBoundaryLaplacianIndex =
      pressureNeumannBoundaryBasis.getAlphaColumnOffset(
          LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);

  auto pressureNeumannBoundaryNeighborListsLengths =
      pressureNeumannBoundaryBasis.getNeighborListsLengths();

  // velocity basis
  if (__particle.vectorBasis == nullptr)
    __particle.vectorBasis =
        new GMLS(DivergenceFreeVectorTaylorPolynomial, VectorPointSample,
                 __polynomialOrder, __dim, "LU", "STANDARD");
  GMLS &velocityBasis = *__particle.vectorBasis;

  velocityBasis.setProblemData(neighborListsDevice, sourceCoordsDevice,
                               targetCoordsDevice, epsilonDevice);

  vector<TargetOperation> velocityOperation(2);
  velocityOperation[0] = CurlCurlOfVectorPointEvaluation;
  velocityOperation[1] = GradientOfVectorPointEvaluation;

  velocityBasis.addTargets(velocityOperation);

  velocityBasis.setWeightingType(WeightingFunctionType::Power);
  velocityBasis.setWeightingPower(2);

  velocityBasis.generateAlphas(number_of_batches);

  auto velocityAlphas = velocityBasis.getAlphas();

  vector<int> velocityCurCurlIndex;
  for (int i = 0; i < __dim; i++) {
    for (int j = 0; j < __dim; j++) {
      velocityCurCurlIndex.push_back(velocityBasis.getAlphaColumnOffset(
          CurlCurlOfVectorPointEvaluation, i, 0, j, 0));
    }
  }
  vector<int> velocityGradientIndex(pow(__dim, 3));
  for (int i = 0; i < __dim; i++) {
    for (int j = 0; j < __dim; j++) {
      for (int k = 0; k < __dim; k++) {
        velocityGradientIndex[(i * __dim + j) * __dim + k] =
            velocityBasis.getAlphaColumnOffset(GradientOfVectorPointEvaluation,
                                               i, j, k, 0);
      }
    }
  }

  auto velocityNeighborListsLengths = velocityBasis.getNeighborListsLengths();

  // matrix assembly
  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "\nGenerating Stokes Matrix...\n");

  int rigidBodyDof = (__dim == 3 ? 6 : 3);
  int numRigidBody = __rigidBody.Ci_X.size();

  int localVelocityDOF = __particle.localParticleNum * __dim;
  int globalVelocityDOF =
      __particle.globalParticleNum * __dim + rigidBodyDof * numRigidBody;
  int localPressureDOF = __particle.localParticleNum;
  int globalPressureDOF = __particle.globalParticleNum + 1;

  if (__myID == __MPISize - 1) {
    localVelocityDOF += rigidBodyDof * numRigidBody;
    localPressureDOF++;
  }

  int localParticleNum = __particle.localParticleNum;
  vector<int> globalParticleNum(__MPISize);
  MPI_Allgather(&localParticleNum, 1, MPI_INT, globalParticleNum.data(), 1,
                MPI_INT, MPI_COMM_WORLD);

  int localRigidBodyOffset = globalParticleNum[__MPISize - 1] * __dim;
  int globalRigidBodyOffset = __particle.globalParticleNum * __dim;

  PetscSparseMatrix LUV(localVelocityDOF, localVelocityDOF, globalVelocityDOF);
  PetscSparseMatrix GXY(localVelocityDOF, localPressureDOF, globalPressureDOF);
  PetscSparseMatrix DXY(localPressureDOF, localVelocityDOF, globalVelocityDOF);
  PetscSparseMatrix PI(localPressureDOF, localPressureDOF, globalPressureDOF);

  for (int i = 0; i < __particle.localParticleNum; i++) {
    const int currentParticleLocalIndex = i;
    const int currentParticleGlobalIndex = __particle.globalIndex[i];
    // velocity block
    if (__particle.particleType[i] == 0) {
      for (int j = 1; j < velocityNeighborListsLengths(i); j++) {
        const int neighborParticleIndex =
            __backgroundParticle.index[neighborLists(i, j + 1)];
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
      }
    } else {
      // wall boundary (including particles on rigid body)
      for (int axes1 = 0; axes1 < __dim; axes1++) {
        const int iVelocityLocal = __dim * currentParticleLocalIndex + axes1;
        const int iVelocityGlobal = __dim * currentParticleGlobalIndex + axes1;

        LUV.increment(iVelocityLocal, iVelocityGlobal, 1.0);
      }

      // particles on rigid body
      if (__particle.particleType[i] == 4) {
        const int currentRigidBody = __particle.attachedRigidBodyIndex[i];
        const int currentRigidBodyLocalOffset =
            localRigidBodyOffset + rigidBodyDof * currentRigidBody;
        const int currentRigidBodyGlobalOffset =
            globalRigidBodyOffset + rigidBodyDof * currentRigidBody;

        vec3 rci = __particle.X[i] - __rigidBody.Ci_X[currentRigidBody];
        // translation
        for (int axes1 = 0; axes1 < __dim; axes1++) {
          const int iVelocityLocal = __dim * currentParticleLocalIndex + axes1;
          LUV.increment(iVelocityLocal, currentRigidBodyGlobalOffset + axes1,
                        -1.0);
        }

        // rotation
        if (__dim == 2) {
        }
        if (__dim == 3) {
          for (int axes1 = 0; axes1 < 3; axes1++) {
            const int iVelocityLocal =
                __dim * currentParticleLocalIndex + axes1;

            LUV.increment(iVelocityLocal,
                          currentRigidBodyGlobalOffset + 3 + (axes1 + 2) % 3,
                          rci[(axes1 + 1) % 3]);
            LUV.increment(iVelocityLocal,
                          currentRigidBodyGlobalOffset + 3 + (axes1 + 1) % 3,
                          -rci[(axes1 + 2) % 3]);
          }
        }

        const int iPressureGlobal = currentParticleGlobalIndex;

        vec3 Ndr = __particle.normal[i] * pow(__particle.d[i], __dim - 1);

        // apply pressure
        for (int axes1 = 0; axes1 < __dim; axes1++) {
          GXY.outProcessIncrement(currentRigidBodyLocalOffset + axes1,
                                  iPressureGlobal, -Ndr[axes1]);
        }

        for (int j = 1; j < velocityNeighborListsLengths(i); j++) {
          const int neighborParticleIndex =
              __backgroundParticle.index[neighborLists(i, j + 1)];

          // force balance
          for (int axes1 = 0; axes1 < __dim; axes1++) {
            // output component 1
            for (int axes2 = 0; axes2 < __dim; axes2++) {
              // output component 2
              for (int axes3 = 0; axes3 < __dim; axes3++) {
                // input component 1
                const int jVelocityGlobal =
                    __dim * neighborParticleIndex + axes3;
                const int iVelocityGlobal =
                    __dim * currentParticleGlobalIndex + axes3;
                const int velocityGradientAlphaIndex1 =
                    velocityGradientIndex[(axes1 * __dim + axes2) * __dim +
                                          axes3];
                const int velocityGradientAlphaIndex2 =
                    velocityGradientIndex[(axes2 * __dim + axes1) * __dim +
                                          axes3];
                const double sigma =
                    __eta * (velocityAlphas(i, velocityGradientAlphaIndex1, j) +
                             velocityAlphas(i, velocityGradientAlphaIndex2, j));
                LUV.outProcessIncrement(currentRigidBodyLocalOffset + axes1,
                                        jVelocityGlobal, sigma * Ndr[axes2]);
                LUV.outProcessIncrement(currentRigidBodyLocalOffset + axes1,
                                        iVelocityGlobal, -sigma * Ndr[axes2]);
              }
            }
          }

          // torque balance
          if (__dim == 2) {
          }
          if (__dim == 3) {
            for (int axes1 = 0; axes1 < __dim; axes1++) {
              // output component 1
              for (int axes2 = 0; axes2 < __dim; axes2++) {
                // output component 2
                for (int axes3 = 0; axes3 < __dim; axes3++) {
                  // input component 1
                  const int jVelocityGlobal =
                      __dim * neighborParticleIndex + axes3;
                  const int iVelocityGlobal =
                      __dim * currentParticleGlobalIndex + axes3;

                  const int velocityGradientAlphaIndex1 = velocityGradientIndex
                      [(((axes1 + 2) % 3) * __dim + axes2) * __dim + axes3];
                  const int velocityGradientAlphaIndex2 = velocityGradientIndex
                      [(((axes1 + 1) % 3) * __dim + axes2) * __dim + axes3];

                  const double f1 =
                      __eta * velocityAlphas(i, velocityGradientAlphaIndex1, j);
                  const double f2 =
                      __eta * velocityAlphas(i, velocityGradientAlphaIndex2, j);

                  LUV.outProcessIncrement(
                      currentRigidBodyLocalOffset + 3 + axes1, jVelocityGlobal,
                      rci[(axes1 + 1) % 3] * f1 * Ndr[axes2]);
                  LUV.outProcessIncrement(
                      currentRigidBodyLocalOffset + 3 + axes1, jVelocityGlobal,
                      -rci[(axes1 + 2) % 3] * f2 * Ndr[axes2]);
                  LUV.outProcessIncrement(
                      currentRigidBodyLocalOffset + 3 + axes1, iVelocityGlobal,
                      -rci[(axes1 + 1) % 3] * f1 * Ndr[axes2]);
                  LUV.outProcessIncrement(
                      currentRigidBodyLocalOffset + 3 + axes1, iVelocityGlobal,
                      rci[(axes1 + 2) % 3] * f2 * Ndr[axes2]);
                }
              }
            }
          }
        }
      }  // end of particles on rigid body
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
              pressureNeumannBoundaryBasis.getAlpha0TensorTo0Tensor(
                  LaplacianOfScalarPointEvaluation, neumannBoudnaryIndex,
                  neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0)) *
              __particle.normal[i][axes1];

          for (int axes2 = 0; axes2 < __dim; axes2++) {
            const int iVelocityGlobal =
                __dim * currentParticleGlobalIndex + axes2;
            const int jVelocityGlobal = __dim * neighborParticleIndex + axes2;

            const double Lij =
                __eta * velocityAlphas(
                            i, velocityCurCurlIndex[axes1 * __dim + axes2], j);

            DXY.increment(iPressureLocal, jVelocityGlobal, -Lij * dpdni);
            DXY.increment(iPressureLocal, iVelocityGlobal, Lij * dpdni);
          }
        }
      }
    }  // end of velocity block

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
    }
    // Lagrangian multipler
    PI.increment(iPressureLocal, __particle.globalParticleNum, 1.0);
    // end of pressure block
  }  // end of fluid particle loop

  // Lagrangian multipler for pressure
  if (__myID == __MPISize - 1) {
    for (int i = 0; i < __particle.globalParticleNum; i++) {
      PI.increment(__particle.localParticleNum, i, 1.0);

      PI.increment(__particle.localParticleNum, __particle.globalParticleNum,
                   0.0);
    }
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

  rhsPressure.clear();
  rhsVelocity.clear();

  rhsPressure.resize(localPressureDOF);
  rhsVelocity.resize(localVelocityDOF);
  xPressure.resize(localPressureDOF);
  xVelocity.resize(localVelocityDOF);

  // boundary condition
  for (int i = 0; i < __particle.localParticleNum; i++) {
    if (__particle.particleType[i] != 0) {
      for (int axes = 0; axes < __dim; axes++) {
        // double Hsqr = __boundingBox[1][1] * __boundingBox[1][1];
        // rhsVelocity[__dim * i + axes] =
        //     2.5 * (1.0 - __particle.X[i][1] * __particle.X[i][1] / Hsqr) *
        //     double(axes == 0);
        // rhsVelocity[__dim * i + axes] = __particle.X[i][1] * double(axes ==
        // 0);
        rhsVelocity[__dim * i + axes] =
            1.0 * double(axes == 0) *
            double(abs(__particle.X[i][1] - __boundingBox[1][1]) < 1e-5);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  Solve(LUV, GXY, DXY, PI, rhsVelocity, rhsPressure, xVelocity, xPressure);
  MPI_Barrier(MPI_COMM_WORLD);

  // copy data
  __particle.pressure.resize(__particle.localParticleNum);
  __particle.velocity.resize(__particle.localParticleNum * __dim);

  for (int i = 0; i < __particle.localParticleNum; i++) {
    __particle.pressure[i] = xPressure[i];
    for (int axes1 = 0; axes1 < __dim; axes1++)
      __particle.velocity[__dim * i + axes1] = xVelocity[__dim * i + axes1];
  }

  if (__myID == __MPISize - 1) {
    for (int i = 0; i < __rigidBody.Ci_X.size(); i++) {
      for (int j = 0; j < __dim; j++) {
        __rigidBody.Ci_V[i][j] = xVelocity[localRigidBodyOffset + i * 6 + j];
        __rigidBody.Ci_Omega[i][j] =
            xVelocity[localRigidBodyOffset + i * 6 + 3 + j];
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(__rigidBody.Ci_V.data(), __dim * __rigidBody.Ci_X.size(),
            MPI_DOUBLE, __MPISize - 1, MPI_COMM_WORLD);
  MPI_Bcast(__rigidBody.Ci_Omega.data(), __dim * __rigidBody.Ci_X.size(),
            MPI_DOUBLE, __MPISize - 1, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = 0; i < __rigidBody.Ci_X.size(); i++) {
    for (int j = 0; j < 3; j++) {
      __rigidBody.Ci_X[i][j] += __rigidBody.Ci_V[i][j] * __dt;
      __rigidBody.Ci_Theta[i][j] += __rigidBody.Ci_Omega[i][j] * __dt;
    }
  }
}