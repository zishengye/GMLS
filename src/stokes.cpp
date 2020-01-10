#include "gmls_solver.h"
#include "sparse_matrix.h"

using namespace std;
using namespace Compadre;

void GMLS_Solver::StokesEquationInitialization() {
  __field.vector.Register("fluid velocity");
  __field.scalar.Register("fluid pressure");

  __field.scalar.Register("rhs velocity");
  __field.scalar.Register("rhs pressure");
  __field.scalar.Register("x velocity");
  __field.scalar.Register("x pressure");

  __gmls.Register("pressure basis",
                  new GMLS(ScalarTaylorPolynomial,
                           StaggeredEdgeAnalyticGradientIntegralSample,
                           __polynomialOrder, __dim, "SVD", "STANDARD"));
  __gmls.Register(
      "pressure basis neumann boundary",
      new GMLS(ScalarTaylorPolynomial,
               StaggeredEdgeAnalyticGradientIntegralSample, __polynomialOrder,
               __dim, "SVD", "STANDARD", "NEUMANN_GRAD_SCALAR"));

  __gmls.Register(
      "velocity basis",
      new GMLS(DivergenceFreeVectorTaylorPolynomial, VectorPointSample,
               __polynomialOrder, __dim, "SVD", "STANDARD"));
}

void GMLS_Solver::StokesEquation() {
  static vector<vec3> &backgroundSourceCoord =
      __background.vector.GetHandle("source coord");
  static vector<int> &backgroundSourceIndex =
      __background.index.GetHandle("source index");
  static vector<vec3> &coord = __field.vector.GetHandle("coord");
  static vector<vec3> &normal = __field.vector.GetHandle("normal");
  static vector<vec3> &particleSize = __field.vector.GetHandle("size");
  static vector<int> &particleNum = __field.index.GetHandle("particle number");
  static vector<int> &particleType = __field.index.GetHandle("particle type");

  static vector<int> &attachedRigidBodyIndex =
      __field.index.GetHandle("attached rigid body index");

  GMLS *all_pressure = new GMLS(ScalarTaylorPolynomial,
                                StaggeredEdgeAnalyticGradientIntegralSample,
                                __polynomialOrder, __dim, "SVD", "STANDARD");
  GMLS *neuman_pressure = new GMLS(
      ScalarTaylorPolynomial, StaggeredEdgeAnalyticGradientIntegralSample,
      __polynomialOrder, __dim, "SVD", "STANDARD", "NEUMANN_GRAD_SCALAR");
  GMLS *all_velocity =
      new GMLS(DivergenceFreeVectorTaylorPolynomial, VectorPointSample,
               __polynomialOrder, __dim, "SVD", "STANDARD");

  GMLS &pressureBasis = *all_pressure;
  GMLS &pressureNeumannBoundaryBasis = *neuman_pressure;
  GMLS &velocityBasis = *all_velocity;

  static vector<vec3> &rigidBodyPosition =
      __rigidBody.vector.GetHandle("position");
  static vector<vec3> &rigidBodyOrientation =
      __rigidBody.vector.GetHandle("orientation");
  static vector<vec3> &rigidBodyVelocity =
      __rigidBody.vector.GetHandle("velocity");
  static vector<vec3> &rigidBodyAngularVelocity =
      __rigidBody.vector.GetHandle("angular velocity");

  int &localParticleNum = particleNum[0];
  int &globalParticleNum = particleNum[1];

  const int number_of_batches = 1;
  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "\nSolving GMLS subproblems...\n");

  // create source coords (full particle set)
  int numSourceCoords = backgroundSourceCoord.size();
  int numTargetCoords = coord.size();
  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
      "source coordinates", numSourceCoords, 3);
  Kokkos::View<double **>::HostMirror sourceCoords =
      Kokkos::create_mirror_view(sourceCoordsDevice);

  for (size_t i = 0; i < backgroundSourceCoord.size(); i++) {
    for (int j = 0; j < 3; j++) {
      sourceCoords(i, j) = backgroundSourceCoord[i][j];
    }
  }

  int neumannBoundaryNumTargetCoords = 0;
  for (int i = 0; i < localParticleNum; i++) {
    if (particleType[i] != 0) {
      neumannBoundaryNumTargetCoords++;
    }
  }

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> targetCoordsDevice(
      "target coordinates", numTargetCoords, 3);
  Kokkos::View<double **>::HostMirror targetCoords =
      Kokkos::create_mirror_view(targetCoordsDevice);
  Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
      neumannBoundaryTargetCoordsDevice("neumann target coordinates",
                                        neumannBoundaryNumTargetCoords, 3);
  Kokkos::View<double **>::HostMirror neumannBoundaryTargetCoords =
      Kokkos::create_mirror_view(neumannBoundaryTargetCoordsDevice);

  // create target coords
  vector<int> fluid2NeumannBoundary;
  int iNeumanBoundary = 0;
  for (int i = 0; i < localParticleNum; i++) {
    for (int j = 0; j < 3; j++) {
      targetCoords(i, j) = coord[i][j];
    }
    fluid2NeumannBoundary.push_back(iNeumanBoundary);
    if (particleType[i] != 0) {
      for (int j = 0; j < 3; j++) {
        neumannBoundaryTargetCoords(iNeumanBoundary, j) = coord[i][j];
      }
      iNeumanBoundary++;
    }
  }

  Kokkos::deep_copy(sourceCoordsDevice, sourceCoords);
  Kokkos::deep_copy(targetCoordsDevice, targetCoords);
  Kokkos::deep_copy(neumannBoundaryTargetCoordsDevice,
                    neumannBoundaryTargetCoords);

  // neighbor search
  auto pointCloudSearch(CreatePointCloudSearch(sourceCoords, __dim));

  const int minNeighbors = Compadre::GMLS::getNP(__polynomialOrder, __dim);

  double epsilonMultiplier = 2.7;
  int estimatedUpperBoundNumberNeighbors =
      8 * pointCloudSearch.getEstimatedNumberNeighborsUpperBound(
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
      false, targetCoords, neighborLists, epsilon, minNeighbors,
      epsilonMultiplier);

  pointCloudSearch.generateNeighborListsFromKNNSearch(
      false, neumannBoundaryTargetCoords, neumannBoundaryNeighborLists,
      neumannBoundaryEpsilon, minNeighbors, epsilonMultiplier);

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
  for (int i = 0; i < localParticleNum; i++) {
    if (particleType[i] != 0) {
      if (__dim == 3) {
        tangentBundles(counter, 0, 0) = 0.0;
        tangentBundles(counter, 0, 1) = 0.0;
        tangentBundles(counter, 0, 2) = 0.0;
        tangentBundles(counter, 1, 0) = 0.0;
        tangentBundles(counter, 1, 1) = 0.0;
        tangentBundles(counter, 1, 2) = 0.0;
        tangentBundles(counter, 2, 0) = normal[i][0];
        tangentBundles(counter, 2, 1) = normal[i][1];
        tangentBundles(counter, 2, 2) = normal[i][2];
      }
      if (__dim == 2) {
        tangentBundles(counter, 0, 0) = 0.0;
        tangentBundles(counter, 0, 1) = 0.0;
        tangentBundles(counter, 1, 0) = normal[i][0];
        tangentBundles(counter, 1, 1) = normal[i][1];
      }
      counter++;
    }
  }

  Kokkos::deep_copy(tangentBundlesDevice, tangentBundles);

  // pressure basis
  pressureBasis.setProblemData(neighborListsDevice, sourceCoordsDevice,
                               targetCoordsDevice, epsilonDevice);

  vector<TargetOperation> pressureOperation(2);
  pressureOperation[0] = LaplacianOfScalarPointEvaluation;
  pressureOperation[1] = GradientOfScalarPointEvaluation;

  pressureBasis.clearTargets();
  pressureBasis.addTargets(pressureOperation);

  pressureBasis.setWeightingType(WeightingFunctionType::Power);
  pressureBasis.setWeightingPower(2);

  pressureBasis.generateAlphas(number_of_batches);

  auto pressureAlphas = pressureBasis.getAlphas();

  const int pressureLaplacianIndex =
      pressureBasis.getAlphaColumnOffset(pressureOperation[0], 0, 0, 0, 0);
  vector<int> pressureGradientIndex;
  for (int i = 0; i < __dim; i++)
    pressureGradientIndex.push_back(
        pressureBasis.getAlphaColumnOffset(pressureOperation[1], i, 0, 0, 0));

  auto pressureNeighborListsLengths = pressureBasis.getNeighborListsLengths();

  // pressure Neumann boundary basis
  pressureNeumannBoundaryBasis.setProblemData(
      neumannBoundaryNeighborListsDevice, sourceCoordsDevice,
      neumannBoundaryTargetCoordsDevice, neumannBoundaryEpsilonDevice);

  pressureNeumannBoundaryBasis.setTangentBundle(tangentBundlesDevice);

  vector<TargetOperation> pressureNeumannBoundaryOperations(1);
  pressureNeumannBoundaryOperations[0] = LaplacianOfScalarPointEvaluation;

  pressureNeumannBoundaryBasis.clearTargets();
  pressureNeumannBoundaryBasis.addTargets(pressureNeumannBoundaryOperations);

  pressureNeumannBoundaryBasis.setWeightingType(WeightingFunctionType::Power);
  pressureNeumannBoundaryBasis.setWeightingPower(2);

  pressureNeumannBoundaryBasis.generateAlphas(number_of_batches);

  auto pressureNeumannBoundaryAlphas = pressureNeumannBoundaryBasis.getAlphas();

  const int pressureNeumannBoundaryLaplacianIndex =
      pressureNeumannBoundaryBasis.getAlphaColumnOffset(
          pressureNeumannBoundaryOperations[0], 0, 0, 0, 0);

  auto pressureNeumannBoundaryNeighborListsLengths =
      pressureNeumannBoundaryBasis.getNeighborListsLengths();

  // velocity basis
  velocityBasis.setProblemData(neighborListsDevice, sourceCoordsDevice,
                               targetCoordsDevice, epsilonDevice);

  vector<TargetOperation> velocityOperation(2);
  velocityOperation[0] = CurlCurlOfVectorPointEvaluation;
  velocityOperation[1] = GradientOfVectorPointEvaluation;

  velocityBasis.clearTargets();
  velocityBasis.addTargets(velocityOperation);

  velocityBasis.setWeightingType(WeightingFunctionType::Power);
  velocityBasis.setWeightingPower(2);

  velocityBasis.generateAlphas(number_of_batches);

  auto velocityAlphas = velocityBasis.getAlphas();

  vector<int> velocityCurlCurlIndex(pow(__dim, 2));
  for (int i = 0; i < __dim; i++) {
    for (int j = 0; j < __dim; j++) {
      velocityCurlCurlIndex[i * __dim + j] = velocityBasis.getAlphaColumnOffset(
          CurlCurlOfVectorPointEvaluation, i, 0, j, 0);
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

  const int translationDof = (__dim == 3 ? 3 : 2);
  const int rotationDof = (__dim == 3 ? 3 : 1);
  const int rigidBodyDof = (__dim == 3 ? 6 : 3);
  const int numRigidBody = rigidBodyPosition.size();

  int localVelocityDof = localParticleNum * __dim;
  int globalVelocityDof =
      globalParticleNum * __dim + rigidBodyDof * numRigidBody;
  int localPressureDof = localParticleNum;
  int globalPressureDof = globalParticleNum + 1;

  if (__myID == __MPISize - 1) {
    localVelocityDof += rigidBodyDof * numRigidBody;
    localPressureDof++;
  }

  int fieldDof = __dim + 1;
  int velocityDof = __dim;

  int localRigidBodyOffset = particleNum[__MPISize + 1] * fieldDof + 1;
  int globalRigidBodyOffset = globalParticleNum * fieldDof + 1;
  int localLagrangeMultiplierOffset = particleNum[__MPISize + 1] * fieldDof;
  int globalLagrangeMultiplierOffset = globalParticleNum * fieldDof;

  int localDof = localVelocityDof + localPressureDof;
  int globalDof = globalVelocityDof + globalPressureDof;

  PetscSparseMatrix A(localDof, localDof, globalDof);

  for (int i = 0; i < localParticleNum; i++) {
    const int currentParticleLocalIndex = i;
    const int currentParticleGlobalIndex = backgroundSourceIndex[i];

    const int iPressureLocal =
        currentParticleLocalIndex * fieldDof + velocityDof;
    const int iPressureGlobal =
        currentParticleGlobalIndex * fieldDof + velocityDof;
    // velocity block
    if (particleType[i] == 0) {
      for (int j = 0; j < velocityNeighborListsLengths(i); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];
        // inner fluid particle

        // curl curl u
        for (int axes1 = 0; axes1 < __dim; axes1++) {
          const int iVelocityLocal = __dim * currentParticleLocalIndex + axes1;
          for (int axes2 = 0; axes2 < __dim; axes2++) {
            const int iVelocityGlobal =
                fieldDof * currentParticleGlobalIndex + axes2;
            const int jVelocityGlobal =
                fieldDof * neighborParticleIndex + axes2;

            const double Lij =
                __eta * velocityAlphas(
                            i, velocityCurlCurlIndex[axes1 * __dim + axes2], j);

            A.increment(iVelocityLocal, jVelocityGlobal, Lij);
          }
        }
      }
    } else {
      // wall boundary (including particles on rigid body)
      for (int axes1 = 0; axes1 < __dim; axes1++) {
        const int iVelocityLocal = fieldDof * currentParticleLocalIndex + axes1;
        const int iVelocityGlobal =
            fieldDof * currentParticleGlobalIndex + axes1;

        A.increment(iVelocityLocal, iVelocityGlobal, 1.0);
      }

      // particles on rigid body
      if (particleType[i] == 4) {
        const int currentRigidBody = attachedRigidBodyIndex[i];
        const int currentRigidBodyLocalOffset =
            localRigidBodyOffset + rigidBodyDof * currentRigidBody;
        const int currentRigidBodyGlobalOffset =
            globalRigidBodyOffset + rigidBodyDof * currentRigidBody;

        vec3 rci = coord[i] - rigidBodyPosition[currentRigidBody];
        // non-slip condition
        // translation
        for (int axes1 = 0; axes1 < translationDof; axes1++) {
          const int iVelocityLocal =
              fieldDof * currentParticleLocalIndex + axes1;
          A.increment(iVelocityLocal, currentRigidBodyGlobalOffset + axes1,
                      -1.0);
        }

        // rotation
        for (int axes1 = 0; axes1 < __dim; axes1++) {
          const int iVelocityLocal =
              fieldDof * currentParticleLocalIndex + axes1;

          A.increment(iVelocityLocal,
                      currentRigidBodyGlobalOffset + translationDof +
                          (axes1 + 2) % rotationDof,
                      rci[(axes1 + 1) % translationDof]);
          A.increment(iVelocityLocal,
                      currentRigidBodyGlobalOffset + translationDof +
                          (axes1 + 1) % rotationDof,
                      -rci[(axes1 + 2) % translationDof]);
        }

        vec3 dA = normal[i] * pow(particleSize[i][0], __dim - 1);

        // apply pressure
        for (int axes1 = 0; axes1 < translationDof; axes1++) {
          A.outProcessIncrement(currentRigidBodyLocalOffset + axes1,
                                iPressureGlobal, -dA[axes1]);
        }
        for (int axes1 = 0; axes1 < rotationDof; axes1++) {
          A.outProcessIncrement(
              currentRigidBodyLocalOffset + translationDof + axes1,
              iPressureGlobal,
              rci[(axes1 + 2) % translationDof] *
                      dA[(axes1 + 1) % translationDof] -
                  rci[(axes1 + 1) % translationDof] *
                      dA[(axes1 + 2) % translationDof]);
        }

        for (int j = 0; j < velocityNeighborListsLengths(i); j++) {
          const int neighborParticleIndex =
              backgroundSourceIndex[neighborLists(i, j + 1)];

          for (int axes3 = 0; axes3 < __dim; axes3++) {
            const int jVelocityGlobal =
                fieldDof * neighborParticleIndex + axes3;

            double *f = new double[__dim];
            for (int axes1 = 0; axes1 < __dim; axes1++) {
              f[axes1] = 0.0;
            }

            for (int axes1 = 0; axes1 < __dim; axes1++) {
              // output component 1
              for (int axes2 = 0; axes2 < __dim; axes2++) {
                // output component 2
                const int velocityGradientAlphaIndex1 =
                    velocityGradientIndex[(axes1 * __dim + axes2) * __dim +
                                          axes3];
                const int velocityGradientAlphaIndex2 =
                    velocityGradientIndex[(axes2 * __dim + axes1) * __dim +
                                          axes3];
                const double sigma =
                    __eta * (velocityAlphas(i, velocityGradientAlphaIndex1, j) +
                             velocityAlphas(i, velocityGradientAlphaIndex2, j));

                f[axes1] += sigma * dA[axes2];
              }
            }

            // force balance
            for (int axes1 = 0; axes1 < translationDof; axes1++) {
              A.outProcessIncrement(currentRigidBodyLocalOffset + axes1,
                                    jVelocityGlobal, f[axes1]);
            }

            // torque balance
            for (int axes1 = 0; axes1 < rotationDof; axes1++) {
              A.outProcessIncrement(
                  currentRigidBodyLocalOffset + translationDof + axes1,
                  jVelocityGlobal,
                  rci[(axes1 + 1) % translationDof] *
                          f[(axes1 + 2) % translationDof] -
                      rci[(axes1 + 2) % translationDof] *
                          f[(axes1 + 1) % translationDof]);
            }
            delete[] f;
          }
        }
      }  // end of particles on rigid body
    }

    // n \cdot grad p
    if (particleType[i] != 0) {
      const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];
      const double bi = pressureNeumannBoundaryBasis.getAlpha0TensorTo0Tensor(
          LaplacianOfScalarPointEvaluation, neumannBoudnaryIndex,
          neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0));
      for (int j = 0; j < neighborLists(i, 0); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];

        for (int axes1 = 0; axes1 < __dim; axes1++) {
          for (int axes2 = 0; axes2 < __dim; axes2++) {
            const int jVelocityGlobal = __dim * neighborParticleIndex + axes2;

            const double Lij =
                __eta * velocityAlphas(
                            i, velocityCurlCurlIndex[axes1 * __dim + axes2], j);

            A.increment(iPressureLocal, jVelocityGlobal,
                        -bi * normal[i][axes1] * Lij);
          }
        }
      }
    }  // end of velocity block

    // pressure block
    if (particleType[i] == 0) {
      for (int j = 0; j < neighborLists(i, 0); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];

        const int jPressureGlobal =
            neighborParticleIndex * fieldDof + velocityDof;

        const double Aij = pressureAlphas(i, pressureLaplacianIndex, j);

        // laplacian p
        A.increment(iPressureLocal, jPressureGlobal, -Aij);
        A.increment(iPressureLocal, iPressureGlobal, Aij);

        for (int axes1 = 0; axes1 < __dim; axes1++) {
          const int iVelocityLocal = __dim * currentParticleLocalIndex + axes1;

          const double Dijx =
              pressureAlphas(i, pressureGradientIndex[axes1], j);

          // grad p
          A.increment(iVelocityLocal, jPressureGlobal, -Dijx);
          A.increment(iVelocityLocal, iPressureGlobal, Dijx);
        }
      }

      // Lagrangian multiplier
      A.increment(iPressureLocal, globalLagrangeMultiplierOffset, 1.0);
      A.outProcessIncrement(localLagrangeMultiplierOffset, iPressureGlobal,
                            1.0);
    }
    if (particleType[i] != 0) {
      const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];

      for (int j = 0; j < neighborLists(i, 0); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];

        const int jPressureGlobal =
            neighborParticleIndex * fieldDof + velocityDof;

        const double Aij = pressureNeumannBoundaryAlphas(
            neumannBoudnaryIndex, pressureNeumannBoundaryLaplacianIndex, j);

        // laplacian p
        A.increment(iPressureLocal, jPressureGlobal, -Aij);
        A.increment(iPressureLocal, iPressureGlobal, Aij);
      }
    }
    // end of pressure block
  }  // end of fluid particle loop

  if (__myID == __MPISize - 1) {
    // Lagrangian multiplier for pressure
    A.increment(localLagrangeMultiplierOffset, globalLagrangeMultiplierOffset,
                0.0);

    // for (int i = 0; i < numRigidBody; i++) {
    //   for (int j = 0; j < translationDof; j++) {
    //     LUV.increment(localRigidBodyOffset + rigidBodyDof * i + j,
    //                   globalRigidBodyOffset + rigidBodyDof * i + j, 1.0);
    //   }
    //   for (int j = 0; j < rotationDof; j++) {
    //     LUV.increment(
    //         localRigidBodyOffset + rigidBodyDof * i + translationDof + j,
    //         globalRigidBodyOffset + rigidBodyDof * i + translationDof +
    //         j, 1.0);
    //   }
    // }
  }

  A.FinalAssemble();

  delete all_pressure;
  delete all_velocity;
  delete neuman_pressure;

  PetscPrintf(PETSC_COMM_WORLD, "\nStokes Matrix Assembled\n");

  vector<double> &rhsPressure = __field.scalar.GetHandle("rhs pressure");
  vector<double> &rhsVelocity = __field.scalar.GetHandle("rhs velocity");
  vector<double> &xPressure = __field.scalar.GetHandle("x pressure");
  vector<double> &xVelocity = __field.scalar.GetHandle("x velocity");

  rhsPressure.clear();
  rhsVelocity.clear();

  rhsPressure.resize(localPressureDof);
  rhsVelocity.resize(localVelocityDof + localPressureDof);
  xPressure.resize(localPressureDof);
  xVelocity.resize(localVelocityDof + localPressureDof);

  for (int i = 0; i < localVelocityDof; i++) {
    xVelocity[i] = 0.0;
    rhsVelocity[i] = 0.0;
  }

  for (int i = 0; i < localPressureDof; i++) {
    xPressure[i] = 0.0;
    rhsPressure[i] = 0.0;
  }

  // boundary condition
  for (int i = 0; i < localParticleNum; i++) {
    if (particleType[i] != 0) {
      for (int axes = 0; axes < __dim; axes++) {
        // double Hsqr = __boundingBox[1][1] * __boundingBox[1][1];
        // rhsVelocity[__dim * i + axes] =
        //     1.5 * (1.0 - coord[i][1] * coord[i][1] / Hsqr) * double(axes ==
        //     0);
        rhsVelocity[fieldDof * i + axes] = coord[i][1] * double(axes == 0);
        // rhsVelocity[__dim * i + axes] =
        //     1.0 * double(axes == 0) *
        //     double(abs(coord[i][1] - __boundingBox[1][1]) < 1e-5);
        // rhsVelocity[__dim * i + axes] = 0.0;
      }
      // double x = coord[i][0] / __boundingBoxSize[0];
      // double y = coord[i][1] / __boundingBoxSize[1];
      // rhsVelocity[__dim * i] =
      //     cos(x * M_PI + M_PI / 2.0) * sin(y * M_PI + M_PI / 2.0);
      // rhsVelocity[__dim * i + 1] =
      //     -sin(x * M_PI + M_PI / 2.0) * cos(y * M_PI + M_PI / 2.0);
      // double x = coord[i][0] / __boundingBoxSize[0];
      // double y = coord[i][1] / __boundingBoxSize[1];
      // double z = coord[i][2] / __boundingBoxSize[2];
      // rhsVelocity[__dim * i] = cos(x * M_PI + M_PI / 2.0) *
      //                          sin(y * M_PI + M_PI / 2.0) *
      //                          sin(z * M_PI + M_PI / 2.0);
      // rhsVelocity[__dim * i + 1] = -2 * sin(x * M_PI + M_PI / 2.0) *
      //                              cos(y * M_PI + M_PI / 2.0) *
      //                              sin(z * M_PI + M_PI / 2.0);
      // rhsVelocity[__dim * i + 2] = sin(x * M_PI + M_PI / 2.0) *
      //                              sin(y * M_PI + M_PI / 2.0) *
      //                              cos(z * M_PI + M_PI / 2.0);
      rhsVelocity[fieldDof * i + velocityDof] = 0.0;
      // const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];
      // const double bi =
      // pressureNeumannBoundaryBasis.getAlpha0TensorTo0Tensor(
      //     LaplacianOfScalarPointEvaluation, neumannBoudnaryIndex,
      //     neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0));
      // rhsPressure[i] = (normal[i][0] + normal[i][1]) * bi;
      // rhsVelocity[__dim * i] = pow(coord[i][0], 2) - pow(coord[i][1], 2);
      // rhsVelocity[__dim * i + 1] = -pow(coord[i][0], 2) + pow(coord[i][1],
      // 2);
    } else {
      for (int axes = 0; axes < __dim; axes++) {
        rhsVelocity[fieldDof * i + axes] = 0.0;
      }
      rhsVelocity[fieldDof * i + velocityDof] = 0.0;
      // rhsVelocity[__dim * i] = 2 * coord[i][0];
      // rhsVelocity[__dim * i + 1] = -2 * coord[i][1];
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  A.Solve(rhsVelocity, xVelocity);
  MPI_Barrier(MPI_COMM_WORLD);
  // copy data
  static vector<vec3> &velocity = __field.vector.GetHandle("fluid velocity");
  static vector<double> &pressure = __field.scalar.GetHandle("fluid pressure");
  pressure.resize(localParticleNum);
  velocity.resize(localParticleNum);

  for (int i = 0; i < localParticleNum; i++) {
    pressure[i] = rhsVelocity[fieldDof * i + velocityDof];
    for (int axes1 = 0; axes1 < __dim; axes1++)
      velocity[i][axes1] = xVelocity[fieldDof * i + axes1];
  }

  if (__myID == __MPISize - 1) {
    for (int i = 0; i < numRigidBody; i++) {
      for (int j = 0; j < translationDof; j++) {
        rigidBodyVelocity[i][j] =
            xVelocity[localRigidBodyOffset + i * rigidBodyDof + j];
      }
      for (int j = 0; j < rotationDof; j++) {
        rigidBodyAngularVelocity[i][j] =
            xVelocity[localRigidBodyOffset + i * rigidBodyDof + translationDof +
                      j];
      }
    }
  }
}
