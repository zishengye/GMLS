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

  if (__dim == 2) {
    __gmls.Register(
        "velocity basis",
        new GMLS(DivergenceFreeVectorTaylorPolynomial, VectorPointSample,
                 __polynomialOrder, __dim, "SVD", "STANDARD"));
  } else {
    __gmls.Register(
        "velocity basis",
        new GMLS(DivergenceFreeVectorTaylorPolynomial, VectorPointSample,
                 __polynomialOrder, __dim, "QR", "STANDARD"));
  }
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

  static GMLS &pressureBasis = __gmls.GetHandle("pressure basis");
  static GMLS &pressureNeumannBoundaryBasis =
      __gmls.GetHandle("pressure basis neumann boundary");
  static GMLS &velocityBasis = __gmls.GetHandle("velocity basis");

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

  double epsilonMultiplier = 1.5;
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

  vector<int> velocityCurCurlIndex(pow(__dim, 2));
  for (int i = 0; i < __dim; i++) {
    for (int j = 0; j < __dim; j++) {
      velocityCurCurlIndex[i * __dim + j] = velocityBasis.getAlphaColumnOffset(
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

  int localRigidBodyOffset = particleNum[__MPISize + 1] * __dim;
  int globalRigidBodyOffset = globalParticleNum * __dim;
  int lagrangeMultiplierOffset = particleNum[__MPISize + 1];

  PetscSparseMatrix LUV(localVelocityDof, localVelocityDof, globalVelocityDof);
  PetscSparseMatrix GXY(localVelocityDof, localPressureDof, globalPressureDof);
  PetscSparseMatrix DXY(localPressureDof, localVelocityDof, globalVelocityDof);
  PetscSparseMatrix PI(localPressureDof, localPressureDof, globalPressureDof);

  for (int i = 0; i < localParticleNum; i++) {
    const int currentParticleLocalIndex = i;
    const int currentParticleGlobalIndex = backgroundSourceIndex[i];

    const int iPressureLocal = currentParticleLocalIndex;
    const int iPressureGlobal = currentParticleGlobalIndex;
    // velocity block
    if (particleType[i] == 0) {
      for (int j = 1; j < velocityNeighborListsLengths(i); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];
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
          const int iVelocityLocal = __dim * currentParticleLocalIndex + axes1;
          LUV.increment(iVelocityLocal, currentRigidBodyGlobalOffset + axes1,
                        -1.0);
        }

        // rotation
        for (int axes1 = 0; axes1 < __dim; axes1++) {
          const int iVelocityLocal = __dim * currentParticleLocalIndex + axes1;

          LUV.increment(iVelocityLocal,
                        currentRigidBodyGlobalOffset + translationDof +
                            (axes1 + 2) % rotationDof,
                        rci[(axes1 + 1) % translationDof]);
          LUV.increment(iVelocityLocal,
                        currentRigidBodyGlobalOffset + translationDof +
                            (axes1 + 1) % rotationDof,
                        -rci[(axes1 + 2) % translationDof]);
        }

        const int iPressureGlobal = currentParticleGlobalIndex;

        vec3 dA = normal[i] * pow(particleSize[i][0], __dim - 1);

        // apply pressure
        for (int axes1 = 0; axes1 < translationDof; axes1++) {
          GXY.outProcessIncrement(currentRigidBodyLocalOffset + axes1,
                                  iPressureGlobal, -dA[axes1]);
        }
        for (int axes1 = 0; axes1 < rotationDof; axes1++) {
          GXY.outProcessIncrement(
              currentRigidBodyLocalOffset + translationDof + axes1,
              iPressureGlobal,
              rci[(axes1 + 2) % translationDof] *
                      dA[(axes1 + 1) % translationDof] -
                  rci[(axes1 + 1) % translationDof] *
                      dA[(axes1 + 2) % translationDof]);
        }

        for (int j = 1; j < velocityNeighborListsLengths(i); j++) {
          const int neighborParticleIndex =
              backgroundSourceIndex[neighborLists(i, j + 1)];

          // force balance
          for (int axes1 = 0; axes1 < __dim; axes1++) {
            // output component 1
            for (int axes3 = 0; axes3 < __dim; axes3++) {
              // input component 1
              const int jVelocityGlobal = __dim * neighborParticleIndex + axes3;
              const int iVelocityGlobal =
                  __dim * currentParticleGlobalIndex + axes3;

              double f = 0;
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

                f += sigma * dA[axes2];
              }
              LUV.outProcessIncrement(currentRigidBodyLocalOffset + axes1,
                                      jVelocityGlobal, f);
              LUV.outProcessIncrement(currentRigidBodyLocalOffset + axes1,
                                      iVelocityGlobal, -f);
            }
          }

          // torque balance
          for (int axes1 = 0; axes1 < __dim; axes1++) {
            // output component 1
            for (int axes3 = 0; axes3 < __dim; axes3++) {
              // input component 1
              const int jVelocityGlobal = __dim * neighborParticleIndex + axes3;
              const int iVelocityGlobal =
                  __dim * currentParticleGlobalIndex + axes3;

              double f = 0;
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

                f += sigma * dA[axes2];
              }
              LUV.outProcessIncrement(
                  currentRigidBodyLocalOffset + translationDof +
                      (axes1 + 1) % rotationDof,
                  jVelocityGlobal, f * rci[(axes1 + 2) % rotationDof]);
              LUV.outProcessIncrement(
                  currentRigidBodyLocalOffset + translationDof +
                      (axes1 + 2) % rotationDof,
                  jVelocityGlobal, -f * rci[(axes1 + 2) % rotationDof]);

              LUV.outProcessIncrement(
                  currentRigidBodyLocalOffset + translationDof +
                      (axes1 + 1) % rotationDof,
                  iVelocityGlobal, -f * rci[(axes1 + 2) % rotationDof]);
              LUV.outProcessIncrement(
                  currentRigidBodyLocalOffset + translationDof +
                      (axes1 + 2) % rotationDof,
                  iVelocityGlobal, f * rci[(axes1 + 2) % rotationDof]);
            }
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
      for (int j = 1; j < neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0);
           j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neumannBoundaryNeighborLists(
                neumannBoudnaryIndex, j + 1)];

        for (int axes1 = 0; axes1 < __dim; axes1++) {
          for (int axes2 = 0; axes2 < __dim; axes2++) {
            const int iVelocityGlobal =
                __dim * currentParticleGlobalIndex + axes2;
            const int jVelocityGlobal = __dim * neighborParticleIndex + axes2;

            const double Lij =
                __eta * velocityAlphas(
                            i, velocityCurCurlIndex[axes1 * __dim + axes2], j);

            DXY.increment(iPressureLocal, jVelocityGlobal,
                          -bi * normal[i][axes1] * Lij);
            DXY.increment(iPressureLocal, iVelocityGlobal,
                          bi * normal[i][axes1] * Lij);
          }
        }
      }
    }  // end of velocity block

    // pressure block
    if (particleType[i] == 0) {
      for (int j = 1; j < neighborLists(i, 0); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];

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

      // Lagrangian multiplier
      PI.increment(iPressureLocal, globalParticleNum, 1.0);
      PI.outProcessIncrement(lagrangeMultiplierOffset, iPressureGlobal, 1.0);
    }
    if (particleType[i] != 0) {
      const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];

      for (int j = 1; j < neighborLists(i, 0); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];

        const int jPressureGlobal = neighborParticleIndex;

        const double Aij = pressureNeumannBoundaryAlphas(
            neumannBoudnaryIndex, pressureNeumannBoundaryLaplacianIndex, j);

        // laplacian p
        PI.increment(iPressureLocal, jPressureGlobal, Aij);
        PI.increment(iPressureLocal, iPressureGlobal, -Aij);
      }
    }
    // end of pressure block
  }  // end of fluid particle loop

  if (__myID == __MPISize - 1) {
    // Lagrangian multiplier for pressure
    PI.increment(lagrangeMultiplierOffset, globalParticleNum, 0.0);

    for (int i = 0; i < numRigidBody; i++) {
      for (int j = 0; j < translationDof; j++) {
        LUV.increment(localRigidBodyOffset + rigidBodyDof * i + j,
                      globalRigidBodyOffset + rigidBodyDof * i + j, 1.0);
      }
      for (int j = 0; j < rotationDof; j++) {
        LUV.increment(
            localRigidBodyOffset + rigidBodyDof * i + translationDof + j,
            globalRigidBodyOffset + rigidBodyDof * i + translationDof + j, 1.0);
      }
    }
  }

  LUV.FinalAssemble();
  DXY.FinalAssemble();
  GXY.FinalAssemble();
  PI.FinalAssemble();

  PetscPrintf(PETSC_COMM_WORLD, "\nStokes Matrix Assembled\n");

  vector<double> &rhsPressure = __field.scalar.GetHandle("rhs pressure");
  vector<double> &rhsVelocity = __field.scalar.GetHandle("rhs velocity");
  vector<double> &xPressure = __field.scalar.GetHandle("x pressure");
  vector<double> &xVelocity = __field.scalar.GetHandle("x velocity");

  rhsPressure.clear();
  rhsVelocity.clear();

  rhsPressure.resize(localPressureDof);
  rhsVelocity.resize(localVelocityDof);
  xPressure.resize(localPressureDof);
  xVelocity.resize(localVelocityDof);

  // boundary condition
  for (int i = 0; i < localParticleNum; i++) {
    double x = coord[i][0];
    double y = coord[i][1];
    if (particleType[i] != 0) {
      for (int axes = 0; axes < __dim; axes++) {
        double Hsqr = __boundingBox[1][1] * __boundingBox[1][1];
        // rhsVelocity[__dim * i + axes] =
        //     1.5 * (1.0 - coord[i][1] * coord[i][1] / Hsqr) * double(axes ==
        //     0);
        // rhsVelocity[__dim * i + axes] = coord[i][1] * double(axes == 0);
        rhsVelocity[__dim * i + axes] =
            1.0 * double(axes == 0) *
            double(abs(coord[i][1] - __boundingBox[1][1]) < 1e-5);
        // rhsVelocity[__dim * i + axes] = 0.0;
      }
      // const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];
      // const double bi =
      // pressureNeumannBoundaryBasis.getAlpha0TensorTo0Tensor(
      //     LaplacianOfScalarPointEvaluation, neumannBoudnaryIndex,
      //     neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0));
      // rhsPressure[i] =
      //     (2 * coord[i][0] * normal[i][0] - 2 * coord[i][1] * normal[i][1]) *
      //     bi;
    } else {
      // rhsVelocity[__dim * i] = 2 * coord[i][0];
      // rhsVelocity[__dim * i + 1] = -2 * coord[i][1];
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  Solve(LUV, GXY, DXY, PI, rhsVelocity, rhsPressure, xVelocity, xPressure);
  MPI_Barrier(MPI_COMM_WORLD);

  // copy data
  static vector<vec3> &velocity = __field.vector.GetHandle("fluid velocity");
  static vector<double> &pressure = __field.scalar.GetHandle("fluid pressure");
  pressure.resize(localParticleNum);
  velocity.resize(localParticleNum);

  for (int i = 0; i < localParticleNum; i++) {
    pressure[i] = xPressure[i];
    for (int axes1 = 0; axes1 < __dim; axes1++)
      velocity[i][axes1] = xVelocity[__dim * i + axes1];
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

  vector<double> communicatedVelocity(numRigidBody * translationDof);
  vector<double> communicatedAngularVelocity(numRigidBody * rotationDof);

  if (__myID == __MPISize - 1) {
    for (int i = 0; i < numRigidBody; ++i) {
      for (int j = 0; j < translationDof; ++j) {
        communicatedVelocity[translationDof * i + j] = rigidBodyVelocity[i][j];
      }
      for (int j = 0; j < rotationDof; ++j) {
        communicatedAngularVelocity[rotationDof * i + j] =
            rigidBodyAngularVelocity[i][j];
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(communicatedVelocity.data(), translationDof * numRigidBody,
            MPI_DOUBLE, __MPISize - 1, MPI_COMM_WORLD);
  MPI_Bcast(communicatedAngularVelocity.data(), rotationDof * numRigidBody,
            MPI_DOUBLE, __MPISize - 1, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = 0; i < numRigidBody; i++) {
    for (int j = 0; j < translationDof; j++) {
      rigidBodyVelocity[i][j] = communicatedVelocity[translationDof * i + j];
      rigidBodyPosition[i][j] +=
          communicatedVelocity[translationDof * i + j] * __dt;
    }
    for (int j = 0; j < rotationDof; ++j) {
      rigidBodyAngularVelocity[i][j] =
          communicatedAngularVelocity[rotationDof * i + j];
      rigidBodyOrientation[i][j] +=
          communicatedAngularVelocity[rotationDof * i + j] * __dt;
    }
  }
}
