#include "gmls_solver.h"
#include "sparse_matrix.h"

using namespace std;
using namespace Compadre;

void GMLS_Solver::StokesEquationInitialization() {
  __field.vector.Register("fluid velocity");
  __field.scalar.Register("fluid pressure");

  __field.scalar.Register("rhs");
  __field.scalar.Register("res");

  __gmls.Register("pressure basis",
                  new GMLS(VectorTaylorPolynomial,
                           StaggeredEdgeAnalyticGradientIntegralSample,
                           __polynomialOrder, __dim, "SVD", "STANDARD"));
  __gmls.Register(
      "pressure basis neumann boundary",
      new GMLS(VectorTaylorPolynomial,
               StaggeredEdgeAnalyticGradientIntegralSample, __polynomialOrder,
               __dim, "SVD", "STANDARD", "NEUMANN_GRAD_SCALAR"));

  __gmls.Register(
      "velocity basis",
      new GMLS(DivergenceFreeVectorTaylorPolynomial, VectorPointSample,
               __polynomialOrder, __dim, "LU", "STANDARD"));
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

  auto minNeighbors = Compadre::GMLS::getNP(__polynomialOrder, __dim);

  double epsilonMultiplier = 2.5;

  int estimatedUpperBoundNumberNeighbors = pow(2 * epsilonMultiplier, __dim);

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

  for (int i = 0; i < numTargetCoords; i++) {
    epsilon(i) = __particleSize0[0] * epsilonMultiplier;
  }
  for (int i = 0; i < neumannBoundaryNumTargetCoords; i++) {
    neumannBoundaryEpsilon(i) = __particleSize0[0] * epsilonMultiplier;
  }

  pointCloudSearch.generateNeighborListsFromRadiusSearch(
      false, targetCoords, neighborLists, epsilon);

  pointCloudSearch.generateNeighborListsFromRadiusSearch(
      false, neumannBoundaryTargetCoords, neumannBoundaryNeighborLists,
      neumannBoundaryEpsilon);

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
  pressureBasis.setWeightingPower(8);

  pressureBasis.generateAlphas(number_of_batches);

  auto pressureAlphas = pressureBasis.getAlphas();

  auto pressurePreStencilWeights = pressureBasis.getPrestencilWeights();

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
  pressureNeumannBoundaryBasis.setWeightingPower(8);

  pressureNeumannBoundaryBasis.generateAlphas(number_of_batches);

  auto pressureNeumannBoundaryAlphas = pressureNeumannBoundaryBasis.getAlphas();

  auto pressureNeumannBoundaryPreStencilWeights =
      pressureNeumannBoundaryBasis.getPrestencilWeights();

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
  velocityBasis.setWeightingPower(8);

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
          const int iVelocityLocal =
              fieldDof * currentParticleLocalIndex + axes1;
          for (int axes2 = 0; axes2 < __dim; axes2++) {
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
        for (int axes1 = 0; axes1 < rotationDof; axes1++) {
          A.increment(fieldDof * currentParticleLocalIndex +
                          (axes1 + 2) % translationDof,
                      currentRigidBodyGlobalOffset + translationDof + axes1,
                      rci[(axes1 + 1) % translationDof]);
          A.increment(fieldDof * currentParticleLocalIndex +
                          (axes1 + 1) % translationDof,
                      currentRigidBodyGlobalOffset + translationDof + axes1,
                      -rci[(axes1 + 2) % translationDof]);
        }

        vec3 dA = (__dim == 3)
                      ? (normal[i] * particleSize[i][0] * particleSize[i][1])
                      : (normal[i] * particleSize[i][0]);

        // apply pressure
        for (int axes1 = 0; axes1 < translationDof; axes1++) {
          A.outProcessIncrement(currentRigidBodyLocalOffset + axes1,
                                iPressureGlobal, -dA[axes1]);
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

      for (int j = 0; j < neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0);
           j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neumannBoundaryNeighborLists(
                neumannBoudnaryIndex, j + 1)];

        for (int axes2 = 0; axes2 < __dim; axes2++) {
          double gradient = 0.0;
          const int jVelocityGlobal = fieldDof * neighborParticleIndex + axes2;
          for (int axes1 = 0; axes1 < __dim; axes1++) {
            const double Lij =
                __eta * velocityAlphas(
                            i, velocityCurlCurlIndex[axes1 * __dim + axes2], j);

            gradient += normal[i][axes1] * Lij;
          }
          A.increment(iPressureLocal, jVelocityGlobal, -bi * gradient);
        }
      }
    }  // end of velocity block

    // pressure block
    if (particleType[i] == 0) {
      for (int j = 0; j < neighborLists(i, 0); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];

        const int jPressureGlobal =
            fieldDof * neighborParticleIndex + velocityDof;

        const double Aij = pressureAlphas(i, pressureLaplacianIndex, j);

        // laplacian p
        A.increment(iPressureLocal, jPressureGlobal, -Aij);
        A.increment(iPressureLocal, iPressureGlobal, Aij);

        for (int axes1 = 0; axes1 < __dim; axes1++) {
          const int iVelocityLocal =
              fieldDof * currentParticleLocalIndex + axes1;

          const double Dijx =
              pressureAlphas(i, pressureGradientIndex[axes1], j);

          // grad p
          A.increment(iVelocityLocal, jPressureGlobal, -Dijx);
          A.increment(iVelocityLocal, iPressureGlobal, Dijx);
        }
      }
    }
    if (particleType[i] != 0) {
      const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];

      for (int j = 0; j < neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0);
           j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neumannBoundaryNeighborLists(
                neumannBoudnaryIndex, j + 1)];

        const int jPressureGlobal =
            fieldDof * neighborParticleIndex + velocityDof;

        const double Aij = pressureNeumannBoundaryAlphas(
            neumannBoudnaryIndex, pressureNeumannBoundaryLaplacianIndex, j);

        // laplacian p
        A.increment(iPressureLocal, jPressureGlobal, -Aij);
        A.increment(iPressureLocal, iPressureGlobal, Aij);
      }
    }

    // Lagrangian multiplier
    A.increment(iPressureLocal, globalLagrangeMultiplierOffset, 1.0);
    A.outProcessIncrement(localLagrangeMultiplierOffset, iPressureGlobal, 1.0);
    // end of pressure block
  }  // end of fluid particle loop1

  if (__myID == __MPISize - 1) {
    // Lagrangian multiplier for pressure
    // add a penalty factor
    A.increment(localLagrangeMultiplierOffset, globalLagrangeMultiplierOffset,
                globalParticleNum);
  }

  A.FinalAssemble();

  vector<int> neighborInclusion;
  int neighborInclusionSize;
  if (__myID == __MPISize - 1) {
    neighborInclusion.insert(neighborInclusion.end(),
                             A.__j.begin() + A.__i[localRigidBodyOffset],
                             A.__j.end());

    sort(neighborInclusion.begin(), neighborInclusion.end());

    auto it = unique(neighborInclusion.begin(), neighborInclusion.end());
    neighborInclusion.resize(distance(neighborInclusion.begin(), it));

    neighborInclusionSize = neighborInclusion.size();
  }
  MPI_Bcast(&neighborInclusionSize, 1, MPI_INT, __MPISize - 1, MPI_COMM_WORLD);
  if (__myID != __MPISize - 1) {
    neighborInclusion.resize(neighborInclusionSize);
  }
  MPI_Bcast(neighborInclusion.data(), neighborInclusionSize, MPI_INT,
            __MPISize - 1, MPI_COMM_WORLD);

  PetscPrintf(PETSC_COMM_WORLD, "\nStokes Matrix Assembled\n");

  vector<double> &rhs = __field.scalar.GetHandle("rhs");
  vector<double> &res = __field.scalar.GetHandle("res");

  rhs.clear();
  res.clear();

  rhs.resize(localDof);
  res.resize(localDof);

  for (int i = 0; i < localDof; i++) {
    rhs[i] = 0.0;
    res[i] = 0.0;
  }

  for (int i = 0; i < localParticleNum; i++) {
    if (particleType[i] != 0 && particleType[i] < 4) {
      // fluid domain boundary
      for (int axes = 0; axes < __dim; axes++) {
        // double Hsqr = __boundingBox[1][1] * __boundingBox[1][1];
        // rhs[fieldDof * i + axes] =
        //     1.5 * (1.0 - coord[i][1] * coord[i][1] / Hsqr) * double(axes ==
        //     0);
        rhs[fieldDof * i + axes] = coord[i][1] * double(axes == 0);
        // rhs[fieldDof * i + axes] =
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
      // double coordX = coord[i][0];
      // double coordY = coord[i][1];
      // rhs[fieldDof * i] = cos(2.0 * M_PI * coordX) * cos(2.0 * M_PI *
      // coordY); rhs[fieldDof * i + 1] =
      //     sin(2.0 * M_PI * coordX) * sin(2.0 * M_PI * coordY);

      // const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];
      // const double bi =
      // pressureNeumannBoundaryBasis.getAlpha0TensorTo0Tensor(
      //     LaplacianOfScalarPointEvaluation, neumannBoudnaryIndex,
      //     neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0));
      // rhs[i * fieldDof + velocityDof] =
      //     bi * (normal[i][0] * (8.0 * pow(M_PI, 2) * cos(2.0 * M_PI * coordX)
      //     *
      //                               cos(2.0 * M_PI * coordY) -
      //                           2.0 * M_PI * sin(2.0 * M_PI * coordX) *
      //                               cos(2.0 * M_PI * coordY)) +
      //           normal[i][1] * (8.0 * pow(M_PI, 2) * sin(2.0 * M_PI * coordX)
      //           *
      //                               sin(2.0 * M_PI * coordY) -
      //                           2.0 * M_PI * cos(2.0 * M_PI * coordX) *
      //                               sin(2.0 * M_PI * coordY)));

      // rhsPressure[i] = (normal[i][0] + normal[i][1]) * bi;
      // rhsVelocity[__dim * i] = pow(coord[i][0], 2) - pow(coord[i][1], 2);
      // rhsVelocity[__dim * i + 1] = -pow(coord[i][0], 2) + pow(coord[i][1],
      // 2);
    } else {
      // fluid interior
      // double coordX = coord[i][0];
      // double coordY = coord[i][1];
      // rhs[fieldDof * i] =
      //     8.0 * pow(M_PI, 2) * cos(2.0 * M_PI * coordX) *
      //         cos(2.0 * M_PI * coordY) -
      //     2.0 * M_PI * sin(2.0 * M_PI * coordX) * cos(2.0 * M_PI * coordY);
      // rhs[fieldDof * i + 1] =
      //     8.0 * pow(M_PI, 2) * sin(2.0 * M_PI * coordX) *
      //         sin(2.0 * M_PI * coordY) -
      //     2.0 * M_PI * cos(2.0 * M_PI * coordX) * sin(2.0 * M_PI * coordY);
      // for (int axes = 0; axes < __dim; axes++) {
      //   rhs[fieldDof * i + axes] = 0.0;
      // }
      // rhs[fieldDof * i + velocityDof] = -8.0 * pow(M_PI, 2) *
      //                                   cos(2 * M_PI * coord[i][0]) *
      //                                   cos(2 * M_PI * coord[i][1]);
      // rhsVelocity[__dim * i] = 2 * coord[i][0];
      // rhsVelocity[__dim * i + 1] = -2 * coord[i][1];
    }
  }

  delete all_pressure;
  delete all_velocity;
  delete neuman_pressure;

  double tStart, tEnd;

  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();
  if (numRigidBody == 0) {
    A.Solve(rhs, res, __dim);
  } else {
    A.Solve(rhs, res, __dim, numRigidBody);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "linear system solving duration: %fs\n",
              tEnd - tStart);

  // copy data
  static vector<vec3> &velocity = __field.vector.GetHandle("fluid velocity");
  static vector<double> &pressure = __field.scalar.GetHandle("fluid pressure");
  pressure.resize(localParticleNum);
  velocity.resize(localParticleNum);

  for (int i = 0; i < localParticleNum; i++) {
    pressure[i] = res[fieldDof * i + velocityDof];
    for (int axes1 = 0; axes1 < __dim; axes1++)
      velocity[i][axes1] = res[fieldDof * i + axes1];
  }

  if (__myID == __MPISize - 1) {
    for (int i = 0; i < numRigidBody; i++) {
      for (int j = 0; j < translationDof; j++) {
        rigidBodyVelocity[i][j] =
            res[localRigidBodyOffset + i * rigidBodyDof + j];
      }
      for (int j = 0; j < rotationDof; j++) {
        rigidBodyAngularVelocity[i][j] =
            res[localRigidBodyOffset + i * rigidBodyDof + translationDof + j];
      }
    }
  }
}
