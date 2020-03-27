#include "gmls_solver.h"
#include "sparse_matrix.h"

using namespace std;
using namespace Compadre;

void GMLS_Solver::StokesEquationInitialization() {
  __field.vector.Register("fluid velocity");
  __field.scalar.Register("fluid pressure");

  __field.scalar.Register("rhs");
  __field.scalar.Register("res");

  __gmls.Register("pressure basis");
  __gmls.Register("pressure basis neumann boundary");
  __gmls.Register("velocity basis");

  auto all_pressure = __gmls.GetPointer("pressure basis");
  auto neumann_pressure = __gmls.GetPointer("pressure basis neumann boundary");
  auto all_velocity = __gmls.GetPointer("velocity basis");

  *all_pressure = nullptr;
  *neumann_pressure = nullptr;
  *all_velocity = nullptr;
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

  auto all_pressure = __gmls.GetPointer("pressure basis");
  auto neumann_pressure = __gmls.GetPointer("pressure basis neumann boundary");
  auto all_velocity = __gmls.GetPointer("velocity basis");

  if (*all_pressure != nullptr)
    delete *all_pressure;
  if (*neumann_pressure != nullptr)
    delete *neumann_pressure;
  if (*all_velocity != nullptr)
    delete *all_velocity;

  *all_pressure = new GMLS(ScalarTaylorPolynomial,
                           StaggeredEdgeAnalyticGradientIntegralSample,
                           __polynomialOrder, __dim, "SVD", "STANDARD");
  *neumann_pressure = new GMLS(
      ScalarTaylorPolynomial, StaggeredEdgeAnalyticGradientIntegralSample,
      __polynomialOrder, __dim, "SVD", "STANDARD", "NEUMANN_GRAD_SCALAR");
  *all_velocity =
      new GMLS(DivergenceFreeVectorTaylorPolynomial, VectorPointSample,
               __polynomialOrder, __dim, "SVD", "STANDARD");

  GMLS &pressureBasis = **all_pressure;
  GMLS &pressureNeumannBoundaryBasis = **neumann_pressure;
  GMLS &velocityBasis = **all_velocity;

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

  double tStart, tEnd;

  const int number_of_batches = __batchSize;
  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();
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

  double epsilonMultiplier = __polynomialOrder + 0.5;

  int estimatedUpperBoundNumberNeighbors =
      pow(2, __dim) * pow(2 * epsilonMultiplier, __dim);

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

  int counter = 0;
  double maxEpsilon = 0.0;
  __epsilon.resize(localParticleNum);
  for (int i = 0; i < numTargetCoords; i++) {
    epsilon(i) = (max(__particleSize0[0] * pow(0.5, __adaptive_step),
                      particleSize[i][0])) *
                     epsilonMultiplier +
                 1e-15;
    __epsilon[i] = epsilon(i);
    if (epsilon(i) > maxEpsilon) {
      maxEpsilon = epsilon(i);
    }
    if (particleType[i] != 0) {
      neumannBoundaryEpsilon(counter++) =
          (max(__particleSize0[0] * pow(0.5, __adaptive_step),
               particleSize[i][0])) *
              epsilonMultiplier +
          1e-15;
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &maxEpsilon, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);

  vector<double> __neumannBoundaryEpsilon(neumannBoundaryNumTargetCoords);
  for (int i = 0; i < neumannBoundaryNumTargetCoords; i++) {
    __neumannBoundaryEpsilon[i] = neumannBoundaryEpsilon(i);
  }

  vector<double> recvEpsilon, backgroundEpsilon;

  DataSwapAmongNeighbor(__epsilon, recvEpsilon);

  backgroundEpsilon.insert(backgroundEpsilon.end(), __epsilon.begin(),
                           __epsilon.end());
  backgroundEpsilon.insert(backgroundEpsilon.end(), recvEpsilon.begin(),
                           recvEpsilon.end());

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace>
      backgroundEpsilonKokkosDevice("background h supports",
                                    backgroundEpsilon.size());
  Kokkos::View<double *>::HostMirror backgroundEpsilonKokkos =
      Kokkos::create_mirror_view(backgroundEpsilonKokkosDevice);

  for (int i = 0; i < backgroundEpsilon.size(); i++) {
    backgroundEpsilonKokkos(i) = backgroundEpsilon[i];
  }

  Kokkos::deep_copy(backgroundEpsilonKokkosDevice, backgroundEpsilonKokkos);

  // pointCloudSearch.generateNeighborListsFromRadiusSearch(
  //     false, targetCoords, neighborLists, epsilon);

  // counter = 0;
  // for (int i = 0; i < numTargetCoords; i++) {
  //   if (particleType[i] != 0) {
  //     neumannBoundaryNeighborLists(counter, 0) = neighborLists(i, 0);
  //     for (int j = 0; j < neighborLists(i, 0); j++) {
  //       neumannBoundaryNeighborLists(counter, j + 1) = neighborLists(i, j +
  //       1);
  //     }
  //     counter++;
  //   }
  // }

  pointCloudSearch.generateSymmetricNeighborListsFromRadiusSearch(
      false, targetCoords, neighborLists, backgroundEpsilonKokkos, 0.0,
      maxEpsilon);

  counter = 0;
  for (int i = 0; i < numTargetCoords; i++) {
    if (particleType[i] != 0) {
      neumannBoundaryNeighborLists(counter, 0) = neighborLists(i, 0);
      for (int j = 0; j < neighborLists(i, 0); j++) {
        neumannBoundaryNeighborLists(counter, j + 1) = neighborLists(i, j + 1);
      }
      counter++;
    }
  }

  Kokkos::deep_copy(neighborListsDevice, neighborLists);
  Kokkos::deep_copy(epsilonDevice, epsilon);
  Kokkos::deep_copy(neumannBoundaryNeighborListsDevice,
                    neumannBoundaryNeighborLists);
  Kokkos::deep_copy(neumannBoundaryEpsilonDevice, neumannBoundaryEpsilon);

  __neighborLists.resize(numTargetCoords);
  for (size_t i = 0; i < __neighborLists.size(); i++) {
    __neighborLists[i].resize(estimatedUpperBoundNumberNeighbors);

    for (size_t j = 0; j < estimatedUpperBoundNumberNeighbors; j++) {
      __neighborLists[i][j] = neighborLists(i, j);
    }
  }

  // tangent bundle for neumann boundary particles
  Kokkos::View<double ***, Kokkos::DefaultExecutionSpace> tangentBundlesDevice(
      "tangent bundles", neumannBoundaryNumTargetCoords, __dim, __dim);
  Kokkos::View<double ***>::HostMirror tangentBundles =
      Kokkos::create_mirror_view(tangentBundlesDevice);

  counter = 0;
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
  pressureBasis.setWeightingPower(__weightFuncOrder);

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
  pressureNeumannBoundaryBasis.setWeightingPower(__weightFuncOrder);

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

  vector<TargetOperation> velocityOperation(3);
  velocityOperation[0] = CurlCurlOfVectorPointEvaluation;
  velocityOperation[1] = GradientOfVectorPointEvaluation;
  velocityOperation[2] = ScalarPointEvaluation;

  velocityBasis.clearTargets();
  velocityBasis.addTargets(velocityOperation);

  velocityBasis.setWeightingType(WeightingFunctionType::Power);
  velocityBasis.setWeightingPower(__weightFuncOrder);

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

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "GMLS solving duration: %fs\n", tEnd - tStart);

  // matrix assembly
  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "\nGenerating Stokes Matrix...\n");

  const int translationDof = (__dim == 3 ? 3 : 2);
  const int rotationDof = (__dim == 3 ? 3 : 1);
  const int rigidBodyDof = (__dim == 3 ? 6 : 3);
  const int numRigidBody = rigidBodyPosition.size();

  int fieldDof = __dim + 1;
  int velocityDof = __dim;

  int localVelocityDof = localParticleNum * __dim;
  int globalVelocityDof =
      globalParticleNum * __dim + rigidBodyDof * numRigidBody;
  int localPressureDof = localParticleNum;
  int globalPressureDof = globalParticleNum + fieldDof;

  if (__myID == __MPISize - 1) {
    localVelocityDof += velocityDof + rigidBodyDof * numRigidBody;
    localPressureDof += 1;
  }

  int localRigidBodyOffset = particleNum[__MPISize + 1] * fieldDof + fieldDof;
  int globalRigidBodyOffset = globalParticleNum * fieldDof + fieldDof;
  int localOutProcessOffset = particleNum[__MPISize + 1] * fieldDof;
  int globalOutProcessOffset = globalParticleNum * fieldDof;
  int localLagrangeMultiplierOffset =
      particleNum[__MPISize + 1] * fieldDof + velocityDof;
  int globalLagrangeMultiplierOffset =
      globalParticleNum * fieldDof + velocityDof;

  int localDof = localVelocityDof + localPressureDof;
  int globalDof = globalVelocityDof + globalPressureDof;

  int outProcessRow = rigidBodyDof * numRigidBody + fieldDof;

  PetscSparseMatrix A(localDof, localDof, globalDof, outProcessRow,
                      localOutProcessOffset);

  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();

  // compute matrix graph
  vector<vector<PetscInt>> outProcessIndex(outProcessRow);

  for (int i = 0; i < localParticleNum; i++) {
    const int currentParticleLocalIndex = i;
    const int currentParticleGlobalIndex = backgroundSourceIndex[i];

    const int iPressureLocal =
        currentParticleLocalIndex * fieldDof + velocityDof;
    const int iPressureGlobal =
        currentParticleGlobalIndex * fieldDof + velocityDof;

    vector<PetscInt> index;
    if (particleType[i] == 0) {
      // velocity block
      index.clear();
      for (int j = 0; j < neighborLists(i, 0); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];

        for (int axes = 0; axes < fieldDof; axes++) {
          index.push_back(fieldDof * neighborParticleIndex + axes);
        }
      }

      for (int axes = 0; axes < velocityDof; axes++) {
        A.setColIndex(currentParticleLocalIndex * fieldDof + axes, index);
      }

      // pressure block
      index.clear();
      for (int j = 0; j < neighborLists(i, 0); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];

        index.push_back(fieldDof * neighborParticleIndex + velocityDof);
      }
      // Lagrange multiplier
      index.push_back(globalLagrangeMultiplierOffset);

      A.setColIndex(currentParticleLocalIndex * fieldDof + velocityDof, index);
    }

    if (particleType[i] != 0 && particleType[i] < 4) {
      // velocity block
      index.clear();
      index.resize(1);
      for (int axes = 0; axes < velocityDof; axes++) {
        index[0] = currentParticleGlobalIndex * fieldDof + axes;
        A.setColIndex(currentParticleLocalIndex * fieldDof + axes, index);
      }

      // pressure block
      index.clear();
      for (int j = 0; j < neighborLists(i, 0); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];

        for (int axes = 0; axes < fieldDof; axes++) {
          index.push_back(fieldDof * neighborParticleIndex + axes);
        }
      }
      // Lagrange multiplier
      index.push_back(globalLagrangeMultiplierOffset);

      A.setColIndex(currentParticleLocalIndex * fieldDof + velocityDof, index);
    }

    if (particleType[i] >= 4) {
      // velocity block
      index.clear();
      index.resize(2 + rotationDof);
      for (int axes = 0; axes < rotationDof; axes++) {
        index[2 + axes] = globalRigidBodyOffset +
                          attachedRigidBodyIndex[i] * rigidBodyDof +
                          translationDof + axes;
      }

      for (int axes = 0; axes < velocityDof; axes++) {
        index[0] = currentParticleGlobalIndex * fieldDof + axes;
        index[1] = globalRigidBodyOffset +
                   attachedRigidBodyIndex[i] * rigidBodyDof + axes;
        A.setColIndex(currentParticleLocalIndex * fieldDof + axes, index);
      }

      // pressure block
      index.clear();
      for (int j = 0; j < neighborLists(i, 0); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];

        for (int axes = 0; axes < fieldDof; axes++) {
          index.push_back(fieldDof * neighborParticleIndex + axes);
        }
      }
      // Lagrange multiplier
      index.push_back(globalLagrangeMultiplierOffset);

      A.setColIndex(currentParticleLocalIndex * fieldDof + velocityDof, index);
    }
  }

  // outprocess graph
  for (int i = 0; i < localParticleNum; i++) {
    const int currentParticleLocalIndex = i;
    const int currentParticleGlobalIndex = backgroundSourceIndex[i];

    // Lagrange multiplier
    outProcessIndex[velocityDof].push_back(
        currentParticleGlobalIndex * fieldDof + velocityDof);

    if (particleType[i] >= 4) {
      vector<PetscInt> index;
      // attached rigid body
      index.clear();
      for (int j = 0; j < neighborLists(i, 0); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];

        for (int axes = 0; axes < velocityDof; axes++) {
          index.push_back(fieldDof * neighborParticleIndex + axes);
        }
      }
      // pressure term
      index.push_back(fieldDof * currentParticleGlobalIndex + velocityDof);

      for (int axes = 0; axes < rigidBodyDof; axes++) {
        vector<PetscInt> &it =
            outProcessIndex[fieldDof +
                            attachedRigidBodyIndex[i] * rigidBodyDof + axes];
        it.insert(it.end(), index.begin(), index.end());
      }
    }
  }

  if (__myID == __MPISize - 1) {
    for (int i = 0; i < fieldDof; i++) {
      outProcessIndex[i].push_back(globalOutProcessOffset + i);
    }
  }

  for (int i = 0; i < outProcessIndex.size(); i++) {
    sort(outProcessIndex[i].begin(), outProcessIndex[i].end());
    outProcessIndex[i].erase(
        unique(outProcessIndex[i].begin(), outProcessIndex[i].end()),
        outProcessIndex[i].end());

    A.setOutProcessColIndex(localOutProcessOffset + i, outProcessIndex[i]);
  }

  // insert matrix entity
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
              A.outProcessIncrement(currentRigidBodyLocalOffset +
                                        translationDof + axes1,
                                    jVelocityGlobal,
                                    rci[(axes1 + 1) % translationDof] *
                                            f[(axes1 + 2) % translationDof] -
                                        rci[(axes1 + 2) % translationDof] *
                                            f[(axes1 + 1) % translationDof]);
            }
            delete[] f;
          }
        }
      } // end of particles on rigid body
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
          A.increment(iPressureLocal, jVelocityGlobal, bi * gradient);
        }
      }
    } // end of velocity block

    // pressure block
    if (particleType[i] == 0) {
      for (int j = 0; j < neighborLists(i, 0); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];

        const int jPressureGlobal =
            fieldDof * neighborParticleIndex + velocityDof;

        const double Aij = pressureAlphas(i, pressureLaplacianIndex, j);

        // laplacian p
        A.increment(iPressureLocal, jPressureGlobal, Aij);
        A.increment(iPressureLocal, iPressureGlobal, -Aij);

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
        A.increment(iPressureLocal, jPressureGlobal, Aij);
        A.increment(iPressureLocal, iPressureGlobal, -Aij);
      }
    }

    // Lagrangian multiplier
    A.increment(iPressureLocal, globalLagrangeMultiplierOffset, 1.0);

    A.outProcessIncrement(localLagrangeMultiplierOffset, iPressureGlobal, 1.0);
    // end of pressure block
  } // end of fluid particle loop

  if (__myID == __MPISize - 1) {
    // Lagrangian multiplier for pressure
    // add a penalty factor
    A.outProcessIncrement(localLagrangeMultiplierOffset,
                          globalLagrangeMultiplierOffset, globalParticleNum);
    for (int i = 0; i < velocityDof; i++) {
      A.outProcessIncrement(localOutProcessOffset + i,
                            globalOutProcessOffset + i, 1.0);
    }
  }

  // A.FinalAssemble(__dim, globalParticleNum, backgroundSourceIndex);
  A.FinalAssemble();

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "Matrix assembly duration: %fs\n",
              tEnd - tStart);

  vector<int> neighborInclusion;
  int neighborInclusionSize;
  if (__myID == __MPISize - 1) {
    neighborInclusion.insert(neighborInclusion.end(),
                             A.__j.begin() + A.__i[localRigidBodyOffset],
                             A.__j.end());

    for (int i = 0; i < rigidBodyDof * numRigidBody; i++) {
      neighborInclusion.push_back(globalRigidBodyOffset + i);
    }
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

  vector<int> interface_flag(globalParticleNum);
  for (int i = 0; i < localParticleNum; i++) {
    if (particleType[i] >= 4) {
      for (int j = 0; j < neighborLists(i, 0); j++) {
        interface_flag[backgroundSourceIndex[neighborLists(i, j + 1)]] = 1;
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, interface_flag.data(), globalParticleNum, MPI_INT,
                MPI_SUM, MPI_COMM_WORLD);

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
      //   if (__dim == 2) {
      //     double x = coord[i][0];
      //     double y = coord[i][1];
      //     rhs[fieldDof * i] = cos(2 * M_PI * x) * sin(2 * M_PI * y);
      //     rhs[fieldDof * i + 1] = -sin(2 * M_PI * x) * cos(2 * M_PI * y);

      //     const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];
      //     const double bi =
      //     pressureNeumannBoundaryBasis.getAlpha0TensorTo0Tensor(
      //         LaplacianOfScalarPointEvaluation, neumannBoudnaryIndex,
      //         neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0));
      //     rhs[fieldDof * i + velocityDof] =
      //         bi * (-normal[i][0] * (8 * pow(M_PI, 2) * cos(2 * M_PI * x) *
      //         sin(2 * M_PI * y)) + normal[i][1] * (8 * pow(M_PI, 2) * sin(2 *
      //         M_PI * x) * cos(2 * M_PI * y)));
      //   } else {
      //     double x = coord[i][0];
      //     double y = coord[i][1];
      //     double z = coord[i][2];
      //     rhs[fieldDof * i] =
      //         cos(2 * M_PI * x) * sin(2 * M_PI * y) * sin(2 * M_PI * z);
      //     rhs[fieldDof * i + 1] =
      //         -2 * sin(2 * M_PI * x) * cos(2 * M_PI * y) * sin(2 * M_PI *
      //         z);
      //     rhs[fieldDof * i + 2] =
      //         sin(2 * M_PI * x) * sin(2 * M_PI * y) * cos(2 * M_PI * z);

      //     const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];
      //     const double bi =
      //     pressureNeumannBoundaryBasis.getAlpha0TensorTo0Tensor(
      //         LaplacianOfScalarPointEvaluation, neumannBoudnaryIndex,
      //         neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0));
      //     rhs[fieldDof * i + velocityDof] =
      //         bi * (-normal[i][0] * (12 * pow(M_PI, 2) * cos(2 * M_PI *
      //         x) * sin(2 * M_PI * y) * sin(2 * M_PI * z)) + normal[i][1]
      //         * (24 * pow(M_PI, 2) * sin(2 * M_PI * x) * cos(2 * M_PI *
      //         y) * sin(2 * M_PI * z)) - normal[i][2] * (12 * pow(M_PI, 2)
      //         * sin(2 * M_PI * x) * sin(2 * M_PI * y) * cos(2 * M_PI *
      //         z)));
      //   }

      // 2-d cavity flow
      // rhs[fieldDof * i] =
      //     1.0 * double(abs(coord[i][1] - __boundingBox[1][1]) < 1e-5);

      // 2-d Taylor-Green vortex-like flow
      if (__dim == 2) {
        double x = coord[i][0] / __boundingBoxSize[0];
        double y = coord[i][1] / __boundingBoxSize[1];

        rhs[fieldDof * i] = cos(M_PI * x) * sin(M_PI * y);
        rhs[fieldDof * i + 1] = -sin(M_PI * x) * cos(M_PI * y);
      }

      // 2-d Poiseuille flow
      // double y = coord[i][1] / __boundingBoxSize[1] * 2;
      // rhs[fieldDof * i] = 0.1 * y * y;

      // // 3-d Taylor-Green vortex-like flow
      if (__dim == 3) {
        double x = coord[i][0] / __boundingBoxSize[0];
        double y = coord[i][1] / __boundingBoxSize[1];
        double z = coord[i][2] / __boundingBoxSize[2];

        rhs[fieldDof * i] = cos(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
        rhs[fieldDof * i + 1] =
            -2 * sin(M_PI * x) * cos(M_PI * y) * sin(M_PI * z);
        rhs[fieldDof * i + 2] = sin(M_PI * x) * sin(M_PI * y) * cos(M_PI * z);
      }
    } else {
      // if (__dim == 3) {
      //   double x = coord[i][0];
      //   double y = coord[i][1];
      //   double z = coord[i][2];
      //   rhs[fieldDof * i] = 12 * pow(M_PI, 2) * cos(2 * M_PI * x) *
      //                       sin(2 * M_PI * y) * sin(2 * M_PI * z);
      //   rhs[fieldDof * i + 1] = -24 * pow(M_PI, 2) * sin(2 * M_PI * x) *
      //                           cos(2 * M_PI * y) * sin(2 * M_PI * z);
      //   rhs[fieldDof * i + 2] = 12 * pow(M_PI, 2) * sin(2 * M_PI * x) *
      //                           sin(2 * M_PI * y) * cos(2 * M_PI * z);
      // } else {
      //   double x = coord[i][0];
      //   double y = coord[i][1];
      //   rhs[fieldDof * i] =
      //       8 * pow(M_PI, 2) * cos(2 * M_PI * x) * sin(2 * M_PI * y);
      //   rhs[fieldDof * i + 1] =
      //       -8 * pow(M_PI, 2) * sin(2 * M_PI * x) * cos(2 * M_PI * y);
      // }

      // rhs[fieldDof * i + velocityDof] = 0.0;
    }
  }

  // A.Write("A.txt");

  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();
  if (numRigidBody == 0) {
    A.Solve(rhs, res, __dim);
    // A.Solve(rhs, res);
  } else {
    // A.Solve(rhs, res, __dim, numRigidBody);
    A.Solve(rhs, res, neighborInclusion, interface_flag, __dim, numRigidBody);
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
