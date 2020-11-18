#include "gmls_solver.h"
#include "sparse_matrix.h"

#include <iomanip>

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

void GMLS_Solver::StokesEquationFinalization() {
  auto all_pressure = __gmls.GetPointer("pressure basis");
  auto neumann_pressure = __gmls.GetPointer("pressure basis neumann boundary");
  auto all_velocity = __gmls.GetPointer("velocity basis");

  delete *all_pressure;
  delete *neumann_pressure;
  delete *all_velocity;
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
  static auto &adaptive_level = __field.index.GetHandle("adaptive level");
  static vector<int> &particleType = __field.index.GetHandle("particle type");

  static vector<int> &attachedRigidBodyIndex =
      __field.index.GetHandle("attached rigid body index");

  auto all_pressure = __gmls.GetPointer("pressure basis");
  auto neumann_pressure = __gmls.GetPointer("pressure basis neumann boundary");
  auto all_velocity = __gmls.GetPointer("velocity basis");

  double tStart, tEnd;

  const int number_of_batches = __batchSize;
  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "\nSolving GMLS subproblems...\n");

  if (*all_pressure != nullptr)
    delete *all_pressure;
  if (*neumann_pressure != nullptr)
    delete *neumann_pressure;
  if (*all_velocity != nullptr)
    delete *all_velocity;

  *all_pressure = new GMLS(VectorTaylorPolynomial, StaggeredEdgeIntegralSample,
                           StaggeredEdgeAnalyticGradientIntegralSample,
                           __polynomialOrder, __dim, "SVD", "STANDARD");
  *neumann_pressure =
      new GMLS(VectorTaylorPolynomial, StaggeredEdgeIntegralSample,
               StaggeredEdgeAnalyticGradientIntegralSample, __polynomialOrder,
               __dim, "SVD", "STANDARD", "NEUMANN_GRAD_SCALAR");
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
    epsilon(i) = (max(__particleSize0[0] * pow(0.5, adaptive_level[i]),
                      particleSize[i][0])) *
                     epsilonMultiplier +
                 1e-15;
    __epsilon[i] = epsilon(i);
    if (epsilon(i) > maxEpsilon) {
      maxEpsilon = epsilon(i);
    }
    if (particleType[i] != 0) {
      neumannBoundaryEpsilon(counter++) =
          (max(__particleSize0[0] * pow(0.5, adaptive_level[i]),
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

  // pointCloudSearch.generateNeighborListsFromKNNSearch(
  //     false, targetCoords, neighborLists, epsilon, 10, 1.05);

  pointCloudSearch.generateSymmetricNeighborListsFromRadiusSearch(
      false, targetCoords, neighborLists, backgroundEpsilonKokkos, 0.0,
      maxEpsilon);

  // pointCloudSearch.generateNeighborListsFromRadiusSearch(
  //     false, targetCoords, neighborLists, epsilon, 0.0, maxEpsilon);

  for (int i = 0; i < localParticleNum; i++) {
    if (particleType[i] != 0) {
      neumannBoundaryNeighborLists(fluid2NeumannBoundary[i], 0) =
          neighborLists(i, 0);
      for (int j = 0; j < neighborLists(i, 0); j++) {
        neumannBoundaryNeighborLists(fluid2NeumannBoundary[i], j + 1) =
            neighborLists(i, j + 1);
      }
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
  pressureOperation[0] = DivergenceOfVectorPointEvaluation;
  pressureOperation[1] = GradientOfScalarPointEvaluation;

  pressureBasis.clearTargets();
  pressureBasis.addTargets(pressureOperation);

  pressureBasis.setWeightingType(WeightingFunctionType::Power);
  pressureBasis.setWeightingPower(__weightFuncOrder);
  pressureBasis.setOrderOfQuadraturePoints(2);
  pressureBasis.setDimensionOfQuadraturePoints(1);
  pressureBasis.setQuadratureType("LINE");

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
  pressureNeumannBoundaryOperations[0] = DivergenceOfVectorPointEvaluation;

  pressureNeumannBoundaryBasis.clearTargets();
  pressureNeumannBoundaryBasis.addTargets(pressureNeumannBoundaryOperations);

  pressureNeumannBoundaryBasis.setWeightingType(WeightingFunctionType::Power);
  pressureNeumannBoundaryBasis.setWeightingPower(__weightFuncOrder);
  pressureNeumannBoundaryBasis.setOrderOfQuadraturePoints(2);
  pressureNeumannBoundaryBasis.setDimensionOfQuadraturePoints(1);
  pressureNeumannBoundaryBasis.setQuadratureType("LINE");

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
  int globalPressureDof = globalParticleNum;

  if (__myID == __MPISize - 1) {
    localVelocityDof += rigidBodyDof * numRigidBody;
  }

  int localRigidBodyOffset = particleNum[__MPISize + 1] * fieldDof;
  int globalRigidBodyOffset = globalParticleNum * fieldDof;
  int localOutProcessOffset = particleNum[__MPISize + 1] * fieldDof;

  int localDof = localVelocityDof + localPressureDof;
  int globalDof = globalVelocityDof + globalPressureDof;

  int outProcessRow = rigidBodyDof * numRigidBody;

  if (__adaptive_step == 0) {
    _multi.clear();
  }

  _multi.add_new_level();

  PetscSparseMatrix &A = _multi.getA(__adaptive_step);
  A.resize(localDof, localDof, globalDof, outProcessRow, localOutProcessOffset);

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

      A.setColIndex(currentParticleLocalIndex * fieldDof + velocityDof, index);
    }
  }

  // outprocess graph
  // for (int i = 0; i < localParticleNum; i++) {
  //   const int currentParticleLocalIndex = i;
  //   const int currentParticleGlobalIndex = backgroundSourceIndex[i];

  //   if (particleType[i] >= 4) {
  //     vector<PetscInt> index;
  //     // attached rigid body
  //     index.clear();
  //     for (int j = 0; j < neighborLists(i, 0); j++) {
  //       const int neighborParticleIndex =
  //           backgroundSourceIndex[neighborLists(i, j + 1)];

  //       for (int axes = 0; axes < velocityDof; axes++) {
  //         index.push_back(fieldDof * neighborParticleIndex + axes);
  //       }
  //     }
  //     // pressure term
  //     index.push_back(fieldDof * currentParticleGlobalIndex + velocityDof);

  //     for (int axes = 0; axes < rigidBodyDof; axes++) {
  //       vector<PetscInt> &it =
  //           outProcessIndex[attachedRigidBodyIndex[i] * rigidBodyDof + axes];
  //       it.insert(it.end(), index.begin(), index.end());
  //     }
  //   }
  // }

  if (__myID == __MPISize - 1) {
    vector<PetscInt> index;
    for (int axes = 0; axes < rigidBodyDof; axes++) {
      index.resize(1);
      index[0] = globalRigidBodyOffset + axes;
      outProcessIndex[axes] = index;
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
      if (particleType[i] >= 4) {
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

        vec3 dA;
        if (particleType[i] == 4) {
          // corner point
          dA = vec3(0.0, 0.0, 0.0);
        } else {
          dA = (__dim == 3)
                   ? (normal[i] * particleSize[i][0] * particleSize[i][1])
                   : (normal[i] * particleSize[i][0]);
        }

        // apply pressure
        // for (int axes1 = 0; axes1 < translationDof; axes1++) {
        //   A.outProcessIncrement(currentRigidBodyLocalOffset + axes1,
        //                         iPressureGlobal, -dA[axes1]);
        // }

        // for (int axes1 = 0; axes1 < rotationDof; axes1++) {
        //   A.outProcessIncrement(currentRigidBodyLocalOffset + translationDof
        //   +
        //                             axes1,
        //                         iPressureGlobal,
        //                         -rci[(axes1 + 1) % translationDof] *
        //                                 dA[(axes1 + 2) % translationDof] +
        //                             rci[(axes1 + 2) % translationDof] *
        //                                 dA[(axes1 + 1) % translationDof]);
        // }

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
            // for (int axes1 = 0; axes1 < translationDof; axes1++) {
            //   A.outProcessIncrement(currentRigidBodyLocalOffset + axes1,
            //                         jVelocityGlobal, f[axes1]);
            // }

            // // torque balance
            // for (int axes1 = 0; axes1 < rotationDof; axes1++) {
            //   A.outProcessIncrement(currentRigidBodyLocalOffset +
            //                             translationDof + axes1,
            //                         jVelocityGlobal,
            //                         rci[(axes1 + 1) % translationDof] *
            //                                 f[(axes1 + 2) % translationDof] -
            //                             rci[(axes1 + 2) % translationDof] *
            //                                 f[(axes1 + 1) % translationDof]);
            // }
            delete[] f;
          }
        }
      } // end of particles on rigid body
    }

    // n \cdot grad p
    if (particleType[i] != 0) {
      const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];
      const double bi = pressureNeumannBoundaryBasis.getAlpha0TensorTo0Tensor(
          DivergenceOfVectorPointEvaluation, neumannBoudnaryIndex,
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
    // end of pressure block
  } // end of fluid particle loop

  if (__myID == __MPISize - 1) {
    for (int axes = 0; axes < rigidBodyDof; axes++) {
      A.outProcessIncrement(localRigidBodyOffset + axes,
                            globalRigidBodyOffset + axes, 1.0);
    }
  }

  vector<int> idx_neighbor;

  // A.FinalAssemble();
  if (numRigidBody == 0) {
    Mat &ff = _multi.getFieldMat(__adaptive_step);
    A.FinalAssemble(ff, fieldDof, numRigidBody, rigidBodyDof);
  } else {
    Mat &ff = _multi.getFieldMat(__adaptive_step);
    A.FinalAssemble(ff, fieldDof, numRigidBody, rigidBodyDof);
    if (__myID == __MPISize - 1) {
      idx_neighbor.resize(3);
      idx_neighbor[0] = globalRigidBodyOffset;
      idx_neighbor[1] = globalRigidBodyOffset + 1;
      idx_neighbor[2] = globalRigidBodyOffset + 2;
    }
    // A.ExtractNeighborIndex(idx_neighbor, __dim, numRigidBody,
    //                        localRigidBodyOffset, globalRigidBodyOffset);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "Matrix assembly duration: %fs\n",
              tEnd - tStart);

  // Interpolation Matrix
  PetscSparseMatrix &I = _multi.getI(__adaptive_step);
  PetscSparseMatrix &R = _multi.getR(__adaptive_step);

  if (numRigidBody != 0 && __adaptive_step != 0) {
    tStart = MPI_Wtime();

    BuildInterpolationAndRestrictionMatrices(I, R, numRigidBody, __dim);

    tEnd = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,
                "Interpolation matrix building duration: %fs\n", tEnd - tStart);
  }

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

  // if (numRigidBody == 0) {
  //   for (int i = 0; i < localParticleNum; i++) {
  //     if (particleType[i] != 0 && particleType[i] < 4) {
  //       // 2-d Taylor-Green vortex-like flow
  //       if (__dim == 2) {
  //         double x = coord[i][0];
  //         double y = coord[i][1];

  //         rhs[fieldDof * i] = cos(M_PI * x) * sin(M_PI * y);
  //         rhs[fieldDof * i + 1] = -sin(M_PI * x) * cos(M_PI * y);

  //         const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];
  //         const double bi =
  //             pressureNeumannBoundaryBasis.getAlpha0TensorTo0Tensor(
  //                 DivergenceOfVectorPointEvaluation, neumannBoudnaryIndex,
  //                 neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0));

  //         rhs[fieldDof * i + velocityDof] =
  //             -4.0 * pow(M_PI, 2.0) *
  //                 (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y)) +
  //             bi * (normal[i][0] * 2.0 * pow(M_PI, 2.0) * cos(M_PI * x) *
  //                       sin(M_PI * y) -
  //                   normal[i][1] * 2.0 * pow(M_PI, 2.0) * sin(M_PI * x) *
  //                       cos(M_PI * y)) +
  //             bi * (normal[i][0] * 2.0 * M_PI * sin(2.0 * M_PI * x) +
  //                   normal[i][1] * 2.0 * M_PI * sin(2.0 * M_PI * y));

  //         // 2-d cavity flow
  //         // rhs[fieldDof * i] =
  //         //     1.0 * double(abs(coord[i][1] - __boundingBox[1][1]) < 1e-5);
  //       }

  //       // 3-d Taylor-Green vortex-like flow
  //       if (__dim == 3) {
  //         double x = coord[i][0];
  //         double y = coord[i][1];
  //         double z = coord[i][2];

  //         rhs[fieldDof * i] = cos(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
  //         rhs[fieldDof * i + 1] =
  //             -2 * sin(M_PI * x) * cos(M_PI * y) * sin(M_PI * z);
  //         rhs[fieldDof * i + 2] = sin(M_PI * x) * sin(M_PI * y) * cos(M_PI *
  //         z);

  //         const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];
  //         const double bi =
  //             pressureNeumannBoundaryBasis.getAlpha0TensorTo0Tensor(
  //                 DivergenceOfVectorPointEvaluation, neumannBoudnaryIndex,
  //                 neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0));

  //         rhs[fieldDof * i + velocityDof] =
  //             -4.0 * pow(M_PI, 2.0) *
  //                 (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y) +
  //                  cos(2.0 * M_PI * z)) +
  //             bi * (normal[i][0] * 3.0 * pow(M_PI, 2.0) * cos(M_PI * x) *
  //                       sin(M_PI * y) * sin(M_PI * z) -
  //                   normal[i][1] * 6.0 * pow(M_PI, 2.0) * sin(M_PI * x) *
  //                       cos(M_PI * y) * sin(M_PI * z) +
  //                   normal[i][2] * 3.0 * pow(M_PI, 2.0) * sin(M_PI * x) *
  //                       sin(M_PI * y) * cos(M_PI * z)) +
  //             bi * (normal[i][0] * 2.0 * M_PI * sin(2.0 * M_PI * x) +
  //                   normal[i][1] * 2.0 * M_PI * sin(2.0 * M_PI * y) +
  //                   normal[i][2] * 2.0 * M_PI * sin(2.0 * M_PI * z));
  //       }
  //     } else if (particleType[i] == 0) {
  //       if (__dim == 2) {
  //         double x = coord[i][0];
  //         double y = coord[i][1];

  //         rhs[fieldDof * i] =
  //             2.0 * pow(M_PI, 2.0) * cos(M_PI * x) * sin(M_PI * y) +
  //             2.0 * M_PI * sin(2.0 * M_PI * x);
  //         rhs[fieldDof * i + 1] =
  //             -2.0 * pow(M_PI, 2.0) * sin(M_PI * x) * cos(M_PI * y) +
  //             2.0 * M_PI * sin(2.0 * M_PI * y);

  //         rhs[fieldDof * i + velocityDof] =
  //             -4.0 * pow(M_PI, 2.0) *
  //             (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y));
  //       }

  //       if (__dim == 3) {
  //         double x = coord[i][0];
  //         double y = coord[i][1];
  //         double z = coord[i][2];

  //         rhs[fieldDof * i] = 3.0 * pow(M_PI, 2) * cos(M_PI * x) *
  //                                 sin(M_PI * y) * sin(M_PI * z) +
  //                             2.0 * M_PI * sin(2.0 * M_PI * x);
  //         rhs[fieldDof * i + 1] = -6.0 * pow(M_PI, 2) * sin(M_PI * x) *
  //                                     cos(M_PI * y) * sin(M_PI * z) +
  //                                 2.0 * M_PI * sin(2.0 * M_PI * y);
  //         rhs[fieldDof * i + 2] = 3.0 * pow(M_PI, 2) * sin(M_PI * x) *
  //                                     sin(M_PI * y) * cos(M_PI * z) +
  //                                 2.0 * M_PI * sin(2.0 * M_PI * z);

  //         rhs[fieldDof * i + velocityDof] =
  //             -4.0 * pow(M_PI, 2.0) *
  //             (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y) + cos(2.0 * M_PI *
  //             z));
  //       }
  //     }
  //   }
  // }

  // if (__myID == __MPISize - 1) {
  //   for (int i = 0; i < numRigidBody; i++) {
  //     rhs[localRigidBodyOffset + i * rigidBodyDof + translationDof] =
  //         pow(-1, i + 1);
  //   }
  // }

  // for (int i = 0; i < localParticleNum; i++) {
  //   if (particleType[i] != 0 && particleType[i] < 4) {
  //     // 2-d Taylor-Green vortex-like flow
  //     if (__dim == 2) {
  //       double x = coord[i][0];
  //       double y = coord[i][1];

  //       rhs[fieldDof * i] = 0.1 * coord[i][1];
  //     }
  //   }
  // }

  vector<double> &rigidBodySize = __rigidBody.scalar.GetHandle("size");

  double u = 1.0;
  double RR = rigidBodySize[0];

  for (int i = 0; i < localParticleNum; i++) {
    if (particleType[i] != 0 && particleType[i] < 4) {
      double x = coord[i][0];
      double y = coord[i][1];
      double z = coord[i][2];

      const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];
      const double bi = pressureNeumannBoundaryBasis.getAlpha0TensorTo0Tensor(
          DivergenceOfVectorPointEvaluation, neumannBoudnaryIndex,
          neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0));

      double r = sqrt(x * x + y * y + z * z);
      double theta = acos(z / r);
      double phi = atan2(y, x);

      double vr = u * cos(theta) *
                  (1 - (3 * RR) / (2 * r) + pow(RR, 3) / (2 * pow(r, 3)));
      double vt = -u * sin(theta) *
                  (1 - (3 * RR) / (4 * r) - pow(RR, 3) / (4 * pow(r, 3)));

      double pr = 3 * RR / pow(r, 3) * u * cos(theta);
      double pt = 3 / 2 * RR / pow(r, 3) * u * sin(theta);

      rhs[fieldDof * i] =
          sin(theta) * cos(phi) * vr + cos(theta) * cos(phi) * vt;
      rhs[fieldDof * i + 1] =
          sin(theta) * sin(phi) * vr + cos(theta) * sin(phi) * vt;
      rhs[fieldDof * i + 2] = cos(theta) * vr - sin(theta) * vt;

      double p1 = sin(theta) * cos(phi) * pr + cos(theta) * cos(phi) * pt;
      double p2 = sin(theta) * sin(phi) * pr + cos(theta) * sin(phi) * pt;
      double p3 = cos(theta) * pr - sin(theta) * pt;

      // rhs[fieldDof * i + 3] =
      //     bi * (normal[i][0] * p1 + normal[i][1] * p2 + normal[i][2] * p3);
    } else if (particleType[i] >= 4) {
      double x = coord[i][0];
      double y = coord[i][1];
      double z = coord[i][2];

      double r = sqrt(x * x + y * y + z * z);
      double theta = acos(z / r);
      double phi = atan2(y, x);

      const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];
      const double bi = pressureNeumannBoundaryBasis.getAlpha0TensorTo0Tensor(
          DivergenceOfVectorPointEvaluation, neumannBoudnaryIndex,
          neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0));

      double pr = 3 * RR / pow(r, 3) * u * cos(theta);
      double pt = 3 / 2 * RR / pow(r, 3) * u * sin(theta);

      double p1 = sin(theta) * cos(phi) * pr + cos(theta) * cos(phi) * pt;
      double p2 = sin(theta) * sin(phi) * pr + cos(theta) * sin(phi) * pt;
      double p3 = cos(theta) * pr - sin(theta) * pt;

      // rhs[fieldDof * i + 3] =
      //     bi * (normal[i][0] * p1 + normal[i][1] * p2 + normal[i][2] * p3);
    }
  }

  if (__myID == __MPISize - 1) {
    // rhs[localRigidBodyOffset + 2] = 6 * M_PI * RR * u;
  }

  // make sure pressure term is orthogonal to the constant
  double rhs_pressure_sum = 0.0;
  for (int i = 0; i < localParticleNum; i++) {
    rhs_pressure_sum += rhs[fieldDof * i + velocityDof];
  }
  MPI_Allreduce(MPI_IN_PLACE, &rhs_pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  rhs_pressure_sum /= globalParticleNum;
  for (int i = 0; i < localParticleNum; i++) {
    rhs[fieldDof * i + velocityDof] -= rhs_pressure_sum;
  }

  // A.Write("A.txt");

  PetscLogDefaultBegin();

  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();
  if (numRigidBody == 0) {
    // A.Solve(rhs, res, __dim);
    // A.Solve(rhs, res);
    A.Solve(rhs, res, fieldDof);
  } else {
    if (__adaptive_step != 0)
      InitialGuessFromPreviousAdaptiveStep(I, res);
    // A.Solve(rhs, res, idx_neighbor, __dim, numRigidBody, __adaptive_step, I,
    // R);
    if (_multi.Solve(rhs, res, idx_neighbor) != 0) {
      ofstream output;
      if (__myID == 0) {
        output.open("traj_new.txt", ios::trunc);
        for (int num = 0; num < numRigidBody; num++) {
          for (int j = 0; j < 2; j++) {
            output << setprecision(15) << rigidBodyPosition[num][j] << '\t';
          }
          output << 0.10 << '\t' << 1 << endl;
        }
        output.close();
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "linear system solving duration: %fs\n",
              tEnd - tStart);

  int innerParticleCount = 0;
  for (int i = 0; i < localParticleNum; i++) {
    if (particleType[i] == 0) {
      innerParticleCount++;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &innerParticleCount, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "inner particle count: %d\n",
              innerParticleCount);

  PetscViewer viewer;
  if (__viewer > 0) {
    PetscViewerASCIIGetStdout(PETSC_COMM_WORLD, &viewer);
    PetscLogView(viewer);
  }

  // copy data
  static vector<vec3> &velocity = __field.vector.GetHandle("fluid velocity");
  static vector<double> &pressure = __field.scalar.GetHandle("fluid pressure");
  pressure.resize(localParticleNum);
  velocity.resize(localParticleNum);

  double pressure_sum = 0.0;
  for (int i = 0; i < localParticleNum; i++) {
    pressure[i] = res[fieldDof * i + velocityDof];
    pressure_sum += pressure[i];
    for (int axes1 = 0; axes1 < __dim; axes1++)
      velocity[i][axes1] = res[fieldDof * i + axes1];
  }

  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  double average_pressure = pressure_sum / globalParticleNum;
  for (int i = 0; i < localParticleNum; i++) {
    pressure[i] -= average_pressure;
  }

  // check data
  double true_pressure_mean = 0.0;
  double pressure_mean = 0.0;
  for (int i = 0; i < localParticleNum; i++) {
    if (__dim == 2) {
      double x = coord[i][0];
      double y = coord[i][1];

      double true_pressure = -cos(2.0 * M_PI * x) - cos(2.0 * M_PI * y);

      true_pressure_mean += true_pressure;
      pressure_mean += pressure[i];
    }

    if (__dim == 3) {
      double x = coord[i][0];
      double y = coord[i][1];
      double z = coord[i][2];

      double r = sqrt(x * x + y * y + z * z);
      double theta = acos(z / r);

      double true_pressure = -3 / 2 * RR / pow(r, 2.0) * u * cos(theta);

      true_pressure_mean += true_pressure;
      pressure_mean += pressure[i];
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &true_pressure_mean, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &pressure_mean, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  true_pressure_mean /= globalParticleNum;
  pressure_mean /= globalParticleNum;

  double error_velocity = 0.0;
  double norm_velocity = 0.0;
  double error_pressure = 0.0;
  double norm_pressure = 0.0;
  for (int i = 0; i < localParticleNum; i++) {
    if (__dim == 2) {
      double x = coord[i][0];
      double y = coord[i][1];

      double true_pressure =
          -cos(2.0 * M_PI * x) - cos(2.0 * M_PI * y) - true_pressure_mean;
      double true_velocity[2];
      true_velocity[0] = cos(M_PI * x) * sin(M_PI * y);
      true_velocity[1] = -sin(M_PI * x) * cos(M_PI * y);

      error_velocity += pow(true_velocity[0] - velocity[i][0], 2) +
                        pow(true_velocity[1] - velocity[i][1], 2);
      error_pressure += pow(true_pressure - pressure[i], 2);

      norm_velocity += pow(true_velocity[0], 2) + pow(true_velocity[1], 2);
      norm_pressure += pow(true_pressure, 2);
    }

    if (__dim == 3) {
      double x = coord[i][0];
      double y = coord[i][1];
      double z = coord[i][2];

      double r = sqrt(x * x + y * y + z * z);
      double theta = acos(z / r);
      double phi = atan2(y, x);

      double vr = u * cos(theta) *
                  (1 - (3 * RR) / (2 * r) + pow(RR, 3) / (2 * pow(r, 3)));
      double vt = -u * sin(theta) *
                  (1 - (3 * RR) / (4 * r) - pow(RR, 3) / (4 * pow(r, 3)));

      double pr = 3 * RR / pow(r, 3) * u * cos(theta);
      double pt = 3 / 2 * RR / pow(r, 3) * u * sin(theta);

      double true_velocity[3];

      true_velocity[0] =
          sin(theta) * cos(phi) * vr + cos(theta) * cos(phi) * vt;
      true_velocity[1] =
          sin(theta) * sin(phi) * vr + cos(theta) * sin(phi) * vt;
      true_velocity[2] = cos(theta) * vr - sin(theta) * vt;

      double true_pressure =
          -3 / 2 * RR / pow(r, 2.0) * u * cos(theta) - true_pressure_mean;

      error_velocity += pow(true_velocity[0] - velocity[i][0], 2) +
                        pow(true_velocity[1] - velocity[i][1], 2) +
                        pow(true_velocity[2] - velocity[i][2], 2);
      error_pressure += pow(true_pressure - pressure[i], 2);

      norm_velocity += pow(true_velocity[0], 2) + pow(true_velocity[1], 2) +
                       pow(true_velocity[2], 2);
      norm_pressure += pow(true_pressure, 2);
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &error_velocity, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &error_pressure, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &norm_velocity, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &norm_pressure, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  PetscPrintf(MPI_COMM_WORLD, "relative pressure error: %.10f\n",
              sqrt(error_pressure / norm_pressure));
  PetscPrintf(MPI_COMM_WORLD, "relative velocity error: %.10f\n",
              sqrt(error_velocity / norm_velocity));

  PetscPrintf(MPI_COMM_WORLD, "RMS pressure error: %.10f\n",
              sqrt(error_pressure / globalParticleNum));
  PetscPrintf(MPI_COMM_WORLD, "RMS velocity error: %.10f\n",
              sqrt(error_velocity / globalParticleNum));

  MPI_Barrier(MPI_COMM_WORLD);

  // vector<vec3> recvVelocity;
  // DataSwapAmongNeighbor(velocity, recvVelocity);
  // vector<vec3> backgroundVelocity;

  // for (int i = 0; i < localParticleNum; i++) {
  //   backgroundVelocity.push_back(velocity[i]);
  // }

  // static vector<int> &offset = __neighbor.index.GetHandle("recv offset");
  // int neighborNum = pow(3, __dim);
  // int totalNeighborParticleNum = offset[neighborNum];

  // for (int i = 0; i < totalNeighborParticleNum; i++) {
  //   backgroundVelocity.push_back(recvVelocity[i]);
  // }

  // // communicate coeffients
  // Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
  //     backgroundVelocityDevice("background velocity",
  //     backgroundVelocity.size(),
  //                              3);
  // Kokkos::View<double **>::HostMirror backgroundVelocityHost =
  //     Kokkos::create_mirror_view(backgroundVelocityDevice);

  // for (size_t i = 0; i < backgroundVelocity.size(); i++) {
  //   backgroundVelocityHost(i, 0) = backgroundVelocity[i][0];
  //   backgroundVelocityHost(i, 1) = backgroundVelocity[i][1];
  //   backgroundVelocityHost(i, 2) = backgroundVelocity[i][2];
  // }

  // Kokkos::deep_copy(backgroundVelocityHost, backgroundVelocityDevice);

  // Evaluator velocityEvaluator(&velocityBasis);

  // auto coefficients =
  //     velocityEvaluator.applyFullPolynomialCoefficientsBasisToDataAllComponents<
  //         double **, Kokkos::HostSpace>(backgroundVelocityDevice);

  // auto gradient =
  //     velocityEvaluator.applyAlphasToDataAllComponentsAllTargetSites<
  //         double **, Kokkos::HostSpace>(backgroundVelocityDevice,
  //                                       GradientOfVectorPointEvaluation);

  // double fz = 0.0;
  // for (int i = 0; i < localParticleNum; i++) {
  //   if (particleType[i] >= 4) {
  //     vec3 dA = (__dim == 3)
  //                   ? (normal[i] * particleSize[i][0] * particleSize[i][1])
  //                   : (normal[i] * particleSize[i][0]);

  //     vector<double> f;
  //     f.resize(3);
  //     for (int axes1 = 0; axes1 < __dim; axes1++) {
  //       f[axes1] = 0.0;
  //     }

  //     for (int axes1 = 0; axes1 < __dim; axes1++) {
  //       // output component 1
  //       for (int axes2 = 0; axes2 < __dim; axes2++) {
  //         // output component 2
  //         const int index = axes1 * __dim + axes2;
  //         const double sigma =
  //             pressure[i] + __eta * (gradient(i, index) + gradient(i,
  //             index));

  //         f[axes1] += sigma * dA[axes2];
  //       }
  //     }

  //     fz += f[2];
  //   }
  // }

  // MPI_Allreduce(MPI_IN_PLACE, &fz, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  // PetscPrintf(PETSC_COMM_WORLD, "force_z: %f, actual force: %f\n", fz,
  //             6 * M_PI * RR * u);
  // PetscPrintf(PETSC_COMM_WORLD, "difference %f\n",
  //             abs(fz - 6 * M_PI * RR * u) / abs(6 * M_PI * RR * u));

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

  // communicate velocity and angular velocity
  vector<double> translation_velocity(numRigidBody * translationDof);
  vector<double> angular_velocity(numRigidBody * rotationDof);

  if (__myID == __MPISize - 1) {
    for (int i = 0; i < numRigidBody; i++) {
      for (int j = 0; j < translationDof; j++) {
        translation_velocity[i * translationDof + j] = rigidBodyVelocity[i][j];
        cout << rigidBodyVelocity[i][j] << endl;
      }
      for (int j = 0; j < rotationDof; j++) {
        angular_velocity[i * rotationDof + j] = rigidBodyAngularVelocity[i][j];
        cout << rigidBodyAngularVelocity[i][j] << endl;
      }
    }
  }

  MPI_Bcast(translation_velocity.data(), numRigidBody * translationDof,
            MPI_DOUBLE, __MPISize - 1, MPI_COMM_WORLD);
  MPI_Bcast(angular_velocity.data(), numRigidBody * rotationDof, MPI_DOUBLE,
            __MPISize - 1, MPI_COMM_WORLD);

  if (__myID != __MPISize - 1) {
    for (int i = 0; i < numRigidBody; i++) {
      for (int j = 0; j < translationDof; j++) {
        rigidBodyVelocity[i][j] = translation_velocity[i * translationDof + j];
      }
      for (int j = 0; j < rotationDof; j++) {
        rigidBodyAngularVelocity[i][j] = angular_velocity[i * rotationDof + j];
      }
    }
  }

  if (__adaptiveRefinement) {
    static auto &old_coord = __field.vector.GetHandle("old coord");
    static auto &old_particle_type =
        __field.index.GetHandle("old particle type");
    static auto &old_background_coord =
        __background.vector.GetHandle("old source coord");
    static auto &old_background_index =
        __background.index.GetHandle("old source index");
    old_coord = coord;
    old_particle_type = particleType;
    old_background_coord = backgroundSourceCoord;
    old_background_index = backgroundSourceIndex;
  }
}
