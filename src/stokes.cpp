#include "gmls_solver.hpp"
#include "petsc_sparse_matrix.hpp"

#include <iomanip>

using namespace std;
using namespace Compadre;

void solution(double r, double phi, double omega, double &u, double &v) {
  double r0 = 1;
  double G = omega * 2;
  double vr = G / 2 * (pow((pow(r, 2.0) - pow(r0, 2.0)), 2.0) / pow(r, 3.0)) *
              sin(2 * phi);
  double vt = G / 2 * (-r + (r - pow(r0, 4) / pow(r, 3.0)) * cos(2 * phi));

  double y = r * sin(phi);

  u = vr * cos(phi) - vt * sin(phi) - G * y;
  v = vr * sin(phi) + vt * cos(phi);
}

void gmls_solver::StokesEquationInitialization() {
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

void gmls_solver::StokesEquationFinalization() {
  auto all_pressure = __gmls.GetPointer("pressure basis");
  auto neumann_pressure = __gmls.GetPointer("pressure basis neumann boundary");
  auto all_velocity = __gmls.GetPointer("velocity basis");

  delete *all_pressure;
  delete *neumann_pressure;
  delete *all_velocity;
}

void gmls_solver::StokesEquation() {
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

  int local_particle_num = particleNum[0];
  int global_particle_num = particleNum[1];

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
  for (int i = 0; i < local_particle_num; i++) {
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
  for (int i = 0; i < local_particle_num; i++) {
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

  auto minNeighbors = Compadre::GMLS::getNP(
      __polynomialOrder, __dim, DivergenceFreeVectorTaylorPolynomial);

  double epsilonMultiplier = __polynomialOrder + 0.5;
  if (__epsilonMultiplier != 0.0)
    epsilonMultiplier = __epsilonMultiplier;

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
  __epsilon.resize(local_particle_num);
  for (int i = 0; i < numTargetCoords; i++) {
    epsilon(i) =
        __particleSize0[0] * pow(0.5, adaptive_level[i]) * epsilonMultiplier +
        1e-15;
    __epsilon[i] = epsilon(i);
    if (epsilon(i) > maxEpsilon) {
      maxEpsilon = epsilon(i);
    }
    if (particleType[i] != 0) {
      neumannBoundaryEpsilon(counter++) = epsilon(i);
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &maxEpsilon, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);

  vector<double> __neumannBoundaryEpsilon(neumannBoundaryNumTargetCoords);
  for (int i = 0; i < neumannBoundaryNumTargetCoords; i++) {
    __neumannBoundaryEpsilon[i] = neumannBoundaryEpsilon(i);
  }

  // ensure every particle has enough neighbors
  bool goodNeighborSearch = false;
  while (!goodNeighborSearch) {
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

    bool passNeighborNumCheck = true;
    int minNeighbor = 100;
    int maxNeighbor = 0;
    for (int i = 0; i < local_particle_num; i++) {
      // if (neighborLists(i, 0) <= minNeighbors) {
      // __epsilon[i] +=
      //     0.5 * (max(__particleSize0[0] * pow(0.5, __adaptive_step),
      //                particleSize[i][0]));
      // epsilon(i) = __epsilon[i];
      // passNeighborNumCheck = false;
      // if (particleType[i] != 0) {
      //   neumannBoundaryEpsilon(fluid2NeumannBoundary[i]) = __epsilon[i];
      // }
      // }
      if (neighborLists(i, 0) < minNeighbor)
        minNeighbor = neighborLists(i, 0);
      if (neighborLists(i, 0) > maxNeighbor)
        maxNeighbor = neighborLists(i, 0);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &minNeighbor, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxNeighbor, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    PetscPrintf(MPI_COMM_WORLD, "min neighbor: %d\n", minNeighbor);
    PetscPrintf(MPI_COMM_WORLD, "max neighbor: %d\n", maxNeighbor);

    int processCounter = 0;
    if (!passNeighborNumCheck) {
      processCounter = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, &processCounter, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "process counter: %d\n", processCounter);

    if (processCounter == 0) {
      goodNeighborSearch = true;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // pointCloudSearch.generateNeighborListsFromRadiusSearch(
  //     false, targetCoords, neighborLists, epsilon, 0.0, maxEpsilon);

  for (int i = 0; i < local_particle_num; i++) {
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
  for (int i = 0; i < local_particle_num; i++) {
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

  const int translation_dof = (__dim == 3 ? 3 : 2);
  const int rotation_dof = (__dim == 3 ? 3 : 1);
  const int rigid_body_dof = (__dim == 3 ? 6 : 3);
  const int num_rigid_body = rigidBodyPosition.size();

  int field_dof = __dim + 1;
  int velocity_dof = __dim;

  int local_velocity_dof = local_particle_num * __dim;
  int global_velocity_dof =
      global_particle_num * __dim + rigid_body_dof * num_rigid_body;
  int local_pressure_dof = local_particle_num;
  int global_pressure_dof = global_particle_num;

  if (__myID == __MPISize - 1) {
    local_velocity_dof += rigid_body_dof * num_rigid_body;
  }

  int local_rigid_body_offset = particleNum[__MPISize + 1] * field_dof;
  int global_rigid_body_offset = global_particle_num * field_dof;
  int local_out_process_offset = particleNum[__MPISize + 1] * field_dof;

  int local_dof = local_velocity_dof + local_pressure_dof;
  int global_dof = global_velocity_dof + global_pressure_dof;

  int out_process_row = rigid_body_dof * num_rigid_body;

  if (__adaptive_step == 0) {
    _multi.clear();
  }

  _multi.add_new_level();

  petsc_sparse_matrix &A = _multi.getA(__adaptive_step);
  A.resize(local_dof, local_dof, global_dof, out_process_row,
           local_out_process_offset);

  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();

  // compute matrix graph
  vector<vector<PetscInt>> outProcessIndex(out_process_row);

  for (int i = 0; i < local_particle_num; i++) {
    const int currentParticleLocalIndex = i;
    const int currentParticleGlobalIndex = backgroundSourceIndex[i];

    const int iPressureLocal =
        currentParticleLocalIndex * field_dof + velocity_dof;
    const int iPressureGlobal =
        currentParticleGlobalIndex * field_dof + velocity_dof;

    vector<PetscInt> index;
    if (particleType[i] == 0) {
      // velocity block
      index.clear();
      for (int j = 0; j < neighborLists(i, 0); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];

        for (int axes = 0; axes < field_dof; axes++) {
          index.push_back(field_dof * neighborParticleIndex + axes);
        }
      }

      for (int axes = 0; axes < velocity_dof; axes++) {
        A.set_col_index(currentParticleLocalIndex * field_dof + axes, index);
      }

      // pressure block
      index.clear();
      for (int j = 0; j < neighborLists(i, 0); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];

        index.push_back(field_dof * neighborParticleIndex + velocity_dof);
      }

      A.set_col_index(currentParticleLocalIndex * field_dof + velocity_dof,
                      index);
    }

    if (particleType[i] != 0 && particleType[i] < 4) {
      // velocity block
      index.clear();
      index.resize(1);
      for (int axes = 0; axes < velocity_dof; axes++) {
        index[0] = currentParticleGlobalIndex * field_dof + axes;
        A.set_col_index(currentParticleLocalIndex * field_dof + axes, index);
      }

      // pressure block
      index.clear();
      for (int j = 0; j < neighborLists(i, 0); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];

        for (int axes = 0; axes < field_dof; axes++) {
          index.push_back(field_dof * neighborParticleIndex + axes);
        }
      }

      A.set_col_index(currentParticleLocalIndex * field_dof + velocity_dof,
                      index);
    }

    if (particleType[i] >= 4) {
      // velocity block
      index.clear();
      index.resize(2 + rotation_dof);
      for (int axes = 0; axes < rotation_dof; axes++) {
        index[2 + axes] = global_rigid_body_offset +
                          attachedRigidBodyIndex[i] * rigid_body_dof +
                          translation_dof + axes;
      }

      for (int axes = 0; axes < velocity_dof; axes++) {
        index[0] = currentParticleGlobalIndex * field_dof + axes;
        index[1] = global_rigid_body_offset +
                   attachedRigidBodyIndex[i] * rigid_body_dof + axes;
        A.set_col_index(currentParticleLocalIndex * field_dof + axes, index);
      }

      // pressure block
      index.clear();
      for (int j = 0; j < neighborLists(i, 0); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];

        for (int axes = 0; axes < field_dof; axes++) {
          index.push_back(field_dof * neighborParticleIndex + axes);
        }
      }

      A.set_col_index(currentParticleLocalIndex * field_dof + velocity_dof,
                      index);
    }
  }

  // outprocess graph
  for (int i = 0; i < local_particle_num; i++) {
    const int currentParticleLocalIndex = i;
    const int currentParticleGlobalIndex = backgroundSourceIndex[i];

    if (particleType[i] >= 4) {
      vector<PetscInt> index;
      // attached rigid body
      index.clear();
      for (int j = 0; j < neighborLists(i, 0); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];

        for (int axes = 0; axes < velocity_dof; axes++) {
          index.push_back(field_dof * neighborParticleIndex + axes);
        }
      }
      // pressure term
      index.push_back(field_dof * currentParticleGlobalIndex + velocity_dof);

      for (int axes = 0; axes < rigid_body_dof; axes++) {
        vector<PetscInt> &it =
            outProcessIndex[attachedRigidBodyIndex[i] * rigid_body_dof + axes];
        it.insert(it.end(), index.begin(), index.end());
      }
    }
  }

  for (int i = 0; i < outProcessIndex.size(); i++) {
    sort(outProcessIndex[i].begin(), outProcessIndex[i].end());
    outProcessIndex[i].erase(
        unique(outProcessIndex[i].begin(), outProcessIndex[i].end()),
        outProcessIndex[i].end());

    A.set_out_process_col_index(local_out_process_offset + i,
                                outProcessIndex[i]);
  }

  // insert matrix entity
  for (int i = 0; i < local_particle_num; i++) {
    const int currentParticleLocalIndex = i;
    const int currentParticleGlobalIndex = backgroundSourceIndex[i];

    const int iPressureLocal =
        currentParticleLocalIndex * field_dof + velocity_dof;
    const int iPressureGlobal =
        currentParticleGlobalIndex * field_dof + velocity_dof;
    // velocity block
    if (particleType[i] == 0) {
      for (int j = 0; j < velocityNeighborListsLengths(i); j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neighborLists(i, j + 1)];
        // inner fluid particle

        // curl curl u
        for (int axes1 = 0; axes1 < __dim; axes1++) {
          const int iVelocityLocal =
              field_dof * currentParticleLocalIndex + axes1;
          for (int axes2 = 0; axes2 < __dim; axes2++) {
            const int jVelocityGlobal =
                field_dof * neighborParticleIndex + axes2;

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
        const int iVelocityLocal =
            field_dof * currentParticleLocalIndex + axes1;
        const int iVelocityGlobal =
            field_dof * currentParticleGlobalIndex + axes1;

        A.increment(iVelocityLocal, iVelocityGlobal, 1.0);
      }

      // particles on rigid body
      if (particleType[i] >= 4) {
        const int currentRigidBody = attachedRigidBodyIndex[i];
        const int currentRigidBodyLocalOffset =
            local_rigid_body_offset + rigid_body_dof * currentRigidBody;
        const int currentRigidBodyGlobalOffset =
            global_rigid_body_offset + rigid_body_dof * currentRigidBody;

        vec3 rci = coord[i] - rigidBodyPosition[currentRigidBody];
        // non-slip condition
        // translation
        for (int axes1 = 0; axes1 < translation_dof; axes1++) {
          const int iVelocityLocal =
              field_dof * currentParticleLocalIndex + axes1;
          A.increment(iVelocityLocal, currentRigidBodyGlobalOffset + axes1,
                      -1.0);
        }

        // rotation
        for (int axes1 = 0; axes1 < rotation_dof; axes1++) {
          A.increment(field_dof * currentParticleLocalIndex +
                          (axes1 + 2) % translation_dof,
                      currentRigidBodyGlobalOffset + translation_dof + axes1,
                      rci[(axes1 + 1) % translation_dof]);
          A.increment(field_dof * currentParticleLocalIndex +
                          (axes1 + 1) % translation_dof,
                      currentRigidBodyGlobalOffset + translation_dof + axes1,
                      -rci[(axes1 + 2) % translation_dof]);
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
        for (int axes1 = 0; axes1 < translation_dof; axes1++) {
          A.out_process_increment(currentRigidBodyLocalOffset + axes1,
                                  iPressureGlobal, -dA[axes1]);
        }

        for (int axes1 = 0; axes1 < rotation_dof; axes1++) {
          A.out_process_increment(currentRigidBodyLocalOffset +
                                      translation_dof + axes1,
                                  iPressureGlobal,
                                  -rci[(axes1 + 1) % translation_dof] *
                                          dA[(axes1 + 2) % translation_dof] +
                                      rci[(axes1 + 2) % translation_dof] *
                                          dA[(axes1 + 1) % translation_dof]);
        }

        for (int j = 0; j < velocityNeighborListsLengths(i); j++) {
          const int neighborParticleIndex =
              backgroundSourceIndex[neighborLists(i, j + 1)];

          for (int axes3 = 0; axes3 < __dim; axes3++) {
            const int jVelocityGlobal =
                field_dof * neighborParticleIndex + axes3;

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
            for (int axes1 = 0; axes1 < translation_dof; axes1++) {
              A.out_process_increment(currentRigidBodyLocalOffset + axes1,
                                      jVelocityGlobal, f[axes1]);
            }

            // torque balance
            for (int axes1 = 0; axes1 < rotation_dof; axes1++) {
              A.out_process_increment(currentRigidBodyLocalOffset +
                                          translation_dof + axes1,
                                      jVelocityGlobal,
                                      rci[(axes1 + 1) % translation_dof] *
                                              f[(axes1 + 2) % translation_dof] -
                                          rci[(axes1 + 2) % translation_dof] *
                                              f[(axes1 + 1) % translation_dof]);
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
          DivergenceOfVectorPointEvaluation, neumannBoudnaryIndex,
          neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0));

      for (int j = 0; j < neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0);
           j++) {
        const int neighborParticleIndex =
            backgroundSourceIndex[neumannBoundaryNeighborLists(
                neumannBoudnaryIndex, j + 1)];

        for (int axes2 = 0; axes2 < __dim; axes2++) {
          double gradient = 0.0;
          const int jVelocityGlobal = field_dof * neighborParticleIndex + axes2;
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
            field_dof * neighborParticleIndex + velocity_dof;

        const double Aij = pressureAlphas(i, pressureLaplacianIndex, j);

        // laplacian p
        A.increment(iPressureLocal, jPressureGlobal, Aij);
        A.increment(iPressureLocal, iPressureGlobal, -Aij);

        for (int axes1 = 0; axes1 < __dim; axes1++) {
          const int iVelocityLocal =
              field_dof * currentParticleLocalIndex + axes1;

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
            field_dof * neighborParticleIndex + velocity_dof;

        const double Aij = pressureNeumannBoundaryAlphas(
            neumannBoudnaryIndex, pressureNeumannBoundaryLaplacianIndex, j);

        // laplacian p
        A.increment(iPressureLocal, jPressureGlobal, Aij);
        A.increment(iPressureLocal, iPressureGlobal, -Aij);
      }
    }
    // end of pressure block
  } // end of fluid particle loop

  vector<int> idx_neighbor;

  // A.assemble();
  if (num_rigid_body == 0) {
    auto &ff = _multi.get_field_mat(__adaptive_step);
    A.assemble(*ff, field_dof, num_rigid_body, rigid_body_dof);
  } else {
    auto &ff = _multi.get_field_mat(__adaptive_step);
    A.assemble(*ff, field_dof, num_rigid_body, rigid_body_dof);
    A.extract_neighbor_index(idx_neighbor, __dim, num_rigid_body,
                             local_rigid_body_offset, global_rigid_body_offset);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "Matrix assembly duration: %fs\n",
              tEnd - tStart);

  // Interpolation Matrix
  petsc_sparse_matrix &I = _multi.getI(__adaptive_step);
  petsc_sparse_matrix &R = _multi.getR(__adaptive_step);

  if (num_rigid_body != 0 && __adaptive_step != 0) {
    tStart = MPI_Wtime();

    BuildInterpolationAndRestrictionMatrices(I, R, num_rigid_body, __dim);

    tEnd = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,
                "Interpolation matrix building duration: %fs\n", tEnd - tStart);
  }

  vector<double> &rhs = __field.scalar.GetHandle("rhs");
  vector<double> &res = __field.scalar.GetHandle("res");

  rhs.clear();
  res.clear();

  rhs.resize(local_dof);
  res.resize(local_dof);

  for (int i = 0; i < local_dof; i++) {
    rhs[i] = 0.0;
    res[i] = 0.0;
  }

  if (num_rigid_body == 0) {
    for (int i = 0; i < local_particle_num; i++) {
      if (particleType[i] != 0 && particleType[i] < 4) {
        // 2-d Taylor-Green vortex-like flow
        if (__dim == 2) {
          double x = coord[i][0];
          double y = coord[i][1];

          rhs[field_dof * i] = cos(M_PI * x) * sin(M_PI * y);
          rhs[field_dof * i + 1] = -sin(M_PI * x) * cos(M_PI * y);

          const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];
          const double bi =
              pressureNeumannBoundaryBasis.getAlpha0TensorTo0Tensor(
                  DivergenceOfVectorPointEvaluation, neumannBoudnaryIndex,
                  neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0));

          rhs[field_dof * i + velocity_dof] =
              -4.0 * pow(M_PI, 2.0) *
                  (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y)) +
              bi * (normal[i][0] * 2.0 * pow(M_PI, 2.0) * cos(M_PI * x) *
                        sin(M_PI * y) -
                    normal[i][1] * 2.0 * pow(M_PI, 2.0) * sin(M_PI * x) *
                        cos(M_PI * y)) +
              bi * (normal[i][0] * 2.0 * M_PI * sin(2.0 * M_PI * x) +
                    normal[i][1] * 2.0 * M_PI * sin(2.0 * M_PI * y));

          // 2-d cavity flow
          // rhs[field_dof * i] =
          //     1.0 * double(abs(coord[i][1] - __boundingBox[1][1]) < 1e-5);
        }

        // 3-d Taylor-Green vortex-like flow
        if (__dim == 3) {
          double x = coord[i][0];
          double y = coord[i][1];
          double z = coord[i][2];

          rhs[field_dof * i] = cos(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
          rhs[field_dof * i + 1] =
              -2 * sin(M_PI * x) * cos(M_PI * y) * sin(M_PI * z);
          rhs[field_dof * i + 2] =
              sin(M_PI * x) * sin(M_PI * y) * cos(M_PI * z);

          const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];
          const double bi =
              pressureNeumannBoundaryBasis.getAlpha0TensorTo0Tensor(
                  DivergenceOfVectorPointEvaluation, neumannBoudnaryIndex,
                  neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0));

          rhs[field_dof * i + velocity_dof] =
              -4.0 * pow(M_PI, 2.0) *
                  (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y) +
                   cos(2.0 * M_PI * z)) +
              bi * (normal[i][0] * 3.0 * pow(M_PI, 2.0) * cos(M_PI * x) *
                        sin(M_PI * y) * sin(M_PI * z) -
                    normal[i][1] * 6.0 * pow(M_PI, 2.0) * sin(M_PI * x) *
                        cos(M_PI * y) * sin(M_PI * z) +
                    normal[i][2] * 3.0 * pow(M_PI, 2.0) * sin(M_PI * x) *
                        sin(M_PI * y) * cos(M_PI * z)) +
              bi * (normal[i][0] * 2.0 * M_PI * sin(2.0 * M_PI * x) +
                    normal[i][1] * 2.0 * M_PI * sin(2.0 * M_PI * y) +
                    normal[i][2] * 2.0 * M_PI * sin(2.0 * M_PI * z));
        }
      } else if (particleType[i] == 0) {
        if (__dim == 2) {
          double x = coord[i][0];
          double y = coord[i][1];

          rhs[field_dof * i] =
              2.0 * pow(M_PI, 2.0) * cos(M_PI * x) * sin(M_PI * y) +
              2.0 * M_PI * sin(2.0 * M_PI * x);
          rhs[field_dof * i + 1] =
              -2.0 * pow(M_PI, 2.0) * sin(M_PI * x) * cos(M_PI * y) +
              2.0 * M_PI * sin(2.0 * M_PI * y);

          rhs[field_dof * i + velocity_dof] =
              -4.0 * pow(M_PI, 2.0) *
              (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y));
        }

        if (__dim == 3) {
          double x = coord[i][0];
          double y = coord[i][1];
          double z = coord[i][2];

          rhs[field_dof * i] = 3.0 * pow(M_PI, 2) * cos(M_PI * x) *
                                   sin(M_PI * y) * sin(M_PI * z) +
                               2.0 * M_PI * sin(2.0 * M_PI * x);
          rhs[field_dof * i + 1] = -6.0 * pow(M_PI, 2) * sin(M_PI * x) *
                                       cos(M_PI * y) * sin(M_PI * z) +
                                   2.0 * M_PI * sin(2.0 * M_PI * y);
          rhs[field_dof * i + 2] = 3.0 * pow(M_PI, 2) * sin(M_PI * x) *
                                       sin(M_PI * y) * cos(M_PI * z) +
                                   2.0 * M_PI * sin(2.0 * M_PI * z);

          rhs[field_dof * i + velocity_dof] =
              -4.0 * pow(M_PI, 2.0) *
              (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y) + cos(2.0 * M_PI * z));
        }
      }
    }
  }

  if (__myID == __MPISize - 1) {
    for (int i = 0; i < num_rigid_body; i++) {
      rhs[local_rigid_body_offset + i * rigid_body_dof + translation_dof] =
          pow(-1, i + 1);
    }
  }

  // for (int i = 0; i < local_particle_num; i++) {
  //   if (particleType[i] != 0 && particleType[i] < 4) {
  //     // 2-d Taylor-Green vortex-like flow
  //     if (__dim == 2) {
  //       double x = coord[i][0];
  //       double y = coord[i][1];

  //       double u1, u2, v1, v2;

  //       vec3 rci1 = coord[i] - rigidBodyPosition[0];
  //       double r1 = sqrt(pow(rci1[0], 2.0) + pow(rci1[1], 2.0));
  //       double phi1 = atan2(rci1[1], rci1[0]);

  //       solution(r1, phi1, omega, u1, v1);

  //       vec3 rci2 = coord[i] - rigidBodyPosition[1];
  //       double r2 = sqrt(pow(rci2[0], 2.0) + pow(rci2[1], 2.0));
  //       double phi2 = atan2(rci2[1], rci2[0]);

  //       solution(r2, phi2, -omega, u2, v2);

  //       // rhs[field_dof * i] = 0.1 * y + u1 + u2;
  //       // rhs[field_dof * i + 1] = v1 + v2;
  //       rhs[field_dof * i] = 0.1 * y;
  //     }
  //   }
  // }

  // vector<double> &rigidBodySize = __rigidBody.scalar.GetHandle("size");

  // double u = 1.0;
  // double RR = rigidBodySize[0];

  // for (int i = 0; i < local_particle_num; i++) {
  //   if (particleType[i] != 0 && particleType[i] < 4) {
  //     double x = coord[i][0];
  //     double y = coord[i][1];
  //     double z = coord[i][2];

  //     const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];
  //     const double bi =
  //     pressureNeumannBoundaryBasis.getAlpha0TensorTo0Tensor(
  //         DivergenceOfVectorPointEvaluation, neumannBoudnaryIndex,
  //         neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0));

  //     double r = sqrt(x * x + y * y + z * z);
  //     double theta = acos(z / r);
  //     double phi = atan2(y, x);

  //     double vr = u * cos(theta) *
  //                 (1 - (3 * RR) / (2 * r) + pow(RR, 3) / (2 * pow(r, 3)));
  //     double vt = -u * sin(theta) *
  //                 (1 - (3 * RR) / (4 * r) - pow(RR, 3) / (4 * pow(r, 3)));

  //     double pr = 3 * RR / pow(r, 3) * u * cos(theta);
  //     double pt = 3 / 2 * RR / pow(r, 3) * u * sin(theta);

  //     rhs[field_dof * i] =
  //         sin(theta) * cos(phi) * vr + cos(theta) * cos(phi) * vt;
  //     rhs[field_dof * i + 1] =
  //         sin(theta) * sin(phi) * vr + cos(theta) * sin(phi) * vt;
  //     rhs[field_dof * i + 2] = cos(theta) * vr - sin(theta) * vt;

  //     double p1 = sin(theta) * cos(phi) * pr + cos(theta) * cos(phi) * pt;
  //     double p2 = sin(theta) * sin(phi) * pr + cos(theta) * sin(phi) * pt;
  //     double p3 = cos(theta) * pr - sin(theta) * pt;

  //     // rhs[field_dof * i + 3] =
  //     //     bi * (normal[i][0] * p1 + normal[i][1] * p2 + normal[i][2] *
  //     p3);
  //   } else if (particleType[i] >= 4) {
  //     double x = coord[i][0];
  //     double y = coord[i][1];
  //     double z = coord[i][2];

  //     double r = sqrt(x * x + y * y + z * z);
  //     double theta = acos(z / r);
  //     double phi = atan2(y, x);

  //     const int neumannBoudnaryIndex = fluid2NeumannBoundary[i];
  //     const double bi =
  //     pressureNeumannBoundaryBasis.getAlpha0TensorTo0Tensor(
  //         DivergenceOfVectorPointEvaluation, neumannBoudnaryIndex,
  //         neumannBoundaryNeighborLists(neumannBoudnaryIndex, 0));

  //     double pr = 3 * RR / pow(r, 3) * u * cos(theta);
  //     double pt = 3 / 2 * RR / pow(r, 3) * u * sin(theta);

  //     double p1 = sin(theta) * cos(phi) * pr + cos(theta) * cos(phi) * pt;
  //     double p2 = sin(theta) * sin(phi) * pr + cos(theta) * sin(phi) * pt;
  //     double p3 = cos(theta) * pr - sin(theta) * pt;

  //     // rhs[field_dof * i + 3] =
  //     //     bi * (normal[i][0] * p1 + normal[i][1] * p2 + normal[i][2] *
  //     p3);
  //   }
  // }

  // if (__myID == __MPISize - 1) {
  //   rhs[local_rigid_body_offset + 2] = 6 * M_PI * RR * u;
  // }

  int inner_counter = 0;
  for (int i = 0; i < local_particle_num; i++) {
    if (particleType[i] == 0)
      inner_counter++;
  }
  MPI_Allreduce(MPI_IN_PLACE, &inner_counter, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "total inner particle count: %d\n",
              inner_counter);

  // make sure pressure term is orthogonal to the constant
  double rhs_pressure_sum = 0.0;
  for (int i = 0; i < local_particle_num; i++) {
    rhs_pressure_sum += rhs[field_dof * i + velocity_dof];
  }
  MPI_Allreduce(MPI_IN_PLACE, &rhs_pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  rhs_pressure_sum /= global_particle_num;
  for (int i = 0; i < local_particle_num; i++) {
    rhs[field_dof * i + velocity_dof] -= rhs_pressure_sum;
  }

  // A.Write("A.txt");

  PetscLogDefaultBegin();

  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();
  if (num_rigid_body == 0) {
    // A.Solve(rhs, res, __dim);
    // A.Solve(rhs, res);
    A.solve(rhs, res, field_dof);
  } else {
    if (__adaptive_step != 0)
      initial_guess_from_previous_adaptive_step(I, res);
    // A.Solve(rhs, res, idx_neighbor, __dim, num_rigid_body, __adaptive_step,
    // I, R);
    if (_multi.solve(rhs, res, idx_neighbor) != 0) {
      ofstream output;
      if (__myID == 0) {
        output.open("traj_new.txt", ios::trunc);
        for (int num = 0; num < num_rigid_body; num++) {
          for (int j = 0; j < 2; j++) {
            output << setprecision(15) << rigidBodyPosition[num][j] << '\t';
          }
          output << 0.10 << '\t' << 1 << endl;
        }
        output.close();
      }
      A.solve(rhs, res);
    }
    // A.Solve(rhs, res);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "linear system solving duration: %fs\n",
              tEnd - tStart);

  PetscViewer viewer;
  if (__viewer > 0) {
    PetscViewerASCIIGetStdout(PETSC_COMM_WORLD, &viewer);
    PetscLogView(viewer);
  }

  // copy data
  static vector<vec3> &velocity = __field.vector.GetHandle("fluid velocity");
  static vector<double> &pressure = __field.scalar.GetHandle("fluid pressure");
  pressure.resize(local_particle_num);
  velocity.resize(local_particle_num);

  double pressure_sum = 0.0;
  for (int i = 0; i < local_particle_num; i++) {
    pressure[i] = res[field_dof * i + velocity_dof];
    pressure_sum += pressure[i];
    for (int axes1 = 0; axes1 < __dim; axes1++)
      velocity[i][axes1] = res[field_dof * i + axes1];
  }

  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  double average_pressure = pressure_sum / global_particle_num;
  for (int i = 0; i < local_particle_num; i++) {
    pressure[i] -= average_pressure;
  }

  // check data
  double true_pressure_mean = 0.0;
  double pressure_mean = 0.0;
  for (int i = 0; i < local_particle_num; i++) {
    if (__dim == 2) {
      double x = coord[i][0];
      double y = coord[i][1];

      double true_pressure = -cos(2.0 * M_PI * x) - cos(2.0 * M_PI * y);

      true_pressure_mean += true_pressure;
      pressure_mean += pressure[i];
    }

    if (__dim == 3) {
      // double x = coord[i][0];
      // double y = coord[i][1];
      // double z = coord[i][2];

      // double r = sqrt(x * x + y * y + z * z);
      // double theta = acos(z / r);

      // double true_pressure = -3 / 2 * RR / pow(r, 2.0) * u * cos(theta);

      // true_pressure_mean += true_pressure;
      // pressure_mean += pressure[i];
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &true_pressure_mean, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &pressure_mean, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  true_pressure_mean /= global_particle_num;
  pressure_mean /= global_particle_num;

  double error_velocity = 0.0;
  double norm_velocity = 0.0;
  double error_pressure = 0.0;
  double norm_pressure = 0.0;
  for (int i = 0; i < local_particle_num; i++) {
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
      // double x = coord[i][0];
      // double y = coord[i][1];
      // double z = coord[i][2];

      // double r = sqrt(x * x + y * y + z * z);
      // double theta = acos(z / r);
      // double phi = atan2(y, x);

      // double vr = u * cos(theta) *
      //             (1 - (3 * RR) / (2 * r) + pow(RR, 3) / (2 * pow(r, 3)));
      // double vt = -u * sin(theta) *
      //             (1 - (3 * RR) / (4 * r) - pow(RR, 3) / (4 * pow(r, 3)));

      // double pr = 3 * RR / pow(r, 3) * u * cos(theta);
      // double pt = 3 / 2 * RR / pow(r, 3) * u * sin(theta);

      // double true_velocity[3];

      // true_velocity[0] =
      //     sin(theta) * cos(phi) * vr + cos(theta) * cos(phi) * vt;
      // true_velocity[1] =
      //     sin(theta) * sin(phi) * vr + cos(theta) * sin(phi) * vt;
      // true_velocity[2] = cos(theta) * vr - sin(theta) * vt;

      // double true_pressure =
      //     -3 / 2 * RR / pow(r, 2.0) * u * cos(theta) - true_pressure_mean;

      // error_velocity += pow(true_velocity[0] - velocity[i][0], 2) +
      //                   pow(true_velocity[1] - velocity[i][1], 2) +
      //                   pow(true_velocity[2] - velocity[i][2], 2);
      // error_pressure += pow(true_pressure - pressure[i], 2);

      // norm_velocity += pow(true_velocity[0], 2) + pow(true_velocity[1], 2) +
      //                  pow(true_velocity[2], 2);
      // norm_pressure += pow(true_pressure, 2);
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
              sqrt(error_pressure / global_particle_num));
  PetscPrintf(MPI_COMM_WORLD, "RMS velocity error: %.10f\n",
              sqrt(error_velocity / global_particle_num));

  MPI_Barrier(MPI_COMM_WORLD);

  // vector<vec3> recvVelocity;
  // DataSwapAmongNeighbor(velocity, recvVelocity);
  // vector<vec3> backgroundVelocity;

  // for (int i = 0; i < local_particle_num; i++) {
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
  // for (int i = 0; i < local_particle_num; i++) {
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
    for (int i = 0; i < num_rigid_body; i++) {
      for (int j = 0; j < translation_dof; j++) {
        rigidBodyVelocity[i][j] =
            res[local_rigid_body_offset + i * rigid_body_dof + j];
      }
      for (int j = 0; j < rotation_dof; j++) {
        rigidBodyAngularVelocity[i][j] =
            res[local_rigid_body_offset + i * rigid_body_dof + translation_dof +
                j];
      }
    }
  }

  // communicate velocity and angular velocity
  vector<double> translation_velocity(num_rigid_body * translation_dof);
  vector<double> angular_velocity(num_rigid_body * rotation_dof);

  if (__myID == __MPISize - 1) {
    for (int i = 0; i < num_rigid_body; i++) {
      for (int j = 0; j < translation_dof; j++) {
        translation_velocity[i * translation_dof + j] = rigidBodyVelocity[i][j];
      }
      for (int j = 0; j < rotation_dof; j++) {
        angular_velocity[i * rotation_dof + j] = rigidBodyAngularVelocity[i][j];
      }
    }
  }

  MPI_Bcast(translation_velocity.data(), num_rigid_body * translation_dof,
            MPI_DOUBLE, __MPISize - 1, MPI_COMM_WORLD);
  MPI_Bcast(angular_velocity.data(), num_rigid_body * rotation_dof, MPI_DOUBLE,
            __MPISize - 1, MPI_COMM_WORLD);

  if (__myID != __MPISize - 1) {
    for (int i = 0; i < num_rigid_body; i++) {
      for (int j = 0; j < translation_dof; j++) {
        rigidBodyVelocity[i][j] = translation_velocity[i * translation_dof + j];
      }
      for (int j = 0; j < rotation_dof; j++) {
        rigidBodyAngularVelocity[i][j] = angular_velocity[i * rotation_dof + j];
      }
    }
  }

  if (__adaptiveRefinement) {
    auto &old_coord = __field.vector.GetHandle("old coord");
    auto &old_particle_type = __field.index.GetHandle("old particle type");
    auto &old_background_coord =
        __background.vector.GetHandle("old source coord");
    auto &old_background_index =
        __background.index.GetHandle("old source index");

    old_coord.clear();
    old_particle_type.clear();
    old_background_coord.clear();
    old_background_index.clear();

    for (int i = 0; i < coord.size(); i++) {
      old_coord.push_back(coord[i]);
      old_particle_type.push_back(particleType[i]);
      old_background_coord.push_back(backgroundSourceCoord[i]);
      old_background_index.push_back(backgroundSourceIndex[i]);
    }
  }
}
