#include "stokes_equation.hpp"
#include "DivergenceFree.hpp"
#include "gmls_solver.hpp"
#include "petsc_sparse_matrix.hpp"

#include <iomanip>

using namespace std;
using namespace Compadre;

double Wab(double r, double h) {
  int p = 4;
  return pow(1.0 - abs(r / h), p) * double((1.0 - abs(r / h)) > 0.0);
}

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

void stokes_equation::init(shared_ptr<particle_geometry> _geo_mgr,
                           shared_ptr<rigid_body_manager> _rb_mgr,
                           const int _poly_order, const int _dim,
                           const int _error_estimation_method,
                           const double _epsilon_multiplier,
                           const double _eta) {
  geo_mgr = _geo_mgr;
  rb_mgr = _rb_mgr;
  eta = _eta;
  dim = _dim;
  poly_order = _poly_order;
  error_esimation_method = _error_estimation_method;

  multi_mgr = make_shared<stokes_multilevel>();

  if (_epsilon_multiplier != 0.0) {
    epsilon_multiplier = _epsilon_multiplier;
  } else {
    epsilon_multiplier = _poly_order + 1.0;
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  current_refinement_level = 0;
  multi_mgr->reset();
  multi_mgr->init(dim, geo_mgr);
  multi_mgr->set_num_rigid_body(rb_mgr->get_rigid_body_num());
}

void stokes_equation::update() {
  multi_mgr->add_new_level();

  build_coefficient_matrix();
  build_rhs();
  solve_step();
  check_solution();
  calculate_error();

  current_refinement_level++;
}

void stokes_equation::reset() {
  current_refinement_level = 0;
  multi_mgr->reset();
}

void stokes_equation::build_coefficient_matrix() {
  // prepare data
  auto &source_coord = *(geo_mgr->get_current_work_ghost_particle_coord());
  auto &source_index = *(geo_mgr->get_current_work_ghost_particle_index());
  auto &coord = *(geo_mgr->get_current_work_particle_coord());
  auto &normal = *(geo_mgr->get_current_work_particle_normal());
  auto &p_spacing = *(geo_mgr->get_current_work_particle_p_spacing());
  auto &spacing = *(geo_mgr->get_current_work_particle_spacing());
  auto &adaptive_level = *(geo_mgr->get_current_work_particle_adaptive_level());
  auto &particle_type = *(geo_mgr->get_current_work_particle_type());
  auto &attached_rigid_body =
      *(geo_mgr->get_current_work_particle_attached_rigid_body());
  auto &num_neighbor = *(geo_mgr->get_current_work_particle_num_neighbor());

  // update basis
  pressure_basis.reset();
  velocity_basis.reset();
  pressure_neumann_basis.reset();

  pressure_basis = make_shared<GMLS>(
      ScalarTaylorPolynomial, StaggeredEdgeAnalyticGradientIntegralSample,
      poly_order, dim, "SVD", "STANDARD");
  velocity_basis =
      make_shared<GMLS>(DivergenceFreeVectorTaylorPolynomial, VectorPointSample,
                        poly_order, dim, "SVD", "STANDARD");
  pressure_neumann_basis = make_shared<GMLS>(
      ScalarTaylorPolynomial, StaggeredEdgeAnalyticGradientIntegralSample,
      poly_order, dim, "SVD", "STANDARD", "NEUMANN_GRAD_SCALAR");

  vector<vec3> &rigid_body_position = rb_mgr->get_position();
  const int num_rigid_body = rb_mgr->get_rigid_body_num();

  const int number_of_batches = 1;

  double timer1, timer2;
  MPI_Barrier(MPI_COMM_WORLD);
  timer1 = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "\nSolving GMLS subproblems...\n");

  int local_particle_num;
  int global_particle_num;

  local_particle_num = coord.size();
  MPI_Allreduce(&local_particle_num, &global_particle_num, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  int num_source_coord = source_coord.size();
  int num_target_coord = coord.size();

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> source_coord_device(
      "source coordinates", num_source_coord, 3);
  Kokkos::View<double **>::HostMirror source_coord_host =
      Kokkos::create_mirror_view(source_coord_device);

  for (size_t i = 0; i < num_source_coord; i++) {
    for (int j = 0; j < 3; j++) {
      source_coord_host(i, j) = source_coord[i][j];
    }
  }

  int num_neumann_target_coord = 0;
  for (int i = 0; i < local_particle_num; i++) {
    if (particle_type[i] != 0) {
      num_neumann_target_coord++;
    }
  }

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> target_coord_device(
      "target coordinates", num_target_coord, 3);
  Kokkos::View<double **>::HostMirror target_coord_host =
      Kokkos::create_mirror_view(target_coord_device);
  Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
      neumann_target_coord_device("neumann target coordinates",
                                  num_neumann_target_coord, 3);
  Kokkos::View<double **>::HostMirror neumann_target_coord_host =
      Kokkos::create_mirror_view(neumann_target_coord_device);

  // create target coords
  int counter;
  neumann_map.resize(local_particle_num);
  counter = 0;
  for (int i = 0; i < local_particle_num; i++) {
    for (int j = 0; j < 3; j++) {
      target_coord_host(i, j) = coord[i][j];
    }
    neumann_map[i] = counter;
    if (particle_type[i] != 0) {
      for (int j = 0; j < 3; j++) {
        neumann_target_coord_host(counter, j) = coord[i][j];
      }
      counter++;
    }
  }

  Kokkos::deep_copy(source_coord_device, source_coord_host);
  Kokkos::deep_copy(target_coord_device, target_coord_host);
  Kokkos::deep_copy(neumann_target_coord_device, neumann_target_coord_host);

  // tangent bundle for neumann boundary particles
  Kokkos::View<double ***, Kokkos::DefaultExecutionSpace> tangent_bundle_device(
      "tangent bundles", num_neumann_target_coord, dim, dim);
  Kokkos::View<double ***>::HostMirror tangent_bundle_host =
      Kokkos::create_mirror_view(tangent_bundle_device);

  counter = 0;
  for (int i = 0; i < local_particle_num; i++) {
    if (particle_type[i] != 0) {
      if (dim == 3) {
        tangent_bundle_host(counter, 0, 0) = 0.0;
        tangent_bundle_host(counter, 0, 1) = 0.0;
        tangent_bundle_host(counter, 0, 2) = 0.0;
        tangent_bundle_host(counter, 1, 0) = 0.0;
        tangent_bundle_host(counter, 1, 1) = 0.0;
        tangent_bundle_host(counter, 1, 2) = 0.0;
        tangent_bundle_host(counter, 2, 0) = normal[i][0];
        tangent_bundle_host(counter, 2, 1) = normal[i][1];
        tangent_bundle_host(counter, 2, 2) = normal[i][2];
      }
      if (dim == 2) {
        tangent_bundle_host(counter, 0, 0) = 0.0;
        tangent_bundle_host(counter, 0, 1) = 0.0;
        tangent_bundle_host(counter, 1, 0) = normal[i][0];
        tangent_bundle_host(counter, 1, 1) = normal[i][1];
      }
      counter++;
    }
  }

  Kokkos::deep_copy(tangent_bundle_device, tangent_bundle_host);

  // neighbor search
  auto point_cloud_search(CreatePointCloudSearch(source_coord_host, dim));

  auto min_num_neighbor = Compadre::GMLS::getNP(
      poly_order, dim, DivergenceFreeVectorTaylorPolynomial);

  int estimated_max_num_neighbor =
      pow(pow(2, dim), 2) * pow(epsilon_multiplier, dim);

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> neighbor_list_device(
      "neighbor lists", num_target_coord, estimated_max_num_neighbor);
  Kokkos::View<int **>::HostMirror neighbor_list_host =
      Kokkos::create_mirror_view(neighbor_list_device);

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
      neumann_neighbor_list_device("neumann boundary neighbor lists",
                                   num_neumann_target_coord,
                                   estimated_max_num_neighbor);
  Kokkos::View<int **>::HostMirror neumann_neighbor_list_host =
      Kokkos::create_mirror_view(neumann_neighbor_list_device);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> epsilon_device(
      "h supports", num_target_coord);
  Kokkos::View<double *>::HostMirror epsilon_host =
      Kokkos::create_mirror_view(epsilon_device);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> neumann_epsilon_device(
      "neumann boundary h supports", num_neumann_target_coord);
  Kokkos::View<double *>::HostMirror neumann_epsilon_host =
      Kokkos::create_mirror_view(neumann_epsilon_device);

  double max_epsilon = 0.0;
  epsilon.resize(local_particle_num);
  for (int i = 0; i < num_target_coord; i++) {
    epsilon_host(i) = spacing[i] * epsilon_multiplier + 1e-15;
    epsilon[i] = epsilon_host(i);
    if (epsilon[i] > max_epsilon) {
      max_epsilon = epsilon_host(i);
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &max_epsilon, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);

  // ensure every particle has enough neighbors
  bool pass_neighbor_search = false;
  while (!pass_neighbor_search) {
    geo_mgr->ghost_forward(epsilon, ghost_epsilon);

    Kokkos::View<double *, Kokkos::DefaultExecutionSpace> ghost_epsilon_device(
        "background h supports", ghost_epsilon.size());
    Kokkos::View<double *>::HostMirror ghost_epsilon_host =
        Kokkos::create_mirror_view(ghost_epsilon_device);

    for (int i = 0; i < ghost_epsilon.size(); i++) {
      ghost_epsilon_host(i) = ghost_epsilon[i];
    }

    // pointCloudSearch.generateNeighborListsFromKNNSearch(
    //     false, targetCoords, neighborLists, epsilon, 10, 1.05);

    point_cloud_search.generate2DSymmetricNeighborListsFromRadiusSearch(
        false, target_coord_host, neighbor_list_host, ghost_epsilon_host, 0.0,
        max_epsilon);

    bool pass_neighbor_num_check = true;
    int min_neighbor = 1000;
    int max_neighbor = 0;
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
      if (neighbor_list_host(i, 0) < min_neighbor)
        min_neighbor = neighbor_list_host(i, 0);
      if (neighbor_list_host(i, 0) > max_neighbor)
        max_neighbor = neighbor_list_host(i, 0);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &min_neighbor, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_neighbor, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    PetscPrintf(MPI_COMM_WORLD, "min neighbor: %d\n", min_neighbor);
    PetscPrintf(MPI_COMM_WORLD, "max neighbor: %d\n", max_neighbor);

    int process_counter = 0;
    if (!pass_neighbor_num_check) {
      process_counter = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, &process_counter, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "process counter: %d\n", process_counter);

    if (process_counter == 0) {
      pass_neighbor_search = true;
    }
  }

  counter = 0;
  for (int i = 0; i < num_target_coord; i++) {
    if (particle_type[i] != 0) {
      neumann_epsilon_host(counter) = epsilon_host(i);
      neumann_neighbor_list_host(counter, 0) = neighbor_list_host(i, 0);
      for (int j = 0; j < neighbor_list_host(i, 0); j++) {
        neumann_neighbor_list_host(counter, j + 1) =
            neighbor_list_host(i, j + 1);
      }

      counter++;
    }
  }

  num_neighbor.resize(num_target_coord);
  for (int i = 0; i < num_target_coord; i++) {
    num_neighbor[i] = neighbor_list_host(i, 0);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  Kokkos::deep_copy(neighbor_list_device, neighbor_list_host);
  Kokkos::deep_copy(epsilon_device, epsilon_host);
  Kokkos::deep_copy(neumann_neighbor_list_device, neumann_neighbor_list_host);
  Kokkos::deep_copy(neumann_epsilon_device, neumann_epsilon_host);

  // pressure basis
  pressure_basis->setProblemData(neighbor_list_device, source_coord_device,
                                 target_coord_device, epsilon_device);

  vector<TargetOperation> pressure_operation(4);
  pressure_operation[0] = DivergenceOfVectorPointEvaluation;
  pressure_operation[1] = GradientOfScalarPointEvaluation;
  pressure_operation[2] = ScalarPointEvaluation;

  pressure_basis->clearTargets();
  pressure_basis->addTargets(pressure_operation);

  pressure_basis->setWeightingType(WeightingFunctionType::Power);
  pressure_basis->setWeightingPower(4);
  pressure_basis->setOrderOfQuadraturePoints(2);
  pressure_basis->setDimensionOfQuadraturePoints(1);
  pressure_basis->setQuadratureType("LINE");

  pressure_basis->generateAlphas(number_of_batches, true);

  auto pressure_alpha = pressure_basis->getAlphas();

  const int pressure_laplacian_index =
      pressure_basis->getAlphaColumnOffset(pressure_operation[0], 0, 0, 0, 0);
  vector<int> pressure_gradient_index;
  for (int i = 0; i < dim; i++)
    pressure_gradient_index.push_back(pressure_basis->getAlphaColumnOffset(
        pressure_operation[1], i, 0, 0, 0));

  // velocity basis
  velocity_basis->setProblemData(neighbor_list_device, source_coord_device,
                                 target_coord_device, epsilon_device);

  vector<TargetOperation> velocity_operation(3);
  velocity_operation[0] = CurlCurlOfVectorPointEvaluation;
  velocity_operation[1] = GradientOfVectorPointEvaluation;
  velocity_operation[2] = ScalarPointEvaluation;

  velocity_basis->clearTargets();
  velocity_basis->addTargets(velocity_operation);

  velocity_basis->setWeightingType(WeightingFunctionType::Power);
  velocity_basis->setWeightingPower(4);

  velocity_basis->generateAlphas(number_of_batches, true);

  auto velocity_alpha = velocity_basis->getAlphas();

  vector<int> velocity_curl_curl_index(pow(dim, 2));
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      velocity_curl_curl_index[i * dim + j] =
          velocity_basis->getAlphaColumnOffset(CurlCurlOfVectorPointEvaluation,
                                               i, 0, j, 0);
    }
  }
  vector<int> velocity_gradient_index(pow(dim, 3));
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        velocity_gradient_index[(i * dim + j) * dim + k] =
            velocity_basis->getAlphaColumnOffset(
                GradientOfVectorPointEvaluation, i, j, k, 0);
      }
    }
  }

  // pressure Neumann boundary basis
  pressure_neumann_basis->setProblemData(
      neumann_neighbor_list_device, source_coord_device,
      neumann_target_coord_device, neumann_epsilon_device);

  pressure_neumann_basis->setTangentBundle(tangent_bundle_device);

  vector<TargetOperation> pressure_neumann_operation(3);
  pressure_neumann_operation[0] = DivergenceOfVectorPointEvaluation;
  pressure_neumann_operation[1] = GradientOfScalarPointEvaluation;
  pressure_neumann_operation[2] = ScalarPointEvaluation;

  pressure_neumann_basis->clearTargets();
  pressure_neumann_basis->addTargets(pressure_neumann_operation);

  pressure_neumann_basis->setWeightingType(WeightingFunctionType::Power);
  pressure_neumann_basis->setWeightingPower(4);
  pressure_neumann_basis->setOrderOfQuadraturePoints(2);
  pressure_neumann_basis->setDimensionOfQuadraturePoints(1);
  pressure_neumann_basis->setQuadratureType("LINE");

  pressure_neumann_basis->generateAlphas(number_of_batches, true);

  auto pressure_neumann_alpha = pressure_neumann_basis->getAlphas();

  const int pressure_neumann_laplacian_index =
      pressure_neumann_basis->getAlphaColumnOffset(
          pressure_neumann_operation[0], 0, 0, 0, 0);

  MPI_Barrier(MPI_COMM_WORLD);
  timer2 = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "GMLS solving duration: %fs\n",
              timer2 - timer1);

  timer1 = timer2;

  // matrix assembly
  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "\nGenerating Stokes Matrix...\n");

  const int translation_dof = (dim == 3 ? 3 : 2);
  const int rotation_dof = (dim == 3 ? 3 : 1);
  const int rigid_body_dof = (dim == 3 ? 6 : 3);

  int field_dof = dim + 1;
  int velocity_dof = dim;

  int local_velocity_dof = local_particle_num * dim;
  int global_velocity_dof =
      global_particle_num * dim + rigid_body_dof * num_rigid_body;
  int local_pressure_dof = local_particle_num;
  int global_pressure_dof = global_particle_num;

  if (rank == size - 1) {
    local_velocity_dof += rigid_body_dof * num_rigid_body;
  }

  vector<int> particle_num_per_process;
  particle_num_per_process.resize(size);

  MPI_Allgather(&local_particle_num, 1, MPI_INT,
                particle_num_per_process.data(), 1, MPI_INT, MPI_COMM_WORLD);

  int local_rigid_body_offset = particle_num_per_process[size - 1] * field_dof;
  int global_rigid_body_offset = global_particle_num * field_dof;
  int local_out_process_offset = particle_num_per_process[size - 1] * field_dof;

  int local_dof = local_velocity_dof + local_pressure_dof;
  int global_dof = global_velocity_dof + global_pressure_dof;

  int out_process_row = rigid_body_dof * num_rigid_body;

  petsc_sparse_matrix &A = *(multi_mgr->getA(current_refinement_level));
  A.resize(local_dof, local_dof, global_dof, out_process_row,
           local_out_process_offset);

  MPI_Barrier(MPI_COMM_WORLD);
  timer1 = MPI_Wtime();

  // compute matrix graph
  vector<vector<PetscInt>> out_process_index(out_process_row);

  for (int i = 0; i < local_particle_num; i++) {
    const int current_particle_local_index = i;
    const int current_particle_global_index = source_index[i];

    const int pressure_local_index =
        current_particle_local_index * field_dof + velocity_dof;
    const int pressure_global_index =
        current_particle_global_index * field_dof + velocity_dof;

    vector<PetscInt> index;
    if (particle_type[i] == 0) {
      // velocity block
      index.clear();
      for (int j = 0; j < neighbor_list_host(i, 0); j++) {
        const int neighbor_particle_index =
            source_index[neighbor_list_host(i, j + 1)];

        for (int axes = 0; axes < field_dof; axes++) {
          index.push_back(field_dof * neighbor_particle_index + axes);
        }
      }

      for (int axes = 0; axes < velocity_dof; axes++) {
        A.set_col_index(current_particle_local_index * field_dof + axes, index);
      }

      // pressure block
      index.clear();
      for (int j = 0; j < neighbor_list_host(i, 0); j++) {
        const int neighbor_particle_index =
            source_index[neighbor_list_host(i, j + 1)];

        index.push_back(field_dof * neighbor_particle_index + velocity_dof);
      }

      A.set_col_index(current_particle_local_index * field_dof + velocity_dof,
                      index);
    }

    if (particle_type[i] != 0 && particle_type[i] < 4) {
      // velocity block
      index.clear();
      index.resize(1);
      for (int axes = 0; axes < velocity_dof; axes++) {
        index[0] = current_particle_global_index * field_dof + axes;
        A.set_col_index(current_particle_local_index * field_dof + axes, index);
      }

      // pressure block
      index.clear();
      for (int j = 0; j < neighbor_list_host(i, 0); j++) {
        const int neighbor_particle_index =
            source_index[neighbor_list_host(i, j + 1)];

        for (int axes = 0; axes < field_dof; axes++) {
          index.push_back(field_dof * neighbor_particle_index + axes);
        }
      }

      A.set_col_index(current_particle_local_index * field_dof + velocity_dof,
                      index);
    }

    if (particle_type[i] >= 4) {
      // velocity block
      index.clear();
      index.resize(2 + rotation_dof);
      for (int axes = 0; axes < rotation_dof; axes++) {
        index[2 + axes] = global_rigid_body_offset +
                          attached_rigid_body[i] * rigid_body_dof +
                          translation_dof + axes;
      }

      for (int axes = 0; axes < velocity_dof; axes++) {
        index[0] = current_particle_global_index * field_dof + axes;
        index[1] = global_rigid_body_offset +
                   attached_rigid_body[i] * rigid_body_dof + axes;
        A.set_col_index(current_particle_local_index * field_dof + axes, index);
      }

      // pressure block
      index.clear();
      for (int j = 0; j < neighbor_list_host(i, 0); j++) {
        const int neighbor_particle_index =
            source_index[neighbor_list_host(i, j + 1)];

        for (int axes = 0; axes < field_dof; axes++) {
          index.push_back(field_dof * neighbor_particle_index + axes);
        }
      }

      A.set_col_index(current_particle_local_index * field_dof + velocity_dof,
                      index);
    }
  }

  // outprocess graph
  for (int i = 0; i < local_particle_num; i++) {
    const int current_particle_local_index = i;
    const int current_particle_global_index = source_index[i];

    if (particle_type[i] >= 4) {
      vector<PetscInt> index;
      // attached rigid body
      index.clear();
      for (int j = 0; j < neighbor_list_host(i, 0); j++) {
        const int neighbor_particle_index =
            source_index[neighbor_list_host(i, j + 1)];

        for (int axes = 0; axes < velocity_dof; axes++) {
          index.push_back(field_dof * neighbor_particle_index + axes);
        }
      }
      // pressure term
      index.push_back(field_dof * current_particle_global_index + velocity_dof);

      for (int axes = 0; axes < rigid_body_dof; axes++) {
        vector<PetscInt> &it =
            out_process_index[attached_rigid_body[i] * rigid_body_dof + axes];
        it.insert(it.end(), index.begin(), index.end());
      }
    }
  }

  for (int i = 0; i < out_process_index.size(); i++) {
    sort(out_process_index[i].begin(), out_process_index[i].end());
    out_process_index[i].erase(
        unique(out_process_index[i].begin(), out_process_index[i].end()),
        out_process_index[i].end());

    A.set_out_process_col_index(local_out_process_offset + i,
                                out_process_index[i]);
  }

  // insert matrix entity
  for (int i = 0; i < local_particle_num; i++) {
    const int current_particle_local_index = i;
    const int current_particle_global_index = source_index[i];

    const int pressure_local_index =
        current_particle_local_index * field_dof + velocity_dof;
    const int pressure_global_index =
        current_particle_global_index * field_dof + velocity_dof;
    // velocity block
    if (particle_type[i] == 0) {
      for (int j = 0; j < neighbor_list_host(i, 0); j++) {
        const int neighbor_particle_index =
            source_index[neighbor_list_host(i, j + 1)];
        // inner fluid particle

        // curl curl u
        for (int axes1 = 0; axes1 < dim; axes1++) {
          const int velocity_local_index =
              field_dof * current_particle_local_index + axes1;
          for (int axes2 = 0; axes2 < dim; axes2++) {
            const int velocity_global_index =
                field_dof * neighbor_particle_index + axes2;

            auto alpha_index = velocity_basis->getAlphaIndexHost(
                i, velocity_curl_curl_index[axes1 * dim + axes2]);
            const double Lij = eta * velocity_alpha(alpha_index + j);

            A.increment(velocity_local_index, velocity_global_index, Lij);
          }
        }
      }
    } else {
      // wall boundary (including particles on rigid body)
      for (int axes1 = 0; axes1 < dim; axes1++) {
        const int velocity_local_index =
            field_dof * current_particle_local_index + axes1;
        const int iVelocityGlobal =
            field_dof * current_particle_global_index + axes1;

        A.increment(velocity_local_index, iVelocityGlobal, 1.0);
      }

      // particles on rigid body
      if (particle_type[i] >= 4) {
        const int current_rigid_body_index = attached_rigid_body[i];
        const int current_rigid_body_local_offset =
            local_rigid_body_offset + rigid_body_dof * current_rigid_body_index;
        const int current_rigid_body_global_offset =
            global_rigid_body_offset +
            rigid_body_dof * current_rigid_body_index;

        vec3 rci = coord[i] - rigid_body_position[current_rigid_body_index];
        // non-slip condition
        // translation
        for (int axes1 = 0; axes1 < translation_dof; axes1++) {
          const int velocity_local_index =
              field_dof * current_particle_local_index + axes1;
          A.increment(velocity_local_index,
                      current_rigid_body_global_offset + axes1, -1.0);
        }

        // rotation
        for (int axes1 = 0; axes1 < rotation_dof; axes1++) {
          A.increment(field_dof * current_particle_local_index +
                          (axes1 + 2) % translation_dof,
                      current_rigid_body_global_offset + translation_dof +
                          axes1,
                      rci[(axes1 + 1) % translation_dof]);
          A.increment(field_dof * current_particle_local_index +
                          (axes1 + 1) % translation_dof,
                      current_rigid_body_global_offset + translation_dof +
                          axes1,
                      -rci[(axes1 + 2) % translation_dof]);
        }

        vec3 dA;
        if (particle_type[i] == 4) {
          // corner point
          dA = vec3(0.0, 0.0, 0.0);
        } else {
          dA = (dim == 3) ? (normal[i] * p_spacing[i][0] * p_spacing[i][1])
                          : (normal[i] * p_spacing[i][0]);
        }

        // apply pressure
        for (int axes1 = 0; axes1 < translation_dof; axes1++) {
          A.out_process_increment(current_rigid_body_local_offset + axes1,
                                  pressure_global_index, -dA[axes1]);
        }

        for (int axes1 = 0; axes1 < rotation_dof; axes1++) {
          A.out_process_increment(current_rigid_body_local_offset +
                                      translation_dof + axes1,
                                  pressure_global_index,
                                  -rci[(axes1 + 1) % translation_dof] *
                                          dA[(axes1 + 2) % translation_dof] +
                                      rci[(axes1 + 2) % translation_dof] *
                                          dA[(axes1 + 1) % translation_dof]);
        }

        for (int j = 0; j < neighbor_list_host(i, 0); j++) {
          const int neighbor_particle_index =
              source_index[neighbor_list_host(i, j + 1)];

          for (int axes3 = 0; axes3 < dim; axes3++) {
            const int velocity_global_index =
                field_dof * neighbor_particle_index + axes3;

            double *f = new double[dim];
            for (int axes1 = 0; axes1 < dim; axes1++) {
              f[axes1] = 0.0;
            }

            for (int axes1 = 0; axes1 < dim; axes1++) {
              // output component 1
              for (int axes2 = 0; axes2 < dim; axes2++) {
                // output component 2
                const int velocity_gradient_index_1 =
                    velocity_gradient_index[(axes1 * dim + axes2) * dim +
                                            axes3];
                const int velocity_gradient_index_2 =
                    velocity_gradient_index[(axes2 * dim + axes1) * dim +
                                            axes3];
                auto alpha_index1 = velocity_basis->getAlphaIndexHost(
                    i, velocity_gradient_index_1);
                auto alpha_index2 = velocity_basis->getAlphaIndexHost(
                    i, velocity_gradient_index_2);
                const double sigma = eta * (velocity_alpha(alpha_index1 + j) +
                                            velocity_alpha(alpha_index2 + j));

                f[axes1] += sigma * dA[axes2];
              }
            }

            // force balance
            for (int axes1 = 0; axes1 < translation_dof; axes1++) {
              A.out_process_increment(current_rigid_body_local_offset + axes1,
                                      velocity_global_index, f[axes1]);
            }

            // torque balance
            for (int axes1 = 0; axes1 < rotation_dof; axes1++) {
              A.out_process_increment(current_rigid_body_local_offset +
                                          translation_dof + axes1,
                                      velocity_global_index,
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
    if (particle_type[i] != 0) {
      const int neumann_index = neumann_map[i];
      const double bi = pressure_neumann_basis->getAlpha0TensorTo0Tensor(
          DivergenceOfVectorPointEvaluation, neumann_index,
          neumann_neighbor_list_host(neumann_index, 0));

      for (int j = 0; j < neumann_neighbor_list_host(neumann_index, 0); j++) {
        const int neighbor_particle_index =
            source_index[neumann_neighbor_list_host(neumann_index, j + 1)];

        for (int axes2 = 0; axes2 < dim; axes2++) {
          double gradient = 0.0;
          const int velocity_global_index =
              field_dof * neighbor_particle_index + axes2;
          for (int axes1 = 0; axes1 < dim; axes1++) {
            auto alpha_index = velocity_basis->getAlphaIndexHost(
                i, velocity_curl_curl_index[axes1 * dim + axes2]);
            const double Lij = eta * velocity_alpha(alpha_index + j);

            gradient += normal[i][axes1] * Lij;
          }
          A.increment(pressure_local_index, velocity_global_index,
                      bi * gradient);
        }
      }
    } // end of velocity block

    // pressure block
    if (particle_type[i] == 0) {
      for (int j = 0; j < neighbor_list_host(i, 0); j++) {
        const int neighbor_particle_index =
            source_index[neighbor_list_host(i, j + 1)];

        const int pressure_neighbor_global_index =
            field_dof * neighbor_particle_index + velocity_dof;

        auto alpha_index =
            pressure_basis->getAlphaIndexHost(i, pressure_laplacian_index);
        const double Aij = pressure_alpha(alpha_index + j);

        // laplacian p
        A.increment(pressure_local_index, pressure_neighbor_global_index, Aij);
        A.increment(pressure_local_index, pressure_global_index, -Aij);

        for (int axes1 = 0; axes1 < dim; axes1++) {
          const int velocity_local_index =
              field_dof * current_particle_local_index + axes1;

          auto alpha_index = pressure_basis->getAlphaIndexHost(
              i, pressure_gradient_index[axes1]);
          const double Dijx = pressure_alpha(alpha_index + j);

          // grad p
          A.increment(velocity_local_index, pressure_neighbor_global_index,
                      -Dijx);
          A.increment(velocity_local_index, pressure_global_index, Dijx);
        }
      }
    }
    if (particle_type[i] != 0) {
      const int neumann_index = neumann_map[i];

      for (int j = 0; j < neumann_neighbor_list_host(neumann_index, 0); j++) {
        const int neighbor_particle_index =
            source_index[neumann_neighbor_list_host(neumann_index, j + 1)];

        const int pressure_neighbor_global_index =
            field_dof * neighbor_particle_index + velocity_dof;

        auto alpha_index = pressure_neumann_basis->getAlphaIndexHost(
            neumann_index, pressure_neumann_laplacian_index);
        const double Aij = pressure_neumann_alpha(alpha_index + j);

        // laplacian p
        A.increment(pressure_local_index, pressure_neighbor_global_index, Aij);
        A.increment(pressure_local_index, pressure_global_index, -Aij);
      }
    }
    // end of pressure block
  } // end of fluid particle loop

  auto ff = multi_mgr->get_field_mat(current_refinement_level);
  A.assemble(*ff, field_dof, num_rigid_body, rigid_body_dof);
  if (num_rigid_body != 0)
    A.extract_neighbor_index(idx_colloid, dim, num_rigid_body,
                             local_rigid_body_offset, global_rigid_body_offset);

  MPI_Barrier(MPI_COMM_WORLD);
  timer2 = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "Matrix assembly duration: %fs\n",
              timer2 - timer1);

  int inner_counter = 0;
  for (int i = 0; i < local_particle_num; i++) {
    if (particle_type[i] == 0)
      inner_counter++;
  }
  MPI_Allreduce(MPI_IN_PLACE, &inner_counter, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "total inner particle count: %d\n",
              inner_counter);
}

void stokes_equation::build_rhs() {
  auto &coord = *(geo_mgr->get_current_work_particle_coord());
  auto &normal = *(geo_mgr->get_current_work_particle_normal());
  auto &particle_type = *(geo_mgr->get_current_work_particle_type());

  auto &rigid_body_position = rb_mgr->get_position();
  auto &rigid_body_size = rb_mgr->get_rigid_body_size();
  const auto num_rigid_body = rb_mgr->get_rigid_body_num();

  int local_particle_num;
  int global_particle_num;

  local_particle_num = coord.size();
  MPI_Allreduce(&local_particle_num, &global_particle_num, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  const int translation_dof = (dim == 3 ? 3 : 2);
  const int rotation_dof = (dim == 3 ? 3 : 1);
  const int rigid_body_dof = (dim == 3 ? 6 : 3);

  int field_dof = dim + 1;
  int velocity_dof = dim;

  int local_velocity_dof = local_particle_num * dim;
  int global_velocity_dof =
      global_particle_num * dim + rigid_body_dof * num_rigid_body;
  int local_pressure_dof = local_particle_num;
  int global_pressure_dof = global_particle_num;

  if (rank == size - 1) {
    local_velocity_dof += rigid_body_dof * num_rigid_body;
  }

  vector<int> particle_num_per_process;
  particle_num_per_process.resize(size);

  MPI_Allgather(&local_particle_num, 1, MPI_INT,
                particle_num_per_process.data(), 1, MPI_INT, MPI_COMM_WORLD);

  int local_rigid_body_offset = particle_num_per_process[size - 1] * field_dof;
  int global_rigid_body_offset = global_particle_num * field_dof;
  int local_out_process_offset = particle_num_per_process[size - 1] * field_dof;

  int local_dof = local_velocity_dof + local_pressure_dof;

  rhs.resize(local_dof);
  res.resize(local_dof);

  for (int i = 0; i < local_dof; i++) {
    rhs[i] = 0.0;
    res[i] = 0.0;
  }

  auto neumann_neighbor_list = pressure_neumann_basis->getNeighborLists();

  // if (num_rigid_body == 0) {
  //   for (int i = 0; i < local_particle_num; i++) {
  //     if (particle_type[i] != 0 && particle_type[i] < 4) {
  //       // 2-d Taylor-Green vortex-like flow
  //       if (dim == 2) {
  //         double x = coord[i][0];
  //         double y = coord[i][1];

  //         rhs[field_dof * i] = cos(M_PI * x) * sin(M_PI * y);
  //         rhs[field_dof * i + 1] = -sin(M_PI * x) * cos(M_PI * y);

  //         const int neumann_index = neumann_map[i];
  //         const double bi = pressure_neumann_basis->getAlpha0TensorTo0Tensor(
  //             DivergenceOfVectorPointEvaluation, neumann_index,
  //             neumann_neighbor_list(neumann_index, 0));

  //         rhs[field_dof * i + velocity_dof] =
  //             -4.0 * pow(M_PI, 2.0) *
  //                 (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y)) +
  //             bi * (normal[i][0] * 2.0 * pow(M_PI, 2.0) * cos(M_PI * x) *
  //                       sin(M_PI * y) -
  //                   normal[i][1] * 2.0 * pow(M_PI, 2.0) * sin(M_PI * x) *
  //                       cos(M_PI * y)) +
  //             bi * (normal[i][0] * 2.0 * M_PI * sin(2.0 * M_PI * x) +
  //                   normal[i][1] * 2.0 * M_PI * sin(2.0 * M_PI * y));
  //       }

  //       // 3-d Taylor-Green vortex-like flow
  //       if (dim == 3) {
  //         double x = coord[i][0];
  //         double y = coord[i][1];
  //         double z = coord[i][2];

  //         rhs[field_dof * i] = cos(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
  //         rhs[field_dof * i + 1] =
  //             -2 * sin(M_PI * x) * cos(M_PI * y) * sin(M_PI * z);
  //         rhs[field_dof * i + 2] =
  //             sin(M_PI * x) * sin(M_PI * y) * cos(M_PI * z);

  //         const int neumann_index = neumann_map[i];
  //         const double bi = pressure_neumann_basis->getAlpha0TensorTo0Tensor(
  //             DivergenceOfVectorPointEvaluation, neumann_index,
  //             neumann_neighbor_list(neumann_index, 0));

  //         rhs[field_dof * i + velocity_dof] =
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
  //     } else if (particle_type[i] == 0) {
  //       if (dim == 2) {
  //         double x = coord[i][0];
  //         double y = coord[i][1];

  //         rhs[field_dof * i] =
  //             2.0 * pow(M_PI, 2.0) * cos(M_PI * x) * sin(M_PI * y) +
  //             2.0 * M_PI * sin(2.0 * M_PI * x);
  //         rhs[field_dof * i + 1] =
  //             -2.0 * pow(M_PI, 2.0) * sin(M_PI * x) * cos(M_PI * y) +
  //             2.0 * M_PI * sin(2.0 * M_PI * y);

  //         rhs[field_dof * i + velocity_dof] =
  //             -4.0 * pow(M_PI, 2.0) *
  //             (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y));
  //       }

  //       if (dim == 3) {
  //         double x = coord[i][0];
  //         double y = coord[i][1];
  //         double z = coord[i][2];

  //         rhs[field_dof * i] = 3.0 * pow(M_PI, 2) * cos(M_PI * x) *
  //                                  sin(M_PI * y) * sin(M_PI * z) +
  //                              2.0 * M_PI * sin(2.0 * M_PI * x);
  //         rhs[field_dof * i + 1] = -6.0 * pow(M_PI, 2) * sin(M_PI * x) *
  //                                      cos(M_PI * y) * sin(M_PI * z) +
  //                                  2.0 * M_PI * sin(2.0 * M_PI * y);
  //         rhs[field_dof * i + 2] = 3.0 * pow(M_PI, 2) * sin(M_PI * x) *
  //                                      sin(M_PI * y) * cos(M_PI * z) +
  //                                  2.0 * M_PI * sin(2.0 * M_PI * z);

  //         rhs[field_dof * i + velocity_dof] =
  //             -4.0 * pow(M_PI, 2.0) *
  //             (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y) + cos(2.0 * M_PI *
  //             z));
  //       }
  //     }
  //   }
  // }

  // if (rank == size - 1) {
  //   for (int i = 0; i < num_rigid_body; i++) {
  //     rhs[local_rigid_body_offset + i * rigid_body_dof + translation_dof] =
  //         pow(-1, i + 1);
  //   }
  // }

  // // make sure pressure term is orthogonal to the constant
  // double rhs_pressure_sum = 0.0;
  // for (int i = 0; i < local_particle_num; i++) {
  //   rhs_pressure_sum += rhs[field_dof * i + velocity_dof];
  // }
  // MPI_Allreduce(MPI_IN_PLACE, &rhs_pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
  //               MPI_COMM_WORLD);
  // rhs_pressure_sum /= global_particle_num;
  // for (int i = 0; i < local_particle_num; i++) {
  //   rhs[field_dof * i + velocity_dof] -= rhs_pressure_sum;
  // }

  double u = 1.0;
  double RR = rigid_body_size[0];

  for (int i = 0; i < local_particle_num; i++) {
    if (particle_type[i] != 0 && particle_type[i] < 4) {
      double x = coord[i][0];
      double y = coord[i][1];
      double z = coord[i][2];

      const int neumann_index = neumann_map[i];
      const double bi = pressure_neumann_basis->getAlpha0TensorTo0Tensor(
          DivergenceOfVectorPointEvaluation, neumann_index,
          neumann_neighbor_list->getNumberOfNeighborsHost(neumann_index));

      double r = sqrt(x * x + y * y + z * z);
      double theta = acos(z / r);
      double phi = atan2(y, x);

      double vr = u * cos(theta) *
                  (1 - (3 * RR) / (2 * r) + pow(RR, 3) / (2 * pow(r, 3)));
      double vt = -u * sin(theta) *
                  (1 - (3 * RR) / (4 * r) - pow(RR, 3) / (4 * pow(r, 3)));

      double pr = 3 * RR / pow(r, 3) * u * cos(theta);
      double pt = 3 / 2 * RR / pow(r, 3) * u * sin(theta);

      rhs[field_dof * i] =
          sin(theta) * cos(phi) * vr + cos(theta) * cos(phi) * vt;
      rhs[field_dof * i + 1] =
          sin(theta) * sin(phi) * vr + cos(theta) * sin(phi) * vt;
      rhs[field_dof * i + 2] = cos(theta) * vr - sin(theta) * vt;

      double p1 = sin(theta) * cos(phi) * pr + cos(theta) * cos(phi) * pt;
      double p2 = sin(theta) * sin(phi) * pr + cos(theta) * sin(phi) * pt;
      double p3 = cos(theta) * pr - sin(theta) * pt;

      // rhs[field_dof * i + 3] =
      //     bi * (normal[i][0] * p1 + normal[i][1] * p2 + normal[i][2] * p3);
    } else if (particle_type[i] >= 4) {
      double x = coord[i][0];
      double y = coord[i][1];
      double z = coord[i][2];

      double r = sqrt(x * x + y * y + z * z);
      double theta = acos(z / r);
      double phi = atan2(y, x);

      const int neumann_index = neumann_map[i];
      const double bi = pressure_neumann_basis->getAlpha0TensorTo0Tensor(
          DivergenceOfVectorPointEvaluation, neumann_index,
          neumann_neighbor_list->getNumberOfNeighborsHost(neumann_index));

      double pr = 3 * RR / pow(r, 3) * u * cos(theta);
      double pt = 3 / 2 * RR / pow(r, 3) * u * sin(theta);

      double p1 = sin(theta) * cos(phi) * pr + cos(theta) * cos(phi) * pt;
      double p2 = sin(theta) * sin(phi) * pr + cos(theta) * sin(phi) * pt;
      double p3 = cos(theta) * pr - sin(theta) * pt;

      // rhs[field_dof * i + 3] =
      //     bi * (normal[i][0] * p1 + normal[i][1] * p2 + normal[i][2] * p3);
    }
  }

  if (rank == size - 1) {
    rhs[local_rigid_body_offset + 2] = 6 * M_PI * RR * u;
  }

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
}

void stokes_equation::solve_step() {
  const int num_rigid_body = rb_mgr->get_rigid_body_num();

  // build interpolation and resitriction operators
  double timer1, timer2;
  if (num_rigid_body != 0 && current_refinement_level != 0) {
    timer1 = MPI_Wtime();

    multi_mgr->build_interpolation_restriction(num_rigid_body, dim);

    timer2 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,
                "Interpolation matrix building duration: %fs\n",
                timer2 - timer1);
  }

  PetscLogDefaultBegin();

  MPI_Barrier(MPI_COMM_WORLD);
  timer1 = MPI_Wtime();
  multi_mgr->solve(rhs, res, idx_colloid);
  MPI_Barrier(MPI_COMM_WORLD);
  timer2 = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "linear system solving duration: %fs\n",
              timer2 - timer1);

  // PetscViewer viewer;
  // PetscViewerASCIIGetStdout(PETSC_COMM_WORLD, &viewer);
  // PetscLogView(viewer);

  // copy data
  vector<vec3> &coord = *(geo_mgr->get_current_work_particle_coord());

  int local_particle_num;
  int global_particle_num;

  local_particle_num = coord.size();
  MPI_Allreduce(&local_particle_num, &global_particle_num, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  const int translation_dof = (dim == 3 ? 3 : 2);
  const int rotation_dof = (dim == 3 ? 3 : 1);
  const int rigid_body_dof = (dim == 3 ? 6 : 3);

  int field_dof = dim + 1;
  int velocity_dof = dim;

  vector<int> particle_num_per_process;
  particle_num_per_process.resize(size);

  MPI_Allgather(&local_particle_num, 1, MPI_INT,
                particle_num_per_process.data(), 1, MPI_INT, MPI_COMM_WORLD);

  int local_rigid_body_offset = particle_num_per_process[size - 1] * field_dof;
  int global_rigid_body_offset = global_particle_num * field_dof;
  int local_out_process_offset = particle_num_per_process[size - 1] * field_dof;

  pressure.resize(local_particle_num);
  velocity.resize(local_particle_num);

  double pressure_sum = 0.0;
  for (int i = 0; i < local_particle_num; i++) {
    pressure[i] = res[field_dof * i + velocity_dof];
    pressure_sum += pressure[i];
    for (int axes1 = 0; axes1 < dim; axes1++)
      velocity[i][axes1] = res[field_dof * i + axes1];
  }

  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  double average_pressure = pressure_sum / global_particle_num;
  for (int i = 0; i < local_particle_num; i++) {
    pressure[i] -= average_pressure;
  }

  vector<vec3> &rigid_body_velocity = rb_mgr->get_velocity();
  vector<vec3> &rigid_body_angular_velocity = rb_mgr->get_angular_velocity();

  if (rank == size - 1) {
    for (int i = 0; i < num_rigid_body; i++) {
      for (int j = 0; j < translation_dof; j++) {
        rigid_body_velocity[i][j] =
            res[local_rigid_body_offset + i * rigid_body_dof + j];
      }
      for (int j = 0; j < rotation_dof; j++) {
        rigid_body_angular_velocity[i][j] =
            res[local_rigid_body_offset + i * rigid_body_dof + translation_dof +
                j];
      }
    }
  }

  // communicate velocity and angular velocity
  vector<double> translation_velocity(num_rigid_body * translation_dof);
  vector<double> angular_velocity(num_rigid_body * rotation_dof);

  if (rank == size - 1) {
    for (int i = 0; i < num_rigid_body; i++) {
      for (int j = 0; j < translation_dof; j++) {
        translation_velocity[i * translation_dof + j] =
            rigid_body_velocity[i][j];

        cout << rigid_body_velocity[i][j] << endl;
      }
      for (int j = 0; j < rotation_dof; j++) {
        angular_velocity[i * rotation_dof + j] =
            rigid_body_angular_velocity[i][j];
      }
    }
  }

  MPI_Bcast(translation_velocity.data(), num_rigid_body * translation_dof,
            MPI_DOUBLE, size - 1, MPI_COMM_WORLD);
  MPI_Bcast(angular_velocity.data(), num_rigid_body * rotation_dof, MPI_DOUBLE,
            size - 1, MPI_COMM_WORLD);

  if (rank != size - 1) {
    for (int i = 0; i < num_rigid_body; i++) {
      for (int j = 0; j < translation_dof; j++) {
        rigid_body_velocity[i][j] =
            translation_velocity[i * translation_dof + j];
      }
      for (int j = 0; j < rotation_dof; j++) {
        rigid_body_angular_velocity[i][j] =
            angular_velocity[i * rotation_dof + j];
      }
    }
  }
}

void stokes_equation::check_solution() {
  vector<vec3> &coord = *(geo_mgr->get_current_work_particle_coord());
  vector<vec3> &normal = *(geo_mgr->get_current_work_particle_normal());
  vector<int> &particle_type = *(geo_mgr->get_current_work_particle_type());

  vector<vec3> &rigid_body_position = rb_mgr->get_position();
  vector<double> &rigid_body_size = rb_mgr->get_rigid_body_size();
  const int num_rigid_body = rb_mgr->get_rigid_body_num();

  int local_particle_num;
  int global_particle_num;

  local_particle_num = coord.size();
  MPI_Allreduce(&local_particle_num, &global_particle_num, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  const int translation_dof = (dim == 3 ? 3 : 2);
  const int rotation_dof = (dim == 3 ? 3 : 1);
  const int rigid_body_dof = (dim == 3 ? 6 : 3);

  int field_dof = dim + 1;
  int velocity_dof = dim;

  int local_velocity_dof = local_particle_num * dim;
  int global_velocity_dof =
      global_particle_num * dim + rigid_body_dof * num_rigid_body;
  int local_pressure_dof = local_particle_num;
  int global_pressure_dof = global_particle_num;

  if (rank == size - 1) {
    local_velocity_dof += rigid_body_dof * num_rigid_body;
  }

  vector<int> particle_num_per_process;
  particle_num_per_process.resize(size);

  MPI_Allgather(&local_particle_num, 1, MPI_INT,
                particle_num_per_process.data(), 1, MPI_INT, MPI_COMM_WORLD);

  int local_rigid_body_offset = particle_num_per_process[size - 1] * field_dof;
  int global_rigid_body_offset = global_particle_num * field_dof;
  int local_out_process_offset = particle_num_per_process[size - 1] * field_dof;

  int local_dof = local_velocity_dof + local_pressure_dof;

  double u = 1.0;
  double RR = rigid_body_size[0];

  // check data
  double true_pressure_mean = 0.0;
  double pressure_mean = 0.0;
  for (int i = 0; i < local_particle_num; i++) {
    if (dim == 2) {
      double x = coord[i][0];
      double y = coord[i][1];

      double true_pressure = -cos(2.0 * M_PI * x) - cos(2.0 * M_PI * y);

      true_pressure_mean += true_pressure;
      pressure_mean += pressure[i];
    }

    if (dim == 3) {
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

  true_pressure_mean /= global_particle_num;
  pressure_mean /= global_particle_num;

  double error_velocity = 0.0;
  double norm_velocity = 0.0;
  double error_pressure = 0.0;
  double norm_pressure = 0.0;
  for (int i = 0; i < local_particle_num; i++) {
    if (dim == 2) {
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

    if (dim == 3) {
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
}

void stokes_equation::calculate_error() {
  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "\nstart of error estimation\n");

  // prepare stage
  auto &source_coord = *(geo_mgr->get_current_work_ghost_particle_coord());
  auto &source_volume = *(geo_mgr->get_current_work_ghost_particle_volume());
  auto &source_index = *(geo_mgr->get_current_work_ghost_particle_index());
  auto &coord = *(geo_mgr->get_current_work_particle_coord());
  auto &volume = *(geo_mgr->get_current_work_particle_volume());

  const int local_particle_num = coord.size();

  double local_error, local_volume, local_direct_gradient_norm;
  double global_direct_gradient_norm;

  local_error = 0.0;
  local_direct_gradient_norm = 0.0;

  error.resize(local_particle_num);
  for (int i = 0; i < local_particle_num; i++) {
    error[i] = 0.0;
  }

  auto neighbor_list = velocity_basis->getNeighborLists();

  // error estimation base on velocity
  if (error_esimation_method == VELOCITY_ERROR_EST) {
    vector<vec3> ghost_velocity;
    geo_mgr->ghost_forward(velocity, ghost_velocity);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        ghost_velocity_device("source velocity", ghost_velocity.size(), 3);
    Kokkos::View<double **>::HostMirror ghost_velocity_host =
        Kokkos::create_mirror_view(ghost_velocity_device);

    for (size_t i = 0; i < ghost_velocity.size(); i++) {
      ghost_velocity_host(i, 0) = ghost_velocity[i][0];
      ghost_velocity_host(i, 1) = ghost_velocity[i][1];
      ghost_velocity_host(i, 2) = ghost_velocity[i][2];
    }

    Kokkos::deep_copy(ghost_velocity_device, ghost_velocity_host);

    Evaluator velocity_evaluator(velocity_basis.get());

    auto coefficients =
        velocity_evaluator
            .applyFullPolynomialCoefficientsBasisToDataAllComponents<
                double **, Kokkos::HostSpace>(ghost_velocity_device);

    auto direct_gradient =
        velocity_evaluator.applyAlphasToDataAllComponentsAllTargetSites<
            double **, Kokkos::HostSpace>(ghost_velocity_device,
                                          GradientOfVectorPointEvaluation);

    auto coefficients_size = velocity_basis->getPolynomialCoefficientsSize();

    vector<vector<double>> coefficients_chunk, ghost_coefficients_chunk;
    coefficients_chunk.resize(local_particle_num);
    for (int i = 0; i < local_particle_num; i++) {
      coefficients_chunk[i].resize(coefficients_size);
      for (int j = 0; j < coefficients_size; j++) {
        coefficients_chunk[i][j] = coefficients(i, j);
      }
    }

    geo_mgr->ghost_forward(coefficients_chunk, ghost_coefficients_chunk,
                           coefficients_size);

    // estimate stage
    auto &recovered_gradient = gradient;
    vector<vector<double>> ghost_recovered_gradient;
    recovered_gradient.resize(local_particle_num);
    const int gradient_component_num = pow(dim, 2);
    for (int i = 0; i < local_particle_num; i++) {
      recovered_gradient[i].resize(gradient_component_num);
      for (int axes1 = 0; axes1 < dim; axes1++) {
        for (int axes2 = 0; axes2 < dim; axes2++) {
          recovered_gradient[i][axes1 * dim + axes2] = 0.0;
        }
      }
    }

    for (int i = 0; i < local_particle_num; i++) {
      int counter = 0;
      for (int j = 0; j < neighbor_list->getNumberOfNeighborsHost(i); j++) {
        const int neighbor_index = neighbor_list->getNeighborHost(i, j);

        vec3 dX = coord[i] - source_coord[neighbor_index];
        if (dX.mag() < ghost_epsilon[neighbor_index]) {
          for (int axes1 = 0; axes1 < dim; axes1++) {
            for (int axes2 = 0; axes2 < dim; axes2++) {
              if (dim == 2)
                recovered_gradient[i][axes1 * dim + axes2] +=
                    calDivFreeBasisGrad(
                        axes1, axes2, dX[0], dX[1], poly_order,
                        ghost_epsilon[neighbor_index],
                        ghost_coefficients_chunk[neighbor_index]);
              if (dim == 3)
                recovered_gradient[i][axes1 * dim + axes2] +=
                    calDivFreeBasisGrad(
                        axes1, axes2, dX[0], dX[1], dX[2], poly_order,
                        ghost_epsilon[neighbor_index],
                        ghost_coefficients_chunk[neighbor_index]);
            }
          }
          counter++;
        }
      }

      for (int axes1 = 0; axes1 < dim; axes1++) {
        for (int axes2 = 0; axes2 < dim; axes2++) {
          recovered_gradient[i][axes1 * dim + axes2] /= counter;
        }
      }
    }

    geo_mgr->ghost_forward(recovered_gradient, ghost_recovered_gradient,
                           gradient_component_num);

    for (int i = 0; i < local_particle_num; i++) {
      vector<double> reconstructed_gradient(gradient_component_num);
      double total_neighbor_vol = 0.0;
      // loop over all neighbors
      for (int j = 0; j < neighbor_list->getNumberOfNeighborsHost(i); j++) {
        const int neighbor_index = neighbor_list->getNeighborHost(i, j);

        vec3 dX = source_coord[neighbor_index] - coord[i];

        if (dX.mag() < epsilon[i]) {
          total_neighbor_vol += source_volume[neighbor_index];
          for (int axes1 = 0; axes1 < dim; axes1++) {
            for (int axes2 = 0; axes2 < dim; axes2++) {
              if (dim == 2)
                reconstructed_gradient[axes1 * dim + axes2] =
                    calDivFreeBasisGrad(axes1, axes2, dX[0], dX[1], poly_order,
                                        ghost_epsilon[i],
                                        ghost_coefficients_chunk[i]);
              if (dim == 3)
                reconstructed_gradient[axes1 * dim + axes2] =
                    calDivFreeBasisGrad(axes1, axes2, dX[0], dX[1], dX[2],
                                        poly_order, ghost_epsilon[i],
                                        ghost_coefficients_chunk[i]);
            }
          }

          for (int axes1 = 0; axes1 < dim; axes1++) {
            for (int axes2 = axes1; axes2 < dim; axes2++) {
              if (axes1 == axes2)
                error[i] +=
                    pow(reconstructed_gradient[axes1 * dim + axes2] -
                            ghost_recovered_gradient[neighbor_index]
                                                    [axes1 * dim + axes2],
                        2) *
                    source_volume[neighbor_index];
              else {
                error[i] +=
                    pow(0.5 * (reconstructed_gradient[axes1 * dim + axes2] -
                               ghost_recovered_gradient[neighbor_index]
                                                       [axes1 * dim + axes2] +
                               reconstructed_gradient[axes2 * dim + axes1] -
                               ghost_recovered_gradient[neighbor_index]
                                                       [axes2 * dim + axes1]),
                        2) *
                    source_volume[neighbor_index];
              }
            }
          }
        }
      }

      error[i] = error[i] / total_neighbor_vol;
      local_error += error[i] * volume[i];

      for (int axes1 = 0; axes1 < dim; axes1++) {
        for (int axes2 = axes1; axes2 < dim; axes2++) {
          if (axes1 == axes2)
            local_direct_gradient_norm +=
                pow(direct_gradient(i, axes1 * dim + axes2), 2) * volume[i];
          else {
            local_direct_gradient_norm +=
                pow(0.5 * (direct_gradient(i, axes1 * dim + axes2) +
                           direct_gradient(i, axes2 * dim + axes1)),
                    2) *
                volume[i];
          }
        }
      }
    }
  }

  // error estimation based on pressure
  if (error_esimation_method == PRESSURE_ERROR_EST) {
    vector<double> ghost_pressure;
    geo_mgr->ghost_forward(pressure, ghost_pressure);

    Kokkos::View<double *, Kokkos::DefaultExecutionSpace> ghost_pressure_device(
        "background pressure", ghost_pressure.size());
    Kokkos::View<double *>::HostMirror ghost_pressure_host =
        Kokkos::create_mirror_view(ghost_pressure_device);

    for (size_t i = 0; i < ghost_pressure.size(); i++) {
      ghost_pressure_host(i) = ghost_pressure[i];
    }

    Kokkos::deep_copy(ghost_pressure_device, ghost_pressure_host);

    Evaluator pressure_evaluator(pressure_basis.get());

    auto coefficients =
        pressure_evaluator
            .applyFullPolynomialCoefficientsBasisToDataAllComponents<
                double **, Kokkos::HostSpace>(ghost_pressure_device);

    auto direct_gradient =
        pressure_evaluator.applyAlphasToDataAllComponentsAllTargetSites<
            double **, Kokkos::HostSpace>(
            ghost_pressure_device, GradientOfScalarPointEvaluation,
            StaggeredEdgeAnalyticGradientIntegralSample);

    auto coefficients_size = pressure_basis->getPolynomialCoefficientsSize();

    // boundary needs special treatment
    {
      Evaluator pressure_neumann_evaluator(pressure_neumann_basis.get());

      auto coefficients_neumann =
          pressure_neumann_evaluator
              .applyFullPolynomialCoefficientsBasisToDataAllComponents<
                  double **, Kokkos::HostSpace>(ghost_pressure_device);

      auto direct_gradient_neumann =
          pressure_neumann_evaluator
              .applyAlphasToDataAllComponentsAllTargetSites<double **,
                                                            Kokkos::HostSpace>(
                  ghost_pressure_device, GradientOfScalarPointEvaluation,
                  StaggeredEdgeAnalyticGradientIntegralSample);

      int counter = 0;

      auto &particle_type = *(geo_mgr->get_current_work_particle_type());
      for (int i = 0; i < local_particle_num; i++) {
        if (particle_type[i] != 0) {
          for (int j = 0; j < coefficients_size; j++) {
            coefficients(i, j) = coefficients_neumann(counter, j);
          }
          for (int j = 0; j < dim; j++) {
            direct_gradient(i, j) = direct_gradient_neumann(counter, j);
          }
          counter++;
        }
      }
    }

    vector<vector<double>> coefficients_chunk(local_particle_num);
    for (int i = 0; i < local_particle_num; i++) {
      coefficients_chunk[i].resize(coefficients_size);
      for (int j = 0; j < coefficients_size; j++) {
        coefficients_chunk[i][j] = coefficients(i, j);
      }
    }

    vector<vector<double>> ghost_coefficients_chunk;

    geo_mgr->ghost_forward(coefficients_chunk, ghost_coefficients_chunk,
                           coefficients_size);

    // estimate stage
    auto &recovered_gradient = gradient;
    vector<vector<double>> ghost_recovered_gradient;
    recovered_gradient.resize(local_particle_num);

    for (int i = 0; i < local_particle_num; i++) {
      recovered_gradient[i].resize(dim);
      for (int axes1 = 0; axes1 < dim; axes1++) {
        recovered_gradient[i][axes1] = 0.0;
      }
    }

    for (int i = 0; i < local_particle_num; i++) {
      int counter = 0;
      for (int j = 0; j < neighbor_list->getNumberOfNeighborsHost(i); j++) {
        const int neighbor_index = neighbor_list->getNeighborHost(i, j);

        vec3 dX = coord[i] - source_coord[neighbor_index];
        if (dX.mag() < ghost_epsilon[neighbor_index]) {
          for (int axes1 = 0; axes1 < dim; axes1++) {
            recovered_gradient[i][axes1] += calStaggeredScalarGrad(
                axes1, dim, dX, poly_order, ghost_epsilon[neighbor_index],
                ghost_coefficients_chunk[neighbor_index]);
          }
          counter++;
        }
      }

      for (int axes1 = 0; axes1 < dim; axes1++) {
        recovered_gradient[i][axes1] /= counter;
      }
    }

    geo_mgr->ghost_forward(recovered_gradient, ghost_recovered_gradient, dim);

    for (int i = 0; i < local_particle_num; i++) {
      vec3 reconstructed_gradient;
      double total_neighbor_vol = 0.0;
      // loop over all neighbors
      for (int j = 0; j < neighbor_list->getNumberOfNeighborsHost(i); j++) {
        const int neighbor_index = neighbor_list->getNeighborHost(i, j);

        vec3 dX = source_coord[neighbor_index] - coord[i];

        if (dX.mag() < epsilon[i]) {
          total_neighbor_vol += source_volume[neighbor_index];
          for (int axes1 = 0; axes1 < dim; axes1++) {
            reconstructed_gradient[axes1] = calStaggeredScalarGrad(
                axes1, dim, dX, poly_order, epsilon[i], coefficients_chunk[i]);
          }

          for (int axes1 = 0; axes1 < dim; axes1++) {
            error[i] += pow(reconstructed_gradient[axes1] -
                                ghost_recovered_gradient[neighbor_index][axes1],
                            2) *
                        source_volume[neighbor_index];
          }
        }
      }

      error[i] = error[i] / total_neighbor_vol;
      local_error += error[i] * volume[i];

      for (int axes1 = 0; axes1 < dim; axes1++) {
        local_direct_gradient_norm +=
            pow(direct_gradient(i, axes1), 2) * volume[i];
      }
    }
  }

  // smooth stage
  for (int ite = 0; ite < 1; ite++) {
    vector<double> ghost_error;
    geo_mgr->ghost_forward(error, ghost_error);

    for (int i = 0; i < local_particle_num; i++) {
      error[i] = 0.0;
      double total_neighbor_vol = 0.0;
      for (int j = 0; j < neighbor_list->getNumberOfNeighborsHost(i); j++) {
        const int neighbor_index = neighbor_list->getNeighborHost(i, j);

        vec3 dX = source_coord[neighbor_index] - coord[i];

        if (dX.mag() < epsilon[i]) {
          double Wabij = Wab(dX.mag(), epsilon[i]);

          error[i] += ghost_error[neighbor_index] *
                      source_volume[neighbor_index] * Wabij;
          total_neighbor_vol += source_volume[neighbor_index] * Wabij;
        }
      }
      error[i] /= total_neighbor_vol;
    }
  }

  for (int i = 0; i < local_particle_num; i++) {
    error[i] *= volume[i];
  }

  MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&local_direct_gradient_norm, &global_direct_gradient_norm, 1,
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  global_error /= global_direct_gradient_norm;
  global_error = sqrt(this->global_error);

  for (int i = 0; i < local_particle_num; i++) {
    error[i] = sqrt(error[i]);
  }
}