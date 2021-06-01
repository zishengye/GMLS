#include "stokes_equation.hpp"
#include "DivergenceFree.hpp"
#include "gmls_solver.hpp"
#include "petsc_sparse_matrix.hpp"

#include <iomanip>
#include <utility>

using namespace std;
using namespace Compadre;

bool compare(const std::pair<int, double> &firstElem,
             const std::pair<int, double> &secondElem) {
  return firstElem.second < secondElem.second;
}

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
                           const double _epsilon_multiplier, const double _eta,
                           const int _compress_memory) {
  geo_mgr = _geo_mgr;
  rb_mgr = _rb_mgr;
  eta = _eta;
  dim = _dim;
  poly_order = _poly_order;
  error_esimation_method = _error_estimation_method;
  compress_memory = _compress_memory;

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
  calculate_error();
  check_solution();

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
  auto &local_idx = *(geo_mgr->get_current_work_particle_local_index());
  auto &particle_type = *(geo_mgr->get_current_work_particle_type());
  auto &attached_rigid_body =
      *(geo_mgr->get_current_work_particle_attached_rigid_body());
  auto &num_neighbor = *(geo_mgr->get_current_work_particle_num_neighbor());

  vector<int> ghost_particle_type;
  geo_mgr->ghost_forward(particle_type, ghost_particle_type);

  PetscLogDouble mem;
  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "Current memory usage before reset %.2f GB\n",
              mem / 1e9);

  // update basis
  pressure_basis.reset();
  velocity_basis.reset();
  pressure_neumann_basis.reset();
  velocity_colloid_basis.reset();

  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "Current memory usage after reset %.2f GB\n",
              mem / 1e9);

  pressure_basis = make_shared<GMLS>(
      ScalarTaylorPolynomial, StaggeredEdgeAnalyticGradientIntegralSample,
      poly_order, dim, "LU", "STANDARD");
  velocity_basis =
      make_shared<GMLS>(DivergenceFreeVectorTaylorPolynomial, VectorPointSample,
                        poly_order, dim, "LU", "STANDARD");
  pressure_neumann_basis = make_shared<GMLS>(
      ScalarTaylorPolynomial, StaggeredEdgeAnalyticGradientIntegralSample,
      poly_order, dim, "LU", "STANDARD", "NEUMANN_GRAD_SCALAR");
  velocity_colloid_basis =
      make_shared<GMLS>(DivergenceFreeVectorTaylorPolynomial, VectorPointSample,
                        poly_order, dim, "LU", "STANDARD");

  vector<vec3> &rigid_body_position = rb_mgr->get_position();
  vector<vec3> &rigid_body_velocity_force_switch =
      rb_mgr->get_velocity_force_switch();
  vector<vec3> &rigid_body_angvelocity_torque_switch =
      rb_mgr->get_angvelocity_torque_switch();
  const int num_rigid_body = rb_mgr->get_rigid_body_num();

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
  int num_colloid_target_coord = 0;
  for (int i = 0; i < local_particle_num; i++) {
    if (particle_type[i] != 0) {
      num_neumann_target_coord++;
    }
    if (particle_type[i] >= 4) {
      num_colloid_target_coord++;
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
  Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
      colloid_target_coord_device("collloid target coordinates",
                                  num_colloid_target_coord, 3);
  Kokkos::View<double **>::HostMirror colloid_target_coord_host =
      Kokkos::create_mirror_view(colloid_target_coord_device);

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

  vector<int> colloid_map;
  colloid_map.resize(local_particle_num);
  counter = 0;
  for (int i = 0; i < local_particle_num; i++) {
    if (particle_type[i] >= 4) {
      colloid_map[i] = counter;
      for (int j = 0; j < 3; j++) {
        colloid_target_coord_host(counter, j) = coord[i][j];
      }
      counter++;
    }
  }

  Kokkos::deep_copy(source_coord_device, source_coord_host);
  Kokkos::deep_copy(target_coord_device, target_coord_host);
  Kokkos::deep_copy(neumann_target_coord_device, neumann_target_coord_host);
  Kokkos::deep_copy(colloid_target_coord_device, colloid_target_coord_host);

  // tangent bundle for neumann boundary particles
  Kokkos::View<double ***, Kokkos::DefaultExecutionSpace> tangent_bundle_device(
      "tangent bundles", num_neumann_target_coord, dim, dim);
  Kokkos::View<double ***>::HostMirror tangent_bundle_host =
      Kokkos::create_mirror_view(tangent_bundle_device);
  Kokkos::View<double ***, Kokkos::DefaultExecutionSpace>
      colloid_tangent_bundle_device("tangent bundles", num_colloid_target_coord,
                                    dim, dim);
  Kokkos::View<double ***>::HostMirror colloid_tangent_bundle_host =
      Kokkos::create_mirror_view(colloid_tangent_bundle_device);

  for (int i = 0; i < local_particle_num; i++) {
    if (particle_type[i] != 0) {
      counter = neumann_map[i];
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

    if (particle_type[i] >= 4) {
      counter = colloid_map[i];
      if (dim == 3) {
        colloid_tangent_bundle_host(counter, 0, 0) = 0.0;
        colloid_tangent_bundle_host(counter, 0, 1) = 0.0;
        colloid_tangent_bundle_host(counter, 0, 2) = 0.0;
        colloid_tangent_bundle_host(counter, 1, 0) = 0.0;
        colloid_tangent_bundle_host(counter, 1, 1) = 0.0;
        colloid_tangent_bundle_host(counter, 1, 2) = 0.0;
        colloid_tangent_bundle_host(counter, 2, 0) = normal[i][0];
        colloid_tangent_bundle_host(counter, 2, 1) = normal[i][1];
        colloid_tangent_bundle_host(counter, 2, 2) = normal[i][2];
      }
      if (dim == 2) {
        colloid_tangent_bundle_host(counter, 0, 0) = 0.0;
        colloid_tangent_bundle_host(counter, 0, 1) = 0.0;
        colloid_tangent_bundle_host(counter, 1, 0) = normal[i][0];
        colloid_tangent_bundle_host(counter, 1, 1) = normal[i][1];
      }
    }
  }

  Kokkos::deep_copy(tangent_bundle_device, tangent_bundle_host);
  Kokkos::deep_copy(colloid_tangent_bundle_device, colloid_tangent_bundle_host);

  // neighbor search
  auto point_cloud_search(CreatePointCloudSearch(source_coord_host, dim));

  int min_num_neighbor =
      max(Compadre::GMLS::getNP(poly_order, dim,
                                DivergenceFreeVectorTaylorPolynomial),
          Compadre::GMLS::getNP(poly_order + 1, dim));
  int satisfied_num_neighbor = 2.0 * min_num_neighbor;

  int estimated_max_num_neighbor = 3.0 * satisfied_num_neighbor;

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> neighbor_list_device(
      "neighbor lists", num_target_coord, estimated_max_num_neighbor);
  Kokkos::View<int **>::HostMirror neighbor_list_host =
      Kokkos::create_mirror_view(neighbor_list_device);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> epsilon_device(
      "h supports", num_target_coord);
  Kokkos::View<double *>::HostMirror epsilon_host =
      Kokkos::create_mirror_view(epsilon_device);

  double max_epsilon = geo_mgr->get_cutoff_distance();
  for (int i = 0; i < num_target_coord; i++) {
    epsilon_host(i) = 1.0005 * spacing[i];
  }

  MPI_Allreduce(MPI_IN_PLACE, &max_epsilon, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);

  // ensure every particle has enough neighbors
  bool pass_neighbor_search = false;
  int ite_counter = 0;
  int mean_neighbor;
  while (!pass_neighbor_search) {
    int num_neighbor_size_needed =
        1 + point_cloud_search.generate2DNeighborListsFromRadiusSearch(
                true, target_coord_host, neighbor_list_host, epsilon_host, 0.0,
                max_epsilon);
    if (estimated_max_num_neighbor < num_neighbor_size_needed) {
      estimated_max_num_neighbor = num_neighbor_size_needed;
      neighbor_list_device =
          Kokkos::View<int **, Kokkos::DefaultExecutionSpace>(
              "neighbor lists", num_target_coord, estimated_max_num_neighbor);
      neighbor_list_host = Kokkos::create_mirror_view(neighbor_list_device);
    }
    point_cloud_search.generate2DNeighborListsFromRadiusSearch(
        false, target_coord_host, neighbor_list_host, epsilon_host, 0.0,
        max_epsilon);

    bool pass_neighbor_num_check = true;
    min_neighbor = 1000;
    max_neighbor = 0;
    mean_neighbor = 0;
    for (int i = 0; i < local_particle_num; i++) {
      int local_num_neighbor = 0;
      for (int j = 0; j < neighbor_list_host(i, 0); j++) {
        int neighbor_index = neighbor_list_host(i, j + 1);
        if (ghost_particle_type[neighbor_index] == 0)
          local_num_neighbor++;
      }
      if (local_num_neighbor <= satisfied_num_neighbor) {
        if (epsilon_host(i) + 0.25 * spacing[i] < max_epsilon) {
          epsilon_host(i) += 0.25 * spacing[i];
          pass_neighbor_num_check = false;
        }
      }
      if (neighbor_list_host(i, 0) < min_neighbor)
        min_neighbor = neighbor_list_host(i, 0);
      if (neighbor_list_host(i, 0) > max_neighbor)
        max_neighbor = neighbor_list_host(i, 0);
      mean_neighbor += neighbor_list_host(i, 0);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &min_neighbor, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_neighbor, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &mean_neighbor, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    int process_counter = 0;
    if (!pass_neighbor_num_check) {
      process_counter = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, &process_counter, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    if (process_counter == 0) {
      pass_neighbor_search = true;
    }
    ite_counter++;
  }

  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "Current memory usage after creation of neighbor list %.2f GB\n",
              mem / 1e9);

  PetscPrintf(MPI_COMM_WORLD,
              "iteration counter: %d min neighbor: %d, max neighbor: %d , mean "
              "neighbor %f\n",
              ite_counter, min_neighbor, max_neighbor,
              (double)mean_neighbor / (double)global_particle_num);

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
      neumann_neighbor_list_device("neumann boundary neighbor lists",
                                   num_neumann_target_coord, max_neighbor + 1);
  Kokkos::View<int **>::HostMirror neumann_neighbor_list_host =
      Kokkos::create_mirror_view(neumann_neighbor_list_device);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> neumann_epsilon_device(
      "neumann boundary h supports", num_neumann_target_coord);
  Kokkos::View<double *>::HostMirror neumann_epsilon_host =
      Kokkos::create_mirror_view(neumann_epsilon_device);

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
      colloid_neighbor_list_device("colloid boundary neighbor lists",
                                   num_colloid_target_coord, max_neighbor + 1);
  Kokkos::View<int **>::HostMirror colloid_neighbor_list_host =
      Kokkos::create_mirror_view(colloid_neighbor_list_device);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> colloid_epsilon_device(
      "colloid boundary h supports", num_colloid_target_coord);
  Kokkos::View<double *>::HostMirror colloid_epsilon_host =
      Kokkos::create_mirror_view(colloid_epsilon_device);

  epsilon.resize(local_particle_num);
  for (int i = 0; i < num_target_coord; i++) {
    epsilon[i] = epsilon_host(i);
    if (particle_type[i] != 0) {
      counter = neumann_map[i];
      neumann_epsilon_host(counter) = epsilon_host(i);
      neumann_neighbor_list_host(counter, 0) = neighbor_list_host(i, 0);
      for (int j = 0; j < neighbor_list_host(i, 0); j++) {
        neumann_neighbor_list_host(counter, j + 1) =
            neighbor_list_host(i, j + 1);
      }
    }
    if (particle_type[i] >= 4) {
      counter = colloid_map[i];
      colloid_epsilon_host(counter) = epsilon_host(i);
      colloid_neighbor_list_host(counter, 0) = neighbor_list_host(i, 0);
      for (int j = 0; j < neighbor_list_host(i, 0); j++) {
        colloid_neighbor_list_host(counter, j + 1) =
            neighbor_list_host(i, j + 1);
      }
    }
  }

  geo_mgr->ghost_forward(epsilon, ghost_epsilon);

  num_neighbor.resize(num_target_coord);
  for (int i = 0; i < num_target_coord; i++) {
    num_neighbor[i] = neighbor_list_host(i, 0);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  Kokkos::deep_copy(neighbor_list_device, neighbor_list_host);
  Kokkos::deep_copy(epsilon_device, epsilon_host);
  Kokkos::deep_copy(neumann_neighbor_list_device, neumann_neighbor_list_host);
  Kokkos::deep_copy(neumann_epsilon_device, neumann_epsilon_host);
  Kokkos::deep_copy(colloid_neighbor_list_device, colloid_neighbor_list_host);
  Kokkos::deep_copy(colloid_epsilon_device, colloid_epsilon_host);

  if (dim == 2)
    number_of_batches = max(local_particle_num / 100, 1);
  else
    number_of_batches = max(local_particle_num / 10, 1);

  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "Current memory usage before pressure basis %.2f GB\n",
              mem / 1e9);

  // pressure basis
  pressure_basis->setProblemData(neighbor_list_device, source_coord_device,
                                 target_coord_device, epsilon_device);

  vector<TargetOperation> pressure_operation(2);
  pressure_operation[0] = LaplacianOfScalarPointEvaluation;
  pressure_operation[1] = GradientOfScalarPointEvaluation;

  pressure_basis->clearTargets();
  pressure_basis->addTargets(pressure_operation);

  pressure_basis->setWeightingType(WeightingFunctionType::Power);
  pressure_basis->setWeightingPower(4);
  pressure_basis->setOrderOfQuadraturePoints(2);
  pressure_basis->setDimensionOfQuadraturePoints(1);
  pressure_basis->setQuadratureType("LINE");

  if (compress_memory == 0)
    pressure_basis->generateAlphas(1, true);
  else
    pressure_basis->generateAlphas(number_of_batches, false);

  auto pressure_alpha = pressure_basis->getAlphas();

  const int pressure_laplacian_index =
      pressure_basis->getAlphaColumnOffset(pressure_operation[0], 0, 0, 0, 0);
  vector<int> pressure_gradient_index;
  for (int i = 0; i < dim; i++)
    pressure_gradient_index.push_back(pressure_basis->getAlphaColumnOffset(
        pressure_operation[1], i, 0, 0, 0));

  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "Current memory usage before velocity basis %.2f GB\n",
              mem / 1e9);

  // velocity basis
  velocity_basis->setProblemData(neighbor_list_device, source_coord_device,
                                 target_coord_device, epsilon_device);

  velocity_basis->clearTargets();
  velocity_basis->addTargets(CurlCurlOfVectorPointEvaluation);

  velocity_basis->setWeightingType(WeightingFunctionType::Power);
  velocity_basis->setWeightingPower(4);

  if (compress_memory == 0)
    velocity_basis->generateAlphas(1, true);
  else
    velocity_basis->generateAlphas(number_of_batches, false);

  auto velocity_alpha = velocity_basis->getAlphas();

  vector<int> velocity_curl_curl_index(pow(dim, 2));
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      velocity_curl_curl_index[i * dim + j] =
          velocity_basis->getAlphaColumnOffset(CurlCurlOfVectorPointEvaluation,
                                               i, 0, j, 0);
    }
  }

  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "Current memory usage before colloid velocity basis %.2f GB\n",
              mem / 1e9);

  // velocity colloid boundary basis
  velocity_colloid_basis->setProblemData(
      colloid_neighbor_list_device, source_coord_device,
      colloid_target_coord_device, colloid_epsilon_device);

  velocity_colloid_basis->clearTargets();
  velocity_colloid_basis->addTargets(GradientOfVectorPointEvaluation);

  velocity_colloid_basis->setWeightingType(WeightingFunctionType::Power);
  velocity_colloid_basis->setWeightingPower(4);

  velocity_colloid_basis->generateAlphas(number_of_batches, false);

  vector<int> velocity_gradient_index(pow(dim, 3));
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        velocity_gradient_index[(i * dim + j) * dim + k] =
            velocity_colloid_basis->getAlphaColumnOffset(
                GradientOfVectorPointEvaluation, i, j, k, 0);
      }
    }
  }

  auto velocity_colloid_alpha = velocity_colloid_basis->getAlphas();

  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "Current memory usage before neumann pressure basis %.2f GB\n",
              mem / 1e9);

  // pressure Neumann boundary basis
  pressure_neumann_basis->setProblemData(
      neumann_neighbor_list_device, source_coord_device,
      neumann_target_coord_device, neumann_epsilon_device);

  pressure_neumann_basis->setTangentBundle(tangent_bundle_device);

  vector<TargetOperation> pressure_neumann_operation(1);
  pressure_neumann_operation[0] = LaplacianOfScalarPointEvaluation;

  pressure_neumann_basis->clearTargets();
  pressure_neumann_basis->addTargets(pressure_neumann_operation);

  pressure_neumann_basis->setWeightingType(WeightingFunctionType::Power);
  pressure_neumann_basis->setWeightingPower(4);
  pressure_neumann_basis->setOrderOfQuadraturePoints(2);
  pressure_neumann_basis->setDimensionOfQuadraturePoints(1);
  pressure_neumann_basis->setQuadratureType("LINE");

  if (compress_memory == 0)
    pressure_neumann_basis->generateAlphas(1, true);
  else
    pressure_neumann_basis->generateAlphas(number_of_batches, false);

  auto pressure_neumann_alpha = pressure_neumann_basis->getAlphas();

  const int pressure_neumann_laplacian_index =
      pressure_neumann_basis->getAlphaColumnOffset(
          pressure_neumann_operation[0], 0, 0, 0, 0);

  MPI_Barrier(MPI_COMM_WORLD);
  timer2 = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "GMLS solving duration: %fs\n",
              timer2 - timer1);

  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "Current memory usage after GMLS solving %.2f GB\n", mem / 1e9);

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
    const int current_particle_local_index = local_idx[i];
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

  double area = 0.0;
  int area_num = 0;

  // outprocess graph
  for (int i = 0; i < local_particle_num; i++) {
    int rigid_body_idx = attached_rigid_body[i];
    const int current_particle_local_index = local_idx[i];
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

      for (int axes = 0; axes < translation_dof; axes++) {
        vector<PetscInt> &it =
            out_process_index[rigid_body_idx * rigid_body_dof + axes];
        if (!rigid_body_velocity_force_switch[rigid_body_idx][axes]) {
          it.insert(it.end(), index.begin(), index.end());
        }
      }

      for (int axes = 0; axes < rotation_dof; axes++) {
        vector<PetscInt> &it =
            out_process_index[rigid_body_idx * rigid_body_dof +
                              translation_dof + axes];
        if (!rigid_body_angvelocity_torque_switch[rigid_body_idx][axes]) {
          it.insert(it.end(), index.begin(), index.end());
        }
      }
    }
  }

  if (rank == size - 1) {
    for (int rigid_body_idx = 0; rigid_body_idx < num_rigid_body;
         rigid_body_idx++) {
      for (int axes = 0; axes < translation_dof; axes++) {
        if (rigid_body_velocity_force_switch[rigid_body_idx][axes]) {
          vector<PetscInt> &it =
              out_process_index[rigid_body_idx * rigid_body_dof + axes];
          it.push_back(global_rigid_body_offset +
                       rigid_body_idx * rigid_body_dof + axes);
        }
      }
      for (int axes = 0; axes < rotation_dof; axes++) {
        if (!rigid_body_angvelocity_torque_switch[rigid_body_idx][axes]) {
          vector<PetscInt> &it =
              out_process_index[rigid_body_idx * rigid_body_dof +
                                translation_dof + axes];
          it.push_back(global_rigid_body_offset +
                       rigid_body_idx * rigid_body_dof + translation_dof +
                       axes);
        }
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
    const int current_particle_local_index = local_idx[i];
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
        const int velocity_global_index =
            field_dof * current_particle_global_index + axes1;

        A.increment(velocity_local_index, velocity_global_index, 1.0);
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

          area += p_spacing[i][0] * p_spacing[i][1];
          area_num++;
        }

        // apply pressure
        for (int axes1 = 0; axes1 < translation_dof; axes1++) {
          if (!rigid_body_velocity_force_switch[current_rigid_body_index]
                                               [axes1]) {
            A.out_process_increment(current_rigid_body_local_offset + axes1,
                                    pressure_global_index, -dA[axes1]);
          }
        }

        for (int axes1 = 0; axes1 < rotation_dof; axes1++) {
          if (!rigid_body_angvelocity_torque_switch[current_rigid_body_index]
                                                   [axes1]) {
            A.out_process_increment(current_rigid_body_local_offset +
                                        translation_dof + axes1,
                                    pressure_global_index,
                                    -rci[(axes1 + 1) % translation_dof] *
                                            dA[(axes1 + 2) % translation_dof] +
                                        rci[(axes1 + 2) % translation_dof] *
                                            dA[(axes1 + 1) % translation_dof]);
          }
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
                auto alpha_index1 = velocity_colloid_basis->getAlphaIndexHost(
                    colloid_map[i], velocity_gradient_index_1);
                auto alpha_index2 = velocity_colloid_basis->getAlphaIndexHost(
                    colloid_map[i], velocity_gradient_index_2);
                const double sigma =
                    eta * (velocity_colloid_alpha(alpha_index1 + j) +
                           velocity_colloid_alpha(alpha_index2 + j));

                f[axes1] += sigma * dA[axes2];
              }
            }

            // force balance
            for (int axes1 = 0; axes1 < translation_dof; axes1++) {
              if (!rigid_body_velocity_force_switch[current_rigid_body_index]
                                                   [axes1]) {
                A.out_process_increment(current_rigid_body_local_offset + axes1,
                                        velocity_global_index, f[axes1]);
              }
            }

            // torque balance
            for (int axes1 = 0; axes1 < rotation_dof; axes1++) {
              if (!rigid_body_angvelocity_torque_switch
                      [current_rigid_body_index][axes1]) {
                A.out_process_increment(
                    current_rigid_body_local_offset + translation_dof + axes1,
                    velocity_global_index,
                    rci[(axes1 + 1) % translation_dof] *
                            f[(axes1 + 2) % translation_dof] -
                        rci[(axes1 + 2) % translation_dof] *
                            f[(axes1 + 1) % translation_dof]);
              }
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
          LaplacianOfScalarPointEvaluation, neumann_index,
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

    if (current_refinement_level == 0) {
      A.increment(pressure_local_index, pressure_global_index, 1e-6);
    }

    // end of pressure block
  } // end of fluid particle loop

  if (rank == size - 1) {
    for (int rigid_body_idx = 0; rigid_body_idx < num_rigid_body;
         rigid_body_idx++) {
      for (int axes = 0; axes < translation_dof; axes++) {
        if (rigid_body_velocity_force_switch[rigid_body_idx][axes]) {
          A.out_process_increment(
              local_rigid_body_offset + rigid_body_idx * rigid_body_dof + axes,
              global_rigid_body_offset + rigid_body_idx * rigid_body_dof + axes,
              1.0);
        }
      }
      for (int axes = 0; axes < rotation_dof; axes++) {
        if (!rigid_body_angvelocity_torque_switch[rigid_body_idx][axes]) {
          A.out_process_increment(
              local_rigid_body_offset + rigid_body_idx * rigid_body_dof +
                  translation_dof + axes,
              global_rigid_body_offset + rigid_body_idx * rigid_body_dof +
                  translation_dof + axes,
              1.0);
        }
      }
    }
  }

  // stabilize the coefficient matrix
  // invert_row_index.clear();
  abandon_this_level = 0;
  for (int i = 0; i < local_particle_num; i++) {
    const int current_particle_local_index = local_idx[i];
    const int current_particle_global_index = source_index[i];

    for (int k = 0; k < field_dof; k++) {
      const int local_index = current_particle_local_index * field_dof + k;
      const int global_index = current_particle_global_index * field_dof + k;

      if (A.get_entity(local_index, global_index) < 0.0) {
        cout << fixed << setprecision(10) << source_index[i] << " " << k << ' '
             << adaptive_level[i] << " " << particle_type[i] << " "
             << epsilon[i] << ' ' << neighbor_list_host(i, 0) << ' ';

        cout << "(";
        for (int k = 0; k < dim; k++) {
          cout << " " << coord[i][k];
        }
        cout << ") " << endl;
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &abandon_this_level, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &area, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "surface area: %f\n", area);
  MPI_Allreduce(MPI_IN_PLACE, &area_num, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "surface area particle num: %d\n", area_num);

  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "Current memory usage before assembly %.2f GB\n", mem / 1e9);

  // release memory
  velocity_basis.reset();

  auto ff = multi_mgr->get_field_mat(current_refinement_level);
  A.assemble(*ff, field_dof, num_rigid_body, rigid_body_dof);

  // if (current_refinement_level == 0) {
  //   A.write(string("A" + to_string(current_refinement_level) + ".txt"));
  // }

  MPI_Barrier(MPI_COMM_WORLD);
  timer2 = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "Matrix assembly duration: %fs\n",
              timer2 - timer1);

  idx_colloid.clear();

  if (num_rigid_body != 0)
    A.extract_neighbor_index(idx_colloid, dim, num_rigid_body,
                             local_rigid_body_offset, global_rigid_body_offset);

  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "Current memory usage after assembly %.2f GB\n",
              mem / 1e9);

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
  auto &local_idx = *(geo_mgr->get_current_work_particle_local_index());

  auto &rigid_body_position = rb_mgr->get_position();
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

  // for (int i = 0; i < local_particle_num; i++) {
  //   int current_particle_local_index = local_idx[i];
  //   if (particle_type[i] != 0 && particle_type[i] < 4) {
  //     double y = coord[i][1];

  //     rhs[field_dof * current_particle_local_index] = 0.1 * y;
  //   }
  // }

  for (int i = 0; i < local_particle_num; i++) {
    int current_particle_local_index = local_idx[i];
    if (particle_type[i] != 0 && particle_type[i] < 4) {
      // 2-d Taylor-Green vortex-like flow
      if (dim == 2) {
        double x = coord[i][0];
        double y = coord[i][1];

        rhs[field_dof * current_particle_local_index] =
            sin(M_PI * x) * cos(M_PI * y);
        rhs[field_dof * current_particle_local_index + 1] =
            -cos(M_PI * x) * sin(M_PI * y);

        const int neumann_index = neumann_map[i];
        const double bi = pressure_neumann_basis->getAlpha0TensorTo0Tensor(
            LaplacianOfScalarPointEvaluation, neumann_index,
            neumann_neighbor_list->getNumberOfNeighborsHost(neumann_index));

        rhs[field_dof * current_particle_local_index + velocity_dof] =
            -4.0 * pow(M_PI, 2.0) *
                (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y)) +
            bi * (normal[i][0] * 2.0 * pow(M_PI, 2.0) * sin(M_PI * x) *
                      cos(M_PI * y) -
                  normal[i][1] * 2.0 * pow(M_PI, 2.0) * cos(M_PI * x) *
                      sin(M_PI * y)) +
            bi * (normal[i][0] * 2.0 * M_PI * sin(2.0 * M_PI * x) +
                  normal[i][1] * 2.0 * M_PI * sin(2.0 * M_PI * y));
      }

      // 3-d Taylor-Green vortex-like flow
      if (dim == 3 && num_rigid_body == 0) {
        double x = coord[i][0];
        double y = coord[i][1];
        double z = coord[i][2];

        rhs[field_dof * i] = cos(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
        rhs[field_dof * i + 1] =
            -2 * sin(M_PI * x) * cos(M_PI * y) * sin(M_PI * z);
        rhs[field_dof * i + 2] = sin(M_PI * x) * sin(M_PI * y) * cos(M_PI * z);

        const int neumann_index = neumann_map[i];
        const double bi = pressure_neumann_basis->getAlpha0TensorTo0Tensor(
            LaplacianOfScalarPointEvaluation, neumann_index,
            neumann_neighbor_list->getNumberOfNeighborsHost(neumann_index));

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
    } else if (particle_type[i] == 0) {
      if (dim == 2) {
        double x = coord[i][0];
        double y = coord[i][1];

        rhs[field_dof * current_particle_local_index] =
            2.0 * pow(M_PI, 2.0) * sin(M_PI * x) * cos(M_PI * y) +
            2.0 * M_PI * sin(2.0 * M_PI * x);
        rhs[field_dof * current_particle_local_index + 1] =
            -2.0 * pow(M_PI, 2.0) * cos(M_PI * x) * sin(M_PI * y) +
            2.0 * M_PI * sin(2.0 * M_PI * y);

        rhs[field_dof * current_particle_local_index + velocity_dof] =
            -4.0 * pow(M_PI, 2.0) * (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y));
      }
      if (dim == 3 && num_rigid_body == 0) {
        double x = coord[i][0];
        double y = coord[i][1];
        double z = coord[i][2];

        rhs[field_dof * i] =
            3.0 * pow(M_PI, 2) * cos(M_PI * x) * sin(M_PI * y) * sin(M_PI * z) +
            2.0 * M_PI * sin(2.0 * M_PI * x);
        rhs[field_dof * i + 1] = -6.0 * pow(M_PI, 2) * sin(M_PI * x) *
                                     cos(M_PI * y) * sin(M_PI * z) +
                                 2.0 * M_PI * sin(2.0 * M_PI * y);
        rhs[field_dof * i + 2] =
            3.0 * pow(M_PI, 2) * sin(M_PI * x) * sin(M_PI * y) * cos(M_PI * z) +
            2.0 * M_PI * sin(2.0 * M_PI * z);

        rhs[field_dof * i + velocity_dof] =
            -4.0 * pow(M_PI, 2.0) *
            (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y) + cos(2.0 * M_PI * z));
      }
    }
  }
  // if (rank == size - 1) {
  //   for (int i = 0; i < num_rigid_body; i++) {
  //     rhs[local_rigid_body_offset + i * rigid_body_dof + 1] = 1.0;
  //   }
  // }

  // if (dim == 3 && num_rigid_body != 0) {
  //   vector<double> &rigid_body_size = rb_mgr->get_rigid_body_size();
  //   vector<vec3> &rigid_body_position = rb_mgr->get_position();

  //   double u = 1.0;
  //   double RR = rigid_body_size[0];

  //   for (int i = 0; i < local_particle_num; i++) {
  //     int current_particle_local_index = local_idx[i];
  //     if (particle_type[i] != 0 && particle_type[i] < 4) {
  //       double x = coord[i][0] - rigid_body_position[0][0];
  //       double y = coord[i][1] - rigid_body_position[0][1];
  //       double z = coord[i][2] - rigid_body_position[0][2];

  //       const int neumann_index = neumann_map[i];
  //       // const double bi =
  //       // pressureNeumannBoundaryBasis.getAlpha0TensorTo0Tensor(
  //       //     LaplacianOfScalarPointEvaluation, neumann_index,
  //       //     neumannBoundaryNeighborLists(neumann_index, 0));

  //       double r = sqrt(x * x + y * y + z * z);
  //       double theta = acos(z / r);
  //       double phi = atan2(y, x);

  //       double vr = u * cos(theta) *
  //                   (1 - (3 * RR) / (2 * r) + pow(RR, 3) / (2 * pow(r, 3)));
  //       double vt = -u * sin(theta) *
  //                   (1 - (3 * RR) / (4 * r) - pow(RR, 3) / (4 * pow(r, 3)));

  //       double pr = 3 * RR / pow(r, 3) * u * cos(theta);
  //       double pt = 3 / 2 * RR / pow(r, 3) * u * sin(theta);

  //       rhs[field_dof * current_particle_local_index] =
  //           sin(theta) * cos(phi) * vr + cos(theta) * cos(phi) * vt;
  //       rhs[field_dof * current_particle_local_index + 1] =
  //           sin(theta) * sin(phi) * vr + cos(theta) * sin(phi) * vt;
  //       rhs[field_dof * current_particle_local_index + 2] =
  //           cos(theta) * vr - sin(theta) * vt;

  //       double p1 = sin(theta) * cos(phi) * pr + cos(theta) * cos(phi) * pt;
  //       double p2 = sin(theta) * sin(phi) * pr + cos(theta) * sin(phi) * pt;
  //       double p3 = cos(theta) * pr - sin(theta) * pt;
  //     } else if (particle_type[i] >= 4) {
  //       double x = coord[i][0] - rigid_body_position[0][0];
  //       double y = coord[i][1] - rigid_body_position[0][1];
  //       double z = coord[i][2] - rigid_body_position[0][2];

  //       double r = sqrt(x * x + y * y + z * z);
  //       double theta = acos(z / r);
  //       double phi = atan2(y, x);

  //       const int neumann_index = neumann_map[i];
  //       // const double bi =
  //       // pressureNeumannBoundaryBasis.getAlpha0TensorTo0Tensor(
  //       //     LaplacianOfScalarPointEvaluation, neumann_index,
  //       //     neumannBoundaryNeighborLists(neumann_index, 0));

  //       double pr = 3 * RR / pow(r, 3) * u * cos(theta);
  //       double pt = 3 / 2 * RR / pow(r, 3) * u * sin(theta);

  //       double p1 = sin(theta) * cos(phi) * pr + cos(theta) * cos(phi) * pt;
  //       double p2 = sin(theta) * sin(phi) * pr + cos(theta) * sin(phi) * pt;
  //       double p3 = cos(theta) * pr - sin(theta) * pt;
  //     }
  //   }

  //   if (rank == size - 1) {
  //     rhs[local_rigid_body_offset + 2] = 6 * M_PI * RR * u;
  //   }
  // }

  if (dim == 3 && num_rigid_body != 0) {
    if (rank == size - 1) {
      rhs[local_rigid_body_offset + 1] = 0.03;
    }
  }

  // make sure pressure term is orthogonal to the constant
  double rhs_pressure_sum = 0.0;
  for (int i = 0; i < local_particle_num; i++) {
    int current_particle_local_index = local_idx[i];
    rhs_pressure_sum +=
        rhs[field_dof * current_particle_local_index + velocity_dof];
  }
  MPI_Allreduce(MPI_IN_PLACE, &rhs_pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  rhs_pressure_sum /= global_particle_num;
  for (int i = 0; i < local_particle_num; i++) {
    int current_particle_local_index = local_idx[i];
    rhs[field_dof * current_particle_local_index + velocity_dof] -=
        rhs_pressure_sum;
  }
}

void stokes_equation::solve_step() {
  const int num_rigid_body = rb_mgr->get_rigid_body_num();

  auto &local_idx = *(geo_mgr->get_current_work_particle_local_index());

  // build interpolation and resitriction operators
  double timer1, timer2;
  if (current_refinement_level != 0) {
    timer1 = MPI_Wtime();

    multi_mgr->build_interpolation_restriction(num_rigid_body, dim, poly_order);
    multi_mgr->initial_guess_from_previous_adaptive_step(
        res, velocity, pressure, rb_mgr->get_velocity(),
        rb_mgr->get_angular_velocity());

    timer2 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,
                "Interpolation matrix building duration: %fs\n",
                timer2 - timer1);
  }

  PetscLogDefaultBegin();

  MPI_Barrier(MPI_COMM_WORLD);
  timer1 = MPI_Wtime();
  // if (current_refinement_level < 4)
  multi_mgr->solve(rhs, res, idx_colloid);
  MPI_Barrier(MPI_COMM_WORLD);
  timer2 = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "linear system solving duration: %fs\n",
              timer2 - timer1);

  if (use_viewer) {
    PetscViewer viewer;
    PetscViewerASCIIGetStdout(PETSC_COMM_WORLD, &viewer);
    PetscLogView(viewer);
  }

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
    int current_particle_local_index = local_idx[i];
    pressure[i] = res[field_dof * current_particle_local_index + velocity_dof];
    pressure_sum += pressure[i];
    for (int axes1 = 0; axes1 < dim; axes1++)
      velocity[i][axes1] =
          res[field_dof * current_particle_local_index + axes1];
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
  auto &volume = *(geo_mgr->get_current_work_particle_volume());

  vector<vec3> &rigid_body_position = rb_mgr->get_position();
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

  vector<vector<double>> &rigid_body_size = rb_mgr->get_rigid_body_size();

  double u, RR;

  if (num_rigid_body != 0) {
    u = 1.0;
    RR = rigid_body_size[0][0];
  }

  int local_dof = local_velocity_dof + local_pressure_dof;

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
      if (num_rigid_body != 0) {
        double x = coord[i][0] - rigid_body_position[0][0];
        double y = coord[i][1] - rigid_body_position[0][1];
        double z = coord[i][2] - rigid_body_position[0][2];

        double r = sqrt(x * x + y * y + z * z);
        double theta = acos(z / r);

        double true_pressure = -1.5 * RR / pow(r, 2.0) * u * cos(theta);

        true_pressure_mean += true_pressure;
        pressure_mean += pressure[i];
      }
      if (num_rigid_body == 0) {
        double x = coord[i][0];
        double y = coord[i][1];
        double z = coord[i][2];

        double true_pressure =
            -cos(2.0 * M_PI * x) - cos(2.0 * M_PI * y) - cos(2.0 * M_PI * z);

        true_pressure_mean += true_pressure;
        pressure_mean += pressure[i];
      }
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
      true_velocity[0] = sin(M_PI * x) * cos(M_PI * y);
      true_velocity[1] = -cos(M_PI * x) * sin(M_PI * y);

      error_velocity += pow(true_velocity[0] - velocity[i][0], 2) * volume[i] +
                        pow(true_velocity[1] - velocity[i][1], 2) * volume[i];
      error_pressure += pow(true_pressure - pressure[i], 2) * volume[i];

      norm_velocity += pow(true_velocity[0], 2) * volume[i] +
                       pow(true_velocity[1], 2) * volume[i];
      norm_pressure += pow(true_pressure, 2) * volume[i];
    }

    if (dim == 3) {
      if (num_rigid_body != 0) {
        double x = coord[i][0] - rigid_body_position[0][0];
        double y = coord[i][1] - rigid_body_position[0][1];
        double z = coord[i][2] - rigid_body_position[0][2];

        double r = sqrt(x * x + y * y + z * z);
        double theta = acos(z / r);
        double phi = atan2(y, x);

        double vr =
            u * cos(theta) *
            (1.0 - (3.0 * RR) / (2.0 * r) + pow(RR, 3.0) / (2.0 * pow(r, 3.0)));
        double vt =
            -u * sin(theta) *
            (1.0 - (3.0 * RR) / (4.0 * r) - pow(RR, 3.0) / (4.0 * pow(r, 3.0)));

        double pr = 3.0 * RR / pow(r, 3.0) * u * cos(theta);
        double pt = 3.0 / 2.0 * RR / pow(r, 3.0) * u * sin(theta);

        vec3 true_velocity;

        true_velocity[0] =
            sin(theta) * cos(phi) * vr + cos(theta) * cos(phi) * vt;
        true_velocity[1] =
            sin(theta) * sin(phi) * vr + cos(theta) * sin(phi) * vt;
        true_velocity[2] = cos(theta) * vr - sin(theta) * vt;

        double true_pressure =
            -1.5 * RR / pow(r, 2.0) * u * cos(theta) - true_pressure_mean;

        error_velocity += pow(true_velocity[0] - velocity[i][0], 2) +
                          pow(true_velocity[1] - velocity[i][1], 2) +
                          pow(true_velocity[2] - velocity[i][2], 2);
        error_pressure += pow(true_pressure - pressure[i], 2);

        norm_velocity += pow(true_velocity[0], 2) + pow(true_velocity[1], 2) +
                         pow(true_velocity[2], 2);
        norm_pressure += pow(true_pressure, 2);
      }
      if (num_rigid_body == 0) {
        double x = coord[i][0];
        double y = coord[i][1];
        double z = coord[i][2];

        vec3 true_velocity;

        true_velocity[0] = cos(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
        true_velocity[1] = -2.0 * sin(M_PI * x) * cos(M_PI * y) * sin(M_PI * z);
        true_velocity[2] = sin(M_PI * x) * sin(M_PI * y) * cos(M_PI * z);

        double true_pressure = -cos(2.0 * M_PI * x) - cos(2.0 * M_PI * y) -
                               cos(2.0 * M_PI * z) - true_pressure_mean;

        error_velocity += pow(true_velocity[0] - velocity[i][0], 2) +
                          pow(true_velocity[1] - velocity[i][1], 2) +
                          pow(true_velocity[2] - velocity[i][2], 2);
        error_pressure += pow(true_pressure - pressure[i], 2);

        norm_velocity += pow(true_velocity[0], 2) + pow(true_velocity[1], 2) +
                         pow(true_velocity[2], 2);
        norm_pressure += pow(true_pressure, 2);
      }
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

  // gradient
  double error_velocity_gradient = 0.0;
  double norm_velocity_gradient = 0.0;
  if (dim == 3 && num_rigid_body != 0) {
    for (int i = 0; i < local_particle_num; i++) {
      double x = coord[i][0];
      double y = coord[i][1];
      double z = coord[i][2];

      double r = sqrt(x * x + y * y + z * z);
      double theta = acos(z / r);
      double phi = atan2(y, x);

      double dvr[3][3];

      dvr[0][0] = u * cos(theta) *
                  (3.0 * RR / 2.0 / pow(r, 2.0) -
                   3.0 * pow(RR, 3.0) / 2.0 / pow(r, 4.0));
      dvr[0][1] = -u * sin(theta) *
                  (1.0 / r - 3.0 * RR / 2.0 / pow(r, 2.0) +
                   pow(RR, 3.0) / 2.0 / pow(r, 4.0));
      dvr[0][2] = 0.0;
      dvr[1][0] = -u * sin(theta) *
                  (3.0 * RR / 4.0 / pow(r, 2.0) +
                   3.0 * pow(RR, 3.0) / 4.0 / pow(r, 4.0));
      dvr[1][1] = -u * cos(theta) *
                  (1.0 / r - 3.0 * RR / 4.0 / pow(r, 2.0) -
                   pow(RR, 3.0) / 4.0 / pow(r, 4.0));
      dvr[1][2] = 0.0;
      dvr[2][0] = 0.0;
      dvr[2][1] = 0.0;
      dvr[2][2] = 0.0;

      double R[3][3];
      R[0][0] = sin(theta) * cos(phi);
      R[0][1] = sin(theta) * sin(phi);
      R[0][2] = cos(theta);
      R[1][0] = cos(theta) * cos(phi);
      R[1][1] = cos(theta) * sin(phi);
      R[1][2] = -sin(theta);
      R[2][0] = -sin(phi);
      R[2][1] = cos(phi);
      R[2][2] = 0.0;

      double du[3][3];
      for (int m = 0; m < 3; m++) {
        for (int n = 0; n < 3; n++) {
          du[m][n] = 0.0;
          for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
              du[m][n] += R[j][m] * R[k][n] * dvr[j][k];
            }
          }
        }
      }

      for (int m = 0; m < 3; m++) {
        for (int n = 0; n < 3; n++) {
          error_velocity_gradient +=
              pow(gradient[i][m * dim + n] - du[m][n], 2.0);
          norm_velocity_gradient += pow(du[m][n], 2.0);
        }
      }
    }
  }

  if (dim == 2) {
    for (int i = 0; i < local_particle_num; i++) {
      double x = coord[i][0];
      double y = coord[i][1];

      double du[2][2];
      du[0][0] = M_PI * cos(M_PI * x) * cos(M_PI * y);
      du[0][1] = -M_PI * sin(M_PI * x) * sin(M_PI * y);
      du[1][0] = M_PI * sin(M_PI * x) * sin(M_PI * y);
      du[1][1] = -M_PI * cos(M_PI * x) * cos(M_PI * y);

      for (int m = 0; m < 2; m++) {
        for (int n = 0; n < 2; n++) {
          error_velocity_gradient +=
              pow(gradient[i][m * dim + n] - du[m][n], 2.0);
          norm_velocity_gradient += pow(du[m][n], 2.0);
        }
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &error_velocity_gradient, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &norm_velocity_gradient, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  PetscPrintf(MPI_COMM_WORLD, "relative velocity gradient error: %.10f\n",
              sqrt(error_velocity_gradient / norm_velocity_gradient));
  PetscPrintf(MPI_COMM_WORLD, "RMS velocity gradient error: %.10f\n",
              sqrt(error_velocity_gradient / global_particle_num));
  PetscPrintf(MPI_COMM_WORLD, "error: %.10f, and norm %.10f\n",
              sqrt(error_velocity_gradient), sqrt(norm_velocity_gradient));

  // error_pressure_gradient = 0.0;
  // norm_pressure_gradient = 0.0;
  // for (int i = 0; i < local_particle_num; i++) {
  //   double x = coord[i][0];
  //   double y = coord[i][1];
  //   double dpdx = 2 * M_PI * sin(2 * M_PI * x);
  //   double dpdy = 2 * M_PI * sin(2 * M_PI * y);
  //   double dpdx_gmls = pressure_gradient(i, 0);
  //   double dpdy_gmls = pressure_gradient(i, 1);
  //   if (particle_type[i] != 0) {
  //     double bx_i = pressure_neumann_basis->getAlpha0TensorTo1Tensor(
  //         GradientOfScalarPointEvaluation, neumann_map[i], 0,
  //         neumann_neighbor_list->getNumberOfNeighborsHost(neumann_map[i]));
  //     double by_i = pressure_neumann_basis->getAlpha0TensorTo1Tensor(
  //         GradientOfScalarPointEvaluation, neumann_map[i], 1,
  //         neumann_neighbor_list->getNumberOfNeighborsHost(neumann_map[i]));
  //     // dpdx_gmls = pressure_gradient_neumann(neumann_map[i], 0) +
  //     //             bx_i * (normal[i][0] * dpdx + normal[i][1] * dpdy);
  //     // dpdy_gmls = pressure_gradient_neumann(neumann_map[i], 1) +
  //     //             by_i * (normal[i][0] * dpdx + normal[i][1] * dpdy);
  //     dpdx_gmls = pressure_gradient_neumann(neumann_map[i], 0);
  //     dpdy_gmls = pressure_gradient_neumann(neumann_map[i], 1);
  //   }

  //   error_pressure_gradient +=
  //       pow(dpdx - dpdx_gmls, 2.0) + pow(dpdy - dpdy_gmls, 2.0);
  //   norm_pressure_gradient += pow(dpdx, 2.0) + pow(dpdy, 2.0);
  // }

  // MPI_Allreduce(MPI_IN_PLACE, &error_pressure_gradient, 1, MPI_DOUBLE,
  // MPI_SUM,
  //               MPI_COMM_WORLD);
  // MPI_Allreduce(MPI_IN_PLACE, &norm_pressure_gradient, 1, MPI_DOUBLE,
  // MPI_SUM,
  //               MPI_COMM_WORLD);

  // PetscPrintf(MPI_COMM_WORLD, "relative pressure gradient error: %.10f\n",
  //             sqrt(error_pressure_gradient / norm_pressure_gradient));
  // PetscPrintf(MPI_COMM_WORLD, "RMS pressure gradient error: %.10f\n",
  //             sqrt(error_pressure_gradient / global_particle_num));
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

  auto neighbor_list = pressure_basis->getNeighborLists();

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

    vector<vector<double>> direct_gradient;
    direct_gradient.resize(local_particle_num);
    const int gradient_component_num = pow(dim, 2);
    for (size_t i = 0; i < local_particle_num; i++) {
      direct_gradient[i].resize(gradient_component_num);
    }

    vector<vector<double>> coefficients_chunk, ghost_coefficients_chunk;
    coefficients_chunk.resize(local_particle_num);

    size_t coefficients_size;

    if (compress_memory == 0) {
      Evaluator velocity_evaluator(velocity_basis.get());

      coefficients_size = velocity_basis->getPolynomialCoefficientsSize();

      auto coefficients =
          velocity_evaluator
              .applyFullPolynomialCoefficientsBasisToDataAllComponents<
                  double **, Kokkos::HostSpace>(ghost_velocity_device);

      for (int i = 0; i < local_particle_num; i++) {
        coefficients_chunk[i].resize(coefficients_size);
        for (int j = 0; j < coefficients_size; j++) {
          coefficients_chunk[i][j] = coefficients(i, j);
        }
      }
    } else {
      Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
          source_coord_device("source coordinates", source_coord.size(), 3);
      Kokkos::View<double **>::HostMirror source_coord_host =
          Kokkos::create_mirror_view(source_coord_device);

      for (size_t i = 0; i < source_coord.size(); i++) {
        for (int j = 0; j < 3; j++) {
          source_coord_host(i, j) = source_coord[i][j];
        }
      }

      Kokkos::deep_copy(source_coord_device, source_coord_host);

      int start_particle = 0;
      int end_particle;
      for (int i = 0; i < number_of_batches; i++) {
        GMLS temp_velocity_basis =
            GMLS(DivergenceFreeVectorTaylorPolynomial, VectorPointSample,
                 poly_order, dim, "LU", "STANDARD");

        int batch_size = local_particle_num / number_of_batches +
                         (i < (local_particle_num % number_of_batches));
        int end_particle = min(local_particle_num, start_particle + batch_size);
        int particle_num = end_particle - start_particle;

        Kokkos::View<double *, Kokkos::DefaultExecutionSpace> epsilon_device(
            "h supports", particle_num);
        Kokkos::View<double *>::HostMirror epsilon_host =
            Kokkos::create_mirror_view(epsilon_device);

        for (int i = 0; i < particle_num; i++) {
          epsilon_host(i) = epsilon[start_particle + i];
        }

        Kokkos::deep_copy(epsilon_device, epsilon_host);

        Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
            neighbor_list_device("neighbor lists", particle_num,
                                 max_neighbor + 1);
        Kokkos::View<int **>::HostMirror neighbor_list_host =
            Kokkos::create_mirror_view(neighbor_list_device);

        for (int i = 0; i < particle_num; i++) {
          neighbor_list_host(i, 0) =
              neighbor_list->getNumberOfNeighborsHost(i + start_particle);
          for (int j = 0; j < neighbor_list_host(i, 0); j++) {
            neighbor_list_host(i, j + 1) =
                neighbor_list->getNeighborHost(i + start_particle, j);
          }
        }

        Kokkos::deep_copy(neighbor_list_device, neighbor_list_host);

        Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
            target_coord_device("target coordinates", particle_num, 3);
        Kokkos::View<double **>::HostMirror target_coord_host =
            Kokkos::create_mirror_view(target_coord_device);

        for (int i = 0; i < particle_num; i++) {
          for (int j = 0; j < 3; j++) {
            target_coord_host(i, j) = coord[i + start_particle][j];
          }
        }

        Kokkos::deep_copy(target_coord_device, target_coord_host);

        temp_velocity_basis.setProblemData(neighbor_list_device,
                                           source_coord_device,
                                           target_coord_device, epsilon_device);

        vector<TargetOperation> velocity_operation(2);
        velocity_operation[0] = ScalarPointEvaluation;
        velocity_operation[1] = GradientOfVectorPointEvaluation;

        temp_velocity_basis.addTargets(velocity_operation);

        temp_velocity_basis.setWeightingType(WeightingFunctionType::Power);
        temp_velocity_basis.setWeightingPower(4);

        temp_velocity_basis.generateAlphas(1, true);

        Evaluator temp_velocity_evaluator(&temp_velocity_basis);

        auto coefficients =
            temp_velocity_evaluator
                .applyFullPolynomialCoefficientsBasisToDataAllComponents<
                    double **, Kokkos::HostSpace>(ghost_velocity_device);

        auto temp_gradient =
            temp_velocity_evaluator
                .applyAlphasToDataAllComponentsAllTargetSites<
                    double **, Kokkos::HostSpace>(
                    ghost_velocity_device, GradientOfVectorPointEvaluation);

        coefficients_size = temp_velocity_basis.getPolynomialCoefficientsSize();

        for (int i = 0; i < particle_num; i++) {
          coefficients_chunk[i + start_particle].resize(coefficients_size);
          for (int j = 0; j < coefficients_size; j++) {
            coefficients_chunk[i + start_particle][j] = coefficients(i, j);
          }
        }

        for (int i = 0; i < particle_num; i++) {
          for (int j = 0; j < gradient_component_num; j++)
            direct_gradient[i + start_particle][j] = temp_gradient(i, j);
        }

        start_particle += batch_size;
      }
    }

    geo_mgr->ghost_forward(coefficients_chunk, ghost_coefficients_chunk,
                           coefficients_size);

    // estimate stage
    auto &recovered_gradient = gradient;
    vector<vector<double>> ghost_recovered_gradient;
    recovered_gradient.resize(local_particle_num);
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
              recovered_gradient[i][axes1 * dim + axes2] +=
                  cal_div_free_grad(axes1, axes2, dim, dX, poly_order,
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
              reconstructed_gradient[axes1 * dim + axes2] = cal_div_free_grad(
                  axes1, axes2, dim, dX, poly_order, ghost_epsilon[i],
                  ghost_coefficients_chunk[i]);
            }
          }

          for (int axes1 = 0; axes1 < dim; axes1++) {
            for (int axes2 = 0; axes2 < dim; axes2++) {
              error[i] += pow(reconstructed_gradient[axes1 * dim + axes2] -
                                  ghost_recovered_gradient[neighbor_index]
                                                          [axes1 * dim + axes2],
                              2) *
                          source_volume[neighbor_index];
            }
          }
        }
      }

      error[i] = error[i] / total_neighbor_vol;
      local_error += error[i] * volume[i];

      for (int axes1 = 0; axes1 < dim; axes1++) {
        for (int axes2 = 0; axes2 < dim; axes2++) {
          local_direct_gradient_norm +=
              pow(direct_gradient[i][axes1 * dim + axes2], 2) * volume[i];
          recovered_gradient[i][axes1 * dim + axes2] =
              direct_gradient[i][axes1 * dim + axes2];
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
            double **, Kokkos::HostSpace>(ghost_pressure_device,
                                          GradientOfScalarPointEvaluation,
                                          PointSample);

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
                  PointSample);

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
  for (int ite = 0; ite < 10; ite++) {
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

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&local_direct_gradient_norm, &global_direct_gradient_norm, 1,
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  PetscPrintf(PETSC_COMM_WORLD,
              "global error: %f, global direct gradient norm: %f\n",
              global_error, global_direct_gradient_norm);

  global_error /= global_direct_gradient_norm;
  global_error = sqrt(global_error);

  for (int i = 0; i < local_particle_num; i++) {
    error[i] = sqrt(error[i]);
  }
}