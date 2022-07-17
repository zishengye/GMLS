#include "StokesEquation.hpp"
#include "DivergenceFree.hpp"
#include "gmls_solver.hpp"
#include "petsc_sparse_matrix.hpp"

#include <iomanip>
#include <mpi.h>
#include <utility>

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

void StokesEquation::Init(std::shared_ptr<ParticleGeometry> geoMgr,
                          std::shared_ptr<rigid_body_manager> rbMgr,
                          const int polyOrder, const int dim,
                          const int errorEstimationMethod,
                          const double epsilonMultiplier, const double eta) {
  geoMgr_ = geoMgr;
  rbMgr_ = rbMgr;
  eta_ = eta;
  dim_ = dim;
  polyOrder_ = polyOrder;
  errorEstimationMethod_ = errorEstimationMethod;

  multiMgr_ = std::make_shared<StokesMultilevelPreconditioning>();

  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);

  currentRefinementLevel_ = 0;
  multiMgr_->reset();
  multiMgr_->init(dim_, geoMgr_);
  multiMgr_->set_num_rigid_body(rbMgr_->get_rigid_body_num());
}

void StokesEquation::Update() {
  multiMgr_->add_new_level();

  BuildCoefficientMatrix();
  ConstructRhs();
  SolveEquation();
  CalculateError();
  // CheckSolution();
  // CollectForce();

  currentRefinementLevel_++;
}

void StokesEquation::Reset() {
  currentRefinementLevel_ = 0;
  multiMgr_->reset();
}

void StokesEquation::BuildCoefficientMatrix() {
  // prepare data
  auto &source_coord = *(geoMgr_->get_current_work_ghost_particle_coord());
  auto &source_index = *(geoMgr_->get_current_work_ghost_particle_index());
  auto &coord = *(geoMgr_->get_current_work_particle_coord());
  auto &normal = *(geoMgr_->get_current_work_particle_normal());
  auto &p_spacing = *(geoMgr_->get_current_work_particle_p_spacing());
  auto &spacing = *(geoMgr_->get_current_work_particle_spacing());
  auto &adaptive_level = *(geoMgr_->get_current_work_particle_adaptive_level());
  auto &local_idx = *(geoMgr_->get_current_work_particle_local_index());
  auto &particle_type = *(geoMgr_->get_current_work_particle_type());
  auto &attached_rigid_body =
      *(geoMgr_->get_current_work_particle_attached_rigid_body());
  auto &num_neighbor = *(geoMgr_->get_current_work_particle_num_neighbor());

  std::vector<int> ghost_particle_type;
  geoMgr_->ghost_forward(particle_type, ghost_particle_type);

  PetscLogDouble mem;

  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "Current memory usage before GMLS estimation %.2f GB\n",
              mem / 1e9);

  std::vector<Vec3> &rigid_body_position = rbMgr_->get_position();
  std::vector<Vec3> &rigid_body_velocity_force_switch =
      rbMgr_->get_velocity_force_switch();
  std::vector<Vec3> &rigid_body_angvelocity_torque_switch =
      rbMgr_->get_angvelocity_torque_switch();
  const int numRigidBody = rbMgr_->get_rigid_body_num();

  double timer1, timer2;
  MPI_Barrier(MPI_COMM_WORLD);
  timer1 = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "\nSolving GMLS subproblems...\n");

  unsigned int numLocalParticle;
  unsigned int numGlobalParticleNum;

  numLocalParticle = coord.size();
  MPI_Allreduce(&numLocalParticle, &numGlobalParticleNum, 1, MPI_UNSIGNED,
                MPI_SUM, MPI_COMM_WORLD);

  int num_source_coord = source_coord.size();

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
      "source coordinates", num_source_coord, 3);
  Kokkos::View<double **>::HostMirror sourceCoordsHost =
      Kokkos::create_mirror_view(sourceCoordsDevice);

  for (size_t i = 0; i < num_source_coord; i++) {
    for (int j = 0; j < 3; j++) {
      sourceCoordsHost(i, j) = source_coord[i][j];
    }
  }

  int num_neumann_target_coord = 0;
  int num_colloid_target_coord = 0;
  for (int i = 0; i < numLocalParticle; i++) {
    if (particle_type[i] != 0) {
      num_neumann_target_coord++;
    }
    if (particle_type[i] >= 4) {
      num_colloid_target_coord++;
    }
  }

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> target_coord_device(
      "target coordinates", numLocalParticle, 3);
  Kokkos::View<double **>::HostMirror target_coord_host =
      Kokkos::create_mirror_view(target_coord_device);
  Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
      neumann_target_coord_device("neumann target coordinates",
                                  num_neumann_target_coord, 3);
  Kokkos::View<double **>::HostMirror neumann_target_coord_host =
      Kokkos::create_mirror_view(neumann_target_coord_device);
  Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
      colloid_target_coord_device("colloid target coordinates",
                                  num_colloid_target_coord, 3);
  Kokkos::View<double **>::HostMirror colloid_target_coord_host =
      Kokkos::create_mirror_view(colloid_target_coord_device);

  // create target coords
  int counter;
  neumann_map.clear();
  neumann_map.resize(numLocalParticle);
  counter = 0;
  for (int i = 0; i < numLocalParticle; i++) {
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

  std::vector<int> colloid_map;
  colloid_map.resize(numLocalParticle);
  counter = 0;
  for (int i = 0; i < numLocalParticle; i++) {
    if (particle_type[i] >= 4) {
      colloid_map[i] = counter;
      for (int j = 0; j < 3; j++) {
        colloid_target_coord_host(counter, j) = coord[i][j];
      }
      counter++;
    }
  }

  Kokkos::deep_copy(sourceCoordsDevice, sourceCoordsHost);
  Kokkos::deep_copy(target_coord_device, target_coord_host);
  Kokkos::deep_copy(neumann_target_coord_device, neumann_target_coord_host);
  Kokkos::deep_copy(colloid_target_coord_device, colloid_target_coord_host);

  // tangent bundle for neumann boundary particles
  Kokkos::View<double ***, Kokkos::DefaultExecutionSpace> tangent_bundle_device(
      "tangent bundles", num_neumann_target_coord, dim_, dim_);
  Kokkos::View<double ***>::HostMirror tangent_bundle_host =
      Kokkos::create_mirror_view(tangent_bundle_device);
  Kokkos::View<double ***, Kokkos::DefaultExecutionSpace>
      colloid_tangent_bundle_device("tangent bundles", num_colloid_target_coord,
                                    dim_, dim_);
  Kokkos::View<double ***>::HostMirror colloid_tangent_bundle_host =
      Kokkos::create_mirror_view(colloid_tangent_bundle_device);

  for (int i = 0; i < numLocalParticle; i++) {
    if (particle_type[i] != 0) {
      counter = neumann_map[i];
      if (dim_ == 3) {
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
      if (dim_ == 2) {
        tangent_bundle_host(counter, 0, 0) = 0.0;
        tangent_bundle_host(counter, 0, 1) = 0.0;
        tangent_bundle_host(counter, 1, 0) = normal[i][0];
        tangent_bundle_host(counter, 1, 1) = normal[i][1];
      }
      counter++;
    }

    if (particle_type[i] >= 4) {
      counter = colloid_map[i];
      if (dim_ == 3) {
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
      if (dim_ == 2) {
        colloid_tangent_bundle_host(counter, 0, 0) = 0.0;
        colloid_tangent_bundle_host(counter, 0, 1) = 0.0;
        colloid_tangent_bundle_host(counter, 1, 0) = normal[i][0];
        colloid_tangent_bundle_host(counter, 1, 1) = normal[i][1];
      }
    }
  }

  Kokkos::deep_copy(tangent_bundle_device, tangent_bundle_host);
  Kokkos::deep_copy(colloid_tangent_bundle_device, colloid_tangent_bundle_host);

  if (dim_ == 2)
    number_of_batches = std::max(numLocalParticle / 10000, (unsigned int)1);
  else
    number_of_batches = std::max(numLocalParticle / 1000, (unsigned int)1);

  // neighbor search
  auto point_cloud_search(
      Compadre::CreatePointCloudSearch(sourceCoordsHost, dim_));

  int min_num_neighbor = std::max(
      Compadre::GMLS::getNP(polyOrder_, dim_,
                            Compadre::DivergenceFreeVectorTaylorPolynomial),
      Compadre::GMLS::getNP(polyOrder_ + 1, dim_));
  int satisfied_num_neighbor = pow(2.0, dim_ / 2.0) * min_num_neighbor;

  Kokkos::resize(neighborLists_, numLocalParticle, satisfied_num_neighbor + 1);
  Kokkos::resize(epsilon_, numLocalParticle);

  double max_epsilon = geoMgr_->get_cutoff_distance();
  for (int i = 0; i < numLocalParticle; i++) {
    epsilon_(i) = 1.50 * spacing[i];
  }

  // ensure every particle has enough neighbors
  std::vector<bool> staggered_check;
  staggered_check.resize(numLocalParticle);
  for (int i = 0; i < numLocalParticle; i++) {
    staggered_check[i] = false;
  }
  bool pass_neighbor_search = false;
  int ite_counter = 0;
  double max_ratio;
  double mean_neighbor;
  // We have two layers of checking for neighbors. First, we ensure the number
  // of neighbors has satisfied the required number of neighbors by the GMLS
  // discretization order. Second, we ensure the positivity of coefficients
  // generated by the staggered GMLS discretization.

  point_cloud_search.generate2DNeighborListsFromKNNSearch(
      true, target_coord_host, neighborLists_, epsilon_, satisfied_num_neighbor,
      1.0);

  for (unsigned int i = 0; i < numLocalParticle; i++) {
    double minEpsilon = 1.50 * spacing[i];
    double minSpacing = 0.25 * spacing[i];
    epsilon_(i) = std::max(minEpsilon, epsilon_(i));
    unsigned int scaling =
        std::max(0.0, std::ceil((epsilon_(i) - minEpsilon) / minSpacing));
    epsilon_(i) = minEpsilon + scaling * minSpacing;
  }

  unsigned int minNeighborLists =
      1 + point_cloud_search.generate2DNeighborListsFromRadiusSearch(
              true, target_coord_host, neighborLists_, epsilon_, 0.0, 0.0);
  if (minNeighborLists > neighborLists_.extent(1)) {
    Kokkos::resize(neighborLists_, numLocalParticle, minNeighborLists);
  }
  point_cloud_search.generate2DNeighborListsFromRadiusSearch(
      false, target_coord_host, neighborLists_, epsilon_, 0.0, 0.0);

  max_ratio = 0.0;
  min_neighbor = 1000;
  max_neighbor = 0;
  mean_neighbor = 0;
  for (int i = 0; i < numLocalParticle; i++) {
    if (neighborLists_(i, 0) < min_neighbor)
      min_neighbor = neighborLists_(i, 0);
    if (neighborLists_(i, 0) > max_neighbor)
      max_neighbor = neighborLists_(i, 0);
    mean_neighbor += neighborLists_(i, 0);

    if (max_ratio < epsilon_(i) / spacing[i]) {
      max_ratio = epsilon_(i) / spacing[i];
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &min_neighbor, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_neighbor, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &mean_neighbor, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_ratio, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);

  while (!pass_neighbor_search) {
    // staggered GMLS discretization check
    // check fluid particles
    int num_normal_check_point = 0;
    std::vector<int> normal_check_point;
    for (int i = 0; i < numLocalParticle; i++) {
      if (particle_type[i] == 0 && !staggered_check[i])
        normal_check_point.push_back(i);
    }
    num_normal_check_point = normal_check_point.size();

    if (num_normal_check_point != 0) {
      Compadre::GMLS temp_staggered_basis =
          Compadre::GMLS(Compadre::ScalarTaylorPolynomial,
                         Compadre::StaggeredEdgeAnalyticGradientIntegralSample,
                         polyOrder_, dim_, "LU", "STANDARD");

      Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
          temp_neighbor_list_device("neighbor lists", num_normal_check_point,
                                    minNeighborLists);
      Kokkos::View<int **>::HostMirror temp_neighbor_list_host =
          Kokkos::create_mirror_view(temp_neighbor_list_device);
      for (int i = 0; i < normal_check_point.size(); i++) {
        int index = normal_check_point[i];
        for (int j = 0; j <= neighborLists_(index, 0); j++) {
          temp_neighbor_list_host(i, j) = neighborLists_(index, j);
        }
      }

      Kokkos::View<double *, Kokkos::DefaultExecutionSpace> temp_epsilon_device(
          "h supports", num_normal_check_point);
      Kokkos::View<double *>::HostMirror temp_epsilon_host =
          Kokkos::create_mirror_view(temp_epsilon_device);

      for (int i = 0; i < normal_check_point.size(); i++) {
        int index = normal_check_point[i];
        temp_epsilon_host(i) = epsilon_(index);
      }

      Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
          temp_target_coord_device("colloid target coordinates",
                                   num_normal_check_point, 3);
      Kokkos::View<double **>::HostMirror temp_target_coord_host =
          Kokkos::create_mirror_view(temp_target_coord_device);

      for (int i = 0; i < normal_check_point.size(); i++) {
        for (int j = 0; j < 3; j++) {
          temp_target_coord_host(i, j) =
              target_coord_host(normal_check_point[i], j);
        }
      }

      Kokkos::deep_copy(temp_target_coord_device, temp_target_coord_host);
      Kokkos::deep_copy(temp_epsilon_device, temp_epsilon_host);
      Kokkos::deep_copy(temp_neighbor_list_device, temp_neighbor_list_host);

      temp_staggered_basis.setProblemData(
          temp_neighbor_list_device, sourceCoordsDevice,
          temp_target_coord_device, temp_epsilon_device);

      temp_staggered_basis.addTargets(
          Compadre::LaplacianOfScalarPointEvaluation);

      temp_staggered_basis.setWeightingType(
          Compadre::WeightingFunctionType::Power);
      temp_staggered_basis.setWeightingParameter(4);
      temp_staggered_basis.setOrderOfQuadraturePoints(2);
      temp_staggered_basis.setDimensionOfQuadraturePoints(1);
      temp_staggered_basis.setQuadratureType("LINE");

      temp_staggered_basis.generateAlphas(1, false);

      auto solution_set = temp_staggered_basis.getSolutionSetHost();
      auto temp_pressure_alpha = solution_set->getAlphas();

      const int temp_pressure_laplacian_index =
          solution_set->getAlphaColumnOffset(
              Compadre::LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);

      for (int i = 0; i < normal_check_point.size(); i++) {
        double Aij = 0.0;
        for (int j = 0; j < temp_neighbor_list_host(i, 0); j++) {
          auto alpha_index =
              solution_set->getAlphaIndex(i, temp_pressure_laplacian_index);
          Aij -= temp_pressure_alpha(alpha_index + j);
        }
        // if (Aij > 0.0)
        staggered_check[normal_check_point[i]] = true;
      }
    }

    // check boundary particles
    int num_neumann_check_point = 0;
    std::vector<int> neumann_check_point;
    for (int i = 0; i < numLocalParticle; i++) {
      if (particle_type[i] != 0 && !staggered_check[i]) {
        neumann_check_point.push_back(i);
      }
    }

    num_neumann_check_point = neumann_check_point.size();
    if (num_neumann_check_point != 0) {
      Compadre::GMLS temp_staggered_basis = Compadre::GMLS(
          Compadre::ScalarTaylorPolynomial,
          Compadre::StaggeredEdgeAnalyticGradientIntegralSample, polyOrder_,
          dim_, "LU", "STANDARD", "NEUMANN_GRAD_SCALAR");

      Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
          temp_neighbor_list_device("neighbor lists", num_neumann_check_point,
                                    minNeighborLists);
      Kokkos::View<int **>::HostMirror temp_neighbor_list_host =
          Kokkos::create_mirror_view(temp_neighbor_list_device);
      for (int i = 0; i < neumann_check_point.size(); i++) {
        int index = neumann_check_point[i];
        for (int j = 0; j <= neighborLists_(index, 0); j++) {
          temp_neighbor_list_host(i, j) = neighborLists_(index, j);
        }
      }

      Kokkos::View<double *, Kokkos::DefaultExecutionSpace> temp_epsilon_device(
          "h supports", num_neumann_check_point);
      Kokkos::View<double *>::HostMirror temp_epsilon_host =
          Kokkos::create_mirror_view(temp_epsilon_device);

      for (int i = 0; i < neumann_check_point.size(); i++) {
        int index = neumann_check_point[i];
        temp_epsilon_host(i) = epsilon_(index);
      }

      Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
          temp_target_coord_device("colloid target coordinates",
                                   num_neumann_check_point, 3);
      Kokkos::View<double **>::HostMirror temp_target_coord_host =
          Kokkos::create_mirror_view(temp_target_coord_device);

      for (int i = 0; i < neumann_check_point.size(); i++) {
        for (int j = 0; j < 3; j++) {
          temp_target_coord_host(i, j) =
              target_coord_host(neumann_check_point[i], j);
        }
      }

      Kokkos::View<double ***, Kokkos::DefaultExecutionSpace>
          temp_tangent_bundle_device("tangent bundles", num_neumann_check_point,
                                     dim_, dim_);
      Kokkos::View<double ***>::HostMirror temp_tangent_bundle_host =
          Kokkos::create_mirror_view(temp_tangent_bundle_device);

      for (int i = 0; i < neumann_check_point.size(); i++) {
        int idx = neumann_check_point[i];
        if (dim_ == 3) {
          temp_tangent_bundle_host(i, 0, 0) = 0.0;
          temp_tangent_bundle_host(i, 0, 1) = 0.0;
          temp_tangent_bundle_host(i, 0, 2) = 0.0;
          temp_tangent_bundle_host(i, 1, 0) = 0.0;
          temp_tangent_bundle_host(i, 1, 1) = 0.0;
          temp_tangent_bundle_host(i, 1, 2) = 0.0;
          temp_tangent_bundle_host(i, 2, 0) = normal[idx][0];
          temp_tangent_bundle_host(i, 2, 1) = normal[idx][1];
          temp_tangent_bundle_host(i, 2, 2) = normal[idx][2];
        }
        if (dim_ == 2) {
          temp_tangent_bundle_host(i, 0, 0) = 0.0;
          temp_tangent_bundle_host(i, 0, 1) = 0.0;
          temp_tangent_bundle_host(i, 1, 0) = normal[idx][0];
          temp_tangent_bundle_host(i, 1, 1) = normal[idx][1];
        }
      }

      Kokkos::deep_copy(temp_target_coord_device, temp_target_coord_host);
      Kokkos::deep_copy(temp_epsilon_device, temp_epsilon_host);
      Kokkos::deep_copy(temp_neighbor_list_device, temp_neighbor_list_host);
      Kokkos::deep_copy(temp_tangent_bundle_device, temp_tangent_bundle_host);

      temp_staggered_basis.setProblemData(
          temp_neighbor_list_device, sourceCoordsDevice,
          temp_target_coord_device, temp_epsilon_device);

      temp_staggered_basis.setTangentBundle(temp_tangent_bundle_device);

      temp_staggered_basis.addTargets(
          Compadre::LaplacianOfScalarPointEvaluation);

      temp_staggered_basis.setWeightingType(
          Compadre::WeightingFunctionType::Power);
      temp_staggered_basis.setWeightingParameter(4);
      temp_staggered_basis.setOrderOfQuadraturePoints(2);
      temp_staggered_basis.setDimensionOfQuadraturePoints(1);
      temp_staggered_basis.setQuadratureType("LINE");

      temp_staggered_basis.generateAlphas(1, false);

      auto solution_set = temp_staggered_basis.getSolutionSetHost();
      auto temp_pressure_alpha = solution_set->getAlphas();

      const int temp_pressure_laplacian_index =
          solution_set->getAlphaColumnOffset(
              Compadre::LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);

      for (int i = 0; i < neumann_check_point.size(); i++) {
        double Aij = 0.0;
        for (int j = 0; j < temp_neighbor_list_host(i, 0); j++) {
          auto alpha_index =
              solution_set->getAlphaIndex(i, temp_pressure_laplacian_index);
          Aij -= temp_pressure_alpha(alpha_index + j);
        }
        // if (Aij > 0.0)
        staggered_check[neumann_check_point[i]] = true;
      }
    }

    int process_counter = 0;
    for (int i = 0; i < numLocalParticle; i++) {
      if (!staggered_check[i]) {
        if (epsilon_(i) + 0.25 * spacing[i] < max_epsilon) {
          epsilon_(i) += 0.25 * spacing[i];
          process_counter = 1;
        }
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &process_counter, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    if (process_counter == 0) {
      pass_neighbor_search = true;
    } else {
      minNeighborLists =
          1 + point_cloud_search.generate2DNeighborListsFromRadiusSearch(
                  true, target_coord_host, neighborLists_, epsilon_, 0.0, 0.0);
      if (minNeighborLists > neighborLists_.extent(1)) {
        Kokkos::resize(neighborLists_, numLocalParticle, minNeighborLists);
      }
      point_cloud_search.generate2DNeighborListsFromRadiusSearch(
          false, target_coord_host, neighborLists_, epsilon_, 0.0, 0.0);

      max_ratio = 0.0;
      min_neighbor = 1000;
      max_neighbor = 0;
      mean_neighbor = 0;
      for (int i = 0; i < numLocalParticle; i++) {
        if (neighborLists_(i, 0) < min_neighbor)
          min_neighbor = neighborLists_(i, 0);
        if (neighborLists_(i, 0) > max_neighbor)
          max_neighbor = neighborLists_(i, 0);
        mean_neighbor += neighborLists_(i, 0);

        if (max_ratio < epsilon_(i) / spacing[i]) {
          max_ratio = epsilon_(i) / spacing[i];
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &min_neighbor, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &max_neighbor, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &mean_neighbor, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &max_ratio, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
    }

    ite_counter++;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(MPI_COMM_WORLD,
              "iteration count: %d min neighbor: %d, max neighbor: %d , mean "
              "neighbor %f, max ratio: %f\n",
              ite_counter, min_neighbor, max_neighbor,
              mean_neighbor / (double)numGlobalParticleNum, max_ratio);

  // matrix assembly
  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "\nGenerating Stokes Matrix...\n");

  const int translationDof = (dim_ == 3 ? 3 : 2);
  const int rotationDof = (dim_ == 3 ? 3 : 1);
  const int rigidBodyDof = (dim_ == 3 ? 6 : 3);

  int fieldDof = dim_ + 1;
  int velocityDof = dim_;

  StokesMatrix &A = *(multiMgr_->getA(currentRefinementLevel_));
  A.SetSize(numLocalParticle, numRigidBody);

  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "Current memory usage after resizing %.2f GB\n",
              mem / 1e9);

  A.SetGraph(local_idx, source_index, particle_type, attached_rigid_body,
             neighborLists_);

  MPI_Barrier(MPI_COMM_WORLD);
  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "Current memory usage after building the graph %.2f GB\n",
              mem / 1e9);

  Kokkos::resize(bi_, numLocalParticle);

  const unsigned int batchSize =
      ((dim_ == 2) ? 500 : 100) * omp_get_max_threads();
  const unsigned int numBatch = numLocalParticle / batchSize +
                                ((numLocalParticle % batchSize > 0) ? 1 : 0);

  for (unsigned int batch = 0; batch < numBatch; batch++) {
    const unsigned int startParticle = batch * batchSize;
    const unsigned int endParticle =
        std::min((batch + 1) * batchSize, (unsigned int)numLocalParticle);

    unsigned int numInteriorParticle, numBoundaryParticle;
    numInteriorParticle = 0;
    numBoundaryParticle = 0;
    for (unsigned int i = startParticle; i < endParticle; i++) {
      if (particle_type[i] == 0)
        numInteriorParticle++;
      else
        numBoundaryParticle++;
    }

    Kokkos::View<std::size_t **, Kokkos::DefaultExecutionSpace>
        interiorNeighborListsDevice("interior particle neighbor list",
                                    numInteriorParticle,
                                    neighborLists_.extent(1));
    Kokkos::View<std::size_t **>::HostMirror interiorNeighborListsHost =
        Kokkos::create_mirror_view(interiorNeighborListsDevice);
    Kokkos::View<std::size_t **, Kokkos::DefaultExecutionSpace>
        boundaryNeighborListsDevice("boundary particle neighbor list",
                                    numBoundaryParticle,
                                    neighborLists_.extent(1));
    Kokkos::View<std::size_t **>::HostMirror boundaryNeighborListsHost =
        Kokkos::create_mirror_view(boundaryNeighborListsDevice);

    Kokkos::View<double *, Kokkos::DefaultExecutionSpace> interiorEpsilonDevice(
        "interior particle epsilon", numInteriorParticle);
    Kokkos::View<double *>::HostMirror interiorEpsilonHost =
        Kokkos::create_mirror_view(interiorEpsilonDevice);
    Kokkos::View<double *, Kokkos::DefaultExecutionSpace> boundaryEpsilonDevice(
        "boundary particle epsilon", numBoundaryParticle);
    Kokkos::View<double *>::HostMirror boundaryEpsilonHost =
        Kokkos::create_mirror_view(boundaryEpsilonDevice);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        interiorParticleCoordsDevice("interior particle coord",
                                     numInteriorParticle, dim_);
    Kokkos::View<double **>::HostMirror interiorParticleCoordsHost =
        Kokkos::create_mirror_view(interiorParticleCoordsDevice);
    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        boundaryParticleCoordsDevice("boundary particle coord",
                                     numBoundaryParticle, dim_);
    Kokkos::View<double **>::HostMirror boundaryParticleCoordsHost =
        Kokkos::create_mirror_view(boundaryParticleCoordsDevice);

    Kokkos::View<double ***, Kokkos::DefaultExecutionSpace> tangentBundleDevice(
        "tangent bundles", numBoundaryParticle, dim_, dim_);
    Kokkos::View<double ***>::HostMirror tangentBundleHost =
        Kokkos::create_mirror_view(tangentBundleDevice);

    unsigned int boundaryCounter, interiorCounter;

    boundaryCounter = 0;
    interiorCounter = 0;
    for (unsigned int i = startParticle; i < endParticle; i++) {
      if (particle_type[i] == 0) {
        interiorEpsilonHost(interiorCounter) = epsilon_(i);
        for (std::size_t j = 0; j <= neighborLists_(i, 0); j++) {
          interiorNeighborListsHost(interiorCounter, j) = neighborLists_(i, j);
        }
        for (unsigned int j = 0; j < dim_; j++) {
          interiorParticleCoordsHost(interiorCounter, j) =
              target_coord_host(i, j);
        }

        interiorCounter++;
      } else {
        boundaryEpsilonHost(boundaryCounter) = epsilon_(i);
        for (std::size_t j = 0; j <= neighborLists_(i, 0); j++) {
          boundaryNeighborListsHost(boundaryCounter, j) = neighborLists_(i, j);
        }
        for (unsigned int j = 0; j < dim_; j++) {
          boundaryParticleCoordsHost(boundaryCounter, j) =
              target_coord_host(i, j);
        }
        if (dim_ == 3) {
          tangentBundleHost(boundaryCounter, 0, 0) = 0.0;
          tangentBundleHost(boundaryCounter, 0, 1) = 0.0;
          tangentBundleHost(boundaryCounter, 0, 2) = 0.0;
          tangentBundleHost(boundaryCounter, 1, 0) = 0.0;
          tangentBundleHost(boundaryCounter, 1, 1) = 0.0;
          tangentBundleHost(boundaryCounter, 1, 2) = 0.0;
          tangentBundleHost(boundaryCounter, 2, 0) = normal[i][0];
          tangentBundleHost(boundaryCounter, 2, 1) = normal[i][1];
          tangentBundleHost(boundaryCounter, 2, 2) = normal[i][2];
        }
        if (dim_ == 2) {
          tangentBundleHost(boundaryCounter, 0, 0) = 0.0;
          tangentBundleHost(boundaryCounter, 0, 1) = 0.0;
          tangentBundleHost(boundaryCounter, 1, 0) = normal[i][0];
          tangentBundleHost(boundaryCounter, 1, 1) = normal[i][1];
        }

        boundaryCounter++;
      }
    }

    Kokkos::deep_copy(interiorNeighborListsDevice, interiorNeighborListsHost);
    Kokkos::deep_copy(boundaryNeighborListsDevice, boundaryNeighborListsHost);
    Kokkos::deep_copy(interiorEpsilonDevice, interiorEpsilonHost);
    Kokkos::deep_copy(boundaryEpsilonDevice, boundaryEpsilonHost);
    Kokkos::deep_copy(interiorParticleCoordsDevice, interiorParticleCoordsHost);
    Kokkos::deep_copy(boundaryParticleCoordsDevice, boundaryParticleCoordsHost);

    Compadre::GMLS interiorVelocityBasis = Compadre::GMLS(
        Compadre::DivergenceFreeVectorTaylorPolynomial,
        Compadre::VectorPointSample, polyOrder_, dim_, "LU", "STANDARD");

    interiorVelocityBasis.setProblemData(
        interiorNeighborListsDevice, sourceCoordsDevice,
        interiorParticleCoordsDevice, interiorEpsilonDevice);

    interiorVelocityBasis.addTargets(Compadre::CurlCurlOfVectorPointEvaluation);

    interiorVelocityBasis.setWeightingType(
        Compadre::WeightingFunctionType::Power);
    interiorVelocityBasis.setWeightingParameter(4);
    interiorVelocityBasis.setOrderOfQuadraturePoints(2);
    interiorVelocityBasis.setDimensionOfQuadraturePoints(1);
    interiorVelocityBasis.setQuadratureType("LINE");

    interiorVelocityBasis.generateAlphas(1, false);

    auto interiorVelocitySolutionSet =
        interiorVelocityBasis.getSolutionSetHost();
    auto interiorVelocityAlpha = interiorVelocitySolutionSet->getAlphas();

    std::vector<unsigned int> interiorCurlCurlIndex(pow(dim_, 2));
    for (unsigned int i = 0; i < dim_; i++)
      for (unsigned int j = 0; j < dim_; j++)
        interiorCurlCurlIndex[i * dim_ + j] =
            interiorVelocitySolutionSet->getAlphaColumnOffset(
                Compadre::CurlCurlOfVectorPointEvaluation, i, 0, j, 0);

    Compadre::GMLS interiorPressureBasis =
        Compadre::GMLS(Compadre::ScalarTaylorPolynomial,
                       Compadre::StaggeredEdgeAnalyticGradientIntegralSample,
                       polyOrder_, dim_, "LU", "STANDARD");

    interiorPressureBasis.setProblemData(
        interiorNeighborListsDevice, sourceCoordsDevice,
        interiorParticleCoordsDevice, interiorEpsilonDevice);

    std::vector<Compadre::TargetOperation> interiorPressureOptions(2);
    interiorPressureOptions[0] = Compadre::LaplacianOfScalarPointEvaluation;
    interiorPressureOptions[1] = Compadre::GradientOfScalarPointEvaluation;
    interiorPressureBasis.addTargets(interiorPressureOptions);

    interiorPressureBasis.setWeightingType(
        Compadre::WeightingFunctionType::Power);
    interiorPressureBasis.setWeightingParameter(4);
    interiorPressureBasis.setOrderOfQuadraturePoints(2);
    interiorPressureBasis.setDimensionOfQuadraturePoints(1);
    interiorPressureBasis.setQuadratureType("LINE");

    interiorPressureBasis.generateAlphas(1, false);

    auto interiorPressureSolutionSet =
        interiorPressureBasis.getSolutionSetHost();
    auto interiorPressureAlpha = interiorPressureSolutionSet->getAlphas();

    const unsigned int interiorPressureLaplacianIndex =
        interiorPressureSolutionSet->getAlphaColumnOffset(
            Compadre::LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);
    std::vector<unsigned int> interiorPressureGradientIndex(dim_);
    for (unsigned int i = 0; i < dim_; i++) {
      interiorPressureGradientIndex[i] =
          interiorPressureSolutionSet->getAlphaColumnOffset(
              Compadre::GradientOfScalarPointEvaluation, i, 0, 0, 0);
    }

    Compadre::GMLS boundaryPressureBasis = Compadre::GMLS(
        Compadre::ScalarTaylorPolynomial,
        Compadre::StaggeredEdgeAnalyticGradientIntegralSample, polyOrder_, dim_,
        "LU", "STANDARD", "NEUMANN_GRAD_SCALAR");

    boundaryPressureBasis.setProblemData(
        boundaryNeighborListsDevice, sourceCoordsDevice,
        boundaryParticleCoordsDevice, boundaryEpsilonDevice);

    boundaryPressureBasis.setTangentBundle(tangentBundleDevice);

    std::vector<Compadre::TargetOperation> boundaryPressureOptions(2);
    boundaryPressureOptions[0] = Compadre::LaplacianOfScalarPointEvaluation;
    boundaryPressureOptions[1] = Compadre::GradientOfScalarPointEvaluation;
    boundaryPressureBasis.addTargets(boundaryPressureOptions);

    boundaryPressureBasis.setWeightingType(
        Compadre::WeightingFunctionType::Power);
    boundaryPressureBasis.setWeightingParameter(4);
    boundaryPressureBasis.setOrderOfQuadraturePoints(2);
    boundaryPressureBasis.setDimensionOfQuadraturePoints(1);
    boundaryPressureBasis.setQuadratureType("LINE");

    boundaryPressureBasis.generateAlphas(1, false);

    auto boundaryPressureSolutionSet =
        boundaryPressureBasis.getSolutionSetHost();
    auto boundaryPressureAlpha = boundaryPressureSolutionSet->getAlphas();

    const unsigned int boundaryPressureLaplacianIndex =
        boundaryPressureSolutionSet->getAlphaColumnOffset(
            Compadre::LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);
    std::vector<unsigned int> boundaryPressureGradientIndex(dim_);
    for (unsigned int i = 0; i < dim_; i++) {
      boundaryPressureGradientIndex[i] =
          boundaryPressureSolutionSet->getAlphaColumnOffset(
              Compadre::GradientOfScalarPointEvaluation, i, 0, 0, 0);
    }

    Compadre::GMLS boundaryVelocityBasis = Compadre::GMLS(
        Compadre::DivergenceFreeVectorTaylorPolynomial,
        Compadre::VectorPointSample, polyOrder_, dim_, "LU", "STANDARD");

    boundaryVelocityBasis.setProblemData(
        boundaryNeighborListsDevice, sourceCoordsDevice,
        boundaryParticleCoordsDevice, boundaryEpsilonDevice);

    std::vector<Compadre::TargetOperation> boundaryVelocityOptions(2);
    boundaryVelocityOptions[0] = Compadre::CurlCurlOfVectorPointEvaluation;
    boundaryVelocityOptions[1] = Compadre::GradientOfVectorPointEvaluation;
    boundaryVelocityBasis.addTargets(boundaryVelocityOptions);

    boundaryVelocityBasis.setWeightingType(
        Compadre::WeightingFunctionType::Power);
    boundaryVelocityBasis.setWeightingParameter(4);
    boundaryVelocityBasis.setOrderOfQuadraturePoints(2);
    boundaryVelocityBasis.setDimensionOfQuadraturePoints(1);
    boundaryVelocityBasis.setQuadratureType("LINE");

    boundaryVelocityBasis.generateAlphas(1, false);

    auto boundaryVelocitySolutionSet =
        boundaryVelocityBasis.getSolutionSetHost();
    auto boundaryVelocityAlpha = boundaryVelocitySolutionSet->getAlphas();

    std::vector<unsigned int> boundaryCurlCurlIndex(pow(dim_, 2));
    for (unsigned int i = 0; i < dim_; i++) {
      for (unsigned int j = 0; j < dim_; j++) {
        boundaryCurlCurlIndex[i * dim_ + j] =
            boundaryVelocitySolutionSet->getAlphaColumnOffset(
                Compadre::CurlCurlOfVectorPointEvaluation, i, 0, j, 0);
      }
    }

    std::vector<unsigned int> boundaryVelocityGradientIndex(pow(dim_, 3));
    for (unsigned int i = 0; i < dim_; i++)
      for (unsigned int j = 0; j < dim_; j++)
        for (unsigned int k = 0; k < dim_; k++)
          boundaryVelocityGradientIndex[(i * dim_ + j) * dim_ + k] =
              boundaryVelocitySolutionSet->getAlphaColumnOffset(
                  Compadre::GradientOfVectorPointEvaluation, i, j, k, 0);

    // assembly
    std::vector<PetscInt> index;
    std::vector<PetscReal> value;

    boundaryCounter = 0;
    interiorCounter = 0;

    const unsigned int blockStorageSize = fieldDof * fieldDof;

    for (unsigned int i = startParticle; i < endParticle; i++) {
      const PetscInt currentParticleIndex = local_idx[i];

      const unsigned int numNeighbor = neighborLists_(i, 0);
      const unsigned int singleRowSize = numNeighbor * fieldDof;

      index.resize(numNeighbor);
      value.resize(numNeighbor * blockStorageSize);

      for (auto &v : value)
        v = 0.0;

      for (unsigned int j = 0; j < numNeighbor; j++) {
        index[j] = source_index[neighborLists_(i, j + 1)];
      }

      if (particle_type[i] == 0) {
        // curl curl u
        for (std::size_t j = 0; j < numNeighbor; j++)
          for (unsigned int axes1 = 0; axes1 < dim_; axes1++)
            for (unsigned int axes2 = 0; axes2 < dim_; axes2++) {
              auto alphaIndex = interiorVelocitySolutionSet->getAlphaIndex(
                  interiorCounter, interiorCurlCurlIndex[axes1 * dim_ + axes2]);
              value[axes1 * singleRowSize + j * fieldDof + axes2] =
                  interiorVelocityAlpha(alphaIndex + j);
            }

        // laplacian p
        double Aij = 0.0;
        for (std::size_t j = 0; j < numNeighbor; j++) {
          auto alphaIndex = interiorPressureSolutionSet->getAlphaIndex(
              interiorCounter, interiorPressureLaplacianIndex);
          value[velocityDof * singleRowSize + j * fieldDof + velocityDof] =
              interiorPressureAlpha(alphaIndex + j);
          Aij -= interiorPressureAlpha(alphaIndex + j);
        }
        value[velocityDof * singleRowSize + velocityDof] = Aij;

        // grad p
        for (unsigned int k = 0; k < dim_; k++) {
          Aij = 0.0;
          for (std::size_t j = 0; j < numNeighbor; j++) {
            auto alphaIndex = interiorPressureSolutionSet->getAlphaIndex(
                interiorCounter, interiorPressureGradientIndex[k]);
            value[k * singleRowSize + j * fieldDof + velocityDof] =
                -interiorPressureAlpha(alphaIndex + j);
            Aij += interiorPressureAlpha(alphaIndex + j);
          }
          value[k * singleRowSize + velocityDof] = Aij;
        }

        interiorCounter++;
      } else {
        double Aij = 0.0;
        const unsigned int numNeighbor = neighborLists_(i, 0);

        // velocity BCs
        for (unsigned int k = 0; k < velocityDof; k++) {
          value[k * singleRowSize + k] = 1.0;
        }

        bi_(i) = boundaryPressureSolutionSet->getAlpha0TensorTo0Tensor(
            Compadre::LaplacianOfScalarPointEvaluation, boundaryCounter,
            numNeighbor);

        // laplacian p and norm times curl curl u
        for (std::size_t j = 0; j < numNeighbor; j++) {
          auto alphaIndex = boundaryPressureSolutionSet->getAlphaIndex(
              boundaryCounter, boundaryPressureLaplacianIndex);
          value[velocityDof * singleRowSize + j * fieldDof + velocityDof] =
              boundaryPressureAlpha(alphaIndex + j);
          Aij -= boundaryPressureAlpha(alphaIndex + j);

          for (unsigned int axes2 = 0; axes2 < dim_; axes2++) {
            double gradient = 0.0;
            for (unsigned int axes1 = 0; axes1 < dim_; axes1++) {
              auto alpha_index = boundaryVelocitySolutionSet->getAlphaIndex(
                  boundaryCounter, boundaryCurlCurlIndex[axes1 * dim_ + axes2]);
              const double Lij = boundaryVelocityAlpha(alpha_index + j);

              gradient += normal[i][axes1] * Lij;
            }
            value[velocityDof * singleRowSize + j * fieldDof + axes2] =
                bi_(i) * gradient;
          }
        }
        value[velocityDof * singleRowSize + velocityDof] = Aij;

        boundaryCounter++;
      }

      A.IncrementFieldField(currentParticleIndex, index, value);
    }

    boundaryCounter = 0;
    for (unsigned int i = startParticle; i < endParticle; i++) {
      const PetscInt currentParticleIndex = local_idx[i];
      const PetscInt currentParticleGlobalIndex = source_index[i];

      const unsigned int currentRigidBodyIndex =
          attached_rigid_body[i] * rigidBodyDof;
      if (particle_type[i] > 0) {
        if (particle_type[i] >= 4) {
          Vec3 rci = coord[i] - rigid_body_position[attached_rigid_body[i]];
          // translation
          for (unsigned int axes1 = 0; axes1 < translationDof; axes1++) {
            A.IncrementFieldRigidBody(fieldDof * currentParticleIndex + axes1,
                                      currentRigidBodyIndex + axes1, -1.0);
          }

          // rotation
          if (dim_ == 2) {
            for (int axes1 = 0; axes1 < rotationDof; axes1++) {
              A.IncrementFieldRigidBody(fieldDof * currentParticleIndex +
                                            (axes1 + 2) % translationDof,
                                        currentRigidBodyIndex + translationDof +
                                            axes1,
                                        rci[(axes1 + 1) % translationDof]);
              A.IncrementFieldRigidBody(fieldDof * currentParticleIndex +
                                            (axes1 + 1) % translationDof,
                                        currentRigidBodyIndex + translationDof +
                                            axes1,
                                        -rci[(axes1 + 2) % translationDof]);
            }
          }
          if (dim_ == 3) {
            for (int axes1 = 0; axes1 < rotationDof; axes1++) {
              A.IncrementFieldRigidBody(fieldDof * currentParticleIndex +
                                            (axes1 + 2) % translationDof,
                                        currentRigidBodyIndex + translationDof +
                                            axes1,
                                        -rci[(axes1 + 1) % translationDof]);
              A.IncrementFieldRigidBody(fieldDof * currentParticleIndex +
                                            (axes1 + 1) % translationDof,
                                        currentRigidBodyIndex + translationDof +
                                            axes1,
                                        rci[(axes1 + 2) % translationDof]);
            }
          }

          Vec3 dA;
          if (particle_type[i] == 4) {
            // corner point
            dA = Vec3(0.0, 0.0, 0.0);
          } else {
            dA = (dim_ == 3) ? (normal[i] * p_spacing[i][0] * p_spacing[i][1])
                             : (normal[i] * p_spacing[i][0]);
          }
          // apply pressure
          for (unsigned int axes1 = 0; axes1 < translationDof; axes1++)
            if (!rigid_body_velocity_force_switch[attached_rigid_body[i]]
                                                 [axes1])
              A.IncrementRigidBodyField(currentRigidBodyIndex + axes1,
                                        fieldDof * currentParticleGlobalIndex +
                                            velocityDof,
                                        -dA[axes1]);

          for (int axes1 = 0; axes1 < rotationDof; axes1++)
            if (!rigid_body_angvelocity_torque_switch[attached_rigid_body[i]]
                                                     [axes1])
              A.IncrementRigidBodyField(
                  currentRigidBodyIndex + translationDof + axes1,
                  fieldDof * currentParticleGlobalIndex + velocityDof,
                  -rci[(axes1 + 1) % translationDof] *
                          dA[(axes1 + 2) % translationDof] +
                      rci[(axes1 + 2) % translationDof] *
                          dA[(axes1 + 1) % translationDof]);

          for (unsigned int j = 0; j < neighborLists_(i, 0); j++) {
            const int neighborParticleIndex =
                source_index[neighborLists_(i, j + 1)];

            for (unsigned int axes3 = 0; axes3 < dim_; axes3++) {
              double *f = new double[dim_];
              for (int axes1 = 0; axes1 < dim_; axes1++)
                f[axes1] = 0.0;

              for (unsigned int axes1 = 0; axes1 < dim_; axes1++)
                // output component 1
                for (unsigned int axes2 = 0; axes2 < dim_; axes2++) {
                  // output component 2
                  const int velocityGradientIndex1 =
                      boundaryVelocityGradientIndex[(axes1 * dim_ + axes2) *
                                                        dim_ +
                                                    axes3];
                  const int velocityGradientIndex2 =
                      boundaryVelocityGradientIndex[(axes2 * dim_ + axes1) *
                                                        dim_ +
                                                    axes3];
                  auto alphaIndex1 = boundaryVelocitySolutionSet->getAlphaIndex(
                      boundaryCounter, velocityGradientIndex1);
                  auto alphaIndex2 = boundaryVelocitySolutionSet->getAlphaIndex(
                      boundaryCounter, velocityGradientIndex2);
                  const double sigma = (boundaryVelocityAlpha(alphaIndex1 + j) +
                                        boundaryVelocityAlpha(alphaIndex2 + j));

                  f[axes1] += sigma * dA[axes2];
                }

              // force balance
              for (int axes1 = 0; axes1 < translationDof; axes1++)
                if (!rigid_body_velocity_force_switch[attached_rigid_body[i]]
                                                     [axes1])
                  A.IncrementRigidBodyField(
                      currentRigidBodyIndex + axes1,
                      neighborParticleIndex * fieldDof + axes3, f[axes1]);

              // torque balance
              if (dim_ == 2)
                for (int axes1 = 0; axes1 < rotationDof; axes1++)
                  if (!rigid_body_angvelocity_torque_switch
                          [attached_rigid_body[i]][axes1])
                    A.IncrementRigidBodyField(
                        currentRigidBodyIndex + translationDof + axes1,
                        neighborParticleIndex * fieldDof + axes3,
                        -rci[(axes1 + 1) % translationDof] *
                                f[(axes1 + 2) % translationDof] +
                            rci[(axes1 + 2) % translationDof] *
                                f[(axes1 + 1) % translationDof]);

              if (dim_ == 3)
                for (int axes1 = 0; axes1 < rotationDof; axes1++)
                  if (!rigid_body_angvelocity_torque_switch
                          [attached_rigid_body[i]][axes1])
                    A.IncrementRigidBodyField(
                        currentRigidBodyIndex + translationDof + axes1,
                        neighborParticleIndex * fieldDof + axes3,
                        rci[(axes1 + 1) % translationDof] *
                                f[(axes1 + 2) % translationDof] -
                            rci[(axes1 + 2) % translationDof] *
                                f[(axes1 + 1) % translationDof]);

              delete[] f;
            }
          }
        }

        boundaryCounter++;
      }
    }
  }

  A.Assemble();

  MPI_Barrier(MPI_COMM_WORLD);
  timer2 = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "Matrix assembly duration: %fs\n",
              timer2 - timer1);

  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "Current memory usage after assembly %.2f GB\n",
              mem / 1e9);

  unsigned int innerParticleCounter = 0;
  for (int i = 0; i < numLocalParticle; i++) {
    if (particle_type[i] == 0)
      innerParticleCounter++;
  }
  MPI_Allreduce(MPI_IN_PLACE, &innerParticleCounter, 1, MPI_UNSIGNED, MPI_SUM,
                MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "total inner particle count: %ld\n",
              innerParticleCounter);
}

void StokesEquation::ConstructRhs() {
  auto &coord = *(geoMgr_->get_current_work_particle_coord());
  auto &normal = *(geoMgr_->get_current_work_particle_normal());
  auto &particle_type = *(geoMgr_->get_current_work_particle_type());
  auto &local_idx = *(geoMgr_->get_current_work_particle_local_index());

  auto &rigid_body_position = rbMgr_->get_position();
  auto &rigid_body_velocity = rbMgr_->get_velocity();
  auto &rigid_body_angular_velocity = rbMgr_->get_angular_velocity();
  auto &rigid_body_force = rbMgr_->get_force();
  auto &rigid_body_torque = rbMgr_->get_torque();
  auto &rigid_body_velocity_force_switch = rbMgr_->get_velocity_force_switch();
  auto &rigid_body_angvelocity_torque_switch =
      rbMgr_->get_angvelocity_torque_switch();
  const auto numRigidBody = rbMgr_->get_rigid_body_num();

  unsigned int numLocalParticle;
  unsigned int numGlobalParticleNum;

  numLocalParticle = coord.size();
  MPI_Allreduce(&numLocalParticle, &numGlobalParticleNum, 1, MPI_UNSIGNED,
                MPI_SUM, MPI_COMM_WORLD);

  const int translationDof = (dim_ == 3 ? 3 : 2);
  const int rotationDof = (dim_ == 3 ? 3 : 1);
  const int rigidBodyDof = (dim_ == 3 ? 6 : 3);

  int fieldDof = dim_ + 1;
  int velocityDof = dim_;

  rhsField_.resize(fieldDof * numLocalParticle);
  resField_.resize(fieldDof * numLocalParticle);

  for (int i = 0; i < fieldDof * numLocalParticle; i++) {
    rhsField_[i] = 0.0;
    resField_[i] = 0.0;
  }

  for (int i = 0; i < numLocalParticle; i++) {
    int current_particle_local_index = local_idx[i];
    if (particle_type[i] != 0 && particle_type[i] < 4) {
      // 2-d Taylor-Green vortex-like flow
      if (dim_ == 2) {
        double x = coord[i][0];
        double y = coord[i][1];

        rhsField_[fieldDof * current_particle_local_index] =
            sin(M_PI * x) * cos(M_PI * y);
        rhsField_[fieldDof * current_particle_local_index + 1] =
            -cos(M_PI * x) * sin(M_PI * y);

        const int neumann_index = neumann_map[i];

        rhsField_[fieldDof * current_particle_local_index + velocityDof] =
            -4.0 * pow(M_PI, 2.0) *
                (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y)) +
            bi_(i) * (normal[i][0] * 2.0 * pow(M_PI, 2.0) * sin(M_PI * x) *
                          cos(M_PI * y) -
                      normal[i][1] * 2.0 * pow(M_PI, 2.0) * cos(M_PI * x) *
                          sin(M_PI * y)) +
            bi_(i) * (normal[i][0] * 2.0 * M_PI * sin(2.0 * M_PI * x) +
                      normal[i][1] * 2.0 * M_PI * sin(2.0 * M_PI * y));
      }

      // 3-d Taylor-Green vortex-like flow
      if (dim_ == 3) {
        double x = coord[i][0];
        double y = coord[i][1];
        double z = coord[i][2];

        // rhsField_[fieldDof * current_particle_local_index] =
        //     cos(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
        // rhsField_[fieldDof * current_particle_local_index + 1] =
        //     -2 * sin(M_PI * x) * cos(M_PI * y) * sin(M_PI * z);
        // rhsField_[fieldDof * current_particle_local_index + 2] =
        //     sin(M_PI * x) * sin(M_PI * y) * cos(M_PI * z);

        rhsField_[fieldDof * current_particle_local_index] =
            sin(M_PI * x) * cos(M_PI * y) * cos(M_PI * z);
        rhsField_[fieldDof * current_particle_local_index + 1] =
            -2 * cos(M_PI * x) * sin(M_PI * y) * cos(M_PI * z);
        rhsField_[fieldDof * current_particle_local_index + 2] =
            cos(M_PI * x) * cos(M_PI * y) * sin(M_PI * z);

        // rhsField_[fieldDof * current_particle_local_index + velocityDof] =
        //     -4.0 * pow(M_PI, 2.0) *
        //         (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y) +
        //          cos(2.0 * M_PI * z)) +
        //     bi_(i) * (normal[i][0] * 3.0 * pow(M_PI, 2.0) * cos(M_PI * x) *
        //                   sin(M_PI * y) * sin(M_PI * z) -
        //               normal[i][1] * 6.0 * pow(M_PI, 2.0) * sin(M_PI * x) *
        //                   cos(M_PI * y) * sin(M_PI * z) +
        //               normal[i][2] * 3.0 * pow(M_PI, 2.0) * sin(M_PI * x) *
        //                   sin(M_PI * y) * cos(M_PI * z)) +
        //     bi_(i) * (normal[i][0] * 2.0 * M_PI * sin(2.0 * M_PI * x) +
        //               normal[i][1] * 2.0 * M_PI * sin(2.0 * M_PI * y) +
        //               normal[i][2] * 2.0 * M_PI * sin(2.0 * M_PI * z));

        rhsField_[fieldDof * current_particle_local_index + velocityDof] =
            -4.0 * pow(M_PI, 2.0) *
                (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y) +
                 cos(2.0 * M_PI * z)) +
            bi_(i) * (normal[i][0] * 3.0 * pow(M_PI, 2.0) * sin(M_PI * x) *
                          cos(M_PI * y) * cos(M_PI * z) -
                      normal[i][1] * 6.0 * pow(M_PI, 2.0) * cos(M_PI * x) *
                          sin(M_PI * y) * cos(M_PI * z) +
                      normal[i][2] * 3.0 * pow(M_PI, 2.0) * cos(M_PI * x) *
                          cos(M_PI * y) * sin(M_PI * z)) +
            bi_(i) * (normal[i][0] * 2.0 * M_PI * sin(2.0 * M_PI * x) +
                      normal[i][1] * 2.0 * M_PI * sin(2.0 * M_PI * y) +
                      normal[i][2] * 2.0 * M_PI * sin(2.0 * M_PI * z));
      }
    } else if (particle_type[i] == 0) {
      if (dim_ == 2) {
        double x = coord[i][0];
        double y = coord[i][1];

        rhsField_[fieldDof * current_particle_local_index] =
            2.0 * pow(M_PI, 2.0) * sin(M_PI * x) * cos(M_PI * y) +
            2.0 * M_PI * sin(2.0 * M_PI * x);
        rhsField_[fieldDof * current_particle_local_index + 1] =
            -2.0 * pow(M_PI, 2.0) * cos(M_PI * x) * sin(M_PI * y) +
            2.0 * M_PI * sin(2.0 * M_PI * y);

        rhsField_[fieldDof * current_particle_local_index + velocityDof] =
            -4.0 * pow(M_PI, 2.0) * (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y));
      }
      if (dim_ == 3) {
        double x = coord[i][0];
        double y = coord[i][1];
        double z = coord[i][2];

        // rhsField_[fieldDof * current_particle_local_index] =
        //     3.0 * pow(M_PI, 2) * cos(M_PI * x) * sin(M_PI * y) * sin(M_PI *
        //     z) + 2.0 * M_PI * sin(2.0 * M_PI * x);
        // rhsField_[fieldDof * current_particle_local_index + 1] =
        //     -6.0 * pow(M_PI, 2) * sin(M_PI * x) * cos(M_PI * y) *
        //         sin(M_PI * z) +
        //     2.0 * M_PI * sin(2.0 * M_PI * y);
        // rhsField_[fieldDof * current_particle_local_index + 2] =
        //     3.0 * pow(M_PI, 2) * sin(M_PI * x) * sin(M_PI * y) * cos(M_PI *
        //     z) + 2.0 * M_PI * sin(2.0 * M_PI * z);

        rhsField_[fieldDof * current_particle_local_index] =
            3.0 * pow(M_PI, 2) * sin(M_PI * x) * cos(M_PI * y) * cos(M_PI * z) +
            2.0 * M_PI * sin(2.0 * M_PI * x);
        rhsField_[fieldDof * current_particle_local_index + 1] =
            -6.0 * pow(M_PI, 2) * cos(M_PI * x) * sin(M_PI * y) *
                cos(M_PI * z) +
            2.0 * M_PI * sin(2.0 * M_PI * y);
        rhsField_[fieldDof * current_particle_local_index + 2] =
            3.0 * pow(M_PI, 2) * cos(M_PI * x) * cos(M_PI * y) * sin(M_PI * z) +
            2.0 * M_PI * sin(2.0 * M_PI * z);

        rhsField_[fieldDof * current_particle_local_index + velocityDof] =
            -4.0 * pow(M_PI, 2.0) *
            (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y) + cos(2.0 * M_PI * z));
      }
    }
  }

  // if (dim_ == 3 && numRigidBody != 0) {
  //   auto &rigid_body_size = rbMgr_->get_rigid_body_size();
  //   auto &rigid_body_velocity = rbMgr_->get_velocity();
  //   std::vector<Vec3> &rigid_body_position = rbMgr_->get_position();

  //   double RR = rigid_body_size[0][0];
  //   double u = 1.0;

  //   for (int i = 0; i < numLocalParticle; i++) {
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

  //       rhsField_[fieldDof * current_particle_local_index] =
  //           sin(theta) * cos(phi) * vr + cos(theta) * cos(phi) * vt;
  //       rhsField_[fieldDof * current_particle_local_index + 1] =
  //           sin(theta) * sin(phi) * vr + cos(theta) * sin(phi) * vt;
  //       rhsField_[fieldDof * current_particle_local_index + 2] =
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
  // }

  numRigidBody_ = numRigidBody;
  numLocalRigidBody_ =
      numRigidBody_ / (unsigned int)mpiSize_ +
      ((numRigidBody_ % (unsigned int)mpiSize_ > mpiRank_) ? 1 : 0);

  rigidBodyStartIndex_ = 0;
  for (int i = 0; i < mpiRank_; i++) {
    rigidBodyStartIndex_ +=
        numRigidBody_ / (unsigned int)mpiSize_ +
        ((numRigidBody_ % (unsigned int)mpiSize_ > i) ? 1 : 0);
  }
  rigidBodyEndIndex_ = rigidBodyStartIndex_ + numLocalRigidBody_;

  rhsRigidBody_.resize(numLocalRigidBody_ * rigidBodyDof);
  resRigidBody_.resize(numLocalRigidBody_ * rigidBodyDof);

  for (unsigned int i = 0; i < numLocalRigidBody_ * rigidBodyDof; i++) {
    rhsRigidBody_[i] = 0.0;
    resRigidBody_[i] = 0.0;
  }

  for (int i = rigidBodyStartIndex_; i < rigidBodyEndIndex_; i++) {
    for (int axes = 0; axes < translationDof; axes++) {
      if (rigid_body_velocity_force_switch[i][axes])
        rhsRigidBody_[(i - rigidBodyStartIndex_) * rigidBodyDof + axes] =
            rigid_body_velocity[i][axes];
      else
        rhsRigidBody_[(i - rigidBodyStartIndex_) * rigidBodyDof + axes] =
            -rigid_body_force[i][axes];
    }
    for (int axes = 0; axes < rotationDof; axes++) {
      if (rigid_body_angvelocity_torque_switch[i][axes])
        rhsRigidBody_[(i - rigidBodyStartIndex_) * rigidBodyDof +
                      translationDof + axes] =
            rigid_body_angular_velocity[i][axes];
      else
        rhsRigidBody_[(i - rigidBodyStartIndex_) * rigidBodyDof +
                      translationDof + axes] = -rigid_body_torque[i][axes];
    }
  }

  // make sure pressure term is orthogonal to the constant
  double rhs_pressure_sum = 0.0;
  for (int i = 0; i < numLocalParticle; i++) {
    int current_particle_local_index = local_idx[i];
    rhs_pressure_sum +=
        rhsField_[fieldDof * current_particle_local_index + velocityDof];
  }
  MPI_Allreduce(MPI_IN_PLACE, &rhs_pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  rhs_pressure_sum /= numGlobalParticleNum;
  for (int i = 0; i < numLocalParticle; i++) {
    int current_particle_local_index = local_idx[i];
    rhsField_[fieldDof * current_particle_local_index + velocityDof] -=
        rhs_pressure_sum;
  }
}

void StokesEquation::SolveEquation() {
  const int numRigidBody = rbMgr_->get_rigid_body_num();

  const unsigned int translationDof = (dim_ == 3 ? 3 : 2);
  const unsigned int rotationDof = (dim_ == 3 ? 3 : 1);
  const unsigned int rigidBodyDof = (dim_ == 3 ? 6 : 3);

  std::vector<Vec3> &rigid_body_velocity = rbMgr_->get_velocity();
  std::vector<Vec3> &rigid_body_angular_velocity =
      rbMgr_->get_angular_velocity();

  auto &local_idx = *(geoMgr_->get_current_work_particle_local_index());

  // build interpolation and restriction operators
  double timer1, timer2;
  if (currentRefinementLevel_ != 0) {
    timer1 = MPI_Wtime();

    multiMgr_->build_interpolation_restriction(numRigidBody, dim_, polyOrder_);
    multiMgr_->initial_guess_from_previous_adaptive_step(resField_, velocity,
                                                         pressure);

    for (int i = rigidBodyStartIndex_; i < rigidBodyEndIndex_; i++) {
      for (int j = 0; j < translationDof; j++) {
        resRigidBody_[(i - rigidBodyStartIndex_) * rigidBodyDof + j] =
            rigid_body_velocity[i][j];
      }
      for (int j = 0; j < rotationDof; j++) {
        resRigidBody_[(i - rigidBodyStartIndex_) * rigidBodyDof +
                      translationDof + j] = rigid_body_angular_velocity[i][j];
      }
    }

    timer2 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,
                "Interpolation matrix building duration: %fs\n",
                timer2 - timer1);
  }

  PetscLogDefaultBegin();

  MPI_Barrier(MPI_COMM_WORLD);
  timer1 = MPI_Wtime();
  if (currentRefinementLevel_ == 0)
    multiMgr_->Solve(rhsField_, resField_, rhsRigidBody_, resRigidBody_);
  MPI_Barrier(MPI_COMM_WORLD);
  timer2 = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "linear system solving duration: %fs\n",
              timer2 - timer1);

  if (useViewer_) {
    PetscViewer viewer;
    PetscViewerASCIIGetStdout(PETSC_COMM_WORLD, &viewer);
    PetscLogView(viewer);
  }

  // copy data
  std::vector<Vec3> &coord = *(geoMgr_->get_current_work_particle_coord());

  unsigned int numLocalParticle;
  unsigned int numGlobalParticleNum;

  numLocalParticle = coord.size();
  MPI_Allreduce(&numLocalParticle, &numGlobalParticleNum, 1, MPI_UNSIGNED,
                MPI_SUM, MPI_COMM_WORLD);

  const unsigned int fieldDof = dim_ + 1;
  const unsigned int velocityDof = dim_;

  std::vector<int> particle_num_per_process;
  particle_num_per_process.resize(mpiSize_);

  MPI_Allgather(&numLocalParticle, 1, MPI_INT, particle_num_per_process.data(),
                1, MPI_INT, MPI_COMM_WORLD);

  int local_rigid_body_offset =
      particle_num_per_process[mpiSize_ - 1] * fieldDof;
  int global_rigid_body_offset = numGlobalParticleNum * fieldDof;
  int local_out_process_offset =
      particle_num_per_process[mpiSize_ - 1] * fieldDof;

  pressure.resize(numLocalParticle);
  velocity.resize(numLocalParticle);

  double pressure_sum = 0.0;
  for (int i = 0; i < numLocalParticle; i++) {
    int current_particle_local_index = local_idx[i];
    pressure[i] =
        resField_[fieldDof * current_particle_local_index + velocityDof];
    pressure_sum += pressure[i];
    for (int axes1 = 0; axes1 < dim_; axes1++)
      velocity[i][axes1] =
          resField_[fieldDof * current_particle_local_index + axes1];
  }

  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  double average_pressure = pressure_sum / numGlobalParticleNum;
  for (int i = 0; i < numLocalParticle; i++) {
    pressure[i] -= average_pressure;
  }

  // communicate velocity and angular velocity
  std::vector<double> translation_velocity(numRigidBody * translationDof);
  std::vector<double> angular_velocity(numRigidBody * rotationDof);

  for (unsigned int i = 0; i < numRigidBody * translationDof; i++) {
    translation_velocity[i] = 0.0;
  }
  for (unsigned int i = 0; i < numRigidBody * rotationDof; i++) {
    angular_velocity[i] = 0.0;
  }

  for (int i = rigidBodyStartIndex_; i < rigidBodyEndIndex_; i++) {
    for (int j = 0; j < translationDof; j++) {
      translation_velocity[i * translationDof + j] =
          resRigidBody_[(i - rigidBodyStartIndex_) * rigidBodyDof + j];
    }
    for (int j = 0; j < rotationDof; j++) {
      angular_velocity[i * rotationDof + j] =
          resRigidBody_[(i - rigidBodyStartIndex_) * rigidBodyDof +
                        translationDof + j];
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, translation_velocity.data(),
                numRigidBody * translationDof, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, angular_velocity.data(),
                numRigidBody * rotationDof, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  for (int i = 0; i < numRigidBody; i++) {
    for (int j = 0; j < translationDof; j++) {
      rigid_body_velocity[i][j] = translation_velocity[i * translationDof + j];
    }
    for (int j = 0; j < rotationDof; j++) {
      rigid_body_angular_velocity[i][j] = angular_velocity[i * rotationDof + j];
    }
  }
}

void StokesEquation::CheckSolution() {
  std::vector<Vec3> &coord = *(geoMgr_->get_current_work_particle_coord());
  std::vector<Vec3> &normal = *(geoMgr_->get_current_work_particle_normal());
  std::vector<int> &particle_type =
      *(geoMgr_->get_current_work_particle_type());
  auto &volume = *(geoMgr_->get_current_work_particle_volume());

  std::vector<Vec3> &rigid_body_position = rbMgr_->get_position();
  const int numRigidBody = rbMgr_->get_rigid_body_num();

  unsigned int numLocalParticle;
  unsigned int numGlobalParticleNum;

  numLocalParticle = coord.size();
  MPI_Allreduce(&numLocalParticle, &numGlobalParticleNum, 1, MPI_UNSIGNED,
                MPI_SUM, MPI_COMM_WORLD);

  const unsigned int translationDof = (dim_ == 3 ? 3 : 2);
  const unsigned int rotationDof = (dim_ == 3 ? 3 : 1);
  const unsigned int rigidBodyDof = (dim_ == 3 ? 6 : 3);

  const unsigned int fieldDof = dim_ + 1;
  const unsigned int velocityDof = dim_;

  unsigned int local_velocity_dof = numLocalParticle * dim_;
  const unsigned int global_velocity_dof =
      numGlobalParticleNum * dim_ + rigidBodyDof * numRigidBody;
  const unsigned int local_pressure_dof = numLocalParticle;
  const unsigned int global_pressure_dof = numGlobalParticleNum;

  if (mpiRank_ == mpiSize_ - 1) {
    local_velocity_dof += rigidBodyDof * numRigidBody;
  }

  std::vector<std::vector<double>> &rigid_body_size =
      rbMgr_->get_rigid_body_size();
  auto &rigid_body_velocity = rbMgr_->get_velocity();
  auto &rigid_body_force = rbMgr_->get_force();

  double u, RR;

  if (numRigidBody != 0) {
    RR = rigid_body_size[0][0];
    u = 1.0;
  }

  int local_dof = local_velocity_dof + local_pressure_dof;

  // check data
  double true_pressure_mean = 0.0;
  double pressure_mean = 0.0;
  for (int i = 0; i < numLocalParticle; i++) {
    if (dim_ == 2) {
      double x = coord[i][0];
      double y = coord[i][1];

      double true_pressure = -cos(2.0 * M_PI * x) - cos(2.0 * M_PI * y);

      true_pressure_mean += true_pressure;
      pressure_mean += pressure[i];
    }

    if (dim_ == 3) {
      if (numRigidBody != 0) {
        double x = coord[i][0] - rigid_body_position[0][0];
        double y = coord[i][1] - rigid_body_position[0][1];
        double z = coord[i][2] - rigid_body_position[0][2];

        double r = sqrt(x * x + y * y + z * z);
        double theta = acos(z / r);

        double true_pressure = -1.5 * RR / pow(r, 2.0) * u * cos(theta);

        true_pressure_mean += true_pressure;
        pressure_mean += pressure[i];
      }
      if (numRigidBody == 0) {
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

  true_pressure_mean /= numGlobalParticleNum;
  pressure_mean /= numGlobalParticleNum;

  double error_velocity = 0.0;
  double norm_velocity = 0.0;
  double error_pressure = 0.0;
  double norm_pressure = 0.0;
  for (int i = 0; i < numLocalParticle; i++) {
    if (dim_ == 2) {
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

    if (dim_ == 3) {
      if (numRigidBody != 0) {
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

        Vec3 true_velocity;

        true_velocity[0] =
            sin(theta) * cos(phi) * vr + cos(theta) * cos(phi) * vt;
        true_velocity[1] =
            sin(theta) * sin(phi) * vr + cos(theta) * sin(phi) * vt;
        true_velocity[2] = cos(theta) * vr - sin(theta) * vt;

        double true_pressure =
            -1.5 * RR / pow(r, 2.0) * u * cos(theta) - true_pressure_mean;

        error_velocity +=
            pow(true_velocity[0] - velocity[i][0], 2.0) * volume[i] +
            pow(true_velocity[1] - velocity[i][1], 2.0) * volume[i] +
            pow(true_velocity[2] - velocity[i][2], 2.0) * volume[i];
        error_pressure += pow(true_pressure - pressure[i], 2.0) * volume[i];

        norm_velocity += pow(true_velocity[0], 2.0) * volume[i] +
                         pow(true_velocity[1], 2.0) * volume[i] +
                         pow(true_velocity[2], 2.0) * volume[i];
        norm_pressure += pow(true_pressure, 2.0) * volume[i];
      }
      if (numRigidBody == 0) {
        double x = coord[i][0];
        double y = coord[i][1];
        double z = coord[i][2];

        Vec3 true_velocity;

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

  // PetscPrintf(MPI_COMM_WORLD, "RMS pressure error: %.10f\n",
  //             sqrt(error_pressure / numGlobalParticleNum));
  // PetscPrintf(MPI_COMM_WORLD, "RMS velocity error: %.10f\n",
  //             sqrt(error_velocity / numGlobalParticleNum));

  MPI_Barrier(MPI_COMM_WORLD);

  // gradient
  double error_velocity_gradient = 0.0;
  double norm_velocity_gradient = 0.0;
  if (dim_ == 3 && numRigidBody != 0) {
    for (int i = 0; i < numLocalParticle; i++) {
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
                       pow(RR, 3.0) / 2.0 / pow(r, 4.0)) +
                  u * sin(theta) *
                      (1.0 / r - 3.0 * RR / 4.0 / pow(r, 2.0) -
                       pow(RR, 3.0) / 4.0 / pow(r, 4.0));
      dvr[0][2] = 0.0;
      dvr[1][0] = -u * sin(theta) *
                  (3.0 * RR / 4.0 / pow(r, 2.0) +
                   3.0 * pow(RR, 3.0) / 4.0 / pow(r, 4.0));
      dvr[1][1] = -u * cos(theta) *
                      (1.0 / r - 3.0 * RR / 4.0 / pow(r, 2.0) -
                       pow(RR, 3.0) / 4.0 / pow(r, 4.0)) +
                  u * cos(theta) *
                      (1.0 / r - 3.0 * RR / 2.0 / pow(r, 2.0) +
                       pow(RR, 3.0) / 2.0 / pow(r, 4.0));
      dvr[1][2] = 0.0;
      dvr[2][0] = 0.0;
      dvr[2][1] = 0.0;
      dvr[2][2] = -u * cos(theta) *
                      (1.0 / r - 3.0 * RR / 4.0 / pow(r, 2.0) -
                       pow(RR, 3.0) / 4.0 / pow(r, 4.0)) +
                  u * cos(theta) *
                      (1.0 / r - 3.0 * RR / 2.0 / pow(r, 2.0) +
                       pow(RR, 3.0) / 2.0 / pow(r, 4.0));

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
              pow(gradient[i][m * dim_ + n] - du[m][n], 2.0) * volume[i];
          norm_velocity_gradient += pow(du[m][n], 2.0) * volume[i];
        }
      }
    }
  }

  if (dim_ == 2) {
    for (int i = 0; i < numLocalParticle; i++) {
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
              pow(gradient[i][m * dim_ + n] - du[m][n], 2.0);
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
}

void StokesEquation::CalculateError() {
  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "\nstart of error estimation\n");

  // prepare stage
  auto &source_coord = *(geoMgr_->get_current_work_ghost_particle_coord());
  auto &source_volume = *(geoMgr_->get_current_work_ghost_particle_volume());
  auto &source_index = *(geoMgr_->get_current_work_ghost_particle_index());
  auto &coord = *(geoMgr_->get_current_work_particle_coord());
  auto &volume = *(geoMgr_->get_current_work_particle_volume());

  const int numLocalParticle = coord.size();

  double local_error, local_volume, local_direct_gradient_norm;
  double global_direct_gradient_norm;

  Kokkos::View<double *, Kokkos::HostSpace> ghostEpsilon;
  geoMgr_->ApplyGhost(epsilon_, ghostEpsilon);

  local_error = 0.0;
  local_direct_gradient_norm = 0.0;

  error.resize(numLocalParticle);
  for (int i = 0; i < numLocalParticle; i++) {
    error[i] = 0.0;
  }

  double u, RR;

  RR = 0.1;
  u = 1.0;

  // error estimation base on velocity
  if (errorEstimationMethod_ == VELOCITY_ERROR_EST) {
    std::vector<Vec3> ghost_velocity;
    geoMgr_->ghost_forward(velocity, ghost_velocity);

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

    std::vector<std::vector<double>> direct_gradient;
    direct_gradient.resize(numLocalParticle);
    const int gradient_component_num = pow(dim_, 2);
    for (size_t i = 0; i < numLocalParticle; i++) {
      direct_gradient[i].resize(gradient_component_num);
    }

    std::vector<std::vector<double>> coefficients_chunk,
        ghost_coefficients_chunk;
    coefficients_chunk.resize(numLocalParticle);

    size_t coefficients_size = 0;

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
        "source coordinates", source_coord.size(), 3);
    Kokkos::View<double **>::HostMirror sourceCoordsHost =
        Kokkos::create_mirror_view(sourceCoordsDevice);

    for (size_t i = 0; i < source_coord.size(); i++) {
      for (int j = 0; j < 3; j++) {
        sourceCoordsHost(i, j) = source_coord[i][j];
      }
    }

    Kokkos::deep_copy(sourceCoordsDevice, sourceCoordsHost);

    int start_particle = 0;
    int end_particle;
    for (int num = 0; num < number_of_batches; num++) {
      Compadre::GMLS temp_velocity_basis = Compadre::GMLS(
          Compadre::DivergenceFreeVectorTaylorPolynomial,
          Compadre::VectorPointSample, polyOrder_, dim_, "LU", "STANDARD");

      int batch_size = numLocalParticle / number_of_batches +
                       (num < (numLocalParticle % number_of_batches));
      int end_particle =
          std::min(numLocalParticle, start_particle + batch_size);
      int particle_num = end_particle - start_particle;

      Kokkos::View<double *, Kokkos::DefaultExecutionSpace> epsilon_device(
          "h supports", particle_num);
      Kokkos::View<double *>::HostMirror epsilon_host =
          Kokkos::create_mirror_view(epsilon_device);

      for (int i = 0; i < particle_num; i++) {
        epsilon_host(i) = epsilon_(start_particle + i);
      }

      Kokkos::deep_copy(epsilon_device, epsilon_host);

      Kokkos::View<int **, Kokkos::DefaultExecutionSpace> neighbor_lists_device(
          "neighbor lists", particle_num, max_neighbor + 1);
      Kokkos::View<int **>::HostMirror neighbor_lists_host =
          Kokkos::create_mirror_view(neighbor_lists_device);

      for (int i = 0; i < particle_num; i++) {
        neighbor_lists_host(i, 0) = neighborLists_(i + start_particle, 0);
        for (int j = 0; j < neighborLists_(i, 0); j++) {
          neighbor_lists_host(i, j + 1) =
              neighborLists_(i + start_particle, j + 1);
        }
      }

      Kokkos::deep_copy(neighbor_lists_device, neighbor_lists_host);

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

      temp_velocity_basis.setProblemData(neighbor_lists_device,
                                         sourceCoordsDevice,
                                         target_coord_device, epsilon_device);

      std::vector<Compadre::TargetOperation> velocity_operation(2);
      velocity_operation[0] = Compadre::ScalarPointEvaluation;
      velocity_operation[1] = Compadre::GradientOfVectorPointEvaluation;

      temp_velocity_basis.addTargets(velocity_operation);

      temp_velocity_basis.setWeightingType(
          Compadre::WeightingFunctionType::Power);
      temp_velocity_basis.setWeightingParameter(4);

      temp_velocity_basis.generateAlphas(1, true);

      Compadre::Evaluator temp_velocity_evaluator(&temp_velocity_basis);

      auto coefficients =
          temp_velocity_evaluator
              .applyFullPolynomialCoefficientsBasisToDataAllComponents<
                  double **, Kokkos::HostSpace>(ghost_velocity_device);

      auto temp_gradient =
          temp_velocity_evaluator.applyAlphasToDataAllComponentsAllTargetSites<
              double **, Kokkos::HostSpace>(
              ghost_velocity_device, Compadre::GradientOfVectorPointEvaluation);

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

    geoMgr_->ghost_forward(coefficients_chunk, ghost_coefficients_chunk,
                           coefficients_size);
    coefficients_chunk.clear();
    coefficients_chunk.shrink_to_fit();

    // estimate stage
    auto &recovered_gradient = gradient;
    std::vector<std::vector<double>> ghost_recovered_gradient;
    recovered_gradient.resize(numLocalParticle);
    for (int i = 0; i < numLocalParticle; i++) {
      recovered_gradient[i].resize(gradient_component_num);
      for (int axes1 = 0; axes1 < dim_; axes1++) {
        for (int axes2 = 0; axes2 < dim_; axes2++) {
          recovered_gradient[i][axes1 * dim_ + axes2] = 0.0;
        }
      }
    }

    for (int i = 0; i < numLocalParticle; i++) {
      int counter = 0;
      for (int j = 0; j < neighborLists_(i, 0); j++) {
        const int neighbor_index = neighborLists_(i, j + 1);

        Vec3 dX = coord[i] - source_coord[neighbor_index];
        for (int axes1 = 0; axes1 < dim_; axes1++) {
          for (int axes2 = 0; axes2 < dim_; axes2++) {
            recovered_gradient[i][axes1 * dim_ + axes2] +=
                cal_div_free_grad(axes1, axes2, dim_, dX, polyOrder_,
                                  ghostEpsilon(neighbor_index),
                                  ghost_coefficients_chunk[neighbor_index]);
          }
        }
        counter++;
      }

      for (int axes1 = 0; axes1 < dim_; axes1++) {
        for (int axes2 = 0; axes2 < dim_; axes2++) {
          recovered_gradient[i][axes1 * dim_ + axes2] /= counter;
        }
      }
    }

    geoMgr_->ghost_forward(recovered_gradient, ghost_recovered_gradient,
                           gradient_component_num);

    for (int i = 0; i < numLocalParticle; i++) {
      std::vector<double> reconstructed_gradient(gradient_component_num);
      double total_neighbor_vol = 0.0;
      // loop over all neighbors
      for (int j = 0; j < neighborLists_(i, 0); j++) {
        const int neighbor_index = neighborLists_(i, j + 1);

        Vec3 dX = source_coord[neighbor_index] - coord[i];
        total_neighbor_vol += source_volume[neighbor_index];
        for (int axes1 = 0; axes1 < dim_; axes1++) {
          for (int axes2 = 0; axes2 < dim_; axes2++) {
            reconstructed_gradient[axes1 * dim_ + axes2] =
                cal_div_free_grad(axes1, axes2, dim_, dX, polyOrder_,
                                  ghostEpsilon(i), ghost_coefficients_chunk[i]);
          }
        }

        for (int axes1 = 0; axes1 < dim_; axes1++) {
          for (int axes2 = 0; axes2 < dim_; axes2++) {
            error[i] += (pow(reconstructed_gradient[axes1 * dim_ + axes2] -
                                 ghost_recovered_gradient[neighbor_index]
                                                         [axes1 * dim_ + axes2],
                             2.0) *
                         source_volume[neighbor_index]);
          }
        }
      }

      error[i] = error[i] / total_neighbor_vol;
      local_error += error[i] * volume[i];

      for (int axes1 = 0; axes1 < dim_; axes1++) {
        for (int axes2 = 0; axes2 < dim_; axes2++) {
          local_direct_gradient_norm +=
              pow(direct_gradient[i][axes1 * dim_ + axes2], 2) * volume[i];
        }
      }
    }
  }

  // smooth stage
  for (int ite = 0; ite < 10; ite++) {
    std::vector<double> ghost_error;
    geoMgr_->ghost_forward(error, ghost_error);

    for (int i = 0; i < numLocalParticle; i++) {
      error[i] = 0.0;
      double total_neighbor_vol = 0.0;
      for (int j = 0; j < neighborLists_(i, 0); j++) {
        const int neighbor_index = neighborLists_(i, j + 1);

        Vec3 dX = source_coord[neighbor_index] - coord[i];

        double WabIJ = Wab(dX.mag(), epsilon_(i));

        error[i] +=
            ghost_error[neighbor_index] * source_volume[neighbor_index] * WabIJ;
        total_neighbor_vol += source_volume[neighbor_index] * WabIJ;
      }
      error[i] /= total_neighbor_vol;
    }
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

  for (int i = 0; i < numLocalParticle; i++) {
    error[i] = sqrt(error[i] * volume[i]);
  }

  PetscLogDouble mem;
  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "Current memory usage after error estimation %.2f GB\n",
              mem / 1e9);
}

void StokesEquation::CollectForce() {
  auto &coord = *(geoMgr_->get_current_work_particle_coord());
  auto &normal = *(geoMgr_->get_current_work_particle_normal());
  auto &p_spacing = *(geoMgr_->get_current_work_particle_p_spacing());
  auto &particle_type = *(geoMgr_->get_current_work_particle_type());
  auto &attached_rigid_body =
      *(geoMgr_->get_current_work_particle_attached_rigid_body());
  auto &rigid_body_velocity_force_switch = rbMgr_->get_velocity_force_switch();
  auto &rigid_body_angvelocity_torque_switch =
      rbMgr_->get_angvelocity_torque_switch();

  int numLocalParticle = coord.size();

  auto &rigid_body_position = rbMgr_->get_position();
  auto &rigid_body_force = rbMgr_->get_force();
  auto &rigid_body_torque = rbMgr_->get_torque();

  auto numRigidBody = rbMgr_->get_rigid_body_num();

  const int translationDof = (dim_ == 3 ? 3 : 2);
  const int rotationDof = (dim_ == 3 ? 3 : 1);
  const int rigidBodyDof = (dim_ == 3 ? 6 : 3);

  std::vector<double> flattened_force;
  std::vector<double> flattened_torque;

  flattened_force.resize(numRigidBody * translationDof);
  flattened_torque.resize(numRigidBody * rotationDof);

  for (int i = 0; i < numRigidBody; i++) {
    for (int j = 0; j < translationDof; j++) {
      flattened_force[i * translationDof + j] = 0.0;
    }
    for (int j = 0; j < rotationDof; j++) {
      flattened_torque[i * rotationDof + j] = 0.0;
    }
  }

  for (int i = 0; i < numLocalParticle; i++) {
    if (particle_type[i] >= 4) {
      int rigid_body_idx = attached_rigid_body[i];
      Vec3 rci = coord[i] - rigid_body_position[rigid_body_idx];
      Vec3 dA = (dim_ == 3) ? (normal[i] * p_spacing[i][0] * p_spacing[i][1])
                            : (normal[i] * p_spacing[i][0]);

      Vec3 f;
      for (int axes1 = 0; axes1 < dim_; axes1++) {
        f[axes1] = -dA[axes1] * pressure[i];
        for (int axes2 = 0; axes2 < dim_; axes2++) {
          f[axes1] += 0.5 *
                      (gradient[i][axes1 * dim_ + axes2] +
                       gradient[i][axes2 * dim_ + axes1]) *
                      dA[axes2];
        }
      }

      for (int axes1 = 0; axes1 < translationDof; axes1++) {
        flattened_force[rigid_body_idx * translationDof + axes1] += f[axes1];
      }

      for (int axes1 = 0; axes1 < rotationDof; axes1++) {
        flattened_torque[rigid_body_idx * rotationDof + axes1] +=
            rci[(axes1 + 1) % translationDof] *
                f[(axes1 + 2) % translationDof] -
            rci[(axes1 + 2) % translationDof] * f[(axes1 + 1) % translationDof];
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, flattened_force.data(),
                numRigidBody * translationDof, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, flattened_torque.data(),
                numRigidBody * rotationDof, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  for (int i = 0; i < numRigidBody; i++) {
    for (int j = 0; j < translationDof; j++) {
      if (rigid_body_velocity_force_switch[i][j])
        rigid_body_force[i][j] = flattened_force[i * translationDof + j];
    }
    for (int j = 0; j < rotationDof; j++) {
      if (rigid_body_velocity_force_switch[i][j])
        rigid_body_torque[i][j] = flattened_torque[i * rotationDof + j];
    }
  }

  if (mpiRank_ == 0) {
    std::ofstream outputForce;
    outputForce.open("force.txt", std::ios::trunc);
    for (int num = 0; num < numRigidBody; num++) {
      for (int j = 0; j < 3; j++) {
        outputForce << rigid_body_force[num][j] << '\t';
      }
      for (int j = 0; j < 3; j++) {
        outputForce << rigid_body_torque[num][j] << '\t';
      }
    }
    outputForce << std::endl;
    outputForce.close();
  }
}