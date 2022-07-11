#include "StokesEquation.hpp"
#include "DivergenceFree.hpp"
#include "gmls_solver.hpp"
#include "petsc_sparse_matrix.hpp"

#include <iomanip>
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
  CollectForce();

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

  Compadre::GMLS pressureBasis =
      Compadre::GMLS(Compadre::ScalarTaylorPolynomial,
                     Compadre::StaggeredEdgeAnalyticGradientIntegralSample,
                     polyOrder_, dim_, "LU", "STANDARD");
  Compadre::GMLS velocityBasis = Compadre::GMLS(
      Compadre::DivergenceFreeVectorTaylorPolynomial,
      Compadre::VectorPointSample, polyOrder_, dim_, "LU", "STANDARD");
  Compadre::GMLS pressureNeumannBasis =
      Compadre::GMLS(Compadre::ScalarTaylorPolynomial,
                     Compadre::StaggeredEdgeAnalyticGradientIntegralSample,
                     polyOrder_, dim_, "LU", "STANDARD", "NEUMANN_GRAD_SCALAR");
  Compadre::GMLS velocityColloidBasis = Compadre::GMLS(
      Compadre::DivergenceFreeVectorTaylorPolynomial,
      Compadre::VectorPointSample, polyOrder_, dim_, "LU", "STANDARD");

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

  int numLocalParticle;
  int numGlobalParticleNum;

  numLocalParticle = coord.size();
  MPI_Allreduce(&numLocalParticle, &numGlobalParticleNum, 1, MPI_UNSIGNED,
                MPI_SUM, MPI_COMM_WORLD);

  int num_source_coord = source_coord.size();

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

  Kokkos::deep_copy(source_coord_device, source_coord_host);
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
    number_of_batches = std::max(numLocalParticle / 10000, 1);
  else
    number_of_batches = std::max(numLocalParticle / 1000, 1);

  // neighbor search
  auto point_cloud_search(
      Compadre::CreatePointCloudSearch(source_coord_host, dim_));

  int min_num_neighbor = std::max(
      Compadre::GMLS::getNP(polyOrder_, dim_,
                            Compadre::DivergenceFreeVectorTaylorPolynomial),
      Compadre::GMLS::getNP(polyOrder_ + 1, dim_));
  int satisfied_num_neighbor = pow(2.0, dim_ / 2.0) * min_num_neighbor;

  Kokkos::resize(neighbor_lists_, numLocalParticle, satisfied_num_neighbor + 1);
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
      true, target_coord_host, neighbor_lists_, epsilon_,
      satisfied_num_neighbor, 1.0);

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
              true, target_coord_host, neighbor_lists_, epsilon_, 0.0, 0.0);
  if (minNeighborLists > neighbor_lists_.extent(1)) {
    Kokkos::resize(neighbor_lists_, numLocalParticle, minNeighborLists);
  }
  point_cloud_search.generate2DNeighborListsFromRadiusSearch(
      false, target_coord_host, neighbor_lists_, epsilon_, 0.0, 0.0);

  max_ratio = 0.0;
  min_neighbor = 1000;
  max_neighbor = 0;
  mean_neighbor = 0;
  for (int i = 0; i < numLocalParticle; i++) {
    if (neighbor_lists_(i, 0) < min_neighbor)
      min_neighbor = neighbor_lists_(i, 0);
    if (neighbor_lists_(i, 0) > max_neighbor)
      max_neighbor = neighbor_lists_(i, 0);
    mean_neighbor += neighbor_lists_(i, 0);

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
        for (int j = 0; j <= neighbor_lists_(index, 0); j++) {
          temp_neighbor_list_host(i, j) = neighbor_lists_(index, j);
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
          temp_neighbor_list_device, source_coord_device,
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
        for (int j = 0; j <= neighbor_lists_(index, 0); j++) {
          temp_neighbor_list_host(i, j) = neighbor_lists_(index, j);
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
          temp_neighbor_list_device, source_coord_device,
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
                  true, target_coord_host, neighbor_lists_, epsilon_, 0.0, 0.0);
      if (minNeighborLists > neighbor_lists_.extent(1)) {
        Kokkos::resize(neighbor_lists_, numLocalParticle, minNeighborLists);
      }
      point_cloud_search.generate2DNeighborListsFromRadiusSearch(
          false, target_coord_host, neighbor_lists_, epsilon_, 0.0, 0.0);

      max_ratio = 0.0;
      min_neighbor = 1000;
      max_neighbor = 0;
      mean_neighbor = 0;
      for (int i = 0; i < numLocalParticle; i++) {
        if (neighbor_lists_(i, 0) < min_neighbor)
          min_neighbor = neighbor_lists_(i, 0);
        if (neighbor_lists_(i, 0) > max_neighbor)
          max_neighbor = neighbor_lists_(i, 0);
        mean_neighbor += neighbor_lists_(i, 0);

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

  for (int i = 0; i < numLocalParticle; i++) {
    if (neighbor_lists_(i, 0) > 1000)
      std::cout << target_coord_host(i, 0) << ' ' << target_coord_host(i, 1)
                << ' ' << target_coord_host(i, 2) << ' ' << particle_type[i]
                << ' ' << epsilon_(i) / spacing[i] << ' ' << epsilon_(i)
                << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(MPI_COMM_WORLD,
              "iteration count: %d min neighbor: %d, max neighbor: %d , mean "
              "neighbor %f, max ratio: %f\n",
              ite_counter, min_neighbor, max_neighbor,
              mean_neighbor / (double)numGlobalParticleNum, max_ratio);

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
      neumann_neighbor_list_device("neumann boundary neighbor lists",
                                   num_neumann_target_coord,
                                   neighbor_lists_.extent(1));
  Kokkos::View<int **>::HostMirror neumann_neighbor_list_host =
      Kokkos::create_mirror_view(neumann_neighbor_list_device);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> neumann_epsilon_device(
      "neumann boundary h supports", num_neumann_target_coord);
  Kokkos::View<double *>::HostMirror neumann_epsilon_host =
      Kokkos::create_mirror_view(neumann_epsilon_device);

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
      colloid_neighbor_list_device("colloid boundary neighbor lists",
                                   num_colloid_target_coord,
                                   neighbor_lists_.extent(1));
  Kokkos::View<int **>::HostMirror colloid_neighbor_list_host =
      Kokkos::create_mirror_view(colloid_neighbor_list_device);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> colloid_epsilon_device(
      "colloid boundary h supports", num_colloid_target_coord);
  Kokkos::View<double *>::HostMirror colloid_epsilon_host =
      Kokkos::create_mirror_view(colloid_epsilon_device);

  for (int i = 0; i < numLocalParticle; i++) {
    if (particle_type[i] != 0) {
      counter = neumann_map[i];
      neumann_epsilon_host(counter) = epsilon_(i);
      neumann_neighbor_list_host(counter, 0) = neighbor_lists_(i, 0);
      for (int j = 0; j < neighbor_lists_(i, 0); j++) {
        neumann_neighbor_list_host(counter, j + 1) = neighbor_lists_(i, j + 1);
      }
    }
    if (particle_type[i] >= 4) {
      counter = colloid_map[i];
      colloid_epsilon_host(counter) = epsilon_(i);
      colloid_neighbor_list_host(counter, 0) = neighbor_lists_(i, 0);
      for (int j = 0; j < neighbor_lists_(i, 0); j++) {
        colloid_neighbor_list_host(counter, j + 1) = neighbor_lists_(i, j + 1);
      }
    }
  }

  num_neighbor.resize(numLocalParticle);
  for (int i = 0; i < numLocalParticle; i++) {
    num_neighbor[i] = neighbor_lists_(i, 0);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> neighbor_lists_device(
      "neighbor lists", numLocalParticle, neighbor_lists_.extent(1));
  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> epsilon_device(
      "epsilon", numLocalParticle);

  Kokkos::deep_copy(neighbor_lists_device, neighbor_lists_);
  Kokkos::deep_copy(epsilon_device, epsilon_);
  Kokkos::deep_copy(neumann_neighbor_list_device, neumann_neighbor_list_host);
  Kokkos::deep_copy(neumann_epsilon_device, neumann_epsilon_host);
  Kokkos::deep_copy(colloid_neighbor_list_device, colloid_neighbor_list_host);
  Kokkos::deep_copy(colloid_epsilon_device, colloid_epsilon_host);

  // pressure basis
  pressureBasis.setProblemData(neighbor_lists_device, source_coord_device,
                               target_coord_device, epsilon_device);

  std::vector<Compadre::TargetOperation> pressure_operation(2);
  pressure_operation[0] = Compadre::LaplacianOfScalarPointEvaluation;
  pressure_operation[1] = Compadre::GradientOfScalarPointEvaluation;

  pressureBasis.addTargets(pressure_operation);

  pressureBasis.setWeightingType(Compadre::WeightingFunctionType::Power);
  pressureBasis.setWeightingParameter(4);
  pressureBasis.setOrderOfQuadraturePoints(2);
  pressureBasis.setDimensionOfQuadraturePoints(1);
  pressureBasis.setQuadratureType("LINE");

  pressureBasis.generateAlphas(1, false);

  auto pressure_solution_set = pressureBasis.getSolutionSetHost();
  auto pressure_alpha = pressure_solution_set->getAlphas();

  const int pressure_laplacian_index =
      pressure_solution_set->getAlphaColumnOffset(pressure_operation[0], 0, 0,
                                                  0, 0);
  std::vector<int> pressure_gradient_index;
  for (int i = 0; i < dim_; i++)
    pressure_gradient_index.push_back(
        pressure_solution_set->getAlphaColumnOffset(pressure_operation[1], i, 0,
                                                    0, 0));

  // velocity basis
  velocityBasis.setProblemData(neighbor_lists_device, source_coord_device,
                               target_coord_device, epsilon_device);

  velocityBasis.addTargets(Compadre::CurlCurlOfVectorPointEvaluation);

  velocityBasis.setWeightingType(Compadre::WeightingFunctionType::Power);
  velocityBasis.setWeightingParameter(4);

  velocityBasis.generateAlphas(1, false);

  auto velocity_solution_set = velocityBasis.getSolutionSetHost();
  auto velocity_alpha = velocity_solution_set->getAlphas();

  std::vector<int> velocity_curl_curl_index(pow(dim_, 2));
  for (int i = 0; i < dim_; i++) {
    for (int j = 0; j < dim_; j++) {
      velocity_curl_curl_index[i * dim_ + j] =
          velocity_solution_set->getAlphaColumnOffset(
              Compadre::CurlCurlOfVectorPointEvaluation, i, 0, j, 0);
    }
  }

  // velocity colloid boundary basis
  velocityColloidBasis.setProblemData(
      colloid_neighbor_list_device, source_coord_device,
      colloid_target_coord_device, colloid_epsilon_device);

  velocityColloidBasis.addTargets(Compadre::GradientOfVectorPointEvaluation);

  velocityColloidBasis.setWeightingType(Compadre::WeightingFunctionType::Power);
  velocityColloidBasis.setWeightingParameter(4);

  velocityColloidBasis.generateAlphas(1, false);

  auto velocity_colloid_solution_set =
      velocityColloidBasis.getSolutionSetHost();
  auto velocity_colloid_alpha = velocity_colloid_solution_set->getAlphas();

  std::vector<int> velocity_gradient_index(pow(dim_, 3));
  for (int i = 0; i < dim_; i++) {
    for (int j = 0; j < dim_; j++) {
      for (int k = 0; k < dim_; k++) {
        velocity_gradient_index[(i * dim_ + j) * dim_ + k] =
            velocity_colloid_solution_set->getAlphaColumnOffset(
                Compadre::GradientOfVectorPointEvaluation, i, j, k, 0);
      }
    }
  }

  // pressure Neumann boundary basis
  pressureNeumannBasis.setProblemData(
      neumann_neighbor_list_device, source_coord_device,
      neumann_target_coord_device, neumann_epsilon_device);

  pressureNeumannBasis.setTangentBundle(tangent_bundle_device);

  std::vector<Compadre::TargetOperation> pressure_neumann_operation(1);
  pressure_neumann_operation[0] = Compadre::LaplacianOfScalarPointEvaluation;

  pressureNeumannBasis.addTargets(pressure_neumann_operation);

  pressureNeumannBasis.setWeightingType(Compadre::WeightingFunctionType::Power);
  pressureNeumannBasis.setWeightingParameter(4);
  pressureNeumannBasis.setOrderOfQuadraturePoints(2);
  pressureNeumannBasis.setDimensionOfQuadraturePoints(1);
  pressureNeumannBasis.setQuadratureType("LINE");

  pressureNeumannBasis.generateAlphas(1, false);

  auto pressure_neumann_solution_set =
      pressureNeumannBasis.getSolutionSetHost();
  auto pressure_neumann_alpha = pressure_neumann_solution_set->getAlphas();

  const int pressure_neumann_laplacian_index =
      pressure_neumann_solution_set->getAlphaColumnOffset(
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

  const int translation_dof = (dim_ == 3 ? 3 : 2);
  const int rotation_dof = (dim_ == 3 ? 3 : 1);
  const int rigid_body_dof = (dim_ == 3 ? 6 : 3);

  int fieldDof = dim_ + 1;
  int velocityDof = dim_;

  int local_velocity_dof = numLocalParticle * dim_;
  int global_velocity_dof =
      numGlobalParticleNum * dim_ + rigid_body_dof * numRigidBody;
  int local_pressure_dof = numLocalParticle;
  int global_pressure_dof = numGlobalParticleNum;

  if (mpiRank_ == mpiSize_ - 1) {
    local_velocity_dof += rigid_body_dof * numRigidBody;
  }

  std::vector<int> particle_num_per_process;
  particle_num_per_process.resize(mpiSize_);

  MPI_Allgather(&numLocalParticle, 1, MPI_INT, particle_num_per_process.data(),
                1, MPI_INT, MPI_COMM_WORLD);

  int local_rigid_body_offset =
      particle_num_per_process[mpiSize_ - 1] * fieldDof;
  int global_rigid_body_offset = numGlobalParticleNum * fieldDof;
  int local_out_process_offset =
      particle_num_per_process[mpiSize_ - 1] * fieldDof;

  int local_dof = local_velocity_dof + local_pressure_dof;
  int global_dof = global_velocity_dof + global_pressure_dof;

  int out_process_row = rigid_body_dof * numRigidBody;

  petsc_sparse_matrix &A = *(multiMgr_->getA(currentRefinementLevel_));
  A.resize(local_dof, local_dof, global_dof, out_process_row,
           local_out_process_offset);

  MPI_Barrier(MPI_COMM_WORLD);
  timer1 = MPI_Wtime();

  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "Current memory usage after resizing %.2f GB\n",
              mem / 1e9);

  // compute matrix graph
  std::vector<std::vector<PetscInt>> out_process_index(out_process_row);

  for (int i = 0; i < numLocalParticle; i++) {
    const int current_particle_local_index = local_idx[i];
    const int current_particle_global_index = source_index[i];

    const int pressure_local_index =
        current_particle_local_index * fieldDof + velocityDof;
    const int pressure_global_index =
        current_particle_global_index * fieldDof + velocityDof;

    std::vector<PetscInt> index;
    if (particle_type[i] == 0) {
      // velocity block
      index.clear();
      for (int j = 0; j < neighbor_lists_(i, 0); j++) {
        const int neighbor_particle_index =
            source_index[neighbor_lists_(i, j + 1)];

        for (int axes = 0; axes < fieldDof; axes++) {
          index.push_back(fieldDof * neighbor_particle_index + axes);
        }
      }

      for (int axes = 0; axes < velocityDof; axes++) {
        A.set_col_index(current_particle_local_index * fieldDof + axes, index);
      }

      // pressure block
      index.clear();
      for (int j = 0; j < neighbor_lists_(i, 0); j++) {
        const int neighbor_particle_index =
            source_index[neighbor_lists_(i, j + 1)];

        index.push_back(fieldDof * neighbor_particle_index + velocityDof);
      }

      A.set_col_index(current_particle_local_index * fieldDof + velocityDof,
                      index);
    }

    if (particle_type[i] != 0 && particle_type[i] < 4) {
      // velocity block
      index.clear();
      index.resize(1);
      for (int axes = 0; axes < velocityDof; axes++) {
        index[0] = current_particle_global_index * fieldDof + axes;
        A.set_col_index(current_particle_local_index * fieldDof + axes, index);
      }

      // pressure block
      index.clear();
      for (int j = 0; j < neighbor_lists_(i, 0); j++) {
        const int neighbor_particle_index =
            source_index[neighbor_lists_(i, j + 1)];

        for (int axes = 0; axes < fieldDof; axes++) {
          index.push_back(fieldDof * neighbor_particle_index + axes);
        }
      }

      A.set_col_index(current_particle_local_index * fieldDof + velocityDof,
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

      for (int axes = 0; axes < velocityDof; axes++) {
        index[0] = current_particle_global_index * fieldDof + axes;
        index[1] = global_rigid_body_offset +
                   attached_rigid_body[i] * rigid_body_dof + axes;
        A.set_col_index(current_particle_local_index * fieldDof + axes, index);
      }

      // pressure block
      index.clear();
      for (int j = 0; j < neighbor_lists_(i, 0); j++) {
        const int neighbor_particle_index =
            source_index[neighbor_lists_(i, j + 1)];

        for (int axes = 0; axes < fieldDof; axes++) {
          index.push_back(fieldDof * neighbor_particle_index + axes);
        }
      }

      A.set_col_index(current_particle_local_index * fieldDof + velocityDof,
                      index);
    }
  }

  double area = 0.0;
  int area_num = 0;

  // out process graph
  for (int i = 0; i < numLocalParticle; i++) {
    int rigid_body_idx = attached_rigid_body[i];
    const int current_particle_local_index = local_idx[i];
    const int current_particle_global_index = source_index[i];

    if (particle_type[i] >= 4) {
      std::vector<PetscInt> index;
      // attached rigid body
      index.clear();
      for (int j = 0; j < neighbor_lists_(i, 0); j++) {
        const int neighbor_particle_index =
            source_index[neighbor_lists_(i, j + 1)];

        for (int axes = 0; axes < velocityDof; axes++) {
          index.push_back(fieldDof * neighbor_particle_index + axes);
        }
      }
      // pressure term
      index.push_back(fieldDof * current_particle_global_index + velocityDof);

      for (int axes = 0; axes < translation_dof; axes++) {
        std::vector<PetscInt> &it =
            out_process_index[rigid_body_idx * rigid_body_dof + axes];
        if (!rigid_body_velocity_force_switch[rigid_body_idx][axes]) {
          it.insert(it.end(), index.begin(), index.end());
        }
      }

      for (int axes = 0; axes < rotation_dof; axes++) {
        std::vector<PetscInt> &it =
            out_process_index[rigid_body_idx * rigid_body_dof +
                              translation_dof + axes];
        if (!rigid_body_angvelocity_torque_switch[rigid_body_idx][axes]) {
          it.insert(it.end(), index.begin(), index.end());
        }
      }
    }
  }

  if (mpiRank_ == mpiSize_ - 1) {
    for (int rigid_body_idx = 0; rigid_body_idx < numRigidBody;
         rigid_body_idx++) {
      for (int axes = 0; axes < translation_dof; axes++) {
        if (rigid_body_velocity_force_switch[rigid_body_idx][axes]) {
          std::vector<PetscInt> &it =
              out_process_index[rigid_body_idx * rigid_body_dof + axes];
          it.push_back(global_rigid_body_offset +
                       rigid_body_idx * rigid_body_dof + axes);
        }
      }
      for (int axes = 0; axes < rotation_dof; axes++) {
        if (rigid_body_angvelocity_torque_switch[rigid_body_idx][axes]) {
          std::vector<PetscInt> &it =
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

  MPI_Barrier(MPI_COMM_WORLD);
  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "Current memory usage after building the graph %.2f GB\n",
              mem / 1e9);

  Kokkos::resize(bi_, numLocalParticle);

  // insert matrix entity
  for (int i = 0; i < numLocalParticle; i++) {
    const int current_particle_local_index = local_idx[i];
    const int current_particle_global_index = source_index[i];

    const int pressure_local_index =
        current_particle_local_index * fieldDof + velocityDof;
    const int pressure_global_index =
        current_particle_global_index * fieldDof + velocityDof;
    // velocity block
    if (particle_type[i] == 0) {
      for (int j = 0; j < neighbor_lists_(i, 0); j++) {
        const int neighbor_particle_index =
            source_index[neighbor_lists_(i, j + 1)];
        // inner fluid particle

        // curl curl u
        for (int axes1 = 0; axes1 < dim_; axes1++) {
          const int velocity_local_index =
              fieldDof * current_particle_local_index + axes1;
          for (int axes2 = 0; axes2 < dim_; axes2++) {
            const int velocity_global_index =
                fieldDof * neighbor_particle_index + axes2;

            auto alpha_index = velocity_solution_set->getAlphaIndex(
                i, velocity_curl_curl_index[axes1 * dim_ + axes2]);
            const double Lij = eta_ * velocity_alpha(alpha_index + j);

            A.increment(velocity_local_index, velocity_global_index, Lij);
          }
        }
      }
    } else {
      // wall boundary (including particles on rigid body)
      for (int axes1 = 0; axes1 < dim_; axes1++) {
        const int velocity_local_index =
            fieldDof * current_particle_local_index + axes1;
        const int velocity_global_index =
            fieldDof * current_particle_global_index + axes1;

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

        Vec3 rci = coord[i] - rigid_body_position[current_rigid_body_index];
        // non-slip condition
        // translation
        for (int axes1 = 0; axes1 < translation_dof; axes1++) {
          const int velocity_local_index =
              fieldDof * current_particle_local_index + axes1;
          A.increment(velocity_local_index,
                      current_rigid_body_global_offset + axes1, -1.0);
        }

        // rotation
        if (dim_ == 2) {
          for (int axes1 = 0; axes1 < rotation_dof; axes1++) {
            A.increment(fieldDof * current_particle_local_index +
                            (axes1 + 2) % translation_dof,
                        current_rigid_body_global_offset + translation_dof +
                            axes1,
                        rci[(axes1 + 1) % translation_dof]);
            A.increment(fieldDof * current_particle_local_index +
                            (axes1 + 1) % translation_dof,
                        current_rigid_body_global_offset + translation_dof +
                            axes1,
                        -rci[(axes1 + 2) % translation_dof]);
          }
        }
        if (dim_ == 3) {
          for (int axes1 = 0; axes1 < rotation_dof; axes1++) {
            A.increment(fieldDof * current_particle_local_index +
                            (axes1 + 2) % translation_dof,
                        current_rigid_body_global_offset + translation_dof +
                            axes1,
                        -rci[(axes1 + 1) % translation_dof]);
            A.increment(fieldDof * current_particle_local_index +
                            (axes1 + 1) % translation_dof,
                        current_rigid_body_global_offset + translation_dof +
                            axes1,
                        rci[(axes1 + 2) % translation_dof]);
          }
        }

        Vec3 dA;
        if (particle_type[i] == 4) {
          // corner point
          dA = Vec3(0.0, 0.0, 0.0);
        } else {
          dA = (dim_ == 3) ? (normal[i] * p_spacing[i][0] * p_spacing[i][1])
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

        for (int j = 0; j < neighbor_lists_(i, 0); j++) {
          const int neighbor_particle_index =
              source_index[neighbor_lists_(i, j + 1)];

          for (int axes3 = 0; axes3 < dim_; axes3++) {
            const int velocity_global_index =
                fieldDof * neighbor_particle_index + axes3;

            double *f = new double[dim_];
            for (int axes1 = 0; axes1 < dim_; axes1++) {
              f[axes1] = 0.0;
            }

            for (int axes1 = 0; axes1 < dim_; axes1++) {
              // output component 1
              for (int axes2 = 0; axes2 < dim_; axes2++) {
                // output component 2
                const int velocity_gradient_index_1 =
                    velocity_gradient_index[(axes1 * dim_ + axes2) * dim_ +
                                            axes3];
                const int velocity_gradient_index_2 =
                    velocity_gradient_index[(axes2 * dim_ + axes1) * dim_ +
                                            axes3];
                auto alpha_index1 =
                    velocity_colloid_solution_set->getAlphaIndex(
                        colloid_map[i], velocity_gradient_index_1);
                auto alpha_index2 =
                    velocity_colloid_solution_set->getAlphaIndex(
                        colloid_map[i], velocity_gradient_index_2);
                const double sigma =
                    eta_ * (velocity_colloid_alpha(alpha_index1 + j) +
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
            if (dim_ == 2) {
              for (int axes1 = 0; axes1 < rotation_dof; axes1++) {
                if (!rigid_body_angvelocity_torque_switch
                        [current_rigid_body_index][axes1]) {
                  A.out_process_increment(
                      current_rigid_body_local_offset + translation_dof + axes1,
                      velocity_global_index,
                      -rci[(axes1 + 1) % translation_dof] *
                              f[(axes1 + 2) % translation_dof] +
                          rci[(axes1 + 2) % translation_dof] *
                              f[(axes1 + 1) % translation_dof]);
                }
              }
            }
            if (dim_ == 3) {
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
            }
            delete[] f;
          }
        }
      } // end of particles on rigid body
    }

    // n \cdot grad p
    if (particle_type[i] != 0) {
      const int neumann_index = neumann_map[i];
      bi_(i) = pressure_neumann_solution_set->getAlpha0TensorTo0Tensor(
          Compadre::LaplacianOfScalarPointEvaluation, neumann_index,
          neumann_neighbor_list_host(neumann_index, 0));

      for (int j = 0; j < neumann_neighbor_list_host(neumann_index, 0); j++) {
        const int neighbor_particle_index =
            source_index[neumann_neighbor_list_host(neumann_index, j + 1)];

        if (neumann_neighbor_list_host(neumann_index, j + 1) !=
            neighbor_lists_(i, j + 1)) {
          std::cout << mpiRank_ << ", index: " << i
                    << ", neumann index: " << neumann_map[i] << std::endl;
        }

        for (int axes2 = 0; axes2 < dim_; axes2++) {
          double gradient = 0.0;
          const int velocity_global_index =
              fieldDof * neighbor_particle_index + axes2;
          for (int axes1 = 0; axes1 < dim_; axes1++) {
            auto alpha_index = velocity_solution_set->getAlphaIndex(
                i, velocity_curl_curl_index[axes1 * dim_ + axes2]);
            const double Lij = eta_ * velocity_alpha(alpha_index + j);

            gradient += normal[i][axes1] * Lij;
          }
          A.increment(pressure_local_index, velocity_global_index,
                      bi_(i) * gradient);
        }
      }
    } // end of velocity block

    // pressure block
    if (particle_type[i] == 0) {
      for (int j = 0; j < neighbor_lists_(i, 0); j++) {
        const int neighbor_particle_index =
            source_index[neighbor_lists_(i, j + 1)];

        const int pressure_neighbor_global_index =
            fieldDof * neighbor_particle_index + velocityDof;

        auto alpha_index =
            pressure_solution_set->getAlphaIndex(i, pressure_laplacian_index);
        const double Aij = pressure_alpha(alpha_index + j);

        // laplacian p
        A.increment(pressure_local_index, pressure_neighbor_global_index, Aij);
        A.increment(pressure_local_index, pressure_global_index, -Aij);

        for (int axes1 = 0; axes1 < dim_; axes1++) {
          const int velocity_local_index =
              fieldDof * current_particle_local_index + axes1;

          auto alpha_index = pressure_solution_set->getAlphaIndex(
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
            fieldDof * neighbor_particle_index + velocityDof;

        auto alpha_index = pressure_neumann_solution_set->getAlphaIndex(
            neumann_index, pressure_neumann_laplacian_index);
        const double Aij = pressure_neumann_alpha(alpha_index + j);

        // laplacian p
        A.increment(pressure_local_index, pressure_neighbor_global_index, Aij);
        A.increment(pressure_local_index, pressure_global_index, -Aij);
      }
    }

    // end of pressure block
  } // end of fluid particle loop

  if (mpiRank_ == mpiSize_ - 1) {
    for (int rigid_body_idx = 0; rigid_body_idx < numRigidBody;
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
        if (rigid_body_angvelocity_torque_switch[rigid_body_idx][axes]) {
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
  for (int i = 0; i < numLocalParticle; i++) {
    const int current_particle_local_index = local_idx[i];
    const int current_particle_global_index = source_index[i];

    for (int k = 0; k < fieldDof; k++) {
      const int local_index = current_particle_local_index * fieldDof + k;
      const int global_index = current_particle_global_index * fieldDof + k;

      if (A.get_entity(local_index, global_index) < 0.0) {
        std::cout << std::fixed << std::setprecision(10)
                  << "source index: " << source_index[i]
                  << ", field index: " << k
                  << ", adaptive level: " << adaptive_level[i]
                  << ", particle type: " << particle_type[i]
                  << ", epsilon: " << epsilon_(i)
                  << ", ratio: " << epsilon_(i) / spacing[i]
                  << ", num of neighbor: " << neighbor_lists_(i, 0);

        std::cout << "(";
        for (int k = 0; k < dim_; k++) {
          std::cout << " " << coord[i][k];
        }
        std::cout << ") "
                  << "Discretization error" << std::endl;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &area, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "surface area: %f\n", area);
  MPI_Allreduce(MPI_IN_PLACE, &area_num, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "surface area particle num: %d\n", area_num);

  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "Current memory usage before assembly %.2f GB\n", mem / 1e9);

  auto ff = multiMgr_->get_field_mat(currentRefinementLevel_);
  auto nn = multiMgr_->get_colloid_mat(currentRefinementLevel_);
  auto nw = multiMgr_->get_colloid_whole_mat(currentRefinementLevel_);

  A.assemble(*ff, fieldDof, numRigidBody, rigid_body_dof);

  MPI_Barrier(MPI_COMM_WORLD);
  timer2 = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "Matrix assembly duration: %fs\n",
              timer2 - timer1);

  idx_colloid.clear();

  if (numRigidBody != 0)
    A.extract_neighbor_index(idx_colloid, dim_, numRigidBody,
                             local_rigid_body_offset, global_rigid_body_offset,
                             *nn, *nw);

  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "Current memory usage after assembly %.2f GB\n",
              mem / 1e9);

  int inner_counter = 0;
  for (int i = 0; i < numLocalParticle; i++) {
    if (particle_type[i] == 0)
      inner_counter++;
  }
  MPI_Allreduce(MPI_IN_PLACE, &inner_counter, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "total inner particle count: %d\n",
              inner_counter);
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

  int numLocalParticle;
  int numGlobalParticleNum;

  numLocalParticle = coord.size();
  MPI_Allreduce(&numLocalParticle, &numGlobalParticleNum, 1, MPI_UNSIGNED,
                MPI_SUM, MPI_COMM_WORLD);

  const int translation_dof = (dim_ == 3 ? 3 : 2);
  const int rotation_dof = (dim_ == 3 ? 3 : 1);
  const int rigid_body_dof = (dim_ == 3 ? 6 : 3);

  int fieldDof = dim_ + 1;
  int velocityDof = dim_;

  int local_velocity_dof = numLocalParticle * dim_;
  int global_velocity_dof =
      numGlobalParticleNum * dim_ + rigid_body_dof * numRigidBody;
  int local_pressure_dof = numLocalParticle;
  int global_pressure_dof = numGlobalParticleNum;

  if (mpiRank_ == mpiSize_ - 1) {
    local_velocity_dof += rigid_body_dof * numRigidBody;
  }

  std::vector<int> particle_num_per_process;
  particle_num_per_process.resize(mpiSize_);

  MPI_Allgather(&numLocalParticle, 1, MPI_INT, particle_num_per_process.data(),
                1, MPI_INT, MPI_COMM_WORLD);

  int local_rigid_body_offset =
      particle_num_per_process[mpiSize_ - 1] * fieldDof;
  int global_rigid_body_offset = numGlobalParticleNum * fieldDof;
  int local_out_process_offset =
      particle_num_per_process[mpiSize_ - 1] * fieldDof;

  int local_dof = local_velocity_dof + local_pressure_dof;

  rhs.resize(local_dof);
  res.resize(local_dof);

  for (int i = 0; i < local_dof; i++) {
    rhs[i] = 0.0;
    res[i] = 0.0;
  }

  // for (int i = 0; i < numLocalParticle; i++) {
  //   int current_particle_local_index = local_idx[i];
  //   if (particle_type[i] != 0 && particle_type[i] < 4) {
  //     double y = coord[i][1];

  //     rhs[fieldDof * current_particle_local_index] = 0.1 * y;
  //   }
  // }

  for (int i = 0; i < numLocalParticle; i++) {
    int current_particle_local_index = local_idx[i];
    if (particle_type[i] != 0 && particle_type[i] < 4) {
      // 2-d Taylor-Green vortex-like flow
      if (dim_ == 2) {
        double x = coord[i][0];
        double y = coord[i][1];

        rhs[fieldDof * current_particle_local_index] =
            sin(M_PI * x) * cos(M_PI * y);
        rhs[fieldDof * current_particle_local_index + 1] =
            -cos(M_PI * x) * sin(M_PI * y);

        const int neumann_index = neumann_map[i];

        rhs[fieldDof * current_particle_local_index + velocityDof] =
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

        rhs[fieldDof * current_particle_local_index] =
            sin(M_PI * x) * cos(M_PI * y) * cos(M_PI * z);
        rhs[fieldDof * current_particle_local_index + 1] =
            -2 * cos(M_PI * x) * sin(M_PI * y) * cos(M_PI * z);
        rhs[fieldDof * current_particle_local_index + 2] =
            cos(M_PI * x) * cos(M_PI * y) * sin(M_PI * z);

        const int neumann_index = neumann_map[i];

        rhs[fieldDof * current_particle_local_index + velocityDof] =
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

        rhs[fieldDof * current_particle_local_index] =
            2.0 * pow(M_PI, 2.0) * sin(M_PI * x) * cos(M_PI * y) +
            2.0 * M_PI * sin(2.0 * M_PI * x);
        rhs[fieldDof * current_particle_local_index + 1] =
            -2.0 * pow(M_PI, 2.0) * cos(M_PI * x) * sin(M_PI * y) +
            2.0 * M_PI * sin(2.0 * M_PI * y);

        rhs[fieldDof * current_particle_local_index + velocityDof] =
            -4.0 * pow(M_PI, 2.0) * (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y));
      }
      if (dim_ == 3) {
        double x = coord[i][0];
        double y = coord[i][1];
        double z = coord[i][2];

        rhs[fieldDof * current_particle_local_index] =
            3.0 * pow(M_PI, 2) * sin(M_PI * x) * cos(M_PI * y) * cos(M_PI * z) +
            2.0 * M_PI * sin(2.0 * M_PI * x);
        rhs[fieldDof * current_particle_local_index + 1] =
            -6.0 * pow(M_PI, 2) * cos(M_PI * x) * sin(M_PI * y) *
                cos(M_PI * z) +
            2.0 * M_PI * sin(2.0 * M_PI * y);
        rhs[fieldDof * current_particle_local_index + 2] =
            3.0 * pow(M_PI, 2) * cos(M_PI * x) * cos(M_PI * y) * sin(M_PI * z) +
            2.0 * M_PI * sin(2.0 * M_PI * z);

        rhs[fieldDof * current_particle_local_index + velocityDof] =
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

  //       rhs[fieldDof * current_particle_local_index] =
  //           sin(theta) * cos(phi) * vr + cos(theta) * cos(phi) * vt;
  //       rhs[fieldDof * current_particle_local_index + 1] =
  //           sin(theta) * sin(phi) * vr + cos(theta) * sin(phi) * vt;
  //       rhs[fieldDof * current_particle_local_index + 2] =
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

  if (mpiRank_ == mpiSize_ - 1) {
    for (int i = 0; i < numRigidBody; i++) {
      for (int axes = 0; axes < translation_dof; axes++) {
        if (rigid_body_velocity_force_switch[i][axes])
          rhs[local_rigid_body_offset + i * rigid_body_dof + axes] =
              rigid_body_velocity[i][axes];
        else
          rhs[local_rigid_body_offset + i * rigid_body_dof + axes] =
              -rigid_body_force[i][axes];
      }
      for (int axes = 0; axes < rotation_dof; axes++) {
        if (rigid_body_angvelocity_torque_switch[i][axes])
          rhs[local_rigid_body_offset + i * rigid_body_dof + translation_dof +
              axes] = rigid_body_angular_velocity[i][axes];
        else
          rhs[local_rigid_body_offset + i * rigid_body_dof + translation_dof +
              axes] = -rigid_body_torque[i][axes];
      }
    }
  }

  // make sure pressure term is orthogonal to the constant
  double rhs_pressure_sum = 0.0;
  for (int i = 0; i < numLocalParticle; i++) {
    int current_particle_local_index = local_idx[i];
    rhs_pressure_sum +=
        rhs[fieldDof * current_particle_local_index + velocityDof];
  }
  MPI_Allreduce(MPI_IN_PLACE, &rhs_pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  rhs_pressure_sum /= numGlobalParticleNum;
  for (int i = 0; i < numLocalParticle; i++) {
    int current_particle_local_index = local_idx[i];
    rhs[fieldDof * current_particle_local_index + velocityDof] -=
        rhs_pressure_sum;
  }
}

void StokesEquation::SolveEquation() {
  const int numRigidBody = rbMgr_->get_rigid_body_num();

  auto &local_idx = *(geoMgr_->get_current_work_particle_local_index());

  // build interpolation and restriction operators
  double timer1, timer2;
  if (currentRefinementLevel_ != 0) {
    timer1 = MPI_Wtime();

    multiMgr_->build_interpolation_restriction(numRigidBody, dim_, polyOrder_);
    multiMgr_->initial_guess_from_previous_adaptive_step(
        res, velocity, pressure, rbMgr_->get_velocity(),
        rbMgr_->get_angular_velocity());

    timer2 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,
                "Interpolation matrix building duration: %fs\n",
                timer2 - timer1);
  }

  PetscLogDefaultBegin();

  MPI_Barrier(MPI_COMM_WORLD);
  timer1 = MPI_Wtime();
  // if (currentRefinementLevel_ < 4)
  multiMgr_->solve(rhs, res, idx_colloid);
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

  const unsigned int translation_dof = (dim_ == 3 ? 3 : 2);
  const unsigned int rotation_dof = (dim_ == 3 ? 3 : 1);
  const unsigned int rigid_body_dof = (dim_ == 3 ? 6 : 3);

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
    pressure[i] = res[fieldDof * current_particle_local_index + velocityDof];
    pressure_sum += pressure[i];
    for (int axes1 = 0; axes1 < dim_; axes1++)
      velocity[i][axes1] = res[fieldDof * current_particle_local_index + axes1];
  }

  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  double average_pressure = pressure_sum / numGlobalParticleNum;
  for (int i = 0; i < numLocalParticle; i++) {
    pressure[i] -= average_pressure;
  }

  std::vector<Vec3> &rigid_body_velocity = rbMgr_->get_velocity();
  std::vector<Vec3> &rigid_body_angular_velocity =
      rbMgr_->get_angular_velocity();

  if (mpiRank_ == mpiSize_ - 1) {
    for (int i = 0; i < numRigidBody; i++) {
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
  std::vector<double> translation_velocity(numRigidBody * translation_dof);
  std::vector<double> angular_velocity(numRigidBody * rotation_dof);

  if (mpiRank_ == mpiSize_ - 1) {
    for (int i = 0; i < numRigidBody; i++) {
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

  MPI_Bcast(translation_velocity.data(), numRigidBody * translation_dof,
            MPI_DOUBLE, mpiSize_ - 1, MPI_COMM_WORLD);
  MPI_Bcast(angular_velocity.data(), numRigidBody * rotation_dof, MPI_DOUBLE,
            mpiSize_ - 1, MPI_COMM_WORLD);

  if (mpiRank_ != mpiSize_ - 1) {
    for (int i = 0; i < numRigidBody; i++) {
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

  const unsigned int translation_dof = (dim_ == 3 ? 3 : 2);
  const unsigned int rotation_dof = (dim_ == 3 ? 3 : 1);
  const unsigned int rigid_body_dof = (dim_ == 3 ? 6 : 3);

  const unsigned int fieldDof = dim_ + 1;
  const unsigned int velocityDof = dim_;

  unsigned int local_velocity_dof = numLocalParticle * dim_;
  const unsigned int global_velocity_dof =
      numGlobalParticleNum * dim_ + rigid_body_dof * numRigidBody;
  const unsigned int local_pressure_dof = numLocalParticle;
  const unsigned int global_pressure_dof = numGlobalParticleNum;

  if (mpiRank_ == mpiSize_ - 1) {
    local_velocity_dof += rigid_body_dof * numRigidBody;
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

    size_t coefficients_size;

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace> source_coord_device(
        "source coordinates", source_coord.size(), 3);
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
        neighbor_lists_host(i, 0) = neighbor_lists_(i + start_particle, 0);
        for (int j = 0; j < neighbor_lists_(i, 0); j++) {
          neighbor_lists_host(i, j + 1) =
              neighbor_lists_(i + start_particle, j + 1);
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
                                         source_coord_device,
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
      for (int j = 0; j < neighbor_lists_(i, 0); j++) {
        const int neighbor_index = neighbor_lists_(i, j + 1);

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
      for (int j = 0; j < neighbor_lists_(i, 0); j++) {
        const int neighbor_index = neighbor_lists_(i, j + 1);

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
      for (int j = 0; j < neighbor_lists_(i, 0); j++) {
        const int neighbor_index = neighbor_lists_(i, j + 1);

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

  const int translation_dof = (dim_ == 3 ? 3 : 2);
  const int rotation_dof = (dim_ == 3 ? 3 : 1);
  const int rigid_body_dof = (dim_ == 3 ? 6 : 3);

  std::vector<double> flattened_force;
  std::vector<double> flattened_torque;

  flattened_force.resize(numRigidBody * translation_dof);
  flattened_torque.resize(numRigidBody * rotation_dof);

  for (int i = 0; i < numRigidBody; i++) {
    for (int j = 0; j < translation_dof; j++) {
      flattened_force[i * translation_dof + j] = 0.0;
    }
    for (int j = 0; j < rotation_dof; j++) {
      flattened_torque[i * rotation_dof + j] = 0.0;
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

      for (int axes1 = 0; axes1 < translation_dof; axes1++) {
        flattened_force[rigid_body_idx * translation_dof + axes1] += f[axes1];
      }

      for (int axes1 = 0; axes1 < rotation_dof; axes1++) {
        flattened_torque[rigid_body_idx * rotation_dof + axes1] +=
            rci[(axes1 + 1) % translation_dof] *
                f[(axes1 + 2) % translation_dof] -
            rci[(axes1 + 2) % translation_dof] *
                f[(axes1 + 1) % translation_dof];
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, flattened_force.data(),
                numRigidBody * translation_dof, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, flattened_torque.data(),
                numRigidBody * rotation_dof, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  for (int i = 0; i < numRigidBody; i++) {
    for (int j = 0; j < translation_dof; j++) {
      if (rigid_body_velocity_force_switch[i][j])
        rigid_body_force[i][j] = flattened_force[i * translation_dof + j];
    }
    for (int j = 0; j < rotation_dof; j++) {
      if (rigid_body_velocity_force_switch[i][j])
        rigid_body_torque[i][j] = flattened_torque[i * rotation_dof + j];
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