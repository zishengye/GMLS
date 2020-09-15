#include "stokes_equation.hpp"

using namespace std;
using namespace Compadre;

void stokes_equation::build_matrix(
    std::shared_ptr<std::vector<particle>> particle_set,
    std::shared_ptr<std::vector<particle>> background_particle_set,
    std::shared_ptr<sparse_matrix> ff) {
  GMLS pressure_basis(VectorTaylorPolynomial, StaggeredEdgeIntegralSample,
                      StaggeredEdgeAnalyticGradientIntegralSample, 2,
                      _dimension, "SVD", "STANDARD");
  GMLS neumann_pressure_basis(
      VectorTaylorPolynomial, StaggeredEdgeIntegralSample,
      StaggeredEdgeAnalyticGradientIntegralSample, 2, _dimension, "SVD",
      "STANDARD", "NEUMANN_GRAD_SCALAR");
  GMLS velocity_basis(DivergenceFreeVectorTaylorPolynomial, VectorPointSample,
                      2, _dimension, "SVD", "STANDARD");

  size_t num_source_coord = background_particle_set->size();
  size_t num_target_coord = particle_set->size();

  vector<particle> &target_particle = *particle_set;
  vector<particle> &source_particle = *background_particle_set;

  // create source coords (full particle set)
  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> source_coord_device(
      "source coordinates", num_source_coord, 3);
  Kokkos::View<double **>::HostMirror source_coord =
      Kokkos::create_mirror_view(source_coord_device);

  for (size_t i = 0; i < source_particle.size(); i++) {
    for (int j = 0; j < 3; j++) {
      source_coord(i, j) = source_particle[i].coord[j];
    }
  }

  int num_neumann_target_coord = 0;
  for (size_t i = 0; i < num_target_coord; i++) {
    if (target_particle[i].particle_type != 0) {
      num_neumann_target_coord++;
    }
  }

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> target_coord_device(
      "target coordinates", num_target_coord, 3);
  Kokkos::View<double **>::HostMirror target_coord =
      Kokkos::create_mirror_view(target_coord_device);
  Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
      neumann_target_coord_device("neumann target coordinates",
                                  num_neumann_target_coord, 3);
  Kokkos::View<double **>::HostMirror neumann_target_coord =
      Kokkos::create_mirror_view(neumann_target_coord_device);

  // create target coords
  vector<int> field_neumann_mapping;
  int index = 0;
  for (size_t i = 0; i < num_target_coord; i++) {
    for (int j = 0; j < 3; j++) {
      target_coord(i, j) = target_particle[i].coord[j];
    }
    field_neumann_mapping.push_back(index);
    if (target_particle[i].particle_type != 0) {
      for (int j = 0; j < 3; j++) {
        neumann_target_coord(index, j) = target_particle[i].coord[j];
      }
      index++;
    }
  }

  Kokkos::deep_copy(source_coord_device, source_coord);
  Kokkos::deep_copy(target_coord_device, target_coord);
  Kokkos::deep_copy(neumann_target_coord_device, neumann_target_coord);

  // neighbor search
  auto point_cloud_search(CreatePointCloudSearch(source_coord, _dimension));

  auto min_neighbor = Compadre::GMLS::getNP(2, _dimension);

  double epsilon_multiplier = 2.5;

  int estimated_upper_bound =
      pow(2, _dimension) * pow(2 * epsilon_multiplier, _dimension);

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> neighbor_list_device(
      "neighbor lists", num_target_coord, estimated_upper_bound);
  Kokkos::View<int **>::HostMirror neighbor_list =
      Kokkos::create_mirror_view(neighbor_list_device);

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
      neumann_neighbor_list_device("neumann boundary neighbor lists",
                                   num_neumann_target_coord,
                                   estimated_upper_bound);
  Kokkos::View<int **>::HostMirror neumann_neighbor_list =
      Kokkos::create_mirror_view(neumann_neighbor_list_device);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> epsilon_device(
      "h supports", num_target_coord);
  Kokkos::View<double *>::HostMirror epsilon =
      Kokkos::create_mirror_view(epsilon_device);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> neumann_epsilon_device(
      "neumann boundary h supports", num_neumann_target_coord);
  Kokkos::View<double *>::HostMirror neumann_epsilon =
      Kokkos::create_mirror_view(neumann_epsilon_device);

  double search_radius = 2.5 * target_particle[0].particle_size[0] + 1e-5;

  point_cloud_search.generateNeighborListsFromRadiusSearch(
      false, target_coord, neighbor_list, epsilon, search_radius,
      search_radius);

  // auto neighbor_needed = Compadre::GMLS::getNP(
  //     2, _dimension, DivergenceFreeVectorTaylorPolynomial);

  // point_cloud_search.generateNeighborListsFromKNNSearch(
  //     false, target_coord, neighbor_list, epsilon, neighbor_needed, 1.5);

  for (size_t i = 0; i < num_target_coord; i++) {
    if (target_particle[i].particle_type != 0) {
      neumann_neighbor_list(field_neumann_mapping[i], 0) = neighbor_list(i, 0);
      for (int j = 0; j < neighbor_list(i, 0); j++) {
        neumann_neighbor_list(field_neumann_mapping[i], j + 1) =
            neighbor_list(i, j + 1);
      }

      neumann_epsilon(field_neumann_mapping[i]) = epsilon(i);
    }
  }

  Kokkos::deep_copy(neighbor_list_device, neighbor_list);
  Kokkos::deep_copy(epsilon_device, epsilon);
  Kokkos::deep_copy(neumann_neighbor_list_device, neumann_neighbor_list);
  Kokkos::deep_copy(neumann_epsilon_device, neumann_epsilon);

  // tangent bundle for neumann boundary particles
  Kokkos::View<double ***, Kokkos::DefaultExecutionSpace> tangent_bundle_device(
      "tangent bundles", num_neumann_target_coord, _dimension, _dimension);
  Kokkos::View<double ***>::HostMirror tangent_bundle =
      Kokkos::create_mirror_view(tangent_bundle_device);

  index = 0;
  for (int i = 0; i < num_target_coord; i++) {
    if (target_particle[i].particle_type != 0) {
      if (_dimension == 3) {
        tangent_bundle(index, 0, 0) = 0.0;
        tangent_bundle(index, 0, 1) = 0.0;
        tangent_bundle(index, 0, 2) = 0.0;
        tangent_bundle(index, 1, 0) = 0.0;
        tangent_bundle(index, 1, 1) = 0.0;
        tangent_bundle(index, 1, 2) = 0.0;
        tangent_bundle(index, 2, 0) = target_particle[i].normal[0];
        tangent_bundle(index, 2, 1) = target_particle[i].normal[1];
        tangent_bundle(index, 2, 2) = target_particle[i].normal[2];
      }
      if (_dimension == 2) {
        tangent_bundle(index, 0, 0) = 0.0;
        tangent_bundle(index, 0, 1) = 0.0;
        tangent_bundle(index, 1, 0) = target_particle[i].normal[0];
        tangent_bundle(index, 1, 1) = target_particle[i].normal[1];
      }
      index++;
    }
  }

  Kokkos::deep_copy(tangent_bundle_device, tangent_bundle);

  int number_of_batches = 20;

  // pressure basis
  pressure_basis.setProblemData(neighbor_list_device, source_coord_device,
                                target_coord_device, epsilon_device);

  vector<TargetOperation> pressure_operation(2);
  pressure_operation[0] = DivergenceOfVectorPointEvaluation;
  pressure_operation[1] = GradientOfScalarPointEvaluation;

  pressure_basis.clearTargets();
  pressure_basis.addTargets(pressure_operation);

  pressure_basis.setWeightingType(WeightingFunctionType::Power);
  pressure_basis.setWeightingPower(4);
  pressure_basis.setOrderOfQuadraturePoints(2);
  pressure_basis.setDimensionOfQuadraturePoints(1);
  pressure_basis.setQuadratureType("LINE");

  pressure_basis.generateAlphas(number_of_batches);

  auto pressure_alpha = pressure_basis.getAlphas();

  const int pressure_laplacian_index =
      pressure_basis.getAlphaColumnOffset(pressure_operation[0], 0, 0, 0, 0);
  vector<int> pressure_gradient_index;
  for (int i = 0; i < _dimension; i++)
    pressure_gradient_index.push_back(
        pressure_basis.getAlphaColumnOffset(pressure_operation[1], i, 0, 0, 0));

  // pressure Neumann boundary basis
  neumann_pressure_basis.setProblemData(
      neumann_neighbor_list_device, source_coord_device,
      neumann_target_coord_device, neumann_epsilon_device);

  neumann_pressure_basis.setTangentBundle(tangent_bundle_device);

  vector<TargetOperation> neumann_pressure_operation(1);
  neumann_pressure_operation[0] = DivergenceOfVectorPointEvaluation;

  neumann_pressure_basis.clearTargets();
  neumann_pressure_basis.addTargets(neumann_pressure_operation);

  neumann_pressure_basis.setWeightingType(WeightingFunctionType::Power);
  neumann_pressure_basis.setWeightingPower(4);
  neumann_pressure_basis.setOrderOfQuadraturePoints(2);
  neumann_pressure_basis.setDimensionOfQuadraturePoints(1);
  neumann_pressure_basis.setQuadratureType("LINE");

  neumann_pressure_basis.generateAlphas(1);

  auto neumann_pressure_alpha = neumann_pressure_basis.getAlphas();

  const int neumann_pressure_laplacian_index =
      neumann_pressure_basis.getAlphaColumnOffset(neumann_pressure_operation[0],
                                                  0, 0, 0, 0);

  // velocity basis
  velocity_basis.setProblemData(neighbor_list_device, source_coord_device,
                                target_coord_device, epsilon_device);

  vector<TargetOperation> velocity_operation(3);
  velocity_operation[0] = CurlCurlOfVectorPointEvaluation;
  velocity_operation[1] = GradientOfVectorPointEvaluation;
  velocity_operation[2] = ScalarPointEvaluation;

  velocity_basis.clearTargets();
  velocity_basis.addTargets(velocity_operation);

  velocity_basis.setWeightingType(WeightingFunctionType::Power);
  velocity_basis.setWeightingPower(4);

  velocity_basis.generateAlphas(number_of_batches);

  auto velocity_alpha = velocity_basis.getAlphas();

  vector<int> velocity_curl_curl_index(pow(_dimension, 2));
  for (int i = 0; i < _dimension; i++) {
    for (int j = 0; j < _dimension; j++) {
      velocity_curl_curl_index[i * _dimension + j] =
          velocity_basis.getAlphaColumnOffset(CurlCurlOfVectorPointEvaluation,
                                              i, 0, j, 0);
    }
  }
  vector<int> velocity_gradient_index(pow(_dimension, 3));
  for (int i = 0; i < _dimension; i++) {
    for (int j = 0; j < _dimension; j++) {
      for (int k = 0; k < _dimension; k++) {
        velocity_gradient_index[(i * _dimension + j) * _dimension + k] =
            velocity_basis.getAlphaColumnOffset(GradientOfVectorPointEvaluation,
                                                i, j, k, 0);
      }
    }
  }

  int field_dof = _dimension + 1;
  int velocity_dof = _dimension;

  size_t local_particle_num = num_target_coord;
  size_t global_particle_num;

  MPI_Allreduce(&local_particle_num, &global_particle_num, 1, MPI_UNSIGNED_LONG,
                MPI_SUM, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  int local_velocity_dof = local_particle_num * velocity_dof;
  int global_velocity_dof = global_particle_num * velocity_dof;
  int local_pressure_dof = local_particle_num;
  int global_pressure_dof = global_particle_num;

  int local_dof = local_velocity_dof + local_pressure_dof;
  int global_dof = global_velocity_dof + global_pressure_dof;

  sparse_matrix &A = *ff;
  A.resize(local_dof, local_dof, global_dof);

  // compute matrix graph
  for (size_t i = 0; i < num_target_coord; i++) {
    const size_t current_particle_local_index = i;
    const size_t current_particle_global_index =
        target_particle[i].global_index;

    const int index_pressure_local =
        current_particle_local_index * field_dof + velocity_dof;
    const int index_pressure_global =
        current_particle_global_index * field_dof + velocity_dof;

    vector<PetscInt> index;
    if (target_particle[i].particle_type == 0) {
      // velocity block
      index.clear();
      for (int j = 0; j < neighbor_list(i, 0); j++) {
        const int neighbor_index =
            source_particle[neighbor_list(i, j + 1)].global_index;

        for (int axes = 0; axes < field_dof; axes++) {
          index.push_back(field_dof * neighbor_index + axes);
        }
      }

      for (int axes = 0; axes < velocity_dof; axes++) {
        A.setColIndex(current_particle_local_index * field_dof + axes, index);
      }

      // pressure block
      index.clear();
      for (int j = 0; j < neighbor_list(i, 0); j++) {
        const int neighbor_index =
            source_particle[neighbor_list(i, j + 1)].global_index;

        index.push_back(field_dof * neighbor_index + velocity_dof);
      }

      A.setColIndex(current_particle_local_index * field_dof + velocity_dof,
                    index);
    }

    if (target_particle[i].particle_type != 0) {
      // velocity block
      index.clear();
      index.resize(1);
      for (int axes = 0; axes < velocity_dof; axes++) {
        index[0] = current_particle_global_index * field_dof + axes;
        A.setColIndex(current_particle_local_index * field_dof + axes, index);
      }

      // pressure block
      index.clear();
      for (int j = 0; j < neighbor_list(i, 0); j++) {
        const int neighbor_index =
            source_particle[neighbor_list(i, j + 1)].global_index;

        for (int axes = 0; axes < field_dof; axes++) {
          index.push_back(field_dof * neighbor_index + axes);
        }
      }

      A.setColIndex(current_particle_local_index * field_dof + velocity_dof,
                    index);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // insert matrix entity
  for (int i = 0; i < num_target_coord; i++) {
    const int current_particle_local_index = i;
    const int current_particle_global_index = source_particle[i].global_index;

    const int index_pressure_local =
        current_particle_local_index * field_dof + velocity_dof;
    const int index_pressure_global =
        current_particle_global_index * field_dof + velocity_dof;
    // velocity block
    if (target_particle[i].particle_type == 0) {
      for (int j = 0; j < neighbor_list(i, 0); j++) {
        const int neighbor_index =
            source_particle[neighbor_list(i, j + 1)].global_index;
        // inner fluid particle

        // curl curl u
        for (int axes1 = 0; axes1 < _dimension; axes1++) {
          const int index_velocity_local =
              field_dof * current_particle_local_index + axes1;
          for (int axes2 = 0; axes2 < _dimension; axes2++) {
            const int index_neighbor_velocity_global =
                field_dof * neighbor_index + axes2;

            const double Lij = velocity_alpha(
                i, velocity_curl_curl_index[axes1 * _dimension + axes2], j);

            A.increment(index_velocity_local, index_neighbor_velocity_global,
                        Lij);
          }
        }
      }
    } else {
      // wall boundary (including particles on rigid body)
      for (int axes1 = 0; axes1 < _dimension; axes1++) {
        const int index_velocity_local =
            field_dof * current_particle_local_index + axes1;
        const int index_velocity_global =
            field_dof * current_particle_global_index + axes1;

        A.increment(index_velocity_local, index_velocity_global, 1.0);
      }
    }

    // n \cdot grad p
    if (target_particle[i].particle_type != 0) {
      const int neumann_index = field_neumann_mapping[i];
      const double bi = neumann_pressure_basis.getAlpha0TensorTo0Tensor(
          DivergenceOfVectorPointEvaluation, neumann_index,
          neumann_neighbor_list(neumann_index, 0));

      for (int j = 0; j < neumann_neighbor_list(neumann_index, 0); j++) {
        const int neighbor_index =
            source_particle[neumann_neighbor_list(neumann_index, j + 1)]
                .global_index;

        for (int axes2 = 0; axes2 < _dimension; axes2++) {
          double gradient = 0.0;
          const int index_neighbor_velocity_global =
              field_dof * neighbor_index + axes2;
          for (int axes1 = 0; axes1 < _dimension; axes1++) {
            const double Lij = velocity_alpha(
                i, velocity_curl_curl_index[axes1 * _dimension + axes2], j);

            gradient += target_particle[i].normal[axes1] * Lij;
          }
          A.increment(index_pressure_local, index_neighbor_velocity_global,
                      bi * gradient);
        }
      }
    } // end of velocity block

    // pressure block
    if (target_particle[i].particle_type == 0) {
      for (int j = 0; j < neighbor_list(i, 0); j++) {
        const int neighbor_index =
            source_particle[neighbor_list(i, j + 1)].global_index;

        const int index_neighbor_pressure_global =
            field_dof * neighbor_index + velocity_dof;

        const double Aij = pressure_alpha(i, pressure_laplacian_index, j);

        // laplacian p
        A.increment(index_pressure_local, index_neighbor_pressure_global, Aij);
        A.increment(index_pressure_local, index_pressure_global, -Aij);

        for (int axes1 = 0; axes1 < _dimension; axes1++) {
          const int index_velocity_local =
              field_dof * current_particle_local_index + axes1;

          const double Dijx =
              pressure_alpha(i, pressure_gradient_index[axes1], j);

          // grad p
          A.increment(index_velocity_local, index_neighbor_pressure_global,
                      -Dijx);
          A.increment(index_velocity_local, index_pressure_global, Dijx);
        }
      }
    }
    if (target_particle[i].particle_type != 0) {
      const int neumann_index = field_neumann_mapping[i];

      for (int j = 0; j < neumann_neighbor_list(neumann_index, 0); j++) {
        const int neighbor_index =
            source_particle[neumann_neighbor_list(neumann_index, j + 1)]
                .global_index;

        const int index_neighbor_pressure_global =
            field_dof * neighbor_index + velocity_dof;

        const double Aij = neumann_pressure_alpha(
            neumann_index, neumann_pressure_laplacian_index, j);

        // laplacian p
        A.increment(index_pressure_local, index_neighbor_pressure_global, Aij);
        A.increment(index_pressure_local, index_pressure_global, -Aij);
      }
    }
    // end of pressure block
  } // end of fluid particle loop

  A.FinalAssemble(field_dof);

  MPI_Barrier(MPI_COMM_WORLD);
}

void stokes_equation::build_interpolation(
    std::shared_ptr<std::vector<particle>> coarse_grid_particle_set,
    std::shared_ptr<std::vector<particle>> fine_grid_particle_set,
    std::shared_ptr<sparse_matrix> interpolation) {
  int velocity_dof = _dimension;
  int field_dof = _dimension + 1;
  int coarse_grid_particle_num = coarse_grid_particle_set->size();
  int fine_grid_particle_num = fine_grid_particle_set->size();

  int coarse_grid_dof = coarse_grid_particle_num * field_dof;
  int fine_grid_dof = fine_grid_particle_num * field_dof;

  int coarse_grid_dof_global;
  MPI_Allreduce(&coarse_grid_dof, &coarse_grid_dof_global, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  std::vector<particle> &coarse_grid_list = *coarse_grid_particle_set;
  std::vector<particle> &fine_grid_list = *fine_grid_particle_set;

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> source_coord_device(
      "old source coordinates", coarse_grid_particle_num, 3);
  Kokkos::View<double **>::HostMirror source_coord =
      Kokkos::create_mirror_view(source_coord_device);

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> target_coord_device(
      "new target coordinates", fine_grid_particle_num, 3);
  Kokkos::View<double **>::HostMirror target_coord =
      Kokkos::create_mirror_view(target_coord_device);

  for (int i = 0; i < coarse_grid_particle_num; i++) {
    for (int j = 0; j < _dimension; j++)
      source_coord(i, j) = coarse_grid_list[i].coord[j];
  }

  for (int i = 0; i < fine_grid_particle_num; i++) {
    for (int j = 0; j < _dimension; j++)
      target_coord(i, j) = fine_grid_list[i].coord[j];
  }

  Kokkos::deep_copy(source_coord_device, source_coord);
  Kokkos::deep_copy(target_coord_device, target_coord);

  auto interpolation_point_search(
      CreatePointCloudSearch(source_coord_device, _dimension));

  int estimated_num_neighbors = pow(2, _dimension) * pow(2 * 2.5, _dimension);

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> neighbor_lists_device(
      "old to new neighbor lists", fine_grid_particle_num,
      estimated_num_neighbors);
  Kokkos::View<int **>::HostMirror neighbor_lists =
      Kokkos::create_mirror_view(neighbor_lists_device);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> epsilon_device(
      "h supports", fine_grid_particle_num);
  Kokkos::View<double *>::HostMirror epsilon =
      Kokkos::create_mirror_view(epsilon_device);

  auto neighbor_needed = Compadre::GMLS::getNP(
      2, _dimension, DivergenceFreeVectorTaylorPolynomial);
  interpolation_point_search.generateNeighborListsFromKNNSearch(
      false, target_coord, neighbor_lists, epsilon, neighbor_needed, 1.2);

  Kokkos::deep_copy(neighbor_lists_device, neighbor_lists);
  Kokkos::deep_copy(epsilon_device, epsilon);

  auto pressure_basis = new GMLS(ScalarTaylorPolynomial, PointSample, 2,
                                 _dimension, "LU", "STANDARD");
  auto velocity_basis =
      new GMLS(DivergenceFreeVectorTaylorPolynomial, VectorPointSample, 2,
               _dimension, "SVD", "STANDARD");

  // pressure field
  pressure_basis->setProblemData(neighbor_lists_device, source_coord_device,
                                 target_coord_device, epsilon_device);

  pressure_basis->addTargets(ScalarPointEvaluation);

  pressure_basis->setWeightingType(WeightingFunctionType::Power);
  pressure_basis->setWeightingPower(4);
  pressure_basis->setOrderOfQuadraturePoints(2);
  pressure_basis->setDimensionOfQuadraturePoints(1);
  pressure_basis->setQuadratureType("LINE");

  pressure_basis->generateAlphas(1);

  auto pressure_alphas = pressure_basis->getAlphas();

  // old to new velocity field transition
  velocity_basis->setProblemData(neighbor_lists_device, source_coord_device,
                                 target_coord_device, epsilon_device);

  velocity_basis->addTargets(VectorPointEvaluation);

  velocity_basis->generateAlphas(1);

  auto velocity_alphas = velocity_basis->getAlphas();

  interpolation->resize(fine_grid_dof, coarse_grid_dof, coarse_grid_dof_global);

  vector<PetscInt> index;
  // compute matrix graph
  for (int i = 0; i < fine_grid_particle_num; i++) {
    // velocity interpolation
    index.resize(neighbor_lists(i, 0) * velocity_dof);
    for (int j = 0; j < neighbor_lists(i, 0); j++) {
      for (int k = 0; k < velocity_dof; k++) {
        index[j * velocity_dof + k] =
            field_dof *
                coarse_grid_list[neighbor_lists(i, j + 1)].global_index +
            k;
      }
    }

    for (int k = 0; k < velocity_dof; k++) {
      interpolation->setColIndex(field_dof * i + k, index);
    }

    // pressure interpolation
    index.resize(neighbor_lists(i, 0));
    for (int j = 0; j < neighbor_lists(i, 0); j++)
      index[j] =
          field_dof * coarse_grid_list[neighbor_lists(i, j + 1)].global_index +
          velocity_dof;
    interpolation->setColIndex(field_dof * i + velocity_dof, index);
  }

  // compute interpolation matrix entity
  const auto pressure_alphas_index =
      pressure_basis->getAlphaColumnOffset(ScalarPointEvaluation, 0, 0, 0, 0);
  vector<int> velocity_alphas_index(pow(_dimension, 2));
  for (int axes1 = 0; axes1 < _dimension; axes1++)
    for (int axes2 = 0; axes2 < _dimension; axes2++)
      velocity_alphas_index[axes1 * _dimension + axes2] =
          velocity_basis->getAlphaColumnOffset(VectorPointEvaluation, axes1, 0,
                                               axes2, 0);

  for (int i = 0; i < fine_grid_particle_num; i++) {
    // velocity interpolation
    for (int j = 0; j < neighbor_lists(i, 0); j++) {
      for (int axes1 = 0; axes1 < _dimension; axes1++)
        for (int axes2 = 0; axes2 < _dimension; axes2++)
          interpolation->increment(
              field_dof * i + axes1,
              field_dof *
                      coarse_grid_list[neighbor_lists(i, j + 1)].global_index +
                  axes2,
              velocity_alphas(
                  i, velocity_alphas_index[axes1 * _dimension + axes2], j));
    }

    // pressure interpolation
    for (int j = 0; j < neighbor_lists(i, 0); j++) {
      interpolation->increment(
          field_dof * i + velocity_dof,
          field_dof * coarse_grid_list[neighbor_lists(i, j + 1)].global_index +
              velocity_dof,
          pressure_alphas(i, pressure_alphas_index, j));
    }
  }

  interpolation->FinalAssemble();
}

void stokes_equation::build_restriction(
    std::shared_ptr<std::vector<particle>> coarse_grid_particle_set,
    std::shared_ptr<std::vector<particle>> fine_grid_particle_set,
    std::shared_ptr<sparse_matrix> restriction,
    std::shared_ptr<std::vector<std::vector<std::size_t>>> hierarchy) {
  int field_dof = _dimension + 1;
  int coarse_grid_particle_num = coarse_grid_particle_set->size();
  int fine_grid_particle_num = fine_grid_particle_set->size();

  int coarse_grid_dof = coarse_grid_particle_num * field_dof;
  int fine_grid_dof = fine_grid_particle_num * field_dof;

  int fine_grid_dof_global;
  MPI_Allreduce(&fine_grid_dof, &fine_grid_dof_global, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  vector<PetscInt> index;

  std::vector<std::vector<std::size_t>> &hierarchy_list = *hierarchy;
  std::vector<particle> &coarse_grid_list = *coarse_grid_particle_set;
  std::vector<particle> &fine_grid_list = *fine_grid_particle_set;

  restriction->resize(coarse_grid_dof, fine_grid_dof, fine_grid_dof_global);
  // compute restriction matrix graph
  for (int i = 0; i < coarse_grid_particle_num; i++) {
    index.resize(hierarchy_list[i].size());
    for (int j = 0; j < field_dof; j++) {
      for (int k = 0; k < hierarchy_list[i].size(); k++) {
        index[k] =
            fine_grid_list[hierarchy_list[i][k]].global_index * field_dof + j;
      }

      restriction->setColIndex(field_dof * i + j, index);
    }
  }

  for (int i = 0; i < coarse_grid_particle_num; i++) {
    for (int j = 0; j < field_dof; j++) {
      for (int k = 0; k < hierarchy_list[i].size(); k++) {
        restriction->increment(
            field_dof * i + j,
            fine_grid_list[hierarchy_list[i][k]].global_index * field_dof + j,
            1.0 / hierarchy_list[i].size());
      }
    }
  }

  restriction->FinalAssemble();
}

void stokes_equation::build_coarse_level_matrix() {
  size_t particle_set_num_layer = _geo->get_num_layer() - 1;

  _ff.resize(particle_set_num_layer);
  _x.resize(particle_set_num_layer);
  _y.resize(particle_set_num_layer);
  _r.resize(particle_set_num_layer);

  for (size_t i = 0; i < particle_set_num_layer; i++) {
    _ff[i] = make_shared<sparse_matrix>();
    build_matrix(_geo->get_particle_set(i),
                 _geo->get_background_particle_set(i), _ff[i]);

    _x[i] = make_shared<Vec>();
    _y[i] = make_shared<Vec>();
    _r[i] = make_shared<Vec>();
    MatCreateVecs(_ff[i]->__mat, _x[i].get(), NULL);
    MatCreateVecs(_ff[i]->__mat, _y[i].get(), NULL);
    MatCreateVecs(_ff[i]->__mat, _r[i].get(), NULL);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

void stokes_equation::build_interpolation_restriction_operators() {
  size_t particle_set_num_layer = _geo->get_num_layer() - 1;

  _interpolation.resize(particle_set_num_layer);
  _restriction.resize(particle_set_num_layer);

  for (size_t i = 0; i < particle_set_num_layer; i++) {
    _interpolation[i] = make_shared<sparse_matrix>();
    _restriction[i] = make_shared<sparse_matrix>();
    build_interpolation(_geo->get_particle_set(i),
                        _geo->get_particle_set(i + 1), _interpolation[i]);
    build_restriction(_geo->get_particle_set(i), _geo->get_particle_set(i + 1),
                      _restriction[i], _geo->get_hierarchy(i));
  }

  MPI_Barrier(MPI_COMM_WORLD);
}