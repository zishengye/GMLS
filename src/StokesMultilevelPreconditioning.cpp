#include "StokesMultilevelPreconditioning.hpp"
#include "StokesCompositePreconditioning.hpp"
#include "gmls_solver.hpp"

#include <algorithm>
#include <iostream>

using namespace std;
using namespace Compadre;

void StokesMultilevelPreconditioning::build_interpolation_restriction(
    const int _num_rigid_body, const int _dimension, const int _poly_order) {
  petsc_sparse_matrix &I = *(getI(current_refinement_level - 1));
  petsc_sparse_matrix &R = *(getR(current_refinement_level - 1));

  int fieldDof = dimension + 1;
  int velocityDof = dimension;
  int pressure_dof = 1;

  int rigidBodyDof = (dimension == 3) ? 6 : 3;

  int translation_dof = dimension;
  int rotation_dof = (dimension == 3) ? 3 : 1;

  double timer1, timer2;
  timer1 = MPI_Wtime();

  {
    auto &coord = *(geo_mgr->get_current_work_particle_coord());
    auto &new_added = *(geo_mgr->get_current_work_particle_new_added());
    auto &spacing = *(geo_mgr->get_current_work_particle_spacing());
    auto &local_idx = *(geo_mgr->get_current_work_particle_local_index());

    auto &old_coord = *(geo_mgr->get_last_work_particle_coord());

    auto &old_source_coord = *(geo_mgr->get_clll_particle_coord());
    auto &old_source_index = *(geo_mgr->get_clll_particle_index());

    int new_local_particle_num = coord.size();

    int new_global_particle_num;

    MPI_Allreduce(&new_local_particle_num, &new_global_particle_num, 1, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);

    int new_local_dof = fieldDof * new_local_particle_num;
    int new_global_dof = fieldDof * new_global_particle_num;

    int old_local_particle_num = old_coord.size();
    int old_global_particle_num;

    MPI_Allreduce(&old_local_particle_num, &old_global_particle_num, 1, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);

    int old_local_dof, old_global_dof;
    old_local_dof = old_local_particle_num * fieldDof;
    old_global_dof = old_global_particle_num * fieldDof;

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        old_source_coords_device("old source coordinates",
                                 old_source_coord.size(), 3);
    Kokkos::View<double **>::HostMirror old_source_coords_host =
        Kokkos::create_mirror_view(old_source_coords_device);

    int actual_new_target = 0;
    vector<int> new_actual_index(coord.size());
    for (int i = 0; i < coord.size(); i++) {
      new_actual_index[i] = actual_new_target;
      if (new_added[i] < 0)
        actual_new_target++;
    }

    int global_actual_new_target;
    int min_new_target, max_new_target;
    MPI_Allreduce(&actual_new_target, &global_actual_new_target, 1, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&actual_new_target, &min_new_target, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&actual_new_target, &max_new_target, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "Total new particle: %d\n",
                global_actual_new_target);
    PetscPrintf(PETSC_COMM_WORLD, "new particle imbalance: %f\n",
                (double)(max_new_target) / (double)(min_new_target));

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        new_target_coords_device("new target coordinates", actual_new_target,
                                 3);
    Kokkos::View<double **>::HostMirror new_target_coords_host =
        Kokkos::create_mirror_view(new_target_coords_device);

    // copy old source coords
    for (int i = 0; i < old_source_coord.size(); i++) {
      for (int j = 0; j < dimension; j++)
        old_source_coords_host(i, j) = old_source_coord[i][j];
    }

    // copy new target coords
    int counter = 0;
    for (int i = 0; i < coord.size(); i++) {
      if (new_added[i] < 0) {
        for (int j = 0; j < dimension; j++) {
          new_target_coords_host(counter, j) = coord[i][j];
        }

        counter++;
      }
    }

    Kokkos::deep_copy(old_source_coords_device, old_source_coords_host);
    Kokkos::deep_copy(new_target_coords_device, new_target_coords_host);

    auto old_to_new_point_search(
        CreatePointCloudSearch(old_source_coords_host, dimension));

    int estimated_num_neighbor_max =
        pow(2, dimension) * pow(2 * (_poly_order + 1.5), dimension);

    Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
        old_to_new_neighbor_lists_device("old to new neighbor lists",
                                         actual_new_target,
                                         estimated_num_neighbor_max);
    Kokkos::View<int **>::HostMirror old_to_new_neighbor_lists_host =
        Kokkos::create_mirror_view(old_to_new_neighbor_lists_device);

    Kokkos::View<double *, Kokkos::DefaultExecutionSpace> old_epsilon_device(
        "h supports", actual_new_target);
    Kokkos::View<double *>::HostMirror old_epsilon_host =
        Kokkos::create_mirror_view(old_epsilon_device);

    for (int i = 0; i < coord.size(); i++) {
      if (new_added[i] < 0) {
        old_epsilon_host[new_actual_index[i]] = spacing[i];
      }
    }

    auto neighbor_needed =
        2.0 * Compadre::GMLS::getNP(_poly_order, dimension,
                                    DivergenceFreeVectorTaylorPolynomial);
    size_t actual_neighbor_max;

    double sub_timer1, sub_timer2;
    sub_timer1 = MPI_Wtime();

    double max_epsilon = geo_mgr->get_old_cutoff_distance();
    int ite_counter = 0;
    int min_neighbor = 1000, max_neighbor = 0;
    while (true) {
      old_to_new_point_search.generate2DNeighborListsFromRadiusSearch(
          false, new_target_coords_host, old_to_new_neighbor_lists_host,
          old_epsilon_host, 0.0, 0.0);

      min_neighbor = 1000;
      max_neighbor = 0;
      int enough_neighbor = 0;
      for (int i = 0; i < coord.size(); i++) {
        if (new_added[i] < 0) {
          int num_neighbor =
              old_to_new_neighbor_lists_host(new_actual_index[i], 0);
          if (num_neighbor <= neighbor_needed) {
            if ((old_epsilon_host[new_actual_index[i]] + 0.25 * spacing[i]) <
                max_epsilon) {
              old_epsilon_host[new_actual_index[i]] += 0.25 * spacing[i];
              enough_neighbor = 1;
            }
          }
          if (min_neighbor > num_neighbor)
            min_neighbor = num_neighbor;
          if (max_neighbor < num_neighbor)
            max_neighbor = num_neighbor;
        }
      }

      MPI_Allreduce(MPI_IN_PLACE, &min_neighbor, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &max_neighbor, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD);

      MPI_Allreduce(MPI_IN_PLACE, &enough_neighbor, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);

      if (enough_neighbor == 0)
        break;

      ite_counter++;
    }

    sub_timer2 = MPI_Wtime();

    PetscPrintf(PETSC_COMM_WORLD,
                "iteration count: %d, min neighbor: %d, max neighbor: %d, time "
                "duration: %fs\n",
                ite_counter, min_neighbor, max_neighbor,
                sub_timer2 - sub_timer1);

    Kokkos::deep_copy(old_to_new_neighbor_lists_device,
                      old_to_new_neighbor_lists_host);
    Kokkos::deep_copy(old_epsilon_device, old_epsilon_host);

    GMLS old_to_new_pressure_basis(ScalarTaylorPolynomial, PointSample,
                                   _poly_order, dimension, "LU", "STANDARD");
    GMLS old_to_new_velocity_basis(DivergenceFreeVectorTaylorPolynomial,
                                   VectorPointSample, _poly_order, dimension,
                                   "LU", "STANDARD");

    // old to new pressure field transition
    old_to_new_pressure_basis.setProblemData(
        old_to_new_neighbor_lists_device, old_source_coords_device,
        new_target_coords_device, old_epsilon_device);

    old_to_new_pressure_basis.addTargets(ScalarPointEvaluation);

    old_to_new_pressure_basis.setWeightingType(WeightingFunctionType::Power);
    old_to_new_pressure_basis.setWeightingParameter(4);

    // ensure each batch contains less than 200 particles
    int num_of_batches = actual_new_target / 100 + 1;
    old_to_new_pressure_basis.generateAlphas(num_of_batches);

    auto old_to_new_pressure_solution_set =
        old_to_new_pressure_basis.getSolutionSetHost();
    auto old_to_new_pressure_alphas =
        old_to_new_pressure_solution_set->getAlphas();

    // old to new velocity field transition
    old_to_new_velocity_basis.setProblemData(
        old_to_new_neighbor_lists_device, old_source_coords_device,
        new_target_coords_device, old_epsilon_device);

    old_to_new_velocity_basis.addTargets(VectorPointEvaluation);

    old_to_new_velocity_basis.setWeightingType(WeightingFunctionType::Power);
    old_to_new_velocity_basis.setWeightingParameter(4);

    old_to_new_velocity_basis.generateAlphas(num_of_batches);

    auto old_to_new_velocity_solution_set =
        old_to_new_velocity_basis.getSolutionSetHost();
    auto old_to_new_velocity_alphas =
        old_to_new_velocity_solution_set->getAlphas();

    // old to new interpolation matrix
    if (mpi_rank == mpi_size - 1)
      I.resize(new_local_dof + rigidBodyDof * num_rigid_body,
               old_local_dof + rigidBodyDof * num_rigid_body,
               old_global_dof + rigidBodyDof * num_rigid_body);
    else
      I.resize(new_local_dof, old_local_dof,
               old_global_dof + rigidBodyDof * num_rigid_body);
    // compute matrix graph
    vector<PetscInt> index;
    for (int i = 0; i < new_local_particle_num; i++) {
      int current_particle_local_index = local_idx[i];
      if (new_added[i] < 0) {
        // velocity interpolation
        index.resize(old_to_new_neighbor_lists_host(new_actual_index[i], 0) *
                     velocityDof);
        for (int j = 0;
             j < old_to_new_neighbor_lists_host(new_actual_index[i], 0); j++) {
          for (int k = 0; k < velocityDof; k++) {
            index[j * velocityDof + k] =
                fieldDof * old_source_index[old_to_new_neighbor_lists_host(
                               new_actual_index[i], j + 1)] +
                k;
          }
        }

        for (int k = 0; k < velocityDof; k++) {
          I.set_col_index(fieldDof * current_particle_local_index + k, index);
        }

        // pressure interpolation
        index.resize(old_to_new_neighbor_lists_host(new_actual_index[i], 0));
        for (int j = 0;
             j < old_to_new_neighbor_lists_host(new_actual_index[i], 0); j++) {
          index[j] = fieldDof * old_source_index[old_to_new_neighbor_lists_host(
                                    new_actual_index[i], j + 1)] +
                     velocityDof;
        }
        I.set_col_index(fieldDof * current_particle_local_index + velocityDof,
                        index);
      } else {
        index.resize(1);
        for (int j = 0; j < fieldDof; j++) {
          index[0] = fieldDof * new_added[i] + j;
          I.set_col_index(fieldDof * current_particle_local_index + j, index);
        }
      }
    }

    // rigid body
    if (mpi_rank == mpi_size - 1) {
      index.resize(1);
      for (int i = 0; i < num_rigid_body; i++) {
        int local_rigid_body_index_offset =
            fieldDof * new_local_particle_num + i * rigidBodyDof;
        for (int j = 0; j < rigidBodyDof; j++) {
          index[0] = fieldDof * old_global_particle_num + i * rigidBodyDof + j;
          I.set_col_index(local_rigid_body_index_offset + j, index);
        }
      }
    }

    // compute interpolation matrix entity
    const auto pressure_old_to_new_alphas_index =
        old_to_new_pressure_solution_set->getAlphaColumnOffset(
            ScalarPointEvaluation, 0, 0, 0, 0);
    vector<int> velocity_old_to_new_alphas_index(pow(dimension, 2));
    for (int axes1 = 0; axes1 < dimension; axes1++)
      for (int axes2 = 0; axes2 < dimension; axes2++)
        velocity_old_to_new_alphas_index[axes1 * dimension + axes2] =
            old_to_new_velocity_solution_set->getAlphaColumnOffset(
                VectorPointEvaluation, axes1, 0, axes2, 0);

    for (int i = 0; i < new_local_particle_num; i++) {
      int current_particle_local_index = local_idx[i];
      if (new_added[i] < 0) {
        for (int j = 0;
             j < old_to_new_neighbor_lists_host(new_actual_index[i], 0); j++) {
          for (int axes1 = 0; axes1 < dimension; axes1++)
            for (int axes2 = 0; axes2 < dimension; axes2++) {
              auto alpha_index =
                  old_to_new_velocity_solution_set->getAlphaIndex(
                      new_actual_index[i],
                      velocity_old_to_new_alphas_index[axes1 * dimension +
                                                       axes2]);
              int neighbor_index =
                  old_source_index[old_to_new_neighbor_lists_host(
                      new_actual_index[i], j + 1)];
              I.increment(fieldDof * current_particle_local_index + axes1,
                          fieldDof * neighbor_index + axes2,
                          old_to_new_velocity_alphas(alpha_index + j));
            }
        }

        for (int j = 0;
             j < old_to_new_neighbor_lists_host(new_actual_index[i], 0); j++) {
          auto alpha_index = old_to_new_pressure_solution_set->getAlphaIndex(
              new_actual_index[i], pressure_old_to_new_alphas_index);
          int neighbor_index = old_source_index[old_to_new_neighbor_lists_host(
              new_actual_index[i], j + 1)];
          I.increment(fieldDof * current_particle_local_index + velocityDof,
                      fieldDof * neighbor_index + velocityDof,
                      old_to_new_pressure_alphas(alpha_index + j));
        }
      } else {
        for (int j = 0; j < fieldDof; j++) {
          I.increment(fieldDof * current_particle_local_index + j,
                      fieldDof * new_added[i] + j, 1.0);
        }
      }
    }

    // rigid body
    if (mpi_rank == mpi_size - 1) {
      for (int i = 0; i < num_rigid_body; i++) {
        int local_rigid_body_index_offset =
            fieldDof * new_local_particle_num + i * rigidBodyDof;
        for (int j = 0; j < rigidBodyDof; j++) {
          I.increment(local_rigid_body_index_offset + j,
                      fieldDof * old_global_particle_num + i * rigidBodyDof + j,
                      1.0);
        }
      }
    }

    I.assemble();
  }

  timer2 = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "Interpolation matrix building duration: %fs\n",
              timer2 - timer1);

  {
    auto &coord = *(geo_mgr->get_current_work_particle_coord());

    auto &source_coord = *(geo_mgr->get_llcl_particle_coord());
    auto &source_index = *(geo_mgr->get_llcl_particle_index());
    auto &source_particle_type = *(geo_mgr->get_llcl_particle_type());

    auto &old_coord = *(geo_mgr->get_last_work_particle_coord());
    auto &old_index = *(geo_mgr->get_last_work_particle_index());
    auto &old_local_index = *(geo_mgr->get_last_work_particle_local_index());
    auto &old_particle_type = *(geo_mgr->get_last_work_particle_type());
    auto &old_spacing = *(geo_mgr->get_last_work_particle_spacing());

    int old_local_particle_num = old_coord.size();
    int old_global_particle_num;

    MPI_Allreduce(&old_local_particle_num, &old_global_particle_num, 1, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);

    int old_local_dof = fieldDof * old_local_particle_num;
    int old_global_dof = fieldDof * old_global_particle_num;

    int new_local_particle_num = coord.size();

    int new_global_particle_num;

    MPI_Allreduce(&new_local_particle_num, &new_global_particle_num, 1, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);

    int new_local_dof = fieldDof * new_local_particle_num;
    int new_global_dof = fieldDof * new_global_particle_num;

    // new to old restriction matrix
    if (mpi_rank == mpi_size - 1)
      R.resize(old_local_dof + num_rigid_body * rigidBodyDof,
               new_local_dof + num_rigid_body * rigidBodyDof,
               new_global_dof + num_rigid_body * rigidBodyDof);
    else
      R.resize(old_local_dof, new_local_dof,
               new_global_dof + num_rigid_body * rigidBodyDof);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace> source_coords_device(
        "old source coordinates", source_coord.size(), 3);
    Kokkos::View<double **>::HostMirror source_coords_host =
        Kokkos::create_mirror_view(source_coords_device);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace> target_coords_device(
        "new target coordinates", old_coord.size(), 3);
    Kokkos::View<double **>::HostMirror target_coords_host =
        Kokkos::create_mirror_view(target_coords_device);

    // copy old source coords
    for (int i = 0; i < source_coord.size(); i++) {
      for (int j = 0; j < dimension; j++)
        source_coords_host(i, j) = source_coord[i][j];
    }

    // copy new target coords
    for (int i = 0; i < old_coord.size(); i++) {
      for (int j = 0; j < dimension; j++) {
        target_coords_host(i, j) = old_coord[i][j];
      }
    }

    Kokkos::deep_copy(target_coords_device, target_coords_host);
    Kokkos::deep_copy(source_coords_device, source_coords_host);

    auto point_search(CreatePointCloudSearch(source_coords_host, dimension));

    int estimated_num_neighbor_max =
        pow(2, dimension) * pow(2 * 3.0, dimension);

    Kokkos::View<int **, Kokkos::DefaultExecutionSpace> neighbor_lists_device(
        "old to new neighbor lists", old_coord.size(),
        estimated_num_neighbor_max);
    Kokkos::View<int **>::HostMirror neighbor_lists_host =
        Kokkos::create_mirror_view(neighbor_lists_device);

    Kokkos::View<double *, Kokkos::DefaultExecutionSpace> epsilon_device(
        "h supports", old_coord.size());
    Kokkos::View<double *>::HostMirror epsilon_host =
        Kokkos::create_mirror_view(epsilon_device);

    for (int i = 0; i < old_coord.size(); i++) {
      epsilon_host(i) = 0.25 * sqrt(dimension) * old_spacing[i] + 1e-15;
    }

    point_search.generate2DNeighborListsFromRadiusSearch(
        false, target_coords_host, neighbor_lists_host, epsilon_host, 0.0, 0.0);

    for (int i = 0; i < old_coord.size(); i++) {
      bool is_boundary = (old_particle_type[i] == 0) ? false : true;
      vector<int> index;
      int corresponding_index = -1;
      for (int j = 0; j < neighbor_lists_host(i, 0); j++) {
        int neighbor_index = neighbor_lists_host(i, j + 1);
        bool is_neighbor_boundary =
            (source_particle_type[neighbor_index] == 0) ? false : true;
        if (is_boundary == is_neighbor_boundary)
          index.push_back(neighbor_index);
        Vec3 dX = source_coord[neighbor_index] - old_coord[i];
        if (dX.mag() < 1e-15) {
          corresponding_index = neighbor_index;
        }
      }
      if (corresponding_index == -1) {
        neighbor_lists_host(i, 0) = index.size();
        for (int j = 0; j < index.size(); j++) {
          neighbor_lists_host(i, j + 1) = index[j];
        }
      } else {
        neighbor_lists_host(i, 0) = 1;
        neighbor_lists_host(i, 1) = corresponding_index;
      }
    }

    Kokkos::deep_copy(neighbor_lists_device, neighbor_lists_host);
    Kokkos::deep_copy(epsilon_device, epsilon_host);

    // compute restriction matrix graph
    vector<PetscInt> index;
    int min_neighbor = 1000, max_neighbor = 0;
    for (int i = 0; i < old_local_particle_num; i++) {
      int current_particle_local_index = old_local_index[i];
      index.resize(neighbor_lists_host(i, 0));
      if (min_neighbor > neighbor_lists_host(i, 0))
        min_neighbor = neighbor_lists_host(i, 0);
      if (max_neighbor < neighbor_lists_host(i, 0))
        max_neighbor = neighbor_lists_host(i, 0);
      for (int j = 0; j < fieldDof; j++) {
        for (int k = 0; k < neighbor_lists_host(i, 0); k++) {
          int neighbor_index = neighbor_lists_host(i, k + 1);
          index[k] = source_index[neighbor_index] * fieldDof + j;
        }

        R.set_col_index(fieldDof * current_particle_local_index + j, index);
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, &min_neighbor, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_neighbor, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "min neighbor: %d, max neighbor: %d\n",
                min_neighbor, max_neighbor);

    // rigid body
    if (mpi_rank == mpi_size - 1) {
      index.resize(1);
      for (int i = 0; i < num_rigid_body; i++) {
        int local_rigid_body_index_offset =
            fieldDof * old_local_particle_num + i * rigidBodyDof;
        for (int j = 0; j < rigidBodyDof; j++) {
          index[0] = fieldDof * new_global_particle_num + i * rigidBodyDof + j;
          R.set_col_index(local_rigid_body_index_offset + j, index);
        }
      }
    }

    for (int i = 0; i < old_local_particle_num; i++) {
      int current_particle_local_index = old_local_index[i];
      for (int j = 0; j < fieldDof; j++) {
        for (int k = 0; k < neighbor_lists_host(i, 0); k++) {
          int neighbor_index = neighbor_lists_host(i, k + 1);
          R.increment(fieldDof * current_particle_local_index + j,
                      source_index[neighbor_index] * fieldDof + j,
                      1.0 / neighbor_lists_host(i, 0));
        }
      }
    }

    // rigid body
    if (mpi_rank == mpi_size - 1) {
      index.resize(1);
      for (int i = 0; i < num_rigid_body; i++) {
        int local_rigid_body_index_offset =
            fieldDof * old_local_particle_num + i * rigidBodyDof;
        for (int j = 0; j < rigidBodyDof; j++) {
          R.increment(
              local_rigid_body_index_offset + j,
              (fieldDof * new_global_particle_num + i * rigidBodyDof + j), 1.0);
        }
      }
    }

    R.assemble();
  }
}

void StokesMultilevelPreconditioning::initial_guess_from_previous_adaptive_step(
    std::vector<double> &initial_guess, std::vector<Vec3> &velocity,
    std::vector<double> &pressure, std::vector<Vec3> &rb_velocity,
    std::vector<Vec3> &rb_angular_velocity) {
  auto &local_idx = *(geo_mgr->get_last_work_particle_local_index());

  petsc_sparse_matrix &I = *(getI(current_refinement_level - 1));
  petsc_sparse_matrix &R = *(getR(current_refinement_level - 1));
  Vec x1, x2;
  MatCreateVecs(I.GetReference(), &x2, &x1);

  const int old_local_particle_num = pressure.size();

  const int fieldDof = dimension + 1;
  const int velocityDof = dimension;
  const int rigidBodyDof = (dimension == 3) ? 6 : 3;

  PetscReal *a;
  VecGetArray(x2, &a);

  for (int i = 0; i < old_local_particle_num; i++) {
    int current_particle_local_index = local_idx[i];
    for (int j = 0; j < velocityDof; j++) {
      a[current_particle_local_index * fieldDof + j] = velocity[i][j];
    }
    a[current_particle_local_index * fieldDof + velocityDof] = pressure[i];
  }

  if (mpi_rank == mpi_size - 1) {
    const int old_local_rigid_body_offset = old_local_particle_num * fieldDof;

    const int rb_velocity_dof = dimension;
    const int rb_angular_velocity_dof = (dimension == 3) ? 3 : 1;

    for (int i = 0; i < num_rigid_body; i++) {
      for (int j = 0; j < rb_velocity_dof; j++) {
        a[old_local_rigid_body_offset + i * rigidBodyDof + j] =
            rb_velocity[i][j];
      }
      for (int j = 0; j < rb_angular_velocity_dof; j++) {
        a[old_local_rigid_body_offset + i * rigidBodyDof + rb_velocity_dof +
          j] = rb_angular_velocity[i][j];
      }
    }
  }

  VecRestoreArray(x2, &a);

  MatMult(I.GetReference(), x2, x1);

  VecGetArray(x1, &a);
  for (int i = 0; i < initial_guess.size(); i++) {
    initial_guess[i] = a[i];
  }
  VecRestoreArray(x1, &a);

  auto &coord = *(geo_mgr->get_current_work_particle_coord());
  int local_particle_num = coord.size();
  int global_particle_num;
  MPI_Allreduce(&local_particle_num, &global_particle_num, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  double pressure_sum = 0.0;
  for (int i = 0; i < local_particle_num; i++) {
    pressure_sum += initial_guess[i * fieldDof + velocityDof];
  }

  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  double average_pressure = pressure_sum / global_particle_num;
  for (int i = 0; i < local_particle_num; i++) {
    initial_guess[i * fieldDof + velocityDof] -= average_pressure;
  }

  VecDestroy(&x1);
  VecDestroy(&x2);
}

int StokesMultilevelPreconditioning::Solve(std::vector<double> &rhs0,
                                           std::vector<double> &x0,
                                           std::vector<double> &rhs1,
                                           std::vector<double> &x1) {
  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "\nstart of linear system solving setup\n");

  int hasRigidBody = rhs1.size();
  MPI_Allreduce(MPI_IN_PLACE, &hasRigidBody, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  int refinementStep = A_list.size() - 1;

  int fieldDof = dimension + 1;
  int velocityDof = dimension;
  int rigidBodyDof = (dimension == 3) ? 6 : 3;

  unsigned int numLocalParticle = rhs0.size() / fieldDof;
  unsigned int numGlobalParticle;
  MPI_Allreduce(&numLocalParticle, &numGlobalParticle, 1, MPI_UNSIGNED, MPI_SUM,
                MPI_COMM_WORLD);

  local_particle_num_list.push_back(numLocalParticle);
  global_particle_num_list.push_back(numGlobalParticle);

  // setup preconditioner for base level
  if (refinementStep == 0) {
    PC pcFieldBase;

    Mat &ff = A_list[refinementStep]->GetMatrix(0, 0)->GetReference();

    KSPCreate(PETSC_COMM_WORLD, &ksp_field_base->GetReference());
    KSPSetOperators(ksp_field_base->GetReference(), ff, ff);
    if (dimension == 3) {
      KSPSetType(ksp_field_base->GetReference(), KSPRICHARDSON);
      KSPSetTolerances(ksp_field_base->GetReference(), 1e-2, 1e-50, 1e50, 5);
      KSPSetResidualHistory(ksp_field_base->GetReference(), NULL, 500,
                            PETSC_TRUE);

      KSPGetPC(ksp_field_base->GetReference(), &pcFieldBase);
      PCSetType(pcFieldBase, PCSOR);
    } else {
      KSPSetType(ksp_field_base->GetReference(), KSPPREONLY);

      KSPGetPC(ksp_field_base->GetReference(), &pcFieldBase);
      PCSetType(pcFieldBase, PCLU);
      PCFactorSetMatSolverType(pcFieldBase, MATSOLVERMUMPS);
    }
    PCSetFromOptions(pcFieldBase);
    PCSetUp(pcFieldBase);
    KSPSetUp(ksp_field_base->GetReference());

    if (hasRigidBody != 0) {
      Mat &nn = A_list[refinementStep]->GetNeighborNeighborMatrix();

      KSPCreate(MPI_COMM_WORLD, &ksp_colloid_base->GetReference());

      KSPSetType(ksp_colloid_base->GetReference(), KSPGMRES);
      KSPGMRESSetRestart(ksp_colloid_base->GetReference(), 100);
      KSPSetTolerances(ksp_colloid_base->GetReference(), 1e-3, 1e-50, 1e50,
                       500);
      KSPSetOperators(ksp_colloid_base->GetReference(), nn, nn);

      PC pcNeighborBase;
      KSPGetPC(ksp_colloid_base->GetReference(), &pcNeighborBase);
      PCSetType(pcNeighborBase, PCFIELDSPLIT);

      IS isg[2];
      MatNestGetISs(nn, isg, PETSC_NULL);

      PCFieldSplitSetIS(pcNeighborBase, "0", isg[0]);
      PCFieldSplitSetIS(pcNeighborBase, "1", isg[1]);

      Mat &S = A_list[refinementStep]->GetNeighborSchurMatrix();
      Mat &sub00 = A_list[refinementStep]->GetNeighborNeighborSubMatrix(0, 0);

      PCFieldSplitSetSchurPre(pcNeighborBase, PC_FIELDSPLIT_SCHUR_PRE_USER, S);
      PCSetFromOptions(pcNeighborBase);
      PCSetUp(pcNeighborBase);

      KSP *fieldsplit_sub_ksp;
      PetscInt n;
      PCFieldSplitGetSubKSP(pcNeighborBase, &n, &fieldsplit_sub_ksp);
      KSPSetOperators(fieldsplit_sub_ksp[1], S, S);
      KSPSetOperators(fieldsplit_sub_ksp[0], sub00, sub00);
      KSPSetUp(fieldsplit_sub_ksp[0]);
      KSPSetUp(fieldsplit_sub_ksp[1]);
      PetscFree(fieldsplit_sub_ksp);
    }
  } else {
    Mat &ff = A_list[refinementStep]->GetMatrix(0, 0)->GetReference();

    // setup relaxation on field for current level
    KSPCreate(MPI_COMM_WORLD,
              field_relaxation_list[refinementStep]->GetPointer());

    KSPSetType(field_relaxation_list[refinementStep]->GetReference(),
               KSPRICHARDSON);
    // KSPGMRESSetRestart(field_relaxation_list[refinementStep]->GetReference(),
    //                    100);
    KSPSetOperators(field_relaxation_list[refinementStep]->GetReference(), ff,
                    ff);
    KSPSetTolerances(field_relaxation_list[refinementStep]->GetReference(),
                     5e-1, 1e-50, 1e10, 1);

    PC field_relaxation_pc;
    KSPGetPC(field_relaxation_list[refinementStep]->GetReference(),
             &field_relaxation_pc);
    PCSetType(field_relaxation_pc, PCSOR);
    PCSetFromOptions(field_relaxation_pc);
    PCSetUp(field_relaxation_pc);

    KSPSetUp(field_relaxation_list[refinementStep]->GetReference());

    if (hasRigidBody != 0) {
      Mat &nn = A_list[refinementStep]->GetNeighborNeighborMatrix();
      // setup relaxation on neighbor for current level
      KSPCreate(MPI_COMM_WORLD,
                colloid_relaxation_list[refinementStep]->GetPointer());

      KSPSetType(colloid_relaxation_list[refinementStep]->GetReference(),
                 KSPGMRES);
      KSPGMRESSetRestart(
          colloid_relaxation_list[refinementStep]->GetReference(), 100);
      KSPSetTolerances(colloid_relaxation_list[refinementStep]->GetReference(),
                       1e-2, 1e-50, 1e10, 500);
      KSPSetOperators(colloid_relaxation_list[refinementStep]->GetReference(),
                      nn, nn);

      PC neighbor_relaxation_pc;
      KSPGetPC(colloid_relaxation_list[refinementStep]->GetReference(),
               &neighbor_relaxation_pc);
      PCSetType(neighbor_relaxation_pc, PCFIELDSPLIT);

      IS isg[2];
      MatNestGetISs(nn, isg, PETSC_NULL);

      PCFieldSplitSetIS(neighbor_relaxation_pc, "0", isg[0]);
      PCFieldSplitSetIS(neighbor_relaxation_pc, "1", isg[1]);

      Mat &S = A_list[refinementStep]->GetNeighborSchurMatrix();
      Mat &sub00 = A_list[refinementStep]->GetNeighborNeighborSubMatrix(0, 0);

      PCFieldSplitSetSchurPre(neighbor_relaxation_pc,
                              PC_FIELDSPLIT_SCHUR_PRE_USER, S);
      PCSetFromOptions(neighbor_relaxation_pc);
      PCSetUp(neighbor_relaxation_pc);

      KSP *fieldsplit_sub_ksp;
      PetscInt n;
      PCFieldSplitGetSubKSP(neighbor_relaxation_pc, &n, &fieldsplit_sub_ksp);
      KSPSetOperators(fieldsplit_sub_ksp[1], S, S);
      KSPSetOperators(fieldsplit_sub_ksp[0], sub00, sub00);
      KSPSetFromOptions(fieldsplit_sub_ksp[0]);
      KSPSetUp(fieldsplit_sub_ksp[0]);
      KSPSetUp(fieldsplit_sub_ksp[1]);
      PetscFree(fieldsplit_sub_ksp);
    }
  }

  Mat &matA = A_list[refinementStep]->GetReference();

  PetscNestedVec rhs(2), x(2);
  rhs.Create(0, rhs0);
  rhs.Create(1, rhs1);
  rhs.Create();

  x.Create(0, x0);
  x.Create(1, x1);
  x.Create();

  x_list.push_back(std::make_shared<PetscNestedVec>(2));
  b_list.push_back(std::make_shared<PetscNestedVec>(2));
  r_list.push_back(std::make_shared<PetscNestedVec>(2));
  t_list.push_back(std::make_shared<PetscNestedVec>(2));

  x_list[refinementStep]->Duplicate(x);
  b_list[refinementStep]->Duplicate(x);
  r_list[refinementStep]->Duplicate(x);
  t_list[refinementStep]->Duplicate(x);

  x_field_list.push_back(std::make_shared<PetscVector>());
  y_field_list.push_back(std::make_shared<PetscVector>());
  b_field_list.push_back(std::make_shared<PetscVector>());
  r_field_list.push_back(std::make_shared<PetscVector>());

  VecDuplicate(x.GetSubVector(0),
               &(x_field_list[refinementStep]->GetReference()));
  VecDuplicate(x.GetSubVector(0),
               &(y_field_list[refinementStep]->GetReference()));
  VecDuplicate(x.GetSubVector(0),
               &(b_field_list[refinementStep]->GetReference()));
  VecDuplicate(x.GetSubVector(0),
               &(r_field_list[refinementStep]->GetReference()));

  KSP ksp;
  KSPCreate(PETSC_COMM_WORLD, &ksp);

  PC pc;

  HypreLUShellPC *shellCtx;

  KSPGetPC(ksp, &pc);
  KSPSetOperators(ksp, matA, matA);
  KSPSetFromOptions(ksp);

  PCSetType(pc, PCSHELL);

  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "start of stokes_multilevel preconditioner setup\n");

  HypreLUShellPCCreate(&shellCtx);
  PCShellSetApply(pc, HypreLUShellPCApplyAdaptive);
  PCShellSetContext(pc, shellCtx);
  PCShellSetDestroy(pc, HypreLUShellPCDestroy);

  HypreLUShellPCSetUp(pc, this, x.GetReference(), numLocalParticle, fieldDof,
                      hasRigidBody);

  PetscPrintf(PETSC_COMM_WORLD, "final solving of linear system\n");
  PetscReal residualNorm, rhsNorm;

  VecNorm(rhs.GetReference(), NORM_2, &rhsNorm);

  Vec residual;
  VecDuplicate(rhs.GetReference(), &residual);
  MatMult(matA, x.GetReference(), residual);
  VecAXPY(residual, -1.0, rhs.GetReference());
  VecNorm(residual, NORM_2, &residualNorm);
  PetscPrintf(PETSC_COMM_WORLD, "relative residual norm: %f\n",
              residualNorm / rhsNorm);

  KSPSolve(ksp, rhs.GetReference(), x.GetReference());

  MatMult(matA, x.GetReference(), residual);
  VecAXPY(residual, -1.0, rhs.GetReference());
  VecNorm(residual, NORM_2, &residualNorm);
  PetscPrintf(PETSC_COMM_WORLD, "relative residual norm: %f\n",
              residualNorm / rhsNorm);

  VecDestroy(&residual);

  KSPDestroy(&ksp);

  x.Copy(0, x0);
  x.Copy(1, x1);

  PetscLogDouble maxMem, mem;
  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(&mem, &maxMem, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "Current memory usage %.2f GB, maximum memory usage: %.2f GB\n",
              mem / 1e9, maxMem / 1e9);

  return 0;
}

void StokesMultilevelPreconditioning::clear() {
  MPI_Barrier(MPI_COMM_WORLD);

  base_level_initialized = false;

  A_list.clear();
  I_list.clear();
  R_list.clear();

  ff_list.clear();
  nn_list.clear();
  nw_list.clear();
  pp_list.clear();
  pw_list.clear();

  isg_field_list.clear();
  isg_colloid_list.clear();
  isg_pressure_list.clear();

  x_list.clear();
  b_list.clear();
  r_list.clear();
  t_list.clear();

  x_field_list.clear();
  y_field_list.clear();
  b_field_list.clear();
  r_field_list.clear();

  field_relaxation_list.clear();
  colloid_relaxation_list.clear();

  if (base_level_initialized) {
    ksp_field_base.reset();
    ksp_colloid_base.reset();
  }

  local_particle_num_list.clear();
  global_particle_num_list.clear();

  current_refinement_level = -1;
}