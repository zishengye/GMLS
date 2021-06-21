#include "stokes_multilevel.hpp"
#include "gmls_solver.hpp"
#include "stokes_composite_preconditioner.hpp"

#include <algorithm>
#include <iostream>

using namespace std;
using namespace Compadre;

void stokes_multilevel::build_interpolation_restriction(
    const int _num_rigid_body, const int _dimension, const int _poly_order) {
  petsc_sparse_matrix &I = *(getI(current_refinement_level - 1));
  petsc_sparse_matrix &R = *(getR(current_refinement_level - 1));

  int field_dof = dimension + 1;
  int velocity_dof = dimension;
  int pressure_dof = 1;

  int rigid_body_dof = (dimension == 3) ? 6 : 3;

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

    int new_local_dof = field_dof * new_local_particle_num;
    int new_global_dof = field_dof * new_global_particle_num;

    int old_local_particle_num = old_coord.size();
    int old_global_particle_num;

    MPI_Allreduce(&old_local_particle_num, &old_global_particle_num, 1, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);

    int old_local_dof, old_global_dof;
    old_local_dof = old_local_particle_num * field_dof;
    old_global_dof = old_global_particle_num * field_dof;

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
    old_to_new_pressure_basis.setWeightingPower(4);

    // ensure each batch contains less than 200 particles
    int num_of_batches = actual_new_target / 100 + 1;
    old_to_new_pressure_basis.generateAlphas(num_of_batches);

    auto old_to_new_pressure_alphas = old_to_new_pressure_basis.getAlphas();

    // old to new velocity field transition
    old_to_new_velocity_basis.setProblemData(
        old_to_new_neighbor_lists_device, old_source_coords_device,
        new_target_coords_device, old_epsilon_device);

    old_to_new_velocity_basis.addTargets(VectorPointEvaluation);

    old_to_new_velocity_basis.setWeightingType(WeightingFunctionType::Power);
    old_to_new_velocity_basis.setWeightingPower(4);

    old_to_new_velocity_basis.generateAlphas(num_of_batches);

    auto old_to_new_velocity_alphas = old_to_new_velocity_basis.getAlphas();

    // old to new interpolation matrix
    if (mpi_rank == mpi_size - 1)
      I.resize(new_local_dof + rigid_body_dof * num_rigid_body,
               old_local_dof + rigid_body_dof * num_rigid_body,
               old_global_dof + rigid_body_dof * num_rigid_body);
    else
      I.resize(new_local_dof, old_local_dof,
               old_global_dof + rigid_body_dof * num_rigid_body);
    // compute matrix graph
    vector<PetscInt> index;
    for (int i = 0; i < new_local_particle_num; i++) {
      int current_particle_local_index = local_idx[i];
      if (new_added[i] < 0) {
        // velocity interpolation
        index.resize(old_to_new_neighbor_lists_host(new_actual_index[i], 0) *
                     velocity_dof);
        for (int j = 0;
             j < old_to_new_neighbor_lists_host(new_actual_index[i], 0); j++) {
          for (int k = 0; k < velocity_dof; k++) {
            index[j * velocity_dof + k] =
                field_dof * old_source_index[old_to_new_neighbor_lists_host(
                                new_actual_index[i], j + 1)] +
                k;
          }
        }

        for (int k = 0; k < velocity_dof; k++) {
          I.set_col_index(field_dof * current_particle_local_index + k, index);
        }

        // pressure interpolation
        index.resize(old_to_new_neighbor_lists_host(new_actual_index[i], 0));
        for (int j = 0;
             j < old_to_new_neighbor_lists_host(new_actual_index[i], 0); j++) {
          index[j] =
              field_dof * old_source_index[old_to_new_neighbor_lists_host(
                              new_actual_index[i], j + 1)] +
              velocity_dof;
        }
        I.set_col_index(field_dof * current_particle_local_index + velocity_dof,
                        index);
      } else {
        index.resize(1);
        for (int j = 0; j < field_dof; j++) {
          index[0] = field_dof * new_added[i] + j;
          I.set_col_index(field_dof * current_particle_local_index + j, index);
        }
      }
    }

    // rigid body
    if (mpi_rank == mpi_size - 1) {
      index.resize(1);
      for (int i = 0; i < num_rigid_body; i++) {
        int local_rigid_body_index_offset =
            field_dof * new_local_particle_num + i * rigid_body_dof;
        for (int j = 0; j < rigid_body_dof; j++) {
          index[0] =
              field_dof * old_global_particle_num + i * rigid_body_dof + j;
          I.set_col_index(local_rigid_body_index_offset + j, index);
        }
      }
    }

    // compute interpolation matrix entity
    const auto pressure_old_to_new_alphas_index =
        old_to_new_pressure_basis.getAlphaColumnOffset(ScalarPointEvaluation, 0,
                                                       0, 0, 0);
    vector<int> velocity_old_to_new_alphas_index(pow(dimension, 2));
    for (int axes1 = 0; axes1 < dimension; axes1++)
      for (int axes2 = 0; axes2 < dimension; axes2++)
        velocity_old_to_new_alphas_index[axes1 * dimension + axes2] =
            old_to_new_velocity_basis.getAlphaColumnOffset(
                VectorPointEvaluation, axes1, 0, axes2, 0);

    for (int i = 0; i < new_local_particle_num; i++) {
      int current_particle_local_index = local_idx[i];
      if (new_added[i] < 0) {
        for (int j = 0;
             j < old_to_new_neighbor_lists_host(new_actual_index[i], 0); j++) {
          for (int axes1 = 0; axes1 < dimension; axes1++)
            for (int axes2 = 0; axes2 < dimension; axes2++) {
              auto alpha_index = old_to_new_velocity_basis.getAlphaIndexHost(
                  new_actual_index[i],
                  velocity_old_to_new_alphas_index[axes1 * dimension + axes2]);
              int neighbor_index =
                  old_source_index[old_to_new_neighbor_lists_host(
                      new_actual_index[i], j + 1)];
              I.increment(field_dof * current_particle_local_index + axes1,
                          field_dof * neighbor_index + axes2,
                          old_to_new_velocity_alphas(alpha_index + j));
            }
        }

        for (int j = 0;
             j < old_to_new_neighbor_lists_host(new_actual_index[i], 0); j++) {
          auto alpha_index = old_to_new_pressure_basis.getAlphaIndexHost(
              new_actual_index[i], pressure_old_to_new_alphas_index);
          int neighbor_index = old_source_index[old_to_new_neighbor_lists_host(
              new_actual_index[i], j + 1)];
          I.increment(field_dof * current_particle_local_index + velocity_dof,
                      field_dof * neighbor_index + velocity_dof,
                      old_to_new_pressure_alphas(alpha_index + j));
        }
      } else {
        for (int j = 0; j < field_dof; j++) {
          I.increment(field_dof * current_particle_local_index + j,
                      field_dof * new_added[i] + j, 1.0);
        }
      }
    }

    // rigid body
    if (mpi_rank == mpi_size - 1) {
      for (int i = 0; i < num_rigid_body; i++) {
        int local_rigid_body_index_offset =
            field_dof * new_local_particle_num + i * rigid_body_dof;
        for (int j = 0; j < rigid_body_dof; j++) {
          I.increment(local_rigid_body_index_offset + j,
                      field_dof * old_global_particle_num + i * rigid_body_dof +
                          j,
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

    int old_local_dof = field_dof * old_local_particle_num;
    int old_global_dof = field_dof * old_global_particle_num;

    int new_local_particle_num = coord.size();

    int new_global_particle_num;

    MPI_Allreduce(&new_local_particle_num, &new_global_particle_num, 1, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);

    int new_local_dof = field_dof * new_local_particle_num;
    int new_global_dof = field_dof * new_global_particle_num;

    // new to old restriction matrix
    if (mpi_rank == mpi_size - 1)
      R.resize(old_local_dof + num_rigid_body * rigid_body_dof,
               new_local_dof + num_rigid_body * rigid_body_dof,
               new_global_dof + num_rigid_body * rigid_body_dof);
    else
      R.resize(old_local_dof, new_local_dof,
               new_global_dof + num_rigid_body * rigid_body_dof);

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
        vec3 dX = source_coord[neighbor_index] - old_coord[i];
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
      for (int j = 0; j < field_dof; j++) {
        for (int k = 0; k < neighbor_lists_host(i, 0); k++) {
          int neighbor_index = neighbor_lists_host(i, k + 1);
          index[k] = source_index[neighbor_index] * field_dof + j;
        }

        R.set_col_index(field_dof * current_particle_local_index + j, index);
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
            field_dof * old_local_particle_num + i * rigid_body_dof;
        for (int j = 0; j < rigid_body_dof; j++) {
          index[0] =
              field_dof * new_global_particle_num + i * rigid_body_dof + j;
          R.set_col_index(local_rigid_body_index_offset + j, index);
        }
      }
    }

    for (int i = 0; i < old_local_particle_num; i++) {
      int current_particle_local_index = old_local_index[i];
      for (int j = 0; j < field_dof; j++) {
        for (int k = 0; k < neighbor_lists_host(i, 0); k++) {
          int neighbor_index = neighbor_lists_host(i, k + 1);
          R.increment(field_dof * current_particle_local_index + j,
                      source_index[neighbor_index] * field_dof + j,
                      1.0 / neighbor_lists_host(i, 0));
        }
      }
    }

    // rigid body
    if (mpi_rank == mpi_size - 1) {
      index.resize(1);
      for (int i = 0; i < num_rigid_body; i++) {
        int local_rigid_body_index_offset =
            field_dof * old_local_particle_num + i * rigid_body_dof;
        for (int j = 0; j < rigid_body_dof; j++) {
          R.increment(
              local_rigid_body_index_offset + j,
              (field_dof * new_global_particle_num + i * rigid_body_dof + j),
              1.0);
        }
      }
    }

    R.assemble();
  }
}

void stokes_multilevel::initial_guess_from_previous_adaptive_step(
    std::vector<double> &initial_guess, std::vector<vec3> &velocity,
    std::vector<double> &pressure, std::vector<vec3> &rb_velocity,
    std::vector<vec3> &rb_angular_velocity) {
  auto &local_idx = *(geo_mgr->get_last_work_particle_local_index());

  petsc_sparse_matrix &I = *(getI(current_refinement_level - 1));
  petsc_sparse_matrix &R = *(getR(current_refinement_level - 1));
  Vec x1, x2;
  MatCreateVecs(I.get_reference(), &x2, &x1);

  const int old_local_particle_num = pressure.size();

  const int field_dof = dimension + 1;
  const int velocity_dof = dimension;
  const int pressure_dof = 1;
  const int rigid_body_dof = (dimension == 3) ? 6 : 3;

  PetscReal *a;
  VecGetArray(x2, &a);

  for (int i = 0; i < old_local_particle_num; i++) {
    int current_particle_local_index = local_idx[i];
    for (int j = 0; j < velocity_dof; j++) {
      a[current_particle_local_index * field_dof + j] = velocity[i][j];
    }
    a[current_particle_local_index * field_dof + velocity_dof] = pressure[i];
  }

  if (mpi_rank == mpi_size - 1) {
    const int old_local_rigid_body_offset = old_local_particle_num * field_dof;

    const int rb_velocity_dof = dimension;
    const int rb_angular_velocity_dof = (dimension == 3) ? 3 : 1;

    for (int i = 0; i < num_rigid_body; i++) {
      for (int j = 0; j < rb_velocity_dof; j++) {
        a[old_local_rigid_body_offset + i * rigid_body_dof + j] =
            rb_velocity[i][j];
      }
      for (int j = 0; j < rb_angular_velocity_dof; j++) {
        a[old_local_rigid_body_offset + i * rigid_body_dof + rb_velocity_dof +
          j] = rb_angular_velocity[i][j];
      }
    }
  }

  VecRestoreArray(x2, &a);

  MatMult(I.get_reference(), x2, x1);

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
    pressure_sum += initial_guess[i * field_dof + velocity_dof];
  }

  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  double average_pressure = pressure_sum / global_particle_num;
  for (int i = 0; i < local_particle_num; i++) {
    initial_guess[i * field_dof + velocity_dof] -= average_pressure;
  }

  VecDestroy(&x1);
  VecDestroy(&x2);
}

int stokes_multilevel::solve(std::vector<double> &rhs, std::vector<double> &x,
                             std::vector<int> &idx_colloid) {
  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "\nstart of linear system solving setup\n");

  int refinement_step = A_list.size() - 1;

  int field_dof = dimension + 1;
  int velocity_dof = dimension;
  int pressure_dof = 1;
  int rigid_body_dof = (dimension == 3) ? 6 : 3;

  vector<int> idx_field;
  vector<int> idx_pressure;

  PetscInt local_n1, local_n2;
  Mat &shell_mat = getA(refinement_step)->get_shell_reference();
  MatGetOwnershipRange(shell_mat, &local_n1, &local_n2);

  int local_particle_num;
  if (mpi_rank != mpi_size - 1) {
    local_particle_num = (local_n2 - local_n1) / field_dof;
    idx_field.resize(field_dof * local_particle_num);
    idx_pressure.resize(local_particle_num);

    for (int i = 0; i < local_particle_num; i++) {
      for (int j = 0; j < dimension; j++) {
        idx_field[field_dof * i + j] = local_n1 + field_dof * i + j;
      }
      idx_field[field_dof * i + velocity_dof] =
          local_n1 + field_dof * i + velocity_dof;

      idx_pressure[i] = local_n1 + field_dof * i + velocity_dof;
    }
  } else {
    local_particle_num =
        (local_n2 - local_n1 - num_rigid_body * rigid_body_dof) / field_dof;
    idx_field.resize(field_dof * local_particle_num);
    idx_pressure.resize(local_particle_num);

    for (int i = 0; i < local_particle_num; i++) {
      for (int j = 0; j < dimension; j++) {
        idx_field[field_dof * i + j] = local_n1 + field_dof * i + j;
      }
      idx_field[field_dof * i + velocity_dof] =
          local_n1 + field_dof * i + velocity_dof;

      idx_pressure[i] = local_n1 + field_dof * i + velocity_dof;
    }
  }

  int global_particle_num;
  MPI_Allreduce(&local_particle_num, &global_particle_num, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  local_particle_num_list.push_back(local_particle_num);
  global_particle_num_list.push_back(global_particle_num);

  auto isg_field = isg_field_list[refinement_step];
  auto isg_colloid = isg_colloid_list[refinement_step];
  auto isg_pressure = isg_pressure_list[refinement_step];

  isg_field->create_local(idx_field);
  isg_colloid->create(idx_colloid);
  isg_pressure->create_local(idx_pressure);

  vector<int> idx_colloid_sub_field;
  vector<int> idx_colloid_sub_colloid;
  vector<int> idx_colloid_field;

  vector<int> idx_colloid_offset, idx_colloid_global_size;
  idx_colloid_offset.resize(mpi_size + 1);
  idx_colloid_global_size.resize(mpi_size);

  int idx_colloid_local_size = idx_colloid.size();
  MPI_Allgather(&idx_colloid_local_size, 1, MPI_INT,
                idx_colloid_global_size.data(), 1, MPI_INT, MPI_COMM_WORLD);

  idx_colloid_offset[0] = 0;
  for (int i = 0; i < mpi_size; i++) {
    idx_colloid_offset[i + 1] =
        idx_colloid_offset[i] + idx_colloid_global_size[i];
  }

  for (int i = 0; i < idx_colloid.size(); i++) {
    if (idx_colloid[i] < global_particle_num * field_dof) {
      idx_colloid_sub_field.push_back(i + idx_colloid_offset[mpi_rank]);
      idx_colloid_field.push_back(idx_colloid[i]);
    } else {
      idx_colloid_sub_colloid.push_back(i + idx_colloid_offset[mpi_rank]);
    }
  }

  IS isg_colloid_sub_field, isg_colloid_sub_colloid, isg_colloid_field;

  petsc_vector _rhs, _x;
  _rhs.create(rhs);
  _x.create(x);

  Mat &ff_shell = ff_list[refinement_step]->get_shell_reference();
  Mat &ff = ff_list[refinement_step]->get_reference();
  Mat &nn = nn_list[refinement_step]->get_reference();
  Mat &nw = nw_list[refinement_step]->get_reference();
  // Mat &pp = pp_list[refinement_step]->get_reference();

  PetscLogDouble mem;

  // MatCreateSubMatrix(mat, isg_colloid->get_reference(),
  //                    isg_colloid->get_reference(), MAT_INITIAL_MATRIX,
  //                    nn_list[refinement_step]->get_pointer());
  // MatCreateSubMatrix(mat, isg_colloid->get_reference(), NULL,
  //                    MAT_INITIAL_MATRIX,
  //                    nw_list[refinement_step]->get_pointer());
  // MatCreateSubMatrix(ff, isg_pressure->get_reference(),
  //                    isg_pressure->get_reference(), MAT_INITIAL_MATRIX,
  //                    pp_list[refinement_step]->get_pointer());
  // MatCreateSubMatrix(mat, isg_pressure->get_reference(), NULL,
  //                    MAT_INITIAL_MATRIX,
  //                    pw_list[refinement_step]->get_pointer());

  // setup current level vectors
  x_list.push_back(make_shared<petsc_vector>());
  b_list.push_back(make_shared<petsc_vector>());
  r_list.push_back(make_shared<petsc_vector>());
  t_list.push_back(make_shared<petsc_vector>());

  x_pressure_list.push_back(make_shared<petsc_vector>());
  y_pressure_list.push_back(make_shared<petsc_vector>());

  x_field_list.push_back(make_shared<petsc_vector>());
  y_field_list.push_back(make_shared<petsc_vector>());
  b_field_list.push_back(make_shared<petsc_vector>());
  r_field_list.push_back(make_shared<petsc_vector>());

  x_colloid_list.push_back(make_shared<petsc_vector>());
  b_colloid_list.push_back(make_shared<petsc_vector>());

  field_relaxation_list.push_back(make_shared<petsc_ksp>());
  colloid_relaxation_list.push_back(make_shared<petsc_ksp>());
  // pressure_relaxation_list.push_back(make_shared<petsc_ksp>());

  MatCreateVecs(shell_mat, NULL, &(x_list[refinement_step]->get_reference()));
  MatCreateVecs(shell_mat, NULL, &(b_list[refinement_step]->get_reference()));
  MatCreateVecs(shell_mat, NULL, &(r_list[refinement_step]->get_reference()));
  MatCreateVecs(shell_mat, NULL, &(t_list[refinement_step]->get_reference()));

  MatCreateVecs(ff, NULL, &(x_field_list[refinement_step]->get_reference()));
  MatCreateVecs(ff, NULL, &(y_field_list[refinement_step]->get_reference()));
  MatCreateVecs(ff, NULL, &(b_field_list[refinement_step]->get_reference()));
  MatCreateVecs(ff, NULL, &(r_field_list[refinement_step]->get_reference()));

  if (num_rigid_body != 0) {
    MatCreateVecs(nn, NULL,
                  &(x_colloid_list[refinement_step]->get_reference()));
    MatCreateVecs(nn, NULL,
                  &(b_colloid_list[refinement_step]->get_reference()));
  }

  // MatCreateVecs(pp, NULL,
  // &x_pressure_list[refinement_step]->get_reference());
  // MatCreateVecs(pp, NULL,
  // &y_pressure_list[refinement_step]->get_reference());

  // field vector scatter
  field_scatter_list.push_back(make_shared<petsc_vecscatter>());
  VecScatterCreate(x_list[refinement_step]->get_reference(),
                   isg_field->get_reference(),
                   x_field_list[refinement_step]->get_reference(), NULL,
                   field_scatter_list[refinement_step]->get_pointer());

  if (num_rigid_body != 0) {
    colloid_scatter_list.push_back(make_shared<petsc_vecscatter>());
    VecScatterCreate(x_list[refinement_step]->get_reference(),
                     isg_colloid->get_reference(),
                     x_colloid_list[refinement_step]->get_reference(), NULL,
                     colloid_scatter_list[refinement_step]->get_pointer());
  }

  // pressure_scatter_list.push_back(make_shared<petsc_vecscatter>());
  // VecScatterCreate(x_list[refinement_step]->get_reference(),
  //                  isg_pressure->get_reference(),
  //                  x_pressure_list[refinement_step]->get_reference(), NULL,
  //                  pressure_scatter_list[refinement_step]->get_pointer());

  // neighbor vector scatter, only needed on base level
  if (refinement_step == 0 && num_rigid_body != 0) {
    MatCreateVecs(nn, NULL, x_colloid->get_pointer());
    MatCreateVecs(nn, NULL, y_colloid->get_pointer());
  }

  Mat sub_ff, sub_fc, sub_cf, sub_cc, fc_s;

  ISCreateGeneral(MPI_COMM_WORLD, idx_colloid_sub_field.size(),
                  idx_colloid_sub_field.data(), PETSC_COPY_VALUES,
                  &isg_colloid_sub_field);
  ISCreateGeneral(MPI_COMM_WORLD, idx_colloid_sub_colloid.size(),
                  idx_colloid_sub_colloid.data(), PETSC_COPY_VALUES,
                  &isg_colloid_sub_colloid);
  ISCreateGeneral(MPI_COMM_WORLD, idx_colloid_field.size(),
                  idx_colloid_field.data(), PETSC_COPY_VALUES,
                  &isg_colloid_field);

  MatCreateSubMatrix(ff, isg_colloid_field, isg_colloid_field,
                     MAT_INITIAL_MATRIX, &sub_ff);
  MatCreateSubMatrix(nn, isg_colloid_sub_field, isg_colloid_sub_colloid,
                     MAT_INITIAL_MATRIX, &sub_fc);
  MatCreateSubMatrix(nn, isg_colloid_sub_colloid, isg_colloid_sub_field,
                     MAT_INITIAL_MATRIX, &sub_cf);
  MatCreateSubMatrix(nn, isg_colloid_sub_colloid, isg_colloid_sub_colloid,
                     MAT_INITIAL_MATRIX, &sub_cc);

  Mat B, C;
  MatCreate(MPI_COMM_WORLD, &B);
  MatSetType(B, MATMPIAIJ);
  MatInvertBlockDiagonalMat(sub_ff, B);

  MatMatMult(B, sub_fc, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);
  MatMatMult(sub_cf, C, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &fc_s);
  MatScale(fc_s, -1.0);
  MatAXPY(fc_s, 1.0, sub_cc, DIFFERENT_NONZERO_PATTERN);

  MatDestroy(&B);
  MatDestroy(&C);

  // setup preconditioner for base level
  if (refinement_step == 0) {
    KSPCreate(PETSC_COMM_WORLD, &ksp_field_base->get_reference());
    KSPSetOperators(ksp_field_base->get_reference(), ff_shell, ff);
    // KSPSetFromOptions(ksp_field_base->get_reference());
    // KSPSetType(ksp_field_base->get_reference(), KSPPREONLY);
    KSPSetType(ksp_field_base->get_reference(), KSPGMRES);
    KSPGMRESSetRestart(ksp_field_base->get_reference(), 100);
    KSPSetTolerances(ksp_field_base->get_reference(), 1e-2, 1e-50, 1e50, 1000);
    KSPSetNormType(ksp_field_base->get_reference(), KSP_NORM_UNPRECONDITIONED);
    KSPSetResidualHistory(ksp_field_base->get_reference(), NULL, 1000,
                          PETSC_TRUE);

    PC pc_field_base;

    KSPGetPC(ksp_field_base->get_reference(), &pc_field_base);
    // PCSetType(pc_field_base, PCLU);
    // PCFactorSetMatSolverType(pc_field_base, MATSOLVERMUMPS);
    PCSetType(pc_field_base, PCSOR);
    PCSetFromOptions(pc_field_base);
    PCSetUp(pc_field_base);
    KSPSetUp(ksp_field_base->get_reference());

    if (num_rigid_body != 0) {
      // KSPCreate(PETSC_COMM_WORLD, &ksp_colloid_base->get_reference());
      // KSPSetOperators(ksp_colloid_base->get_reference(), nn, nn);
      // KSPSetType(ksp_colloid_base->get_reference(), KSPPREONLY);

      // PC pc_neighbor_base;

      // KSPGetPC(ksp_colloid_base->get_reference(), &pc_neighbor_base);
      // PCSetType(pc_neighbor_base, PCLU);
      // PCFactorSetMatSolverType(pc_neighbor_base, MATSOLVERMUMPS);
      // PCSetFromOptions(pc_neighbor_base);
      // PCSetUp(pc_neighbor_base);

      // KSPSetUp(ksp_colloid_base->get_reference());

      KSPCreate(MPI_COMM_WORLD, &ksp_colloid_base->get_reference());

      KSPSetType(ksp_colloid_base->get_reference(), KSPGMRES);
      KSPGMRESSetRestart(ksp_colloid_base->get_reference(), 100);
      KSPSetTolerances(ksp_colloid_base->get_reference(), 1e-2, 1e-50, 1e50,
                       500);
      KSPSetOperators(ksp_colloid_base->get_reference(), nn, nn);

      PC pc_neighbor_base;
      KSPGetPC(ksp_colloid_base->get_reference(), &pc_neighbor_base);
      PCSetType(pc_neighbor_base, PCFIELDSPLIT);

      PCFieldSplitSetIS(pc_neighbor_base, "0", isg_colloid_sub_field);
      PCFieldSplitSetIS(pc_neighbor_base, "1", isg_colloid_sub_colloid);

      PCFieldSplitSetSchurPre(pc_neighbor_base, PC_FIELDSPLIT_SCHUR_PRE_USER,
                              fc_s);
      PCSetUp(pc_neighbor_base);

      KSP *fieldsplit_sub_ksp;
      PetscInt n;
      PCFieldSplitGetSubKSP(pc_neighbor_base, &n, &fieldsplit_sub_ksp);
      KSPSetOperators(fieldsplit_sub_ksp[1], fc_s, fc_s);
      KSPSetOperators(fieldsplit_sub_ksp[0], sub_ff, sub_ff);
      KSPSetFromOptions(fieldsplit_sub_ksp[0]);
      KSPSetUp(fieldsplit_sub_ksp[0]);
      KSPSetUp(fieldsplit_sub_ksp[1]);
      PetscFree(fieldsplit_sub_ksp);

      KSPSetUp(ksp_colloid_base->get_reference());
    }
  } else {
    // setup relaxation on field for current level
    KSPCreate(MPI_COMM_WORLD,
              field_relaxation_list[refinement_step]->get_pointer());

    KSPSetType(field_relaxation_list[refinement_step]->get_reference(),
               KSPRICHARDSON);
    KSPSetOperators(field_relaxation_list[refinement_step]->get_reference(),
                    ff_shell, ff);
    KSPSetTolerances(field_relaxation_list[refinement_step]->get_reference(),
                     1e-3, 1e-50, 1e10, 1);

    PC field_relaxation_pc;
    KSPGetPC(field_relaxation_list[refinement_step]->get_reference(),
             &field_relaxation_pc);
    PCSetType(field_relaxation_pc, PCSOR);
    PCSetFromOptions(field_relaxation_pc);
    PCSetUp(field_relaxation_pc);

    KSPSetUp(field_relaxation_list[refinement_step]->get_reference());

    if (num_rigid_body != 0) {
      // setup relaxation on neighbor for current level
      KSPCreate(MPI_COMM_WORLD,
                colloid_relaxation_list[refinement_step]->get_pointer());

      KSPSetType(colloid_relaxation_list[refinement_step]->get_reference(),
                 KSPRICHARDSON);
      KSPSetTolerances(
          colloid_relaxation_list[refinement_step]->get_reference(), 1e-3,
          1e-50, 1e10, 1);
      KSPSetOperators(colloid_relaxation_list[refinement_step]->get_reference(),
                      nn, nn);

      PC neighbor_relaxation_pc;
      KSPGetPC(colloid_relaxation_list[refinement_step]->get_reference(),
               &neighbor_relaxation_pc);
      PCSetType(neighbor_relaxation_pc, PCFIELDSPLIT);

      PCFieldSplitSetIS(neighbor_relaxation_pc, "0", isg_colloid_sub_field);
      PCFieldSplitSetIS(neighbor_relaxation_pc, "1", isg_colloid_sub_colloid);

      PCFieldSplitSetSchurPre(neighbor_relaxation_pc,
                              PC_FIELDSPLIT_SCHUR_PRE_USER, fc_s);
      PCSetUp(neighbor_relaxation_pc);

      KSP *fieldsplit_sub_ksp;
      PetscInt n;
      PCFieldSplitGetSubKSP(neighbor_relaxation_pc, &n, &fieldsplit_sub_ksp);
      KSPSetOperators(fieldsplit_sub_ksp[1], fc_s, fc_s);
      KSPSetOperators(fieldsplit_sub_ksp[0], sub_ff, sub_ff);
      KSPSetFromOptions(fieldsplit_sub_ksp[0]);
      KSPSetUp(fieldsplit_sub_ksp[0]);
      KSPSetUp(fieldsplit_sub_ksp[1]);
      PetscFree(fieldsplit_sub_ksp);

      KSPSetUp(colloid_relaxation_list[refinement_step]->get_reference());

      // // setup relaxation on pressure for current level
      // KSPCreate(MPI_COMM_WORLD,
      //           pressure_relaxation_list[refinement_step]->get_pointer());

      // KSPSetType(pressure_relaxation_list[refinement_step]->get_reference(),
      //            KSPRICHARDSON);
      // KSPSetTolerances(
      //     pressure_relaxation_list[refinement_step]->get_reference(), 1e-3,
      //     1e-50, 1e10, 1);
      // KSPSetOperators(
      //     pressure_relaxation_list[refinement_step]->get_reference(), pp,
      //     pp);

      // PC pressure_relaxation_pc;
      // KSPGetPC(pressure_relaxation_list[refinement_step]->get_reference(),
      //          &pressure_relaxation_pc);
      // PCSetType(pressure_relaxation_pc, PCSOR);

      // KSPSetUp(pressure_relaxation_list[refinement_step]->get_reference());
    }
  }

  KSP _ksp;
  KSPCreate(PETSC_COMM_WORLD, &_ksp);

  PC _pc;

  HypreLUShellPC *shell_ctx;

  static PetscInt restart;

  KSPGetPC(_ksp, &_pc);
  KSPSetOperators(_ksp, shell_mat, shell_mat);
  KSPSetFromOptions(_ksp);

  PCSetType(_pc, PCSHELL);

  if (refinement_step == 0) {
    KSPGMRESGetRestart(_ksp, &restart);
  } else {
    KSPGMRESSetRestart(_ksp, restart);
  }

  KSPSetUp(_ksp);

  HypreLUShellPCCreate(&shell_ctx);

  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "start of stokes_multilevel preconditioner setup\n");

  PCShellSetApply(_pc, HypreLUShellPCApplyAdaptive);
  PCShellSetContext(_pc, shell_ctx);
  PCShellSetDestroy(_pc, HypreLUShellPCDestroy);

  HypreLUShellPCSetUp(_pc, this, _x.get_reference(), local_particle_num,
                      field_dof, num_rigid_body);
  // }

  PetscPrintf(PETSC_COMM_WORLD, "final solving of linear system\n");
  PetscReal residual_norm, rhs_norm;
  VecNorm(_rhs.get_reference(), NORM_2, &rhs_norm);
  residual_norm = global_particle_num;
  Vec residual;
  VecDuplicate(_rhs.get_reference(), &residual);
  MatMult(shell_mat, _x.get_reference(), residual);
  VecAXPY(residual, -1.0, _rhs.get_reference());
  VecNorm(residual, NORM_2, &residual_norm);
  PetscPrintf(PETSC_COMM_WORLD, "relative residual norm: %f\n",
              residual_norm / rhs_norm);
  int counter = 0;
  double initial_residual = residual_norm / rhs_norm;
  while (residual_norm / rhs_norm > 1e-3 && counter < 5) {
    KSPSolve(_ksp, _rhs.get_reference(), _x.get_reference());

    KSPConvergedReason convergence_reason;
    KSPGetConvergedReason(_ksp, &convergence_reason);

    MatMult(shell_mat, _x.get_reference(), residual);
    VecAXPY(residual, -1.0, _rhs.get_reference());
    VecNorm(residual, NORM_2, &residual_norm);
    PetscPrintf(PETSC_COMM_WORLD, "relative residual norm: %f\n",
                residual_norm / rhs_norm);
    counter++;

    if (convergence_reason != KSP_CONVERGED_RTOL) {
      restart += 50;
      KSPGMRESSetRestart(_ksp, restart);

      KSPSetUp(_ksp);
    }
    if (convergence_reason == KSP_CONVERGED_RTOL &&
        residual_norm / rhs_norm > 1e-3) {
      KSPSetTolerances(_ksp, pow(10, -6 - counter), 1e-50, 1e50, 500);
    }
    KSPSetInitialGuessNonzero(_ksp, PETSC_TRUE);
  }
  PetscPrintf(PETSC_COMM_WORLD, "ksp solving finished\n");

  if (residual_norm / rhs_norm < initial_residual || refinement_step == 0) {
    _x.copy(x);
  }

  VecDestroy(&residual);
  KSPDestroy(&_ksp);

  if (num_rigid_body != 0) {
    MatDestroy(&sub_ff);
    MatDestroy(&sub_fc);
    MatDestroy(&sub_cf);
    MatDestroy(&sub_cc);
    MatDestroy(&fc_s);

    ISDestroy(&isg_colloid_sub_field);
    ISDestroy(&isg_colloid_sub_colloid);
    ISDestroy(&isg_colloid_field);
  }

  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "Current memory usage %.2f GB\n", mem / 1e9);

  return 0;
}

void stokes_multilevel::clear() {
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
  y_list.clear();
  b_list.clear();
  r_list.clear();
  t_list.clear();

  x_field_list.clear();
  y_field_list.clear();
  b_field_list.clear();
  r_field_list.clear();
  t_field_list.clear();

  x_colloid_list.clear();
  y_colloid_list.clear();
  b_colloid_list.clear();
  r_colloid_list.clear();
  t_colloid_list.clear();

  x_pressure_list.clear();
  y_pressure_list.clear();

  field_scatter_list.clear();
  colloid_scatter_list.clear();
  pressure_scatter_list.clear();

  field_relaxation_list.clear();
  colloid_relaxation_list.clear();
  pressure_relaxation_list.clear();

  if (base_level_initialized) {
    x_colloid.reset();
    y_colloid.reset();

    ksp_field_base.reset();
    ksp_colloid_base.reset();
  }

  local_particle_num_list.clear();
  global_particle_num_list.clear();

  current_refinement_level = -1;
}