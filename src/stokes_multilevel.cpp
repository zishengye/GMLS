#include "stokes_multilevel.hpp"
#include "gmls_solver.hpp"
#include "stokes_composite_preconditioner.hpp"

#include <algorithm>
#include <iostream>

using namespace std;
using namespace Compadre;

void stokes_multilevel::build_interpolation_restriction(int _num_rigid_body,
                                                        int _dimension) {
  petsc_sparse_matrix &I = *(getI(current_refinement_level - 1));
  petsc_sparse_matrix &R = *(getR(current_refinement_level - 1));

  int field_dof = dimension + 1;
  int velocity_dof = dimension;
  int pressure_dof = 1;

  int rigid_body_dof = (dimension == 3) ? 6 : 3;

  int translation_dof = dimension;
  int rotation_dof = (dimension == 3) ? 3 : 1;

  {
    auto &coord = *(geo_mgr->get_current_work_particle_coord());
    auto &new_added = *(geo_mgr->get_current_work_particle_new_added());
    auto &spacing = *(geo_mgr->get_current_work_particle_spacing());

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
        pow(2, dimension) * pow(2 * 3.0, dimension);

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
        old_epsilon_host[new_actual_index[i]] = 3.0 * spacing[i];
      }
    }

    auto neighbor_needed = Compadre::GMLS::getNP(
        2, dimension, DivergenceFreeVectorTaylorPolynomial);
    size_t actual_neighbor_max;

    while (true) {
      actual_neighbor_max =
          old_to_new_point_search.generateNeighborListsFromRadiusSearch(
              true, new_target_coords_host, old_to_new_neighbor_lists_host,
              old_epsilon_host, 0.0, 0.0);
      while (actual_neighbor_max > estimated_num_neighbor_max) {
        estimated_num_neighbor_max *= 2;
        old_to_new_neighbor_lists_device =
            Kokkos::View<int **, Kokkos::DefaultExecutionSpace>(
                "old to new neighbor lists", actual_new_target,
                estimated_num_neighbor_max);
        old_to_new_neighbor_lists_host =
            Kokkos::create_mirror_view(old_to_new_neighbor_lists_device);
      }
      old_to_new_point_search.generateNeighborListsFromRadiusSearch(
          false, new_target_coords_host, old_to_new_neighbor_lists_host,
          old_epsilon_host, 0.0, 0.0);

      bool enough_neighbor = true;
      for (int i = 0; i < coord.size(); i++) {
        if (new_added[i] < 0) {
          if (old_to_new_neighbor_lists_host(new_actual_index[i], 0) <
              neighbor_needed) {
            old_epsilon_host[new_actual_index[i]] += 0.5 * spacing[i];
            enough_neighbor = false;
          }
        }
      }

      if (enough_neighbor)
        break;
    }

    Kokkos::deep_copy(old_to_new_neighbor_lists_device,
                      old_to_new_neighbor_lists_host);
    Kokkos::deep_copy(old_epsilon_device, old_epsilon_host);

    GMLS old_to_new_pressusre_basis(ScalarTaylorPolynomial, PointSample, 2,
                                    dimension, "SVD", "STANDARD");
    GMLS old_to_new_velocity_basis(DivergenceFreeVectorTaylorPolynomial,
                                   VectorPointSample, 2, dimension, "SVD",
                                   "STANDARD");

    // old to new pressure field transition
    old_to_new_pressusre_basis.setProblemData(
        old_to_new_neighbor_lists_device, old_source_coords_device,
        new_target_coords_device, old_epsilon_device);

    old_to_new_pressusre_basis.addTargets(ScalarPointEvaluation);

    old_to_new_pressusre_basis.setWeightingType(WeightingFunctionType::Power);
    old_to_new_pressusre_basis.setWeightingPower(4);

    old_to_new_pressusre_basis.generateAlphas(1);

    auto old_to_new_pressure_alphas = old_to_new_pressusre_basis.getAlphas();

    // old to new velocity field transition
    old_to_new_velocity_basis.setProblemData(
        old_to_new_neighbor_lists_device, old_source_coords_device,
        new_target_coords_device, old_epsilon_device);

    old_to_new_velocity_basis.addTargets(VectorPointEvaluation);

    old_to_new_velocity_basis.setWeightingType(WeightingFunctionType::Power);
    old_to_new_velocity_basis.setWeightingPower(4);

    old_to_new_velocity_basis.generateAlphas(1);

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
          I.set_col_index(field_dof * i + k, index);
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
        I.set_col_index(field_dof * i + velocity_dof, index);
      } else {
        index.resize(1);
        for (int j = 0; j < field_dof; j++) {
          index[0] = field_dof * new_added[i] + j;
          I.set_col_index(field_dof * i + j, index);
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
        old_to_new_pressusre_basis.getAlphaColumnOffset(ScalarPointEvaluation,
                                                        0, 0, 0, 0);
    vector<int> velocity_old_to_new_alphas_index(pow(dimension, 2));
    for (int axes1 = 0; axes1 < dimension; axes1++)
      for (int axes2 = 0; axes2 < dimension; axes2++)
        velocity_old_to_new_alphas_index[axes1 * dimension + axes2] =
            old_to_new_velocity_basis.getAlphaColumnOffset(
                VectorPointEvaluation, axes1, 0, axes2, 0);

    for (int i = 0; i < new_local_particle_num; i++) {
      if (new_added[i] < 0) {
        for (int j = 0;
             j < old_to_new_neighbor_lists_host(new_actual_index[i], 0); j++) {
          for (int axes1 = 0; axes1 < dimension; axes1++)
            for (int axes2 = 0; axes2 < dimension; axes2++)
              I.increment(
                  field_dof * i + axes1,
                  field_dof * old_source_index[old_to_new_neighbor_lists_host(
                                  new_actual_index[i], j + 1)] +
                      axes2,
                  old_to_new_velocity_alphas(
                      new_actual_index[i],
                      velocity_old_to_new_alphas_index[axes1 * dimension +
                                                       axes2],
                      j));
        }

        for (int j = 0;
             j < old_to_new_neighbor_lists_host(new_actual_index[i], 0); j++) {
          I.increment(
              field_dof * i + velocity_dof,
              field_dof * old_source_index[old_to_new_neighbor_lists_host(
                              new_actual_index[i], j + 1)] +
                  velocity_dof,
              old_to_new_pressure_alphas(new_actual_index[i],
                                         pressure_old_to_new_alphas_index, j));
        }
      } else {
        for (int j = 0; j < field_dof; j++) {
          I.increment(field_dof * i + j, field_dof * new_added[i] + j, 1.0);
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

  {
    auto &coord = *(geo_mgr->get_current_work_particle_coord());

    auto &source_coord = *(geo_mgr->get_llcl_particle_coord());
    auto &source_index = *(geo_mgr->get_llcl_particle_index());
    auto &source_particle_type = *(geo_mgr->get_llcl_particle_type());

    auto &old_coord = *(geo_mgr->get_last_work_particle_coord());
    auto &old_index = *(geo_mgr->get_last_work_particle_index());
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
      epsilon_host(i) = old_spacing[i];
    }

    point_search.generateNeighborListsFromRadiusSearch(
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
    for (int i = 0; i < old_local_particle_num; i++) {
      index.resize(neighbor_lists_host(i, 0));
      for (int j = 0; j < field_dof; j++) {
        for (int k = 0; k < neighbor_lists_host(i, 0); k++) {
          int neighbor_index = neighbor_lists_host(i, k + 1);
          index[k] = source_index[neighbor_index] * field_dof + j;
        }

        R.set_col_index(field_dof * i + j, index);
      }
    }

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
      for (int j = 0; j < field_dof; j++) {
        for (int k = 0; k < neighbor_lists_host(i, 0); k++) {
          int neighbor_index = neighbor_lists_host(i, k + 1);
          R.increment(field_dof * i + j,
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
          R.increment(local_rigid_body_index_offset + j,
                      field_dof * new_global_particle_num + i * rigid_body_dof +
                          j,
                      1.0);
        }
      }
    }

    R.assemble();
  }
}

void stokes_multilevel::initial_guess_from_previous_adaptive_step(
    std::vector<double> &initial_guess) {}

int stokes_multilevel::solve(std::vector<double> &rhs, std::vector<double> &x,
                             std::vector<int> &idx_colloid) {
  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "\nstart of linear system solving setup\n");

  int refinement_step = A_list.size() - 1;

  int filed_dof = dimension + 1;
  int velocity_dof = dimension;
  int pressure_dof = 1;
  int rigid_body_dof = (dimension == 3) ? 6 : 3;

  vector<int> idx_field;
  vector<int> idx_pressure;

  PetscInt local_n1, local_n2;
  Mat &mat = getA(refinement_step)->get_reference();
  MatGetOwnershipRange(mat, &local_n1, &local_n2);

  int local_particle_num;
  if (mpi_rank != mpi_size - 1) {
    local_particle_num = (local_n2 - local_n1) / filed_dof;
    idx_field.resize(filed_dof * local_particle_num);
    idx_pressure.resize(local_particle_num);

    for (int i = 0; i < local_particle_num; i++) {
      for (int j = 0; j < dimension; j++) {
        idx_field[filed_dof * i + j] = local_n1 + filed_dof * i + j;
      }
      idx_field[filed_dof * i + velocity_dof] =
          local_n1 + filed_dof * i + velocity_dof;

      idx_pressure[i] = local_n1 + filed_dof * i + velocity_dof;
    }
  } else {
    local_particle_num =
        (local_n2 - local_n1 - num_rigid_body * rigid_body_dof) / filed_dof;
    idx_field.resize(filed_dof * local_particle_num);
    idx_pressure.resize(local_particle_num);

    for (int i = 0; i < local_particle_num; i++) {
      for (int j = 0; j < dimension; j++) {
        idx_field[filed_dof * i + j] = local_n1 + filed_dof * i + j;
      }
      idx_field[filed_dof * i + velocity_dof] =
          local_n1 + filed_dof * i + velocity_dof;

      idx_pressure[i] = local_n1 + filed_dof * i + velocity_dof;
    }
  }

  int global_particle_num;
  MPI_Allreduce(&local_particle_num, &global_particle_num, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  auto isg_field = isg_field_list[refinement_step];
  auto isg_colloid = isg_colloid_list[refinement_step];
  auto isg_pressure = isg_pressure_list[refinement_step];

  isg_field->create_local(idx_field);
  isg_colloid->create(idx_colloid);
  isg_pressure->create_local(idx_pressure);

  petsc_vector _rhs, _x;
  _rhs.create(rhs);
  _x.create(x);

  Mat &ff = ff_list[refinement_step]->get_reference();
  Mat &nn = nn_list[refinement_step]->get_reference();
  Mat &nw = nw_list[refinement_step]->get_reference();
  Mat pp;

  MatCreateSubMatrix(mat, isg_colloid->get_reference(),
                     isg_colloid->get_reference(), MAT_INITIAL_MATRIX,
                     nn_list[refinement_step]->get_pointer());
  MatCreateSubMatrix(mat, isg_colloid->get_reference(), NULL,
                     MAT_INITIAL_MATRIX,
                     nw_list[refinement_step]->get_pointer());
  MatCreateSubMatrix(mat, isg_pressure->get_reference(),
                     isg_pressure->get_reference(), MAT_INITIAL_MATRIX, &pp);

  // setup current level vectors
  x_list.push_back(make_shared<petsc_vector>());
  b_list.push_back(make_shared<petsc_vector>());
  r_list.push_back(make_shared<petsc_vector>());
  t_list.push_back(make_shared<petsc_vector>());

  x_field_list.push_back(make_shared<petsc_vector>());
  y_field_list.push_back(make_shared<petsc_vector>());
  b_field_list.push_back(make_shared<petsc_vector>());
  r_field_list.push_back(make_shared<petsc_vector>());

  x_colloid_list.push_back(make_shared<petsc_vector>());
  b_colloid_list.push_back(make_shared<petsc_vector>());

  x_pressure_list.push_back(make_shared<petsc_vector>());

  field_relaxation_list.push_back(make_shared<petsc_ksp>());
  colloid_relaxation_list.push_back(make_shared<petsc_ksp>());

  MatCreateVecs(mat, NULL, &(x_list[refinement_step]->get_reference()));
  MatCreateVecs(mat, NULL, &(b_list[refinement_step]->get_reference()));
  MatCreateVecs(mat, NULL, &(r_list[refinement_step]->get_reference()));
  MatCreateVecs(mat, NULL, &(t_list[refinement_step]->get_reference()));

  MatCreateVecs(ff, NULL, &(x_field_list[refinement_step]->get_reference()));
  MatCreateVecs(ff, NULL, &(y_field_list[refinement_step]->get_reference()));
  MatCreateVecs(ff, NULL, &(b_field_list[refinement_step]->get_reference()));
  MatCreateVecs(ff, NULL, &(r_field_list[refinement_step]->get_reference()));

  MatCreateVecs(nn, NULL, &(x_colloid_list[refinement_step]->get_reference()));
  MatCreateVecs(nn, NULL, &(b_colloid_list[refinement_step]->get_reference()));

  MatCreateVecs(pp, NULL, &(x_pressure_list[refinement_step]->get_reference()));

  // field vector scatter
  field_scatter_list.push_back(make_shared<petsc_vecscatter>());
  VecScatterCreate(x_list[refinement_step]->get_reference(),
                   isg_field->get_reference(),
                   x_field_list[refinement_step]->get_reference(), NULL,
                   field_scatter_list[refinement_step]->get_pointer());

  colloid_scatter_list.push_back(make_shared<petsc_vecscatter>());
  VecScatterCreate(x_list[refinement_step]->get_reference(),
                   isg_colloid->get_reference(),
                   x_colloid_list[refinement_step]->get_reference(), NULL,
                   colloid_scatter_list[refinement_step]->get_pointer());
  pressure_scatter_list.push_back(make_shared<petsc_vecscatter>());
  VecScatterCreate(x_list[refinement_step]->get_reference(),
                   isg_pressure->get_reference(),
                   x_pressure_list[refinement_step]->get_reference(), NULL,
                   pressure_scatter_list[refinement_step]->get_pointer());

  // setup nullspace_field
  Vec null_whole, null_field;
  VecDuplicate(_rhs.get_reference(), &null_whole);
  VecDuplicate(x_field_list[refinement_step]->get_reference(), &null_field);

  VecSet(null_whole, 0.0);
  VecSet(null_field, 0.0);

  VecSet(x_pressure_list[refinement_step]->get_reference(), 1.0);

  VecScatterBegin(pressure_scatter_list[refinement_step]->get_reference(),
                  x_pressure_list[refinement_step]->get_reference(), null_whole,
                  INSERT_VALUES, SCATTER_REVERSE);
  VecScatterEnd(pressure_scatter_list[refinement_step]->get_reference(),
                x_pressure_list[refinement_step]->get_reference(), null_whole,
                INSERT_VALUES, SCATTER_REVERSE);

  MatNullSpace nullspace_whole;
  MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &null_whole,
                     &nullspace_whole);
  MatSetNearNullSpace(mat, nullspace_whole);

  Vec field_pressure;
  VecGetSubVector(null_field, isg_pressure->get_reference(), &field_pressure);
  VecSet(field_pressure, 1.0);
  VecRestoreSubVector(null_field, isg_pressure->get_reference(),
                      &field_pressure);

  MatNullSpace nullspace_field;
  MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &null_field,
                     &nullspace_field);
  MatSetNearNullSpace(ff, nullspace_field);

  // neighbor vector scatter, only needed on base level
  if (refinement_step == 0) {
    MatCreateVecs(nn, NULL, x_colloid->get_pointer());
    MatCreateVecs(nn, NULL, y_colloid->get_pointer());
  }

  // setup preconditioner for base level
  if (refinement_step == 0) {
    KSPCreate(PETSC_COMM_WORLD, &ksp_field_base->get_reference());
    KSPCreate(PETSC_COMM_WORLD, &ksp_colloid_base->get_reference());

    KSPSetOperators(ksp_field_base->get_reference(), ff, ff);
    KSPSetOperators(ksp_colloid_base->get_reference(), nn, nn);

    KSPSetType(ksp_field_base->get_reference(), KSPPREONLY);
    KSPSetType(ksp_colloid_base->get_reference(), KSPPREONLY);

    PC pc_field_base;
    PC pc_neighbor_base;

    KSPGetPC(ksp_field_base->get_reference(), &pc_field_base);
    PCSetType(pc_field_base, PCHYPRE);
    PCSetFromOptions(pc_field_base);
    PCSetUp(pc_field_base);

    KSPGetPC(ksp_colloid_base->get_reference(), &pc_neighbor_base);
    PCSetType(pc_neighbor_base, PCBJACOBI);
    PCSetUp(pc_neighbor_base);
    PetscInt local_row, local_col;
    MatGetLocalSize(nn, &local_row, &local_col);
    if (local_row > 0) {
      KSP *bjacobi_ksp;
      PCBJacobiGetSubKSP(pc_neighbor_base, NULL, NULL, &bjacobi_ksp);
      KSPSetType(bjacobi_ksp[0], KSPPREONLY);
      PC bjacobi_pc;
      KSPGetPC(bjacobi_ksp[0], &bjacobi_pc);
      PCSetType(bjacobi_pc, PCLU);
      PCFactorSetMatSolverType(bjacobi_pc, MATSOLVERMUMPS);
      // PetscOptionsSetValue(NULL, "-pc_hypre_type", "euclid");
      PCSetFromOptions(bjacobi_pc);
      PCSetUp(bjacobi_pc);
      KSPSetUp(bjacobi_ksp[0]);
    }

    KSPSetUp(ksp_field_base->get_reference());
    KSPSetUp(ksp_colloid_base->get_reference());
  }

  // setup relaxation on field for current level
  KSPCreate(MPI_COMM_WORLD,
            field_relaxation_list[refinement_step]->get_pointer());

  KSPSetType(field_relaxation_list[refinement_step]->get_reference(),
             KSPPREONLY);
  KSPSetOperators(field_relaxation_list[refinement_step]->get_reference(), ff,
                  ff);

  PC field_relaxation_pc;
  KSPGetPC(field_relaxation_list[refinement_step]->get_reference(),
           &field_relaxation_pc);
  PCSetType(field_relaxation_pc, PCSOR);
  PCSetFromOptions(field_relaxation_pc);
  PCSetUp(field_relaxation_pc);

  KSPSetUp(field_relaxation_list[refinement_step]->get_reference());

  // setup relaxation on neighbor for current level
  KSPCreate(MPI_COMM_WORLD,
            colloid_relaxation_list[refinement_step]->get_pointer());

  KSPSetType(colloid_relaxation_list[refinement_step]->get_reference(),
             KSPPREONLY);
  KSPSetOperators(colloid_relaxation_list[refinement_step]->get_reference(), nn,
                  nn);

  PC neighbor_relaxation_pc;
  KSPGetPC(colloid_relaxation_list[refinement_step]->get_reference(),
           &neighbor_relaxation_pc);
  PCSetType(neighbor_relaxation_pc, PCBJACOBI);
  PCSetUp(neighbor_relaxation_pc);
  PetscInt local_row, local_col;
  MatGetLocalSize(nn, &local_row, &local_col);
  if (local_row > 0) {
    KSP *neighbor_relaxation_sub_ksp;
    PCBJacobiGetSubKSP(neighbor_relaxation_pc, NULL, NULL,
                       &neighbor_relaxation_sub_ksp);
    KSPSetType(neighbor_relaxation_sub_ksp[0], KSPGMRES);
    PC neighbor_relaxation_sub_pc;
    KSPGetPC(neighbor_relaxation_sub_ksp[0], &neighbor_relaxation_sub_pc);
    PCSetType(neighbor_relaxation_sub_pc, PCLU);
    PCFactorSetMatSolverType(neighbor_relaxation_sub_pc, MATSOLVERMUMPS);
    PCSetUp(neighbor_relaxation_sub_pc);
    KSPSetUp(neighbor_relaxation_sub_ksp[0]);
  }

  KSPSetUp(colloid_relaxation_list[refinement_step]->get_reference());

  Mat &shell_mat = (*(A_list.end() - 1))->get_shell_reference();

  KSP _ksp;
  KSPCreate(PETSC_COMM_WORLD, &_ksp);
  KSPSetOperators(_ksp, shell_mat, shell_mat);
  KSPSetFromOptions(_ksp);

  PC _pc;

  KSPGetPC(_ksp, &_pc);
  PCSetType(_pc, PCSHELL);

  HypreLUShellPC *shell_ctx;
  HypreLUShellPCCreate(&shell_ctx);
  if (A_list.size() == 1) {
    PCShellSetApply(_pc, HypreLUShellPCApply);
    PCShellSetContext(_pc, shell_ctx);
    PCShellSetDestroy(_pc, HypreLUShellPCDestroy);

    HypreLUShellPCSetUp(_pc, this, _x.get_reference(), local_particle_num,
                        filed_dof);
  } else {
    MPI_Barrier(MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD,
                "start of stokes_multilevel preconditioner setup\n");

    PCShellSetApply(_pc, HypreLUShellPCApplyAdaptive);
    PCShellSetContext(_pc, shell_ctx);
    PCShellSetDestroy(_pc, HypreLUShellPCDestroy);

    HypreLUShellPCSetUp(_pc, this, _x.get_reference(), local_particle_num,
                        filed_dof);
  }

  KSPSetInitialGuessNonzero(_ksp, PETSC_TRUE);
  PetscPrintf(PETSC_COMM_WORLD, "final solving of linear system\n");
  PetscReal residual_norm, rhs_norm;
  VecNorm(_rhs.get_reference(), NORM_2, &rhs_norm);
  residual_norm = global_particle_num;
  Vec residual;
  VecDuplicate(_rhs.get_reference(), &residual);
  PetscReal rtol = 1e-6;
  int counter;
  counter = 0;
  rtol = 1e-6;
  bool diverged = false;
  do {
    KSPSetTolerances(_ksp, rtol, 1e-50, 1e20, 1000);
    KSPSolve(_ksp, _rhs.get_reference(), _x.get_reference());
    MatMult(shell_mat, _x.get_reference(), residual);
    VecAXPY(residual, -1.0, _rhs.get_reference());
    VecNorm(residual, NORM_2, &residual_norm);
    PetscPrintf(PETSC_COMM_WORLD, "relative residual norm: %f\n",
                residual_norm / rhs_norm / (double)global_particle_num);
    rtol *= 1e-2;
    counter++;

    KSPConvergedReason convergence_reason;
    KSPGetConvergedReason(_ksp, &convergence_reason);

    if (counter >= 10)
      break;
    if (residual_norm / rhs_norm / (double)global_particle_num > 1e3)
      diverged = true;
    if (convergence_reason < 0)
      diverged = true;
    if (diverged)
      break;
  } while (residual_norm / rhs_norm / (double)global_particle_num > 1);
  VecDestroy(&residual);
  PetscPrintf(PETSC_COMM_WORLD, "ksp solving finished\n");

  KSPConvergedReason reason;
  KSPGetConvergedReason(_ksp, &reason);

  _x.copy(x);

  KSPDestroy(&_ksp);

  VecDestroy(&null_field);
  VecDestroy(&null_whole);

  MatDestroy(&pp);
  MatNullSpaceDestroy(&nullspace_whole);
  MatNullSpaceDestroy(&nullspace_field);

  return 0;
}

void stokes_multilevel::clear() {
  MPI_Barrier(MPI_COMM_WORLD);

  if (base_level_initialized) {
    ksp_field_base.reset();
    ksp_colloid_base.reset();
  }

  base_level_initialized = false;

  x_pressure_list.clear();

  isg_field_list.clear();
  isg_colloid_list.clear();
  isg_pressure_list.clear();

  field_scatter_list.clear();
  colloid_scatter_list.clear();
  pressure_scatter_list.clear();

  A_list.clear();
  I_list.clear();
  R_list.clear();

  ff_list.clear();
  nn_list.clear();
  nw_list.clear();

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

  current_refinement_level = -1;
}