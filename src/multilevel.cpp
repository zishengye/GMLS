#include "multilevel.h"
#include "composite_preconditioner.h"
#include "gmls_solver.h"

#include <algorithm>
#include <iostream>

using namespace std;
using namespace Compadre;

void GMLS_Solver::BuildInterpolationAndRestrictionMatrices(PetscSparseMatrix &I,
                                                           PetscSparseMatrix &R,
                                                           int num_rigid_body,
                                                           int dimension) {
  static auto &coord = __field.vector.GetHandle("coord");
  static auto &adaptive_level = __field.index.GetHandle("adaptive level");
  static auto &background_coord = __background.vector.GetHandle("source coord");
  static auto &background_index = __background.index.GetHandle("source index");
  static auto &particleType = __field.index.GetHandle("particle type");
  static auto &newAdded = __field.index.GetHandle("new added particle flag");
  static auto &particleSize = __field.vector.GetHandle("size");

  static auto &old_coord = __field.vector.GetHandle("old coord");
  static auto &old_particle_type = __field.index.GetHandle("old particle type");
  static auto &old_background_coord =
      __background.vector.GetHandle("old source coord");
  static auto &old_background_index =
      __background.index.GetHandle("old source index");

  vector<int> recvParticleType;
  DataSwapAmongNeighbor(particleType, recvParticleType);

  vector<int> backgroundParticleType = particleType;
  backgroundParticleType.insert(backgroundParticleType.end(),
                                recvParticleType.begin(),
                                recvParticleType.end());

  int field_dof = dimension + 1;
  int velocity_dof = dimension;
  int pressure_dof = 1;

  int rigid_body_dof = (dimension == 3) ? 6 : 3;

  int translation_dof = dimension;
  int rotation_dof = (dimension == 3) ? 3 : 1;

  int old_local_particle_num = old_coord.size();
  int new_local_particle_num = coord.size();

  int old_global_particle_num, new_global_particle_num;

  MPI_Allreduce(&old_local_particle_num, &old_global_particle_num, 1, MPI_INT,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&new_local_particle_num, &new_global_particle_num, 1, MPI_INT,
                MPI_SUM, MPI_COMM_WORLD);

  int old_local_dof = field_dof * old_local_particle_num;
  int old_global_dof = field_dof * old_global_particle_num;
  int new_local_dof = field_dof * new_local_particle_num;
  int new_global_dof = field_dof * new_global_particle_num;

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
      old_source_coords_device("old source coordinates",
                               old_background_coord.size(), 3);
  Kokkos::View<double **>::HostMirror old_source_coords =
      Kokkos::create_mirror_view(old_source_coords_device);

  int actual_new_target = 0;
  vector<int> new_actual_index(coord.size());
  for (int i = 0; i < coord.size(); i++) {
    new_actual_index[i] = actual_new_target;
    if (newAdded[i] == 1)
      actual_new_target++;
  }

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
      new_target_coords_device("new target coordinates", actual_new_target, 3);
  Kokkos::View<double **>::HostMirror new_target_coords =
      Kokkos::create_mirror_view(new_target_coords_device);

  // copy old source coords
  for (int i = 0; i < old_background_coord.size(); i++) {
    for (int j = 0; j < dimension; j++)
      old_source_coords(i, j) = old_background_coord[i][j];
  }

  // copy new target coords
  int counter = 0;
  for (int i = 0; i < coord.size(); i++) {
    if (newAdded[i] == 1) {
      for (int j = 0; j < dimension; j++) {
        new_target_coords(counter, j) = coord[i][j];
      }

      counter++;
    }
  }

  Kokkos::deep_copy(old_source_coords_device, old_source_coords);
  Kokkos::deep_copy(new_target_coords_device, new_target_coords);

  auto old_to_new_point_search(
      CreatePointCloudSearch(old_source_coords_device, dimension));

  // 2.5 = polynomial_order + 0.5 = 2 + 0.5
  int estimatedUpperBoundNumberNeighbors =
      pow(2, dimension) * pow(2 * 3.5, dimension);

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
      old_to_new_neighbor_lists_device("old to new neighbor lists",
                                       actual_new_target,
                                       estimatedUpperBoundNumberNeighbors);
  Kokkos::View<int **>::HostMirror old_to_new_neighbor_lists =
      Kokkos::create_mirror_view(old_to_new_neighbor_lists_device);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> old_epsilon_device(
      "h supports", actual_new_target);
  Kokkos::View<double *>::HostMirror old_epsilon =
      Kokkos::create_mirror_view(old_epsilon_device);

  for (int i = 0; i < coord.size(); i++) {
    if (newAdded[i] == 1) {
      old_epsilon[new_actual_index[i]] = 3.5 * particleSize[i][0];
    }
  }

  auto neighbor_needed =
      Compadre::GMLS::getNP(2, __dim, DivergenceFreeVectorTaylorPolynomial);
  // old_to_new_point_search.generateNeighborListsFromKNNSearch(
  //     false, new_target_coords, old_to_new_neighbor_lists, old_epsilon,
  //     neighbor_needed, 1.2);
  size_t actual_neighbor_max;

  while (true) {
    actual_neighbor_max =
        old_to_new_point_search.generateNeighborListsFromRadiusSearch(
            true, new_target_coords, old_to_new_neighbor_lists, old_epsilon,
            0.0, 0.0);
    while (actual_neighbor_max > estimatedUpperBoundNumberNeighbors) {
      estimatedUpperBoundNumberNeighbors *= 2;
      old_to_new_neighbor_lists_device =
          Kokkos::View<int **, Kokkos::DefaultExecutionSpace>(
              "old to new neighbor lists", actual_new_target,
              estimatedUpperBoundNumberNeighbors);
      old_to_new_neighbor_lists =
          Kokkos::create_mirror_view(old_to_new_neighbor_lists_device);
    }
    old_to_new_point_search.generateNeighborListsFromRadiusSearch(
        false, new_target_coords, old_to_new_neighbor_lists, old_epsilon, 0.0,
        0.0);

    bool enough_neighbor = true;
    for (int i = 0; i < coord.size(); i++) {
      if (newAdded[i] == 1) {
        if (old_to_new_neighbor_lists(new_actual_index[i], 0) <
            neighbor_needed) {
          old_epsilon[new_actual_index[i]] += 0.5 * particleSize[i][0];
          enough_neighbor = false;
        }
      }
    }

    if (enough_neighbor)
      break;
  }

  Kokkos::deep_copy(old_to_new_neighbor_lists_device,
                    old_to_new_neighbor_lists);
  Kokkos::deep_copy(old_epsilon_device, old_epsilon);

  GMLS old_to_new_pressusre_basis(ScalarTaylorPolynomial, PointSample, 2,
                                  dimension, "SVD", "STANDARD");
  auto old_to_new_velocity_basis =
      new GMLS(DivergenceFreeVectorTaylorPolynomial, VectorPointSample, 2,
               dimension, "SVD", "STANDARD");

  // old to new pressure field transition
  old_to_new_pressusre_basis.setProblemData(
      old_to_new_neighbor_lists_device, old_source_coords_device,
      new_target_coords_device, old_epsilon_device);

  old_to_new_pressusre_basis.addTargets(ScalarPointEvaluation);

  old_to_new_pressusre_basis.setWeightingType(WeightingFunctionType::Power);
  old_to_new_pressusre_basis.setWeightingPower(__weightFuncOrder);

  old_to_new_pressusre_basis.generateAlphas(1);

  auto old_to_new_pressure_alphas = old_to_new_pressusre_basis.getAlphas();

  // old to new velocity field transition
  old_to_new_velocity_basis->setProblemData(
      old_to_new_neighbor_lists_device, old_source_coords_device,
      new_target_coords_device, old_epsilon_device);

  old_to_new_velocity_basis->addTargets(VectorPointEvaluation);

  old_to_new_velocity_basis->setWeightingType(WeightingFunctionType::Power);
  old_to_new_velocity_basis->setWeightingPower(__weightFuncOrder);

  old_to_new_velocity_basis->generateAlphas(1);

  auto old_to_new_velocity_alphas = old_to_new_velocity_basis->getAlphas();

  // old to new interpolation matrix
  if (__myID == __MPISize - 1)
    I.resize(new_local_dof + rigid_body_dof * num_rigid_body,
             old_local_dof + rigid_body_dof * num_rigid_body,
             old_global_dof + rigid_body_dof * num_rigid_body);
  else
    I.resize(new_local_dof, old_local_dof,
             old_global_dof + rigid_body_dof * num_rigid_body);
  // compute matrix graph
  vector<PetscInt> index;
  for (int i = 0; i < new_local_particle_num; i++) {
    if (newAdded[i] == 1) {
      // velocity interpolation
      index.resize(old_to_new_neighbor_lists(new_actual_index[i], 0) *
                   velocity_dof);
      for (int j = 0; j < old_to_new_neighbor_lists(new_actual_index[i], 0);
           j++) {
        for (int k = 0; k < velocity_dof; k++) {
          index[j * velocity_dof + k] =
              field_dof * old_background_index[old_to_new_neighbor_lists(
                              new_actual_index[i], j + 1)] +
              k;
        }
      }

      for (int k = 0; k < velocity_dof; k++) {
        I.setColIndex(field_dof * i + k, index);
      }

      // pressure interpolation
      index.resize(old_to_new_neighbor_lists(new_actual_index[i], 0));
      for (int j = 0; j < old_to_new_neighbor_lists(new_actual_index[i], 0);
           j++) {
        index[j] = field_dof * old_background_index[old_to_new_neighbor_lists(
                                   new_actual_index[i], j + 1)] +
                   velocity_dof;
      }
      I.setColIndex(field_dof * i + velocity_dof, index);
    } else {
      index.resize(1);
      for (int j = 0; j < field_dof; j++) {
        index[0] = field_dof * old_background_index[i] + j;
        I.setColIndex(field_dof * i + j, index);
      }
    }
  }

  // rigid body
  if (__myID == __MPISize - 1) {
    index.resize(1);
    for (int i = 0; i < num_rigid_body; i++) {
      int local_rigid_body_index_offset =
          field_dof * new_local_particle_num + i * rigid_body_dof;
      for (int j = 0; j < rigid_body_dof; j++) {
        index[0] = field_dof * old_global_particle_num + i * rigid_body_dof + j;
        I.setColIndex(local_rigid_body_index_offset + j, index);
      }
    }
  }

  // compute interpolation matrix entity
  const auto pressure_old_to_new_alphas_index =
      old_to_new_pressusre_basis.getAlphaColumnOffset(ScalarPointEvaluation, 0,
                                                      0, 0, 0);
  vector<int> velocity_old_to_new_alphas_index(pow(dimension, 2));
  for (int axes1 = 0; axes1 < dimension; axes1++)
    for (int axes2 = 0; axes2 < dimension; axes2++)
      velocity_old_to_new_alphas_index[axes1 * dimension + axes2] =
          old_to_new_velocity_basis->getAlphaColumnOffset(VectorPointEvaluation,
                                                          axes1, 0, axes2, 0);

  for (int i = 0; i < new_local_particle_num; i++) {
    if (newAdded[i] == 1) {
      for (int j = 0; j < old_to_new_neighbor_lists(new_actual_index[i], 0);
           j++) {
        for (int axes1 = 0; axes1 < dimension; axes1++)
          for (int axes2 = 0; axes2 < dimension; axes2++)
            I.increment(
                field_dof * i + axes1,
                field_dof * old_background_index[old_to_new_neighbor_lists(
                                new_actual_index[i], j + 1)] +
                    axes2,
                old_to_new_velocity_alphas(
                    new_actual_index[i],
                    velocity_old_to_new_alphas_index[axes1 * dimension + axes2],
                    j));
      }

      for (int j = 0; j < old_to_new_neighbor_lists(new_actual_index[i], 0);
           j++) {
        I.increment(field_dof * i + velocity_dof,
                    field_dof * old_background_index[old_to_new_neighbor_lists(
                                    new_actual_index[i], j + 1)] +
                        velocity_dof,
                    old_to_new_pressure_alphas(new_actual_index[i],
                                               pressure_old_to_new_alphas_index,
                                               j));
      }
    } else {
      for (int j = 0; j < field_dof; j++) {
        I.increment(field_dof * i + j, field_dof * old_background_index[i] + j,
                    1.0);
      }
    }
  }

  // rigid body
  if (__myID == __MPISize - 1) {
    for (int i = 0; i < num_rigid_body; i++) {
      int local_rigid_body_index_offset =
          field_dof * new_local_particle_num + i * rigid_body_dof;
      for (int j = 0; j < rigid_body_dof; j++) {
        I.increment(
            local_rigid_body_index_offset + j,
            field_dof * old_global_particle_num + i * rigid_body_dof + j, 1.0);
      }
    }
  }

  I.FinalAssemble();

  static vector<double> &pressure = __field.scalar.GetHandle("fluid pressure");

  // if (new_local_particle_num > 11608) {
  //   cout << coord[11607][0] << ' ' << coord[11607][1] << endl << endl;
  //   for (int i = 0; i < old_to_new_neighbor_lists(new_actual_index[11607],
  //   0);
  //        i++) {
  //     int index = old_to_new_neighbor_lists(new_actual_index[11607], i + 1);
  //     cout << old_background_coord[index][0] << ' '
  //          << old_background_coord[index][1] << ' '
  //          << old_to_new_pressure_alphas(new_actual_index[11607],
  //                                        pressure_old_to_new_alphas_index, i)
  //          << endl;
  //   }
  //   cout << endl;
  //   for (int i = 0; i < old_to_new_neighbor_lists(new_actual_index[11607],
  //   0);
  //        i++) {
  //     int index = old_to_new_neighbor_lists(new_actual_index[11607], i + 1);
  //     cout << pressure[index] << endl;
  //   }

  //   double pressure_sum = 0.0;
  //   for (int i = 0; i < old_to_new_neighbor_lists(new_actual_index[11607],
  //   0);
  //        i++) {
  //     int index = old_to_new_neighbor_lists(new_actual_index[11607], i + 1);
  //     pressure_sum +=
  //         pressure[index] *
  //         old_to_new_pressure_alphas(new_actual_index[11607],
  //                                    pressure_old_to_new_alphas_index, i);
  //   }

  //   cout << pressure_sum << endl;
  // }

  // new to old restriction matrix
  if (__myID == __MPISize - 1)
    R.resize(old_local_dof + num_rigid_body * rigid_body_dof,
             new_local_dof + num_rigid_body * rigid_body_dof,
             new_global_dof + num_rigid_body * rigid_body_dof);
  else
    R.resize(old_local_dof, new_local_dof,
             new_global_dof + num_rigid_body * rigid_body_dof);

  // compute restriction matrix graph
  for (int i = 0; i < old_local_particle_num; i++) {
    if (fieldParticleSplitTag[i]) {
      index.resize(splitList[i].size());
      for (int j = 0; j < field_dof; j++) {
        for (int k = 0; k < splitList[i].size(); k++) {
          index[k] = background_index[splitList[i][k]] * field_dof + j;
        }

        R.setColIndex(field_dof * i + j, index);
      }
    } else {
      index.resize(1);
      for (int k = 0; k < field_dof; k++) {
        index[0] = field_dof * background_index[i] + k;
        R.setColIndex(field_dof * i + k, index);
      }
    }
  }

  // rigid body
  if (__myID == __MPISize - 1) {
    index.resize(1);
    for (int i = 0; i < num_rigid_body; i++) {
      int local_rigid_body_index_offset =
          field_dof * old_local_particle_num + i * rigid_body_dof;
      for (int j = 0; j < rigid_body_dof; j++) {
        index[0] = field_dof * new_global_particle_num + i * rigid_body_dof + j;
        R.setColIndex(local_rigid_body_index_offset + j, index);
      }
    }
  }

  for (int i = 0; i < old_local_particle_num; i++) {
    if (fieldParticleSplitTag[i]) {
      for (int j = 0; j < field_dof; j++) {
        for (int k = 0; k < splitList[i].size(); k++) {
          R.increment(field_dof * i + j,
                      background_index[splitList[i][k]] * field_dof + j,
                      1.0 / splitList[i].size());
        }
      }
    } else {
      for (int k = 0; k < field_dof; k++) {
        R.increment(field_dof * i + k, field_dof * background_index[i] + k,
                    1.0);
      }
    }
  }

  // rigid body
  if (__myID == __MPISize - 1) {
    index.resize(1);
    for (int i = 0; i < num_rigid_body; i++) {
      int local_rigid_body_index_offset =
          field_dof * old_local_particle_num + i * rigid_body_dof;
      for (int j = 0; j < rigid_body_dof; j++) {
        R.increment(
            local_rigid_body_index_offset + j,
            field_dof * new_global_particle_num + i * rigid_body_dof + j, 1.0);
      }
    }
  }

  R.FinalAssemble();

  delete old_to_new_velocity_basis;
}

void multilevel::InitialGuessFromPreviousAdaptiveStep(
    std::vector<double> &initial_guess) {}

int multilevel::Solve(std::vector<double> &rhs, std::vector<double> &x,
                      std::vector<int> &idx_neighbor) {
  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "\nstart of linear system solving setup\n");

  int adaptive_step = A_list.size() - 1;

  int fieldDof = dimension + 1;
  int velocityDof = dimension;
  int pressureDof = 1;
  int rigidBodyDof = (dimension == 3) ? 6 : 3;

  vector<int> idx_field;
  vector<int> idx_pressure;

  PetscInt localN1, localN2;
  Mat &mat = (*(A_list.end() - 1))->__mat;
  MatGetOwnershipRange(mat, &localN1, &localN2);

  int localParticleNum;
  if (myid != mpi_size - 1) {
    localParticleNum = (localN2 - localN1) / fieldDof;
    idx_field.resize(fieldDof * localParticleNum);
    idx_pressure.resize(localParticleNum);

    for (int i = 0; i < localParticleNum; i++) {
      for (int j = 0; j < dimension; j++) {
        idx_field[fieldDof * i + j] = localN1 + fieldDof * i + j;
      }
      idx_field[fieldDof * i + velocityDof] =
          localN1 + fieldDof * i + velocityDof;

      idx_pressure[i] = localN1 + fieldDof * i + velocityDof;
    }
  } else {
    localParticleNum =
        (localN2 - localN1 - num_rigid_body * rigidBodyDof) / fieldDof;
    idx_field.resize(fieldDof * localParticleNum);
    idx_pressure.resize(localParticleNum);

    for (int i = 0; i < localParticleNum; i++) {
      for (int j = 0; j < dimension; j++) {
        idx_field[fieldDof * i + j] = localN1 + fieldDof * i + j;
      }
      idx_field[fieldDof * i + velocityDof] =
          localN1 + fieldDof * i + velocityDof;

      idx_pressure[i] = localN1 + fieldDof * i + velocityDof;
    }
  }

  int globalParticleNum;
  MPI_Allreduce(&localParticleNum, &globalParticleNum, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  IS &isg_field_lag = *isg_field_list[adaptive_step];
  IS &isg_neighbor = *isg_neighbor_list[adaptive_step];
  IS &isg_pressure = *isg_pressure_list[adaptive_step];

  ISCreateGeneral(MPI_COMM_SELF, idx_field.size(), idx_field.data(),
                  PETSC_COPY_VALUES, &isg_field_lag);
  ISCreateGeneral(MPI_COMM_WORLD, idx_neighbor.size(), idx_neighbor.data(),
                  PETSC_COPY_VALUES, &isg_neighbor);
  ISCreateGeneral(MPI_COMM_SELF, idx_pressure.size(), idx_pressure.data(),
                  PETSC_COPY_VALUES, &isg_pressure);

  Vec _rhs, _x;
  VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, rhs.size(), PETSC_DECIDE,
                        rhs.data(), &_rhs);
  VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, x.size(), PETSC_DECIDE, x.data(),
                        &_x);

  Mat &ff = *ff_list[adaptive_step];
  Mat &nn = *nn_list[adaptive_step];
  Mat &nw = *nw_list[adaptive_step];
  Mat pp;

  MatCreateSubMatrix(mat, isg_neighbor, isg_neighbor, MAT_INITIAL_MATRIX, &nn);
  MatCreateSubMatrix(mat, isg_pressure, isg_pressure, MAT_INITIAL_MATRIX, &pp);
  MatCreateSubMatrix(mat, isg_neighbor, NULL, MAT_INITIAL_MATRIX, &nw);

  // setup current level vectors
  x_list.push_back(new Vec);
  y_list.push_back(new Vec);
  b_list.push_back(new Vec);
  r_list.push_back(new Vec);
  t_list.push_back(new Vec);

  x_field_list.push_back(new Vec);
  y_field_list.push_back(new Vec);
  b_field_list.push_back(new Vec);
  r_field_list.push_back(new Vec);
  t_field_list.push_back(new Vec);

  x_neighbor_list.push_back(new Vec);
  y_neighbor_list.push_back(new Vec);
  b_neighbor_list.push_back(new Vec);
  r_neighbor_list.push_back(new Vec);
  t_neighbor_list.push_back(new Vec);

  x_pressure_list.push_back(new Vec);

  field_relaxation_list.push_back(new KSP);
  neighbor_relaxation_list.push_back(new KSP);

  MatCreateVecs(mat, NULL, x_list[adaptive_step]);
  MatCreateVecs(mat, NULL, y_list[adaptive_step]);
  MatCreateVecs(mat, NULL, b_list[adaptive_step]);
  MatCreateVecs(mat, NULL, r_list[adaptive_step]);
  MatCreateVecs(mat, NULL, t_list[adaptive_step]);

  MatCreateVecs(ff, NULL, x_field_list[adaptive_step]);
  MatCreateVecs(ff, NULL, y_field_list[adaptive_step]);
  MatCreateVecs(ff, NULL, b_field_list[adaptive_step]);
  MatCreateVecs(ff, NULL, r_field_list[adaptive_step]);
  MatCreateVecs(ff, NULL, t_field_list[adaptive_step]);

  MatCreateVecs(nn, NULL, x_neighbor_list[adaptive_step]);
  MatCreateVecs(nn, NULL, y_neighbor_list[adaptive_step]);
  MatCreateVecs(nn, NULL, b_neighbor_list[adaptive_step]);
  MatCreateVecs(nn, NULL, r_neighbor_list[adaptive_step]);
  MatCreateVecs(nn, NULL, t_neighbor_list[adaptive_step]);

  MatCreateVecs(pp, NULL, x_pressure_list[adaptive_step]);

  // field vector scatter
  field_scatter_list.push_back(new VecScatter);
  VecScatterCreate(*x_list[adaptive_step], isg_field_lag,
                   *x_field_list[adaptive_step], NULL,
                   field_scatter_list[adaptive_step]);

  neighbor_scatter_list.push_back(new VecScatter);
  VecScatterCreate(*x_list[adaptive_step], isg_neighbor,
                   *x_neighbor_list[adaptive_step], NULL,
                   neighbor_scatter_list[adaptive_step]);
  pressure_scatter_list.push_back(new VecScatter);
  VecScatterCreate(*x_list[adaptive_step], isg_pressure,
                   *x_pressure_list[adaptive_step], NULL,
                   pressure_scatter_list[adaptive_step]);

  // setup nullspace_field
  Vec null_whole, null_field;
  VecDuplicate(_rhs, &null_whole);
  VecDuplicate(*x_field_list[adaptive_step], &null_field);

  VecSet(null_whole, 0.0);
  VecSet(null_field, 0.0);

  VecSet(*x_pressure_list[adaptive_step], 1.0);

  VecScatterBegin(*pressure_scatter_list[adaptive_step],
                  *x_pressure_list[adaptive_step], null_whole, INSERT_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(*pressure_scatter_list[adaptive_step],
                *x_pressure_list[adaptive_step], null_whole, INSERT_VALUES,
                SCATTER_REVERSE);

  nullspace_whole_list.push_back(new MatNullSpace);
  MatNullSpace &nullspace_whole = *nullspace_whole_list[adaptive_step];
  MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &null_whole,
                     &nullspace_whole);
  MatSetNearNullSpace(mat, nullspace_whole);

  Vec field_pressure;
  VecGetSubVector(null_field, isg_pressure, &field_pressure);
  VecSet(field_pressure, 1.0);
  VecRestoreSubVector(null_field, isg_pressure, &field_pressure);

  nullspace_field_list.push_back(new MatNullSpace);
  MatNullSpace &nullspace_field = *nullspace_field_list[adaptive_step];
  MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &null_field,
                     &nullspace_field);
  MatSetNearNullSpace(ff, nullspace_field);

  // neighbor vector scatter, only needed on base level
  if (adaptive_step == 0) {
    MatCreateVecs(nn, NULL, &x_neighbor);
    MatCreateVecs(nn, NULL, &y_neighbor);
  }

  // setup preconditioner for base level
  if (adaptive_step == 0) {
    KSPCreate(PETSC_COMM_WORLD, &ksp_field_base);
    KSPCreate(PETSC_COMM_WORLD, &ksp_neighbor_base);

    KSPSetOperators(ksp_field_base, ff, ff);
    KSPSetOperators(ksp_neighbor_base, nn, nn);

    KSPSetType(ksp_field_base, KSPPREONLY);
    KSPSetType(ksp_neighbor_base, KSPPREONLY);

    PC pc_field_base;
    PC pc_neighbor_base;

    KSPGetPC(ksp_field_base, &pc_field_base);
    PCSetType(pc_field_base, PCHYPRE);
    PCSetFromOptions(pc_field_base);
    PCSetUp(pc_field_base);

    KSPGetPC(ksp_neighbor_base, &pc_neighbor_base);
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
      // PCFactorSetMatSolverType(bjacobi_pc, MATSOLVERMUMPS);
      // PetscOptionsSetValue(NULL, "-pc_hypre_type", "euclid");
      PCSetFromOptions(bjacobi_pc);
      PCSetUp(bjacobi_pc);
      KSPSetUp(bjacobi_ksp[0]);
    }

    KSPSetUp(ksp_field_base);
    KSPSetUp(ksp_neighbor_base);
  }

  // setup relaxation on field for current level
  KSPCreate(MPI_COMM_WORLD, field_relaxation_list[adaptive_step]);

  KSPSetType(*field_relaxation_list[adaptive_step], KSPPREONLY);
  KSPSetOperators(*field_relaxation_list[adaptive_step], ff, ff);

  PC field_relaxation_pc;
  KSPGetPC(*field_relaxation_list[adaptive_step], &field_relaxation_pc);
  PCSetType(field_relaxation_pc, PCSOR);
  PCSetFromOptions(field_relaxation_pc);
  PCSetUp(field_relaxation_pc);

  KSPSetUp(*field_relaxation_list[adaptive_step]);

  // setup relaxation on neighbor for current level
  KSPCreate(MPI_COMM_WORLD, neighbor_relaxation_list[adaptive_step]);

  KSPSetType(*neighbor_relaxation_list[adaptive_step], KSPPREONLY);
  KSPSetOperators(*neighbor_relaxation_list[adaptive_step], nn, nn);

  PC neighbor_relaxation_pc;
  KSPGetPC(*neighbor_relaxation_list[adaptive_step], &neighbor_relaxation_pc);
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

  KSPSetUp(*neighbor_relaxation_list[adaptive_step]);

  Mat shell_mat = (*(A_list.end() - 1))->__shell_mat;

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

    HypreLUShellPCSetUp(_pc, this, _x, localParticleNum, fieldDof);
  } else {
    MPI_Barrier(MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "start of multilevel preconditioner setup\n");

    PCShellSetApply(_pc, HypreLUShellPCApplyAdaptive);
    PCShellSetContext(_pc, shell_ctx);
    PCShellSetDestroy(_pc, HypreLUShellPCDestroy);

    HypreLUShellPCSetUp(_pc, this, _x, localParticleNum, fieldDof);
  }

  double tStart, tEnd;
  PetscScalar *a;

  KSPSetInitialGuessNonzero(_ksp, PETSC_TRUE);
  MPI_Barrier(MPI_COMM_WORLD);
  tStart = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "final solving of linear system\n");
  PetscReal residual_norm, rhs_norm;
  VecNorm(_rhs, NORM_2, &rhs_norm);
  residual_norm = globalParticleNum;
  Vec residual;
  VecDuplicate(_rhs, &residual);
  PetscReal rtol = 1e-6;
  int counter;
  counter = 0;
  rtol = 1e-6;
  bool diverged = false;
  do {
    KSPSetTolerances(_ksp, rtol, 1e-50, 1e20, 200);
    KSPSolve(_ksp, _rhs, _x);
    MatMult(shell_mat, _x, residual);
    VecAXPY(residual, -1.0, _rhs);
    VecNorm(residual, NORM_2, &residual_norm);
    PetscPrintf(PETSC_COMM_WORLD, "relative residual norm: %f\n",
                residual_norm / rhs_norm / (double)globalParticleNum);
    rtol *= 1e-2;
    counter++;

    KSPConvergedReason convergence_reason;
    KSPGetConvergedReason(_ksp, &convergence_reason);

    if (counter >= 10)
      break;
    if (residual_norm / rhs_norm / (double)globalParticleNum > 1e3)
      diverged = true;
    if (convergence_reason < 0)
      diverged = true;
    if (diverged)
      break;
  } while (residual_norm / rhs_norm / (double)globalParticleNum > 1);
  // KSPSolve(_ksp, _rhs, _x);
  VecDestroy(&residual);
  PetscPrintf(PETSC_COMM_WORLD, "ksp solving finished\n");
  tEnd = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "pc apply time: %fs\n", tEnd - tStart);

  KSPConvergedReason reason;
  KSPGetConvergedReason(_ksp, &reason);

  VecGetArray(_x, &a);
  if (reason >= 0 && counter < 10 && diverged == false)
    for (size_t i = 0; i < rhs.size(); i++) {
      x[i] = a[i];
    }
  VecRestoreArray(_x, &a);

  KSPDestroy(&_ksp);

  MatDestroy(&pp);

  VecDestroy(&_rhs);
  VecDestroy(&_x);
  VecDestroy(&null_field);
  VecDestroy(&null_whole);

  if (reason < 0 || counter == 10 || diverged) {
    if (A_list.size() == 1)
      return -1;
  }

  return 0;
}

void multilevel::clear() {
  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = 0; i < current_adaptive_level; i++) {
    KSPDestroy(field_relaxation_list[i]);
    KSPDestroy(neighbor_relaxation_list[i]);

    delete field_relaxation_list[i];
    delete neighbor_relaxation_list[i];
  }
  field_relaxation_list.clear();
  neighbor_relaxation_list.clear();

  if (base_level_initialized) {
    KSPDestroy(&ksp_field_base);
    KSPDestroy(&ksp_neighbor_base);

    VecDestroy(&x_neighbor);
    VecDestroy(&y_neighbor);

    base_level_initialized = false;
  }

  // mat clearance
  for (int i = 0; i < current_adaptive_level; i++) {
    MatNullSpaceDestroy(nullspace_whole_list[i]);
    MatNullSpaceDestroy(nullspace_field_list[i]);

    delete nullspace_whole_list[i];
    delete nullspace_field_list[i];

    MatSetNearNullSpace(A_list[i]->__mat, NULL);
    delete A_list[i];
    delete I_list[i];
    delete R_list[i];

    MatSetNearNullSpace(*ff_list[i], NULL);
    MatDestroy(ff_list[i]);
    MatDestroy(nn_list[i]);
    MatDestroy(nw_list[i]);

    delete ff_list[i];
    delete nn_list[i];
    delete nw_list[i];

    VecDestroy(x_list[i]);
    VecDestroy(y_list[i]);
    VecDestroy(b_list[i]);
    VecDestroy(r_list[i]);
    VecDestroy(t_list[i]);

    delete x_list[i];
    delete y_list[i];
    delete b_list[i];
    delete r_list[i];
    delete t_list[i];

    VecDestroy(x_field_list[i]);
    VecDestroy(y_field_list[i]);
    VecDestroy(b_field_list[i]);
    VecDestroy(r_field_list[i]);
    VecDestroy(t_field_list[i]);

    delete x_field_list[i];
    delete y_field_list[i];
    delete b_field_list[i];
    delete r_field_list[i];
    delete t_field_list[i];

    VecDestroy(x_neighbor_list[i]);
    VecDestroy(y_neighbor_list[i]);
    VecDestroy(b_neighbor_list[i]);
    VecDestroy(r_neighbor_list[i]);
    VecDestroy(t_neighbor_list[i]);

    delete x_neighbor_list[i];
    delete y_neighbor_list[i];
    delete b_neighbor_list[i];
    delete r_neighbor_list[i];
    delete t_neighbor_list[i];

    VecDestroy(x_pressure_list[i]);

    delete x_pressure_list[i];

    ISDestroy(isg_field_list[i]);
    ISDestroy(isg_neighbor_list[i]);
    ISDestroy(isg_pressure_list[i]);

    delete isg_field_list[i];
    delete isg_neighbor_list[i];
    delete isg_pressure_list[i];

    VecScatterDestroy(field_scatter_list[i]);
    VecScatterDestroy(neighbor_scatter_list[i]);
    VecScatterDestroy(pressure_scatter_list[i]);

    delete field_scatter_list[i];
    delete neighbor_scatter_list[i];
    delete pressure_scatter_list[i];
  }

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

  x_neighbor_list.clear();
  y_neighbor_list.clear();
  b_neighbor_list.clear();
  r_neighbor_list.clear();
  t_neighbor_list.clear();

  x_pressure_list.clear();

  isg_field_list.clear();
  isg_neighbor_list.clear();
  isg_pressure_list.clear();

  field_scatter_list.clear();
  neighbor_scatter_list.clear();
  pressure_scatter_list.clear();

  nullspace_whole_list.clear();
  nullspace_field_list.clear();

  current_adaptive_level = 0;
}