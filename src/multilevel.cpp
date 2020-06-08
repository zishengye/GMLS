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

  static auto &old_coord = __field.vector.GetHandle("old coord");
  static auto &old_particle_type = __field.index.GetHandle("old particle type");
  static auto &old_background_coord =
      __background.vector.GetHandle("old source coord");
  static auto &old_background_index =
      __background.index.GetHandle("old source index");

  vector<int> recvParticleType;
  DataSwapAmongNeighbor(particleType, recvParticleType);

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

  int old_local_dof = (__myID == __MPISize - 1)
                          ? field_dof * (old_local_particle_num + 1)
                          : field_dof * old_local_particle_num;
  int old_global_dof = field_dof * (old_global_particle_num + 1);
  int new_local_dof = (__myID == __MPISize - 1)
                          ? field_dof * (new_local_particle_num + 1)
                          : field_dof * new_local_particle_num;
  int new_global_dof = field_dof * (new_global_particle_num + 1);

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
      new_source_coords_device("new source coordinates",
                               background_coord.size(), 3);
  Kokkos::View<double **>::HostMirror new_source_coords =
      Kokkos::create_mirror_view(new_source_coords_device);

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
      old_source_coords_device("old source coordinates",
                               old_background_coord.size(), 3);
  Kokkos::View<double **>::HostMirror old_source_coords =
      Kokkos::create_mirror_view(old_source_coords_device);

  int actual_new_target = 0;
  vector<int> new_actual_index(coord.size());
  for (int i = 0; i < coord.size(); i++) {
    new_actual_index[i] = actual_new_target;
    if (adaptive_level[i] == __adaptive_step)
      actual_new_target++;
  }

  int actual_old_target = 0;
  vector<int> old_actual_index(old_coord.size());
  for (int i = 0; i < old_coord.size(); i++) {
    old_actual_index[i] = actual_old_target;
    if (fieldParticleSplitTag[i]) {
      actual_old_target++;
    }
  }

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
      new_target_coords_device("new target coordinates", actual_new_target, 3);
  Kokkos::View<double **>::HostMirror new_target_coords =
      Kokkos::create_mirror_view(new_target_coords_device);

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
      old_target_coords_device("old target coordinates", actual_old_target, 3);
  Kokkos::View<double **>::HostMirror old_target_coords =
      Kokkos::create_mirror_view(old_target_coords_device);

  // copy old source coords
  for (int i = 0; i < old_background_coord.size(); i++) {
    for (int j = 0; j < dimension; j++)
      old_source_coords(i, j) = old_background_coord[i][j];
  }

  // copy new source coords
  for (int i = 0; i < background_coord.size(); i++) {
    for (int j = 0; j < dimension; j++)
      new_source_coords(i, j) = background_coord[i][j];
  }

  // copy old target coords
  int counter = 0;
  for (int i = 0; i < old_coord.size(); i++) {
    if (fieldParticleSplitTag[i]) {
      for (int j = 0; j < dimension; j++)
        old_target_coords(counter, j) = old_coord[i][j];

      counter++;
    }
  }

  // copy new target coords
  counter = 0;
  for (int i = 0; i < coord.size(); i++) {
    if (adaptive_level[i] == __adaptive_step) {
      for (int j = 0; j < dimension; j++) {
        new_target_coords(counter, j) = coord[i][j];
      }

      counter++;
    }
  }

  Kokkos::deep_copy(old_source_coords_device, old_source_coords);
  Kokkos::deep_copy(new_source_coords_device, new_source_coords);
  Kokkos::deep_copy(old_target_coords_device, old_target_coords);
  Kokkos::deep_copy(new_target_coords_device, new_target_coords);

  auto new_to_old_point_search(
      CreatePointCloudSearch(new_source_coords_device, dimension));
  auto old_to_new_point_search(
      CreatePointCloudSearch(old_source_coords_device, dimension));

  // 2.5 = polynomial_order + 0.5 = 2 + 0.5
  int estimatedUpperBoundNumberNeighbors =
      pow(2, dimension) * pow(2 * 2.5, dimension);

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
      new_to_old_neighbor_lists_device("new to old neighbor lists",
                                       actual_old_target,
                                       estimatedUpperBoundNumberNeighbors);
  Kokkos::View<int **>::HostMirror new_to_old_neighbor_lists =
      Kokkos::create_mirror_view(new_to_old_neighbor_lists_device);

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
      old_to_new_neighbor_lists_device("old to new neighbor lists",
                                       actual_new_target,
                                       estimatedUpperBoundNumberNeighbors);
  Kokkos::View<int **>::HostMirror old_to_new_neighbor_lists =
      Kokkos::create_mirror_view(old_to_new_neighbor_lists_device);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> new_epsilon_device(
      "h supports", actual_old_target);
  Kokkos::View<double *>::HostMirror new_epsilon =
      Kokkos::create_mirror_view(new_epsilon_device);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> old_epsilon_device(
      "h supports", actual_new_target);
  Kokkos::View<double *>::HostMirror old_epsilon =
      Kokkos::create_mirror_view(old_epsilon_device);

  auto neighbor_needed = Compadre::GMLS::getNP(
      __polynomialOrder, __dim, DivergenceFreeVectorTaylorPolynomial);
  new_to_old_point_search.generateNeighborListsFromKNNSearch(
      false, old_target_coords, new_to_old_neighbor_lists, new_epsilon,
      neighbor_needed, 1.2);
  old_to_new_point_search.generateNeighborListsFromKNNSearch(
      false, new_target_coords, old_to_new_neighbor_lists, old_epsilon,
      neighbor_needed, 1.2);

  Kokkos::deep_copy(new_to_old_neighbor_lists_device,
                    new_to_old_neighbor_lists);
  Kokkos::deep_copy(new_epsilon_device, new_epsilon);
  Kokkos::deep_copy(old_to_new_neighbor_lists_device,
                    old_to_new_neighbor_lists);
  Kokkos::deep_copy(old_epsilon_device, old_epsilon);

  auto new_to_old_pressusre_basis = new GMLS(
      ScalarTaylorPolynomial, PointSample, 2, dimension, "LU", "STANDARD");
  auto new_to_old_velocity_basis =
      new GMLS(DivergenceFreeVectorTaylorPolynomial, VectorPointSample, 2,
               dimension, "SVD", "STANDARD");
  auto old_to_new_pressusre_basis = new GMLS(
      ScalarTaylorPolynomial, PointSample, 2, dimension, "LU", "STANDARD");
  auto old_to_new_velocity_basis =
      new GMLS(DivergenceFreeVectorTaylorPolynomial, VectorPointSample, 2,
               dimension, "SVD", "STANDARD");

  // old to new pressure field transition
  old_to_new_pressusre_basis->setProblemData(
      old_to_new_neighbor_lists_device, old_source_coords_device,
      new_target_coords_device, old_epsilon);

  old_to_new_pressusre_basis->addTargets(ScalarPointEvaluation);

  old_to_new_pressusre_basis->setWeightingType(WeightingFunctionType::Power);
  old_to_new_pressusre_basis->setWeightingPower(__weightFuncOrder);
  old_to_new_pressusre_basis->setOrderOfQuadraturePoints(2);
  old_to_new_pressusre_basis->setDimensionOfQuadraturePoints(1);
  old_to_new_pressusre_basis->setQuadratureType("LINE");

  old_to_new_pressusre_basis->generateAlphas(1);

  auto old_to_new_pressure_alphas = old_to_new_pressusre_basis->getAlphas();

  // old to new velocity field transition
  old_to_new_velocity_basis->setProblemData(
      old_to_new_neighbor_lists_device, old_source_coords_device,
      new_target_coords_device, old_epsilon);

  old_to_new_velocity_basis->addTargets(VectorPointEvaluation);

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
    if (adaptive_level[i] == __adaptive_step) {
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

  // lagrange multiplier
  if (__myID == __MPISize - 1) {
    index.resize(1);
    for (int j = 0; j < field_dof; j++) {
      index[0] = field_dof * old_global_particle_num + j;
      I.setColIndex(field_dof * new_local_particle_num + j, index);
    }
  }

  // rigid body
  if (__myID == __MPISize - 1) {
    index.resize(1);
    for (int i = 0; i < num_rigid_body; i++) {
      int local_rigid_body_index_offset =
          field_dof * (new_local_particle_num + 1) + i * rigid_body_dof;
      for (int j = 0; j < rigid_body_dof; j++) {
        index[0] =
            field_dof * (old_global_particle_num + 1) + i * rigid_body_dof + j;
        I.setColIndex(local_rigid_body_index_offset + j, index);
      }
    }
  }

  // compute matrix entity
  const auto pressure_old_to_new_alphas_index =
      old_to_new_pressusre_basis->getAlphaColumnOffset(ScalarPointEvaluation, 0,
                                                       0, 0, 0);
  vector<int> velocity_old_to_new_alphas_index(pow(dimension, 2));
  for (int axes1 = 0; axes1 < dimension; axes1++)
    for (int axes2 = 0; axes2 < dimension; axes2++)
      velocity_old_to_new_alphas_index[axes1 * dimension + axes2] =
          old_to_new_velocity_basis->getAlphaColumnOffset(VectorPointEvaluation,
                                                          axes1, 0, axes2, 0);

  for (int i = 0; i < new_local_particle_num; i++) {
    if (adaptive_level[i] == __adaptive_step) {
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

  // lagrange multiplier
  if (__myID == __MPISize - 1) {
    for (int j = 0; j < field_dof; j++) {
      I.increment(field_dof * new_local_particle_num + j,
                  field_dof * old_global_particle_num + j,
                  old_global_particle_num / new_global_particle_num);
    }
  }

  // rigid body
  if (__myID == __MPISize - 1) {
    for (int i = 0; i < num_rigid_body; i++) {
      int local_rigid_body_index_offset =
          field_dof * (new_local_particle_num + 1) + i * rigid_body_dof;
      for (int j = 0; j < rigid_body_dof; j++) {
        I.increment(local_rigid_body_index_offset + j,
                    field_dof * (old_global_particle_num + 1) +
                        i * rigid_body_dof + j,
                    1.0);
      }
    }
  }

  I.FinalAssemble();

  // new to old relaxation matrix
  // new to old pressure field transition
  new_to_old_pressusre_basis->setProblemData(
      new_to_old_neighbor_lists_device, new_source_coords_device,
      old_target_coords_device, new_epsilon);

  new_to_old_pressusre_basis->addTargets(ScalarPointEvaluation);

  new_to_old_pressusre_basis->setWeightingType(WeightingFunctionType::Power);
  new_to_old_pressusre_basis->setWeightingPower(__weightFuncOrder);
  new_to_old_pressusre_basis->setOrderOfQuadraturePoints(2);
  new_to_old_pressusre_basis->setDimensionOfQuadraturePoints(1);
  new_to_old_pressusre_basis->setQuadratureType("LINE");

  new_to_old_pressusre_basis->generateAlphas(1);

  auto new_to_old_pressure_alphas = new_to_old_pressusre_basis->getAlphas();

  // new to old velocity field transition
  new_to_old_velocity_basis->setProblemData(
      new_to_old_neighbor_lists_device, new_source_coords_device,
      old_target_coords_device, new_epsilon);

  new_to_old_velocity_basis->addTargets(VectorPointEvaluation);

  new_to_old_velocity_basis->generateAlphas(1);

  auto new_to_old_velocity_alphas = new_to_old_velocity_basis->getAlphas();

  // new to old relaxation matrix
  if (__myID == __MPISize - 1)
    R.resize(old_local_dof + num_rigid_body * rigid_body_dof,
             new_local_dof + num_rigid_body * rigid_body_dof,
             new_global_dof + num_rigid_body * rigid_body_dof);
  else
    R.resize(old_local_dof, new_local_dof,
             new_global_dof + num_rigid_body * rigid_body_dof);

  // compute restriction amtrix graph
  for (int i = 0; i < old_local_particle_num; i++) {
    // velocity interpolation
    // index.resize(new_to_old_neighbor_lists(i, 0) * velocity_dof);
    // for (int j = 0; j < new_to_old_neighbor_lists(i, 0); j++) {
    //   for (int k = 0; k < velocity_dof; k++) {
    //     index[j * velocity_dof + k] =
    //         field_dof * background_index[new_to_old_neighbor_lists(i, j + 1)]
    //         + k;
    //   }
    // }

    // for (int k = 0; k < velocity_dof; k++) {
    //   R.setColIndex(field_dof * i + k, index);
    // }

    // pressure interpolation
    if (fieldParticleSplitTag[i]) {
      if (old_particle_type[i] == 0) {
        index.resize(new_to_old_neighbor_lists(old_actual_index[i], 0));
        for (int k = 0; k < field_dof; k++) {
          for (int j = 0; j < new_to_old_neighbor_lists(old_actual_index[i], 0);
               j++) {
            index[j] = field_dof * background_index[new_to_old_neighbor_lists(
                                       old_actual_index[i], j + 1)] +
                       k;
          }
          R.setColIndex(field_dof * i + k, index);
        }
      } else {
        index.resize(new_to_old_neighbor_lists(old_actual_index[i], 0));
        for (int j = 0; j < new_to_old_neighbor_lists(old_actual_index[i], 0);
             j++) {
          index[j] = field_dof * background_index[new_to_old_neighbor_lists(
                                     old_actual_index[i], j + 1)] +
                     velocity_dof;
        }
        R.setColIndex(field_dof * i + velocity_dof, index);
      }
    } else {
      index.resize(1);
      for (int k = 0; k < field_dof; k++) {
        index[0] = field_dof * background_index[i] + k;
        R.setColIndex(field_dof * i + k, index);
      }
    }
  }

  // lagrange multiplier
  if (__myID == __MPISize - 1) {
    index.resize(1);
    for (int j = 0; j < field_dof; j++) {
      index[0] = field_dof * new_global_particle_num + j;
      R.setColIndex(field_dof * old_local_particle_num + j, index);
    }
  }

  // rigid body
  if (__myID == __MPISize - 1) {
    index.resize(1);
    for (int i = 0; i < num_rigid_body; i++) {
      int local_rigid_body_index_offset =
          field_dof * (old_local_particle_num + 1) + i * rigid_body_dof;
      for (int j = 0; j < rigid_body_dof; j++) {
        index[0] =
            field_dof * (new_global_particle_num + 1) + i * rigid_body_dof + j;
        R.setColIndex(local_rigid_body_index_offset + j, index);
      }
    }
  }

  const auto pressure_new_to_old_alphas_index =
      new_to_old_pressusre_basis->getAlphaColumnOffset(ScalarPointEvaluation, 0,
                                                       0, 0, 0);
  vector<int> velocity_new_to_old_alphas_index(pow(dimension, 2));
  for (int axes1 = 0; axes1 < dimension; axes1++)
    for (int axes2 = 0; axes2 < dimension; axes2++)
      velocity_new_to_old_alphas_index[axes1 * dimension + axes2] =
          new_to_old_velocity_basis->getAlphaColumnOffset(VectorPointEvaluation,
                                                          axes1, 0, axes2, 0);

  for (int i = 0; i < old_local_particle_num; i++) {
    // for (int j = 0; j < new_to_old_neighbor_lists(i, 0); j++) {
    //   for (int axes1 = 0; axes1 < dimension; axes1++)
    //     for (int axes2 = 0; axes2 < dimension; axes2++)
    //       R.increment(
    //           field_dof * i + axes1,
    //           field_dof *
    //                   background_index[new_to_old_neighbor_lists(i, j + 1)] +
    //               axes2,
    //           new_to_old_velocity_alphas(
    //               i,
    //               velocity_new_to_old_alphas_index[axes1 * dimension +
    //               axes2], j));
    // }

    if (fieldParticleSplitTag[i]) {
      if (old_particle_type[i] == 0) {
        for (int j = 0; j < new_to_old_neighbor_lists(old_actual_index[i], 0);
             j++) {
          for (int k = 0; k < field_dof; k++)
            R.increment(
                field_dof * i + k,
                field_dof * background_index[new_to_old_neighbor_lists(
                                old_actual_index[i], j + 1)] +
                    k,
                new_to_old_pressure_alphas(
                    old_actual_index[i], pressure_new_to_old_alphas_index, j));
        }
      } else {
        for (int j = 0; j < new_to_old_neighbor_lists(old_actual_index[i], 0);
             j++) {
          R.increment(
              field_dof * i + velocity_dof,
              field_dof * background_index[new_to_old_neighbor_lists(
                              old_actual_index[i], j + 1)] +
                  velocity_dof,
              new_to_old_pressure_alphas(old_actual_index[i],
                                         pressure_new_to_old_alphas_index, j));
        }
      }
    } else {
      for (int k = 0; k < field_dof; k++) {
        R.increment(field_dof * i + k, field_dof * background_index[i] + k,
                    1.0);
      }
    }
  }

  // lagrange multiplier
  if (__myID == __MPISize - 1) {
    for (int j = 0; j < field_dof; j++) {
      R.increment(field_dof * old_local_particle_num + j,
                  field_dof * new_global_particle_num + j,
                  new_global_particle_num / old_global_particle_num);
    }
  }

  // rigid body
  if (__myID == __MPISize - 1) {
    index.resize(1);
    for (int i = 0; i < num_rigid_body; i++) {
      int local_rigid_body_index_offset =
          field_dof * (old_local_particle_num + 1) + i * rigid_body_dof;
      for (int j = 0; j < rigid_body_dof; j++) {
        R.increment(local_rigid_body_index_offset + j,
                    field_dof * (new_global_particle_num + 1) +
                        i * rigid_body_dof + j,
                    1.0);
      }
    }
  }

  R.FinalAssemble();
}

void multilevel::InitialGuessFromPreviousAdaptiveStep(
    std::vector<double> &initial_guess) {}

void multilevel::Solve(std::vector<double> &rhs, std::vector<double> &x,
                       std::vector<int> &idx_neighbor) {
  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "\nstart of linear system solving setup\n");

  int adaptive_step = A_list.size() - 1;

  int fieldDof = dimension + 1;
  int velocityDof = dimension;
  int pressureDof = 1;
  int rigidBodyDof = (dimension == 3) ? 6 : 3;

  vector<int> idx_field;
  vector<int> idx_global;

  PetscInt localN1, localN2;
  Mat &mat = (*(A_list.end() - 1))->__mat;
  MatGetOwnershipRange(mat, &localN1, &localN2);

  if (myid != mpi_size - 1) {
    int localParticleNum = (localN2 - localN1) / fieldDof;
    idx_field.resize(fieldDof * localParticleNum);

    for (int i = 0; i < localParticleNum; i++) {
      for (int j = 0; j < dimension; j++) {
        idx_field[fieldDof * i + j] = localN1 + fieldDof * i + j;
      }
      idx_field[fieldDof * i + velocityDof] =
          localN1 + fieldDof * i + velocityDof;
    }

    idx_global = idx_field;
  } else {
    int localParticleNum =
        (localN2 - localN1 - 1 - num_rigid_body * rigidBodyDof) / fieldDof + 1;
    idx_field.resize(fieldDof * localParticleNum);

    for (int i = 0; i < localParticleNum; i++) {
      for (int j = 0; j < dimension; j++) {
        idx_field[fieldDof * i + j] = localN1 + fieldDof * i + j;
      }
      idx_field[fieldDof * i + velocityDof] =
          localN1 + fieldDof * i + velocityDof;
    }

    idx_global = idx_field;

    // idx_field.push_back(localN1 + fieldDof * localParticleNum +
    // velocityDof);
  }

  IS &isg_field_lag = *isg_field_lag_list[adaptive_step];
  IS &isg_neighbor = *isg_neighbor_list[adaptive_step];

  ISCreateGeneral(MPI_COMM_WORLD, idx_field.size(), idx_field.data(),
                  PETSC_COPY_VALUES, &isg_field_lag);
  ISCreateGeneral(MPI_COMM_WORLD, idx_neighbor.size(), idx_neighbor.data(),
                  PETSC_COPY_VALUES, &isg_neighbor);

  Vec _rhs, _x;
  VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, rhs.size(), PETSC_DECIDE,
                        rhs.data(), &_rhs);
  VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, x.size(), PETSC_DECIDE, x.data(),
                        &_x);

  Mat &ff = *ff_lag_list[adaptive_step];
  Mat &nn = *nn_list[adaptive_step];

  MatCreateSubMatrix(mat, isg_field_lag, isg_field_lag, MAT_INITIAL_MATRIX,
                     &ff);
  MatCreateSubMatrix(mat, isg_neighbor, isg_neighbor, MAT_INITIAL_MATRIX, &nn);

  MatSetBlockSize(ff, fieldDof);

  // setup current level vectors and relaxation
  x_list.push_back(new Vec);
  y_list.push_back(new Vec);
  b_list.push_back(new Vec);
  r_list.push_back(new Vec);
  t_list.push_back(new Vec);

  relaxation_list.push_back(new KSP);

  MatCreateVecs(ff, NULL, x_list[adaptive_step]);
  MatCreateVecs(ff, NULL, y_list[adaptive_step]);
  MatCreateVecs(ff, NULL, b_list[adaptive_step]);
  MatCreateVecs(ff, NULL, r_list[adaptive_step]);
  MatCreateVecs(ff, NULL, t_list[adaptive_step]);

  KSPCreate(MPI_COMM_WORLD, relaxation_list[adaptive_step]);

  KSPSetType(*relaxation_list[adaptive_step], KSPPREONLY);
  KSPSetOperators(*relaxation_list[adaptive_step], ff, ff);

  PC relaxation;
  KSPGetPC(*relaxation_list[adaptive_step], &relaxation);
  PCSetType(relaxation, PCSOR);
  PCSetFromOptions(relaxation);
  PCSetUp(relaxation);

  KSPSetUp(*relaxation_list[adaptive_step]);

  KSP &_ksp = getKsp(adaptive_step);
  KSPCreate(PETSC_COMM_WORLD, &_ksp);
  KSPSetOperators(_ksp, mat, mat);
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

    HypreLUShellPCSetUp(_pc, &mat, &ff, &nn, &isg_field_lag, &isg_neighbor, _x);
  } else {
    MPI_Barrier(MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "start of multilevel preconditioner setup\n");

    PCShellSetApply(_pc, HypreLUShellPCApplyAdaptive);
    PCShellSetContext(_pc, shell_ctx);
    PCShellSetDestroy(_pc, HypreLUShellPCDestroy);

    HypreLUShellPCSetUpAdaptive(_pc, &mat, &ff, ff_lag_list[0], &nn,
                                &isg_field_lag, &isg_neighbor, this, _x);
  }

  Vec x_initial;
  if (A_list.size() > 1) {
    VecDuplicate(_x, &x_initial);
    VecCopy(_x, x_initial);
  }

  KSPSetInitialGuessNonzero(_ksp, PETSC_TRUE);

  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "final solving of linear system\n");
  // if (adaptive_step < 2)
  KSPSolve(_ksp, _rhs, _x);
  PetscPrintf(PETSC_COMM_WORLD, "ksp solving finished\n");

  // if (adatptive_step > 0) {
  //   VecAXPY(_x, -1.0, x_initial);
  //   VecAbs(_x);
  // }

  PetscScalar *a;
  VecGetArray(_x, &a);
  for (size_t i = 0; i < rhs.size(); i++) {
    x[i] = a[i];
  }
  VecRestoreArray(_x, &a);

  VecDestroy(&_rhs);
  VecDestroy(&_x);
}