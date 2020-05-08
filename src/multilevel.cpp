#include "multilevel.h"
#include "composite_preconditioner.h"
#include "gmls_solver.h"

#include <algorithm>
#include <iostream>

using namespace std;
using namespace Compadre;

void GMLS_Solver::BuildInterpolationAndRelaxationMatrices(PetscSparseMatrix &I,
                                                          PetscSparseMatrix &R,
                                                          int num_rigid_body,
                                                          int dimension) {
  static auto &coord = __field.vector.GetHandle("coord");
  static auto &adaptive_level = __field.index.GetHandle("adaptive level");
  static auto &background_coord = __background.vector.GetHandle("source coord");
  static auto &background_index = __background.index.GetHandle("source index");

  static auto &old_coord = __field.vector.GetHandle("old coord");
  static auto &old_background_coord =
      __background.vector.GetHandle("old source coord");
  static auto &old_background_index =
      __background.index.GetHandle("old source index");

  int field_dof = dimension + 1;
  int velocity_dof = dimension;
  int pressure_dof = 1;

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
  vector<int> new_to_actual_index(coord.size());
  for (int i = 0; i < coord.size(); i++) {
    new_to_actual_index[i] = actual_new_target;
    if (adaptive_level[i] == __adaptive_step)
      actual_new_target++;
  }

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
      new_target_coords_device("new target coordinates", actual_new_target, 3);
  Kokkos::View<double **>::HostMirror new_target_coords =
      Kokkos::create_mirror_view(new_target_coords_device);

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
      old_target_coords_device("old target coordinates", old_coord.size(), 3);
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
  for (int i = 0; i < old_coord.size(); i++) {
    for (int j = 0; j < dimension; j++)
      old_target_coords(i, j) = old_coord[i][j];
  }

  // copy new target coords
  int counter = 0;
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
      new_to_old_neighbor_lists_device("neighbor lists", old_coord.size(),
                                       estimatedUpperBoundNumberNeighbors);
  Kokkos::View<int **>::HostMirror new_to_old_neighbor_lists =
      Kokkos::create_mirror_view(new_to_old_neighbor_lists_device);

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
      old_to_new_neighbor_lists_device("neighbor lists", actual_new_target,
                                       estimatedUpperBoundNumberNeighbors);
  Kokkos::View<int **>::HostMirror old_to_new_neighbor_lists =
      Kokkos::create_mirror_view(old_to_new_neighbor_lists_device);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> new_epsilon_device(
      "h supports", old_coord.size());
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
      ScalarTaylorPolynomial, PointSample, 2, dimension, "SVD", "STANDARD");
  auto new_to_old_velocity_basis =
      new GMLS(DivergenceFreeVectorTaylorPolynomial, VectorPointSample, 2,
               dimension, "SVD", "STANDARD");
  auto old_to_new_pressusre_basis = new GMLS(
      ScalarTaylorPolynomial, PointSample, 2, dimension, "SVD", "STANDARD");
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

  old_to_new_pressusre_basis->generateAlphas(20);

  auto old_to_new_pressure_alphas = old_to_new_pressusre_basis->getAlphas();

  // old to new velocity field transition
  old_to_new_velocity_basis->setProblemData(
      old_to_new_neighbor_lists_device, old_source_coords_device,
      new_target_coords_device, old_epsilon);

  old_to_new_velocity_basis->addTargets(VectorPointEvaluation);

  old_to_new_velocity_basis->generateAlphas(20);

  auto old_to_new_velocity_alphas = old_to_new_velocity_basis->getAlphas();

  // old to new interpolation matrix
  I.resize(new_local_dof, old_local_dof, old_global_dof);
  // compute matrix graph
  for (int i = 0; i < new_local_particle_num; i++) {
    vector<PetscInt> index;
    if (adaptive_level[i] == __adaptive_step) {
      // velocity interpolation
      index.resize(old_to_new_neighbor_lists(new_to_actual_index[i], 0) *
                   velocity_dof);
      for (int j = 0; j < old_to_new_neighbor_lists(new_to_actual_index[i], 0);
           j++) {
        for (int k = 0; k < velocity_dof; k++) {
          index[j * velocity_dof + k] =
              field_dof * old_background_index[old_to_new_neighbor_lists(
                              new_to_actual_index[i], j + 1)] +
              k;
        }
      }

      for (int k = 0; k < velocity_dof; k++) {
        I.setColIndex(field_dof * i + k, index);
      }

      // pressure interpolation
      index.resize(old_to_new_neighbor_lists(new_to_actual_index[i], 0));
      for (int j = 0; j < old_to_new_neighbor_lists(new_to_actual_index[i], 0);
           j++) {
        index[j] = field_dof * old_background_index[old_to_new_neighbor_lists(
                                   new_to_actual_index[i], j + 1)] +
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
      for (int j = 0; j < old_to_new_neighbor_lists(new_to_actual_index[i], 0);
           j++) {
        for (int axes1 = 0; axes1 < dimension; axes1++)
          for (int axes2 = 0; axes2 < dimension; axes2++)
            I.increment(
                field_dof * i + axes1,
                field_dof * old_background_index[old_to_new_neighbor_lists(
                                new_to_actual_index[i], j + 1)] +
                    axes2,
                old_to_new_velocity_alphas(
                    new_to_actual_index[i],
                    velocity_old_to_new_alphas_index[axes1 * dimension + axes2],
                    j));
      }

      for (int j = 0; j < old_to_new_neighbor_lists(new_to_actual_index[i], 0);
           j++) {
        I.increment(field_dof * i + velocity_dof,
                    field_dof * old_background_index[old_to_new_neighbor_lists(
                                    new_to_actual_index[i], j + 1)] +
                        velocity_dof,
                    old_to_new_pressure_alphas(new_to_actual_index[i],
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

  PetscPrintf(MPI_COMM_WORLD, "start of interpolation matrix assembly\n");
  I.FinalAssemble();

  // new to old relaxation matrix
  R.resize(old_local_dof, new_local_dof, new_global_dof);
}

void multilevel::Solve(std::vector<double> &rhs, std::vector<double> &x,
                       std::vector<int> &idx_neighbor) {
  int adaptive_step = A_list.size() - 1;

  int fieldDof = dimension + 1;
  int velocityDof = dimension;
  int pressureDof = 1;
  int rigidBodyDof = (dimension == 3) ? 6 : 3;

  vector<int> idx_field;
  vector<int> idx_global;

  PetscInt localN1, localN2;
  Mat &mat = (A_list.end() - 1)->__mat;
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

    // idx_field.push_back(localN1 + fieldDof * localParticleNum + velocityDof);
  }

  IS isg_field, isg_neighbor;
  IS isg_global;

  ISCreateGeneral(MPI_COMM_WORLD, idx_field.size(), idx_field.data(),
                  PETSC_COPY_VALUES, &isg_field);
  ISCreateGeneral(MPI_COMM_WORLD, idx_neighbor.size(), idx_neighbor.data(),
                  PETSC_COPY_VALUES, &isg_neighbor);
  ISCreateGeneral(MPI_COMM_WORLD, idx_global.size(), idx_global.data(),
                  PETSC_COPY_VALUES, &isg_global);

  Vec _rhs, _x;
  VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, rhs.size(), PETSC_DECIDE,
                        rhs.data(), &_rhs);
  VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, x.size(), PETSC_DECIDE, x.data(),
                        &_x);

  Mat ff, nn, gg;

  MatCreateSubMatrix(mat, isg_field, isg_field, MAT_INITIAL_MATRIX, &ff);
  MatCreateSubMatrix(mat, isg_neighbor, isg_neighbor, MAT_INITIAL_MATRIX, &nn);
  MatCreateSubMatrix(mat, isg_global, isg_global, MAT_INITIAL_MATRIX, &gg);

  MatSetBlockSize(ff, fieldDof);

  KSP _ksp;
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

    HypreLUShellPCSetUp(_pc, &mat, &ff, &nn, &isg_field, &isg_neighbor, _x);
  } else {
    PCShellSetApply(_pc, HypreLUShellPCApply);
    PCShellSetContext(_pc, shell_ctx);
    PCShellSetDestroy(_pc, HypreLUShellPCDestroy);

    HypreLUShellPCSetUp(_pc, &mat, &ff, &nn, &isg_field, &isg_neighbor, _x);
  }

  if (A_list.size() > 1) {
    KSP smoother_ksp;
    KSPCreate(PETSC_COMM_WORLD, &smoother_ksp);
    KSPSetOperators(smoother_ksp, nn, nn);
    KSPSetType(smoother_ksp, KSPPREONLY);

    PC smoother_pc;
    KSPGetPC(smoother_ksp, &smoother_pc);
    PCSetType(smoother_pc, PCLU);
    PCSetFromOptions(smoother_pc);

    PCSetUp(smoother_pc);
    KSPSetUp(smoother_ksp);

    Vec r, delta_x;
    VecDuplicate(_x, &r);
    VecDuplicate(_x, &delta_x);
    MatMult(mat, _x, r);
    VecAXPY(r, -1.0, _rhs);

    // KSPSolve(smoother_ksp, r, delta_x);
    // VecAXPY(_x, -1.0, delta_x);

    // KSPSetInitialGuessNonzero(smoother_ksp, PETSC_TRUE);

    Vec r_f, x_f, delta_x_f;
    VecGetSubVector(r, isg_neighbor, &r_f);
    VecGetSubVector(_x, isg_neighbor, &x_f);
    VecDuplicate(x_f, &delta_x_f);
    KSPSolve(smoother_ksp, r_f, delta_x_f);
    VecAXPY(x_f, -1.0, delta_x_f);
    VecRestoreSubVector(_rhs, isg_neighbor, &r_f);
    VecRestoreSubVector(_x, isg_neighbor, &x_f);
  }

  Vec x_initial;
  if (A_list.size() > 1) {
    VecDuplicate(_x, &x_initial);
    VecCopy(_x, x_initial);
  }

  KSPSetInitialGuessNonzero(_ksp, PETSC_TRUE);

  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "final solving of linear system\n");
  KSPSolve(_ksp, _rhs, _x);
  PetscPrintf(PETSC_COMM_WORLD, "ksp solving finished\n");

  // if (adatptive_step > 0) {
  //   VecAXPY(_x, -1.0, x_initial);
  //   VecAbs(_x);
  // }

  KSPDestroy(&_ksp);

  PetscScalar *a;
  VecGetArray(_x, &a);
  for (size_t i = 0; i < rhs.size(); i++) {
    x[i] = a[i];
  }
  VecRestoreArray(_x, &a);

  VecDestroy(&_rhs);
  VecDestroy(&_x);

  ISDestroy(&isg_field);
  ISDestroy(&isg_neighbor);
  ISDestroy(&isg_global);

  MatDestroy(&ff);
  MatDestroy(&nn);
  MatDestroy(&gg);
}