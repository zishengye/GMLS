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
    if (adaptive_level[i] == __adaptive_step)
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
    if (adaptive_level[i] == __adaptive_step) {
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
      pow(2, dimension) * pow(2 * 2.5, dimension);

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

  auto neighbor_needed = Compadre::GMLS::getNP(
      __polynomialOrder, __dim, DivergenceFreeVectorTaylorPolynomial);
  old_to_new_point_search.generateNeighborListsFromKNNSearch(
      false, new_target_coords, old_to_new_neighbor_lists, old_epsilon,
      neighbor_needed, 1.2);

  Kokkos::deep_copy(old_to_new_neighbor_lists_device,
                    old_to_new_neighbor_lists);
  Kokkos::deep_copy(old_epsilon_device, old_epsilon);

  auto old_to_new_pressusre_basis = new GMLS(
      ScalarTaylorPolynomial, PointSample, 2, dimension, "LU", "STANDARD");
  auto old_to_new_velocity_basis =
      new GMLS(DivergenceFreeVectorTaylorPolynomial, VectorPointSample, 2,
               dimension, "SVD", "STANDARD");

  // old to new pressure field transition
  old_to_new_pressusre_basis->setProblemData(
      old_to_new_neighbor_lists_device, old_source_coords_device,
      new_target_coords_device, old_epsilon_device);

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
      new_target_coords_device, old_epsilon_device);

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

  IS &isg_field_lag = *isg_field_list[adaptive_step];
  IS &isg_neighbor = *isg_neighbor_list[adaptive_step];
  IS &isg_pressure = *isg_pressure_list[adaptive_step];

  ISCreateGeneral(MPI_COMM_WORLD, idx_field.size(), idx_field.data(),
                  PETSC_COPY_VALUES, &isg_field_lag);
  ISCreateGeneral(MPI_COMM_WORLD, idx_neighbor.size(), idx_neighbor.data(),
                  PETSC_COPY_VALUES, &isg_neighbor);
  ISCreateGeneral(MPI_COMM_WORLD, idx_pressure.size(), idx_pressure.data(),
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

  MatNullSpace nullspace_whole;
  MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &null_whole,
                     &nullspace_whole);
  MatSetNearNullSpace(mat, nullspace_whole);

  Vec field_pressure;
  VecGetSubVector(null_field, isg_pressure, &field_pressure);
  VecSet(field_pressure, 1.0);
  VecRestoreSubVector(null_field, isg_pressure, &field_pressure);

  MatNullSpace nullspace_field;
  MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &null_field,
                     &nullspace_field);
  MatSetNearNullSpace(ff, nullspace_field);

  // neighbor vector scatter, only needed on base level
  if (adaptive_step == 0) {
    MatCreateVecs(nn, NULL, &x_neighbor);
    MatCreateVecs(nn, NULL, &y_neighbor);

    neighbor_scatter_list.push_back(new VecScatter);

    VecScatterCreate(*x_list[adaptive_step], isg_neighbor, x_neighbor, NULL,
                     neighbor_scatter_list[adaptive_step]);
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
    KSP *bjacobi_ksp;
    PCBJacobiGetSubKSP(pc_neighbor_base, NULL, NULL, &bjacobi_ksp);
    KSPSetType(bjacobi_ksp[0], KSPPREONLY);
    PC bjacobi_pc;
    KSPGetPC(bjacobi_ksp[0], &bjacobi_pc);
    PCSetType(bjacobi_pc, PCLU);
    PCFactorSetMatSolverType(bjacobi_pc, MATSOLVERMUMPS);
    PCSetUp(bjacobi_pc);

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
  KSP *neighbor_relaxation_sub_ksp;
  PCBJacobiGetSubKSP(neighbor_relaxation_pc, NULL, NULL,
                     &neighbor_relaxation_sub_ksp);
  KSPSetType(neighbor_relaxation_sub_ksp[0], KSPPREONLY);
  PC neighbor_relaxation_sub_pc;
  KSPGetPC(neighbor_relaxation_sub_ksp[0], &neighbor_relaxation_sub_pc);
  PCSetType(neighbor_relaxation_sub_pc, PCLU);
  PCFactorSetMatSolverType(neighbor_relaxation_sub_pc, MATSOLVERMUMPS);
  PCSetUp(neighbor_relaxation_sub_pc);

  KSPSetUp(*neighbor_relaxation_list[adaptive_step]);

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

    HypreLUShellPCSetUp(_pc, this, _x);
  } else {
    MPI_Barrier(MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "start of multilevel preconditioner setup\n");

    PCShellSetApply(_pc, HypreLUShellPCApplyAdaptive);
    PCShellSetContext(_pc, shell_ctx);
    PCShellSetDestroy(_pc, HypreLUShellPCDestroy);

    HypreLUShellPCSetUp(_pc, this, _x);
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

  MatDestroy(&pp);

  VecDestroy(&_rhs);
  VecDestroy(&_x);
}