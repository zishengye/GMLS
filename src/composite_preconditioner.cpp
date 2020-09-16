#include "composite_preconditioner.h"

using namespace std;

PetscErrorCode HypreLUShellPCCreate(HypreLUShellPC **shell) {
  HypreLUShellPC *newctx;

  PetscNew(&newctx);
  *shell = newctx;

  return 0;
}

PetscErrorCode HypreLUShellPCSetUp(PC pc, multilevel *multi, Vec x,
                                   PetscInt local_particle_num,
                                   PetscInt field_dof) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  shell->multi = multi;

  shell->adaptive_level = shell->multi->GetInterpolationList()->size();

  shell->local_particle_num = local_particle_num;
  shell->field_dof = field_dof;

  PetscInt global_particle_num;
  MPI_Allreduce(&local_particle_num, &global_particle_num, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  shell->global_particle_num = global_particle_num;

  return 0;
}

PetscErrorCode HypreLUShellPCApply(PC pc, Vec x, Vec y) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  PetscReal *a, *b, pressure_sum;

  double time_begin, time_end;

  PetscInt pressure_offset = shell->field_dof - 1;

  PetscInt field_size = shell->local_particle_num * shell->field_dof;

  // orthogonalize to constant vector
  VecGetArray(x, &a);
  pressure_sum = 0.0;
  for (PetscInt i = 0; i < shell->local_particle_num; i++)
    pressure_sum += a[shell->field_dof * i + pressure_offset];
  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  pressure_sum /= shell->global_particle_num;
  for (PetscInt i = 0; i < shell->local_particle_num; i++)
    a[shell->field_dof * i + pressure_offset] -= pressure_sum;
  VecRestoreArray(x, &a);

  // stage 1
  VecSet(y, 0.0);

  VecGetArray(x, &a);
  VecGetArray(*((*shell->multi->GetXFieldList())[0]), &b);
  for (PetscInt i = 0; i < field_size; i++)
    b[i] = a[i];
  VecRestoreArray(x, &a);
  VecRestoreArray(*((*shell->multi->GetXFieldList())[0]), &b);

  KSPSolve(shell->multi->getFieldBase(), *((*shell->multi->GetXFieldList())[0]),
           *((*shell->multi->GetYFieldList())[0]));

  VecGetArray(y, &a);
  VecGetArray(*((*shell->multi->GetYFieldList())[0]), &b);
  for (PetscInt i = 0; i < field_size; i++)
    a[i] = b[i];
  VecRestoreArray(y, &a);
  VecRestoreArray(*((*shell->multi->GetYFieldList())[0]), &b);

  // stage 2
  MatMult(shell->multi->getNeighborWholeMat(0), y,
          *shell->multi->getXNeighbor());

  VecScatterBegin(*((*shell->multi->GetNeighborScatterList())[0]), x,
                  *shell->multi->getYNeighbor(), INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(*((*shell->multi->GetNeighborScatterList())[0]), x,
                *shell->multi->getYNeighbor(), INSERT_VALUES, SCATTER_FORWARD);

  VecAXPY(*shell->multi->getYNeighbor(), -1.0, *shell->multi->getXNeighbor());

  KSPSolve(shell->multi->getNeighborBase(), *shell->multi->getYNeighbor(),
           *shell->multi->getXNeighbor());

  VecScatterBegin(*((*shell->multi->GetNeighborScatterList())[0]),
                  *shell->multi->getXNeighbor(), y, ADD_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(*((*shell->multi->GetNeighborScatterList())[0]),
                *shell->multi->getXNeighbor(), y, ADD_VALUES, SCATTER_REVERSE);

  // orthogonalize to constant vector
  VecGetArray(x, &a);
  pressure_sum = 0.0;
  for (PetscInt i = 0; i < shell->local_particle_num; i++)
    pressure_sum += a[shell->field_dof * i + pressure_offset];
  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  pressure_sum /= shell->global_particle_num;
  for (PetscInt i = 0; i < shell->local_particle_num; i++)
    a[shell->field_dof * i + pressure_offset] -= pressure_sum;
  VecRestoreArray(x, &a);

  return 0;
}

PetscErrorCode HypreLUShellPCApplyAdaptive(PC pc, Vec x, Vec y) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  PetscReal *a;

  PetscInt local_size, global_size;

  // orthogonalize to constant vector
  VecScatterBegin(
      *((*shell->multi->GetPressureScatterList())[shell->adaptive_level - 1]),
      x, *((*shell->multi->GetXPressureList())[shell->adaptive_level - 1]),
      INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(
      *((*shell->multi->GetPressureScatterList())[shell->adaptive_level - 1]),
      x, *((*shell->multi->GetXPressureList())[shell->adaptive_level - 1]),
      INSERT_VALUES, SCATTER_FORWARD);

  PetscReal pressure_sum = 0.0;
  PetscInt size;
  VecSum(*((*shell->multi->GetXPressureList())[shell->adaptive_level - 1]),
         &pressure_sum);
  VecGetSize(*((*shell->multi->GetXPressureList())[shell->adaptive_level - 1]),
             &size);
  VecSet(*((*shell->multi->GetXPressureList())[shell->adaptive_level - 1]),
         -pressure_sum / size);

  VecScatterBegin(
      *((*shell->multi->GetPressureScatterList())[shell->adaptive_level - 1]),
      *((*shell->multi->GetXPressureList())[shell->adaptive_level - 1]), x,
      ADD_VALUES, SCATTER_REVERSE);
  VecScatterEnd(
      *((*shell->multi->GetPressureScatterList())[shell->adaptive_level - 1]),
      *((*shell->multi->GetXPressureList())[shell->adaptive_level - 1]), x,
      ADD_VALUES, SCATTER_REVERSE);

  VecCopy(x, *((*shell->multi->GetBList())[shell->adaptive_level - 1]));

  // sweep down
  for (int i = shell->adaptive_level - 1; i > 0; i--) {
    // pre-smooth
    // fluid part smoothing
    VecScatterBegin(*((*shell->multi->GetFieldScatterList())[i]),
                    *((*shell->multi->GetBList())[i]),
                    *((*shell->multi->GetBFieldList())[i]), INSERT_VALUES,
                    SCATTER_FORWARD);
    VecScatterEnd(*((*shell->multi->GetFieldScatterList())[i]),
                  *((*shell->multi->GetBList())[i]),
                  *((*shell->multi->GetBFieldList())[i]), INSERT_VALUES,
                  SCATTER_FORWARD);

    KSPSolve(shell->multi->getFieldRelaxation(i),
             *((*shell->multi->GetBFieldList())[i]),
             *((*shell->multi->GetXFieldList())[i]));

    VecSet(*((*shell->multi->GetXList())[i]), 0.0);

    VecScatterBegin(*((*shell->multi->GetFieldScatterList())[i]),
                    *((*shell->multi->GetXFieldList())[i]),
                    *((*shell->multi->GetXList())[i]), INSERT_VALUES,
                    SCATTER_REVERSE);
    VecScatterEnd(*((*shell->multi->GetFieldScatterList())[i]),
                  *((*shell->multi->GetXFieldList())[i]),
                  *((*shell->multi->GetXList())[i]), INSERT_VALUES,
                  SCATTER_REVERSE);

    // neighbor part smoothing
    MatMult(shell->multi->getNeighborWholeMat(i),
            *((*shell->multi->GetXList())[i]),
            *((*shell->multi->GetXNeighborList())[i]));

    VecScatterBegin(*((*shell->multi->GetNeighborScatterList())[i]),
                    *((*shell->multi->GetBList())[i]),
                    *((*shell->multi->GetBNeighborList())[i]), INSERT_VALUES,
                    SCATTER_FORWARD);
    VecScatterEnd(*((*shell->multi->GetNeighborScatterList())[i]),
                  *((*shell->multi->GetBList())[i]),
                  *((*shell->multi->GetBNeighborList())[i]), INSERT_VALUES,
                  SCATTER_FORWARD);

    VecAXPY(*((*shell->multi->GetBNeighborList())[i]), -1.0,
            *((*shell->multi->GetXNeighborList())[i]));

    KSPSolve(shell->multi->getNeighborRelaxation(i),
             *((*shell->multi->GetBNeighborList())[i]),
             *((*shell->multi->GetXNeighborList())[i]));

    VecScatterBegin(*((*shell->multi->GetNeighborScatterList())[i]),
                    *((*shell->multi->GetXNeighborList())[i]),
                    *((*shell->multi->GetXList())[i]), ADD_VALUES,
                    SCATTER_REVERSE);
    VecScatterEnd(*((*shell->multi->GetNeighborScatterList())[i]),
                  *((*shell->multi->GetXNeighborList())[i]),
                  *((*shell->multi->GetXList())[i]), ADD_VALUES,
                  SCATTER_REVERSE);

    // restriction
    MatMult(shell->multi->getA(i).__mat, *((*shell->multi->GetXList())[i]),
            *((*shell->multi->GetRList())[i]));

    VecAXPY(*((*shell->multi->GetRList())[i]), -1.0,
            *((*shell->multi->GetBList())[i]));

    VecScale(*((*shell->multi->GetRList())[i]), -1.0);

    Mat *R = &(*shell->multi->GetRestrictionList())[i]->__mat;
    Vec *v1 = (*shell->multi->GetRList())[i];
    Vec *v2 = (*shell->multi->GetBList())[i - 1];
    MatMult(*R, *v1, *v2);
  }

  // solve on coarest-level
  // stage 1
  VecSet(*((*shell->multi->GetXList())[0]), 0.0);
  VecScatterBegin(*((*shell->multi->GetFieldScatterList())[0]),
                  *((*shell->multi->GetBList())[0]),
                  *((*shell->multi->GetBFieldList())[0]), INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(*((*shell->multi->GetFieldScatterList())[0]),
                *((*shell->multi->GetBList())[0]),
                *((*shell->multi->GetBFieldList())[0]), INSERT_VALUES,
                SCATTER_FORWARD);

  KSPSolve(shell->multi->getFieldBase(), *((*shell->multi->GetBFieldList())[0]),
           *((*shell->multi->GetXFieldList())[0]));

  VecScatterBegin(*((*shell->multi->GetFieldScatterList())[0]),
                  *((*shell->multi->GetXFieldList())[0]),
                  *((*shell->multi->GetXList())[0]), INSERT_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(*((*shell->multi->GetFieldScatterList())[0]),
                *((*shell->multi->GetXFieldList())[0]),
                *((*shell->multi->GetXList())[0]), INSERT_VALUES,
                SCATTER_REVERSE);

  // stage 2
  MatMult(shell->multi->getNeighborWholeMat(0),
          *((*shell->multi->GetXList())[0]), *shell->multi->getXNeighbor());

  VecScatterBegin(*((*shell->multi->GetNeighborScatterList())[0]),
                  *((*shell->multi->GetBList())[0]),
                  *shell->multi->getYNeighbor(), INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(*((*shell->multi->GetNeighborScatterList())[0]),
                *((*shell->multi->GetBList())[0]),
                *shell->multi->getYNeighbor(), INSERT_VALUES, SCATTER_FORWARD);

  VecAXPY(*shell->multi->getYNeighbor(), -1.0, *shell->multi->getXNeighbor());

  KSPSolve(shell->multi->getNeighborBase(), *shell->multi->getYNeighbor(),
           *shell->multi->getXNeighbor());

  VecScatterBegin(*((*shell->multi->GetNeighborScatterList())[0]),
                  *shell->multi->getXNeighbor(),
                  *((*shell->multi->GetXList())[0]), ADD_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(*((*shell->multi->GetNeighborScatterList())[0]),
                *shell->multi->getXNeighbor(),
                *((*shell->multi->GetXList())[0]), ADD_VALUES, SCATTER_REVERSE);

  // sweep up
  for (int i = 1; i < shell->adaptive_level; i++) {
    Mat *I = &(*shell->multi->GetInterpolationList())[i]->__mat;
    Vec *v1 = (*shell->multi->GetTList())[i];
    Vec *v2 = (*shell->multi->GetXList())[i - 1];
    MatMult(*I, *v2, *v1);

    // post-smooth
    // fluid part smoothing
    VecAXPY(*((*shell->multi->GetXList())[i]), 1.0,
            *((*shell->multi->GetTList())[i]));

    MatMult(shell->multi->getA(i).__mat, *((*shell->multi->GetXList())[i]),
            *((*shell->multi->GetRList())[i]));

    VecAXPY(*((*shell->multi->GetRList())[i]), -1.0,
            *((*shell->multi->GetBList())[i]));

    VecScale(*((*shell->multi->GetRList())[i]), -1.0);

    VecScatterBegin(*((*shell->multi->GetFieldScatterList())[i]),
                    *((*shell->multi->GetRList())[i]),
                    *((*shell->multi->GetRFieldList())[i]), INSERT_VALUES,
                    SCATTER_FORWARD);
    VecScatterEnd(*((*shell->multi->GetFieldScatterList())[i]),
                  *((*shell->multi->GetRList())[i]),
                  *((*shell->multi->GetRFieldList())[i]), INSERT_VALUES,
                  SCATTER_FORWARD);

    KSPSolve(shell->multi->getFieldRelaxation(i),
             *((*shell->multi->GetRFieldList())[i]),
             *((*shell->multi->GetXFieldList())[i]));

    VecScatterBegin(*((*shell->multi->GetFieldScatterList())[i]),
                    *((*shell->multi->GetXFieldList())[i]),
                    *((*shell->multi->GetXList())[i]), ADD_VALUES,
                    SCATTER_REVERSE);
    VecScatterEnd(*((*shell->multi->GetFieldScatterList())[i]),
                  *((*shell->multi->GetXFieldList())[i]),
                  *((*shell->multi->GetXList())[i]), ADD_VALUES,
                  SCATTER_REVERSE);

    // neighbor part smoothing
    MatMult(shell->multi->getNeighborWholeMat(i),
            *((*shell->multi->GetXList())[i]),
            *((*shell->multi->GetXNeighborList())[i]));

    VecScatterBegin(*((*shell->multi->GetNeighborScatterList())[i]),
                    *((*shell->multi->GetBList())[i]),
                    *((*shell->multi->GetBNeighborList())[i]), INSERT_VALUES,
                    SCATTER_FORWARD);
    VecScatterEnd(*((*shell->multi->GetNeighborScatterList())[i]),
                  *((*shell->multi->GetBList())[i]),
                  *((*shell->multi->GetBNeighborList())[i]), INSERT_VALUES,
                  SCATTER_FORWARD);

    VecAXPY(*((*shell->multi->GetBNeighborList())[i]), -1.0,
            *((*shell->multi->GetXNeighborList())[i]));

    KSPSolve(shell->multi->getNeighborRelaxation(i),
             *((*shell->multi->GetBNeighborList())[i]),
             *((*shell->multi->GetXNeighborList())[i]));

    VecScatterBegin(*((*shell->multi->GetNeighborScatterList())[i]),
                    *((*shell->multi->GetXNeighborList())[i]),
                    *((*shell->multi->GetXList())[i]), ADD_VALUES,
                    SCATTER_REVERSE);
    VecScatterEnd(*((*shell->multi->GetNeighborScatterList())[i]),
                  *((*shell->multi->GetXNeighborList())[i]),
                  *((*shell->multi->GetXList())[i]), ADD_VALUES,
                  SCATTER_REVERSE);
  }

  VecCopy(*((*shell->multi->GetXList())[shell->adaptive_level - 1]), y);

  // orthogonalize to constant vector
  VecScatterBegin(
      *((*shell->multi->GetPressureScatterList())[shell->adaptive_level - 1]),
      y, *((*shell->multi->GetXPressureList())[shell->adaptive_level - 1]),
      INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(
      *((*shell->multi->GetPressureScatterList())[shell->adaptive_level - 1]),
      y, *((*shell->multi->GetXPressureList())[shell->adaptive_level - 1]),
      INSERT_VALUES, SCATTER_FORWARD);

  VecSum(*((*shell->multi->GetXPressureList())[shell->adaptive_level - 1]),
         &pressure_sum);
  VecGetSize(*((*shell->multi->GetXPressureList())[shell->adaptive_level - 1]),
             &size);
  VecSet(*((*shell->multi->GetXPressureList())[shell->adaptive_level - 1]),
         -pressure_sum / size);

  VecScatterBegin(
      *((*shell->multi->GetPressureScatterList())[shell->adaptive_level - 1]),
      *((*shell->multi->GetXPressureList())[shell->adaptive_level - 1]), y,
      ADD_VALUES, SCATTER_REVERSE);
  VecScatterEnd(
      *((*shell->multi->GetPressureScatterList())[shell->adaptive_level - 1]),
      *((*shell->multi->GetXPressureList())[shell->adaptive_level - 1]), y,
      ADD_VALUES, SCATTER_REVERSE);

  return 0;
}

PetscErrorCode HypreLUShellPCDestroy(PC pc) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  PetscFree(shell);

  return 0;
}

PetscErrorCode HypreLUShellPCDestroyAdaptive(PC pc) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  PetscFree(shell);

  return 0;
}

PetscErrorCode HypreConstConstraintPCCreate(HypreConstConstraintPC **pc) {
  HypreConstConstraintPC *newctx;

  PetscNew(&newctx);
  *pc = newctx;

  return 0;
}

PetscErrorCode HypreConstConstraintPCSetUp(PC pc, Mat *mat,
                                           PetscInt block_size) {
  HypreConstConstraintPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  std::shared_ptr<sparse_matrix> a =
      shell->multi->_stokes.get_coefficient_matrix(0);

  KSPCreate(PETSC_COMM_WORLD, &shell->ksp_base);
  KSPSetOperators(shell->ksp_base, a->__mat, a->__mat);
  KSPSetType(shell->ksp_base, KSPPREONLY);

  PC pc_asm;
  KSPGetPC(shell->ksp_base, &pc_asm);
  PCSetType(pc_asm, PCASM);
  PCSetFromOptions(pc_asm);
  PCSetUp(pc_asm);

  KSPSetUp(shell->ksp_base);

  stokes_equation &stokes = shell->multi->_stokes;
  shell->num_layer = stokes.get_num_layer();

  shell->ksp_sor.resize(shell->num_layer);
  for (int i = 1; i < shell->num_layer; i++) {
    KSPCreate(PETSC_COMM_WORLD, &(shell->ksp_sor[i - 1]));
    KSPSetOperators(shell->ksp_sor[i - 1],
                    stokes.get_coefficient_matrix(i)->__mat,
                    stokes.get_coefficient_matrix(i)->__mat);
    KSPSetType(shell->ksp_sor[i - 1], KSPPREONLY);

    PC pc_sor;
    KSPGetPC(shell->ksp_sor[i - 1], &pc_sor);
    PCSetType(pc_sor, PCSOR);
    PCSORSetIterations(pc_sor, 3, 1);
    PCSetUp(pc_sor);

    KSPSetUp(shell->ksp_sor[i - 1]);
  }
  KSPCreate(PETSC_COMM_WORLD, &(shell->ksp_sor[shell->num_layer - 1]));
  KSPSetOperators(shell->ksp_sor[shell->num_layer - 1], *mat, *mat);
  KSPSetType(shell->ksp_sor[shell->num_layer - 1], KSPPREONLY);

  PC pc_sor;
  KSPGetPC(shell->ksp_sor[shell->num_layer - 1], &pc_sor);
  PCSetType(pc_sor, PCSOR);
  PCSORSetIterations(pc_sor, 3, 1);
  PCSetUp(pc_sor);

  KSPSetUp(shell->ksp_sor[shell->num_layer - 1]);

  shell->block_size = block_size;
  shell->offset = block_size - 1;

  shell->a = mat;

  return 0;
}

PetscErrorCode HypreConstConstraintPCApply(PC pc, Vec x, Vec y) {
  HypreConstConstraintPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  PetscReal *a;
  PetscInt size;

  VecGetLocalSize(x, &size);

  PetscInt local_particle_num = size / shell->block_size;

  PetscReal sum = 0.0;
  VecGetArray(x, &a);
  for (PetscInt i = 0; i < local_particle_num; i++) {
    sum += a[shell->block_size * i + shell->offset];
  }
  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  VecGetSize(x, &size);

  sum /= (size / shell->block_size);
  for (PetscInt i = 0; i < local_particle_num; i++) {
    a[shell->block_size * i + shell->offset] -= sum;
  }
  VecRestoreArray(x, &a);

  Vec r, t;
  VecDuplicate(x, &r);
  VecDuplicate(x, &t);

  KSPSolve(shell->ksp_sor[shell->num_layer - 1], x, y);
  MatMult(*shell->a, y, r);
  VecAXPY(r, -1.0, x);
  VecScale(r, -1.0);

  stokes_equation &stokes = shell->multi->_stokes;

  MatMult(stokes.get_restriction(shell->num_layer - 1)->__mat, r,
          *stokes.get_x(shell->num_layer - 1));
  // MatMultTranspose(stokes.get_interpolation(shell->num_layer - 1)->__mat, r,
  //                  *stokes.get_x(shell->num_layer - 1));

  for (int i = shell->num_layer - 2; i >= 0; i--) {
    KSPSolve(shell->ksp_sor[i], *stokes.get_x(i + 1), *stokes.get_y(i + 1));
    MatMult(stokes.get_coefficient_matrix(i + 1)->__mat, *stokes.get_y(i + 1),
            *stokes.get_r(i + 1));
    VecAXPY(*stokes.get_r(i + 1), -1.0, *stokes.get_x(i + 1));
    VecScale(*stokes.get_r(i + 1), -1.0);
    MatMult(stokes.get_restriction(i)->__mat, *stokes.get_x(i + 1),
            *stokes.get_x(i));
  }

  KSPSolve(shell->ksp_base, *stokes.get_x(0), *stokes.get_y(0));

  for (int i = 0; i < shell->num_layer - 2; i++) {
    MatMultAdd(stokes.get_interpolation(i)->__mat, *stokes.get_y(i),
               *stokes.get_y(i + 1), *stokes.get_y(i + 1));
    MatMult(stokes.get_coefficient_matrix(i + 1)->__mat, *stokes.get_y(i + 1),
            *stokes.get_r(i + 1));
    VecAXPY(*stokes.get_r(i + 1), -1.0, *stokes.get_x(i + 1));
    VecScale(*stokes.get_r(i + 1), -1.0);
    KSPSolve(shell->ksp_sor[i], *stokes.get_r(i + 1), *stokes.get_x(i + 1));
    VecAXPY(*stokes.get_y(i + 1), 1.0, *stokes.get_x(i + 1));
  }

  MatMultAdd(stokes.get_interpolation(shell->num_layer - 1)->__mat,
             *stokes.get_y(shell->num_layer - 1), y, y);
  MatMult(*shell->a, y, r);
  VecAXPY(r, -1.0, x);
  VecScale(r, -1.0);
  KSPSolve(shell->ksp_sor[shell->num_layer - 1], r, t);
  VecAXPY(y, 1.0, t);

  sum = 0.0;
  VecGetArray(y, &a);
  for (PetscInt i = 0; i < local_particle_num; i++) {
    sum += a[shell->block_size * i + shell->offset];
  }
  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  VecGetSize(y, &size);

  sum /= (size / shell->block_size);
  for (PetscInt i = 0; i < local_particle_num; i++) {
    a[shell->block_size * i + shell->offset] -= sum;
  }
  VecRestoreArray(y, &a);

  VecDestroy(&t);
  VecDestroy(&r);

  return 0;
}

PetscErrorCode HypreConstConstraintPCDestroy(PC pc) {
  HypreConstConstraintPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  KSPDestroy(&shell->ksp_base);
  for (int i = 0; i < shell->ksp_sor.size(); i++) {
    KSPDestroy(&(shell->ksp_sor[i]));
  }

  return 0;
}