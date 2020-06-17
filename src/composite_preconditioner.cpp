#include "composite_preconditioner.h"

using namespace std;

PetscErrorCode HypreLUShellPCCreate(HypreLUShellPC **shell) {
  HypreLUShellPC *newctx;

  PetscNew(&newctx);
  *shell = newctx;

  return 0;
}

PetscErrorCode HypreLUShellPCSetUp(PC pc, multilevel *multi, Vec x) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  shell->multi = multi;

  shell->adaptive_level = shell->multi->GetInterpolationList()->size();

  VecGetLocalSize(*((*shell->multi->GetXPressureList())[0]),
                  &shell->local_pressure_size);
  VecGetSize(*((*shell->multi->GetXPressureList())[0]),
             &shell->global_pressure_size);

  return 0;
}

PetscErrorCode HypreLUShellPCApply(PC pc, Vec x, Vec y) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  PetscReal *a;

  // orthogonalize to constant vector
  VecScatterBegin(*((*shell->multi->GetPressureScatterList())[0]), x,
                  *((*shell->multi->GetXPressureList())[0]), INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(*((*shell->multi->GetPressureScatterList())[0]), x,
                *((*shell->multi->GetXPressureList())[0]), INSERT_VALUES,
                SCATTER_FORWARD);

  PetscReal pressure_sum = 0.0;
  VecGetArray(*((*shell->multi->GetXPressureList())[0]), &a);
  for (PetscInt i = 0; i < shell->local_pressure_size; i++) {
    pressure_sum += a[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  pressure_sum /= shell->global_pressure_size;
  for (PetscInt i = 0; i < shell->local_pressure_size; i++) {
    a[i] -= pressure_sum;
  }
  VecRestoreArray(*((*shell->multi->GetXPressureList())[0]), &a);

  VecScatterBegin(*((*shell->multi->GetPressureScatterList())[0]),
                  *((*shell->multi->GetXPressureList())[0]), x, INSERT_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(*((*shell->multi->GetPressureScatterList())[0]),
                *((*shell->multi->GetXPressureList())[0]), x, INSERT_VALUES,
                SCATTER_REVERSE);

  // stage 1
  VecSet(y, 0.0);
  VecScatterBegin(*((*shell->multi->GetFieldScatterList())[0]), x,
                  *((*shell->multi->GetXFieldList())[0]), INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(*((*shell->multi->GetFieldScatterList())[0]), x,
                *((*shell->multi->GetXFieldList())[0]), INSERT_VALUES,
                SCATTER_FORWARD);

  KSPSolve(shell->multi->getFieldBase(), *((*shell->multi->GetXFieldList())[0]),
           *((*shell->multi->GetYFieldList())[0]));

  VecScatterBegin(*((*shell->multi->GetFieldScatterList())[0]),
                  *((*shell->multi->GetYFieldList())[0]), y, INSERT_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(*((*shell->multi->GetFieldScatterList())[0]),
                *((*shell->multi->GetYFieldList())[0]), y, INSERT_VALUES,
                SCATTER_REVERSE);

  // stage 2
  MatMult(shell->multi->getA(0).__mat, y, *(*shell->multi->GetRList())[0]);
  VecAXPY(*(*shell->multi->GetRList())[0], -1.0, x);
  VecScale(*(*shell->multi->GetRList())[0], -1.0);

  VecScatterBegin(*((*shell->multi->GetNeighborScatterList())[0]),
                  *((*shell->multi->GetRList())[0]),
                  *shell->multi->getXNeighbor(), INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(*((*shell->multi->GetNeighborScatterList())[0]),
                *((*shell->multi->GetRList())[0]),
                *shell->multi->getXNeighbor(), INSERT_VALUES, SCATTER_FORWARD);

  KSPSolve(shell->multi->getNeighborBase(), *shell->multi->getXNeighbor(),
           *shell->multi->getYNeighbor());

  VecScatterBegin(*((*shell->multi->GetNeighborScatterList())[0]),
                  *shell->multi->getYNeighbor(), y, ADD_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(*((*shell->multi->GetNeighborScatterList())[0]),
                *shell->multi->getYNeighbor(), y, ADD_VALUES, SCATTER_REVERSE);

  // orthogonalize to constant vector
  VecScatterBegin(*((*shell->multi->GetPressureScatterList())[0]), y,
                  *((*shell->multi->GetXPressureList())[0]), INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(*((*shell->multi->GetPressureScatterList())[0]), y,
                *((*shell->multi->GetXPressureList())[0]), INSERT_VALUES,
                SCATTER_FORWARD);

  pressure_sum = 0.0;
  VecGetArray(*((*shell->multi->GetXPressureList())[0]), &a);
  for (PetscInt i = 0; i < shell->local_pressure_size; i++) {
    pressure_sum += a[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  pressure_sum /= shell->global_pressure_size;
  for (PetscInt i = 0; i < shell->local_pressure_size; i++) {
    a[i] -= pressure_sum;
  }
  VecRestoreArray(*((*shell->multi->GetXPressureList())[0]), &a);

  VecScatterBegin(*((*shell->multi->GetPressureScatterList())[0]),
                  *((*shell->multi->GetXPressureList())[0]), y, INSERT_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(*((*shell->multi->GetPressureScatterList())[0]),
                *((*shell->multi->GetXPressureList())[0]), y, INSERT_VALUES,
                SCATTER_REVERSE);

  return 0;
}

PetscErrorCode HypreLUShellPCApplyAdaptive(PC pc, Vec x, Vec y) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  PetscReal *a;

  // orthogonalize to constant vector
  VecScatterBegin(*((*shell->multi->GetPressureScatterList())[0]), x,
                  *((*shell->multi->GetXPressureList())[0]), INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(*((*shell->multi->GetPressureScatterList())[0]), x,
                *((*shell->multi->GetXPressureList())[0]), INSERT_VALUES,
                SCATTER_FORWARD);

  PetscReal pressure_sum = 0.0;
  VecGetArray(*((*shell->multi->GetXPressureList())[0]), &a);
  for (PetscInt i = 0; i < shell->local_pressure_size; i++) {
    pressure_sum += a[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  pressure_sum /= shell->global_pressure_size;
  for (PetscInt i = 0; i < shell->local_pressure_size; i++) {
    a[i] -= pressure_sum;
  }
  VecRestoreArray(*((*shell->multi->GetXPressureList())[0]), &a);

  VecScatterBegin(*((*shell->multi->GetPressureScatterList())[0]),
                  *((*shell->multi->GetXPressureList())[0]), x, INSERT_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(*((*shell->multi->GetPressureScatterList())[0]),
                *((*shell->multi->GetXPressureList())[0]), x, INSERT_VALUES,
                SCATTER_REVERSE);

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
    MatMult(shell->multi->getA(i).__mat, *((*shell->multi->GetXList())[i]),
            *((*shell->multi->GetRList())[i]));

    VecAXPY(*((*shell->multi->GetRList())[i]), -1.0,
            *((*shell->multi->GetBList())[i]));

    VecScale(*((*shell->multi->GetRList())[i]), -1.0);

    VecScatterBegin(*((*shell->multi->GetNeighborScatterList())[i]),
                    *((*shell->multi->GetRList())[i]),
                    *((*shell->multi->GetBNeighborList())[i]), INSERT_VALUES,
                    SCATTER_FORWARD);
    VecScatterEnd(*((*shell->multi->GetNeighborScatterList())[i]),
                  *((*shell->multi->GetRList())[i]),
                  *((*shell->multi->GetBNeighborList())[i]), INSERT_VALUES,
                  SCATTER_FORWARD);

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
  MatMult(shell->multi->getA(0).__mat, *((*shell->multi->GetXList())[0]),
          *(*shell->multi->GetRList())[0]);
  VecAXPY(*(*shell->multi->GetRList())[0], -1.0,
          *((*shell->multi->GetBList())[0]));
  VecScale(*(*shell->multi->GetRList())[0], -1.0);

  VecScatterBegin(*((*shell->multi->GetNeighborScatterList())[0]),
                  *((*shell->multi->GetRList())[0]),
                  *shell->multi->getXNeighbor(), INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(*((*shell->multi->GetNeighborScatterList())[0]),
                *((*shell->multi->GetRList())[0]),
                *shell->multi->getXNeighbor(), INSERT_VALUES, SCATTER_FORWARD);

  KSPSolve(shell->multi->getNeighborBase(), *shell->multi->getXNeighbor(),
           *shell->multi->getYNeighbor());

  VecScatterBegin(*((*shell->multi->GetNeighborScatterList())[0]),
                  *shell->multi->getYNeighbor(),
                  *((*shell->multi->GetXList())[0]), ADD_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(*((*shell->multi->GetNeighborScatterList())[0]),
                *shell->multi->getYNeighbor(),
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
    MatMult(shell->multi->getA(i).__mat, *((*shell->multi->GetXList())[i]),
            *((*shell->multi->GetRList())[i]));

    VecAXPY(*((*shell->multi->GetRList())[i]), -1.0,
            *((*shell->multi->GetBList())[i]));

    VecScale(*((*shell->multi->GetRList())[i]), -1.0);

    VecScatterBegin(*((*shell->multi->GetNeighborScatterList())[i]),
                    *((*shell->multi->GetRList())[i]),
                    *((*shell->multi->GetBNeighborList())[i]), INSERT_VALUES,
                    SCATTER_FORWARD);
    VecScatterEnd(*((*shell->multi->GetNeighborScatterList())[i]),
                  *((*shell->multi->GetRList())[i]),
                  *((*shell->multi->GetBNeighborList())[i]), INSERT_VALUES,
                  SCATTER_FORWARD);

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
  VecScatterBegin(*((*shell->multi->GetPressureScatterList())[0]), y,
                  *((*shell->multi->GetXPressureList())[0]), INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(*((*shell->multi->GetPressureScatterList())[0]), y,
                *((*shell->multi->GetXPressureList())[0]), INSERT_VALUES,
                SCATTER_FORWARD);

  pressure_sum = 0.0;
  VecGetArray(*((*shell->multi->GetXPressureList())[0]), &a);
  for (PetscInt i = 0; i < shell->local_pressure_size; i++) {
    pressure_sum += a[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  pressure_sum /= shell->global_pressure_size;
  for (PetscInt i = 0; i < shell->local_pressure_size; i++) {
    a[i] -= pressure_sum;
  }
  VecRestoreArray(*((*shell->multi->GetXPressureList())[0]), &a);

  VecScatterBegin(*((*shell->multi->GetPressureScatterList())[0]),
                  *((*shell->multi->GetXPressureList())[0]), y, INSERT_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(*((*shell->multi->GetPressureScatterList())[0]),
                *((*shell->multi->GetXPressureList())[0]), y, INSERT_VALUES,
                SCATTER_REVERSE);

  return 0;
}

PetscErrorCode HypreLUShellPCDestroy(PC pc) {
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

PetscErrorCode HypreConstConstraintPCSetUp(PC pc, Mat *a, PetscInt block_size) {
  HypreConstConstraintPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  KSPCreate(PETSC_COMM_WORLD, &shell->ksp_hypre);
  KSPSetOperators(shell->ksp_hypre, *a, *a);
  KSPSetType(shell->ksp_hypre, KSPPREONLY);

  PC pc_hypre;
  KSPGetPC(shell->ksp_hypre, &pc_hypre);
  PCSetType(pc_hypre, PCHYPRE);
  PCSetFromOptions(pc_hypre);
  PCSetUp(pc_hypre);

  shell->block_size = block_size;
  shell->offset = block_size - 1;
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

  int localsize = local_particle_num;
  MPI_Allreduce(MPI_IN_PLACE, &localsize, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "size: %d\n", localsize);

  KSPSolve(shell->ksp_hypre, x, y);

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
}

PetscErrorCode HypreConstConstraintPCDestroy(PC pc) {
  HypreConstConstraintPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  KSPDestroy(&shell->ksp_hypre);
}