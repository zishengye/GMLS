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
  static double amg_duration = 0.0;
  static double matvec_duration = 0.0;
  static double lu_duration = 0.0;
  static double neighbor_vec_duration = 0.0;

  double tStart, tEnd;
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

  VecScatterBegin(*((*shell->multi->GetFieldScatterList())[0]), x,
                  *((*shell->multi->GetXFieldList())[0]), INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(*((*shell->multi->GetFieldScatterList())[0]), x,
                *((*shell->multi->GetXFieldList())[0]), INSERT_VALUES,
                SCATTER_FORWARD);

  tStart = MPI_Wtime();
  KSPSolve(shell->multi->getFieldBase(), *((*shell->multi->GetXFieldList())[0]),
           *((*shell->multi->GetYFieldList())[0]));
  tEnd = MPI_Wtime();
  amg_duration += tEnd - tStart;

  VecScatterBegin(*((*shell->multi->GetFieldScatterList())[0]),
                  *((*shell->multi->GetYFieldList())[0]), y, INSERT_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(*((*shell->multi->GetFieldScatterList())[0]),
                *((*shell->multi->GetYFieldList())[0]), y, INSERT_VALUES,
                SCATTER_REVERSE);

  // orthogonalize to constant vector
  VecGetArray(y, &a);
  pressure_sum = 0.0;
  for (PetscInt i = 0; i < shell->local_particle_num; i++)
    pressure_sum += a[shell->field_dof * i + pressure_offset];
  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  pressure_sum /= shell->global_particle_num;
  for (PetscInt i = 0; i < shell->local_particle_num; i++)
    a[shell->field_dof * i + pressure_offset] -= pressure_sum;
  VecRestoreArray(y, &a);

  // stage 2
  tStart = MPI_Wtime();
  MatMult(shell->multi->getNeighborWholeMat(0), y,
          *shell->multi->getXNeighbor());
  tEnd = MPI_Wtime();
  matvec_duration += tEnd - tStart;

  VecScatterBegin(*((*shell->multi->GetNeighborScatterList())[0]), x,
                  *shell->multi->getYNeighbor(), INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(*((*shell->multi->GetNeighborScatterList())[0]), x,
                *shell->multi->getYNeighbor(), INSERT_VALUES, SCATTER_FORWARD);

  VecAXPY(*shell->multi->getYNeighbor(), -1.0, *shell->multi->getXNeighbor());

  tStart = MPI_Wtime();
  KSPSolve(shell->multi->getNeighborBase(), *shell->multi->getYNeighbor(),
           *shell->multi->getXNeighbor());
  tEnd = MPI_Wtime();
  lu_duration += tEnd - tStart;

  tStart = MPI_Wtime();
  VecScatterBegin(*((*shell->multi->GetNeighborScatterList())[0]),
                  *shell->multi->getXNeighbor(), y, ADD_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(*((*shell->multi->GetNeighborScatterList())[0]),
                *shell->multi->getXNeighbor(), y, ADD_VALUES, SCATTER_REVERSE);
  tEnd = MPI_Wtime();
  neighbor_vec_duration += tEnd - tStart;

  // orthogonalize to constant vector
  VecGetArray(y, &a);
  pressure_sum = 0.0;
  for (PetscInt i = 0; i < shell->local_particle_num; i++)
    pressure_sum += a[shell->field_dof * i + pressure_offset];
  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  pressure_sum /= shell->global_particle_num;
  for (PetscInt i = 0; i < shell->local_particle_num; i++)
    a[shell->field_dof * i + pressure_offset] -= pressure_sum;
  VecRestoreArray(y, &a);

  return 0;
}

PetscErrorCode HypreLUShellPCApplyAdaptive(PC pc, Vec x, Vec y) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  PetscReal *a;
  PetscInt local_size, global_size;
  PetscReal pressure_sum;
  PetscInt size;

  VecCopy(x, *((*shell->multi->GetBList())[shell->adaptive_level - 1]));

  // sweep down
  for (int i = shell->adaptive_level - 1; i > 0; i--) {
    // pre-smooth
    // orthogonalize to constant vector
    // VecScatterBegin(*((*shell->multi->GetPressureScatterList())[i]),
    //                 *((*shell->multi->GetBList())[i]),
    //                 *((*shell->multi->GetXPressureList())[i]), INSERT_VALUES,
    //                 SCATTER_FORWARD);
    // VecScatterEnd(*((*shell->multi->GetPressureScatterList())[i]),
    //               *((*shell->multi->GetBList())[i]),
    //               *((*shell->multi->GetXPressureList())[i]), INSERT_VALUES,
    //               SCATTER_FORWARD);

    // VecSum(*((*shell->multi->GetXPressureList())[i]), &pressure_sum);
    // VecGetSize(*((*shell->multi->GetXPressureList())[i]), &size);
    // VecSet(*((*shell->multi->GetXPressureList())[i]), -pressure_sum / size);

    // VecScatterBegin(*((*shell->multi->GetPressureScatterList())[i]),
    //                 *((*shell->multi->GetXPressureList())[i]),
    //                 *((*shell->multi->GetBList())[i]), ADD_VALUES,
    //                 SCATTER_REVERSE);
    // VecScatterEnd(*((*shell->multi->GetPressureScatterList())[i]),
    //               *((*shell->multi->GetXPressureList())[i]),
    //               *((*shell->multi->GetBList())[i]), ADD_VALUES,
    //               SCATTER_REVERSE);

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

    // orthogonalize to constant vector
    // VecScatterBegin(*((*shell->multi->GetPressureScatterList())[i]),
    //                 *((*shell->multi->GetXList())[i]),
    //                 *((*shell->multi->GetXPressureList())[i]), INSERT_VALUES,
    //                 SCATTER_FORWARD);
    // VecScatterEnd(*((*shell->multi->GetPressureScatterList())[i]),
    //               *((*shell->multi->GetXList())[i]),
    //               *((*shell->multi->GetXPressureList())[i]), INSERT_VALUES,
    //               SCATTER_FORWARD);

    // VecSum(*((*shell->multi->GetXPressureList())[i]), &pressure_sum);
    // VecGetSize(*((*shell->multi->GetXPressureList())[i]), &size);
    // VecSet(*((*shell->multi->GetXPressureList())[i]), -pressure_sum / size);

    // VecScatterBegin(*((*shell->multi->GetPressureScatterList())[i]),
    //                 *((*shell->multi->GetXPressureList())[i]),
    //                 *((*shell->multi->GetXList())[i]), ADD_VALUES,
    //                 SCATTER_REVERSE);
    // VecScatterEnd(*((*shell->multi->GetPressureScatterList())[i]),
    //               *((*shell->multi->GetXPressureList())[i]),
    //               *((*shell->multi->GetXList())[i]), ADD_VALUES,
    //               SCATTER_REVERSE);

    // restriction
    MatMult(shell->multi->getA(i).__mat, *((*shell->multi->GetXList())[i]),
            *((*shell->multi->GetRList())[i]));

    VecAXPY(*((*shell->multi->GetRList())[i]), -1.0,
            *((*shell->multi->GetBList())[i]));

    // PetscReal norm;
    // VecNorm(*((*shell->multi->GetBNeighborList())[i]), NORM_2, &norm);
    // PetscPrintf(PETSC_COMM_WORLD, "b neighbor norm: %f\n", norm);
    // VecNorm(*((*shell->multi->GetXNeighborList())[i]), NORM_2, &norm);
    // PetscPrintf(PETSC_COMM_WORLD, "x neighbor norm: %f\n", norm);
    // VecNorm(*((*shell->multi->GetBList())[i]), NORM_2, &norm);
    // PetscPrintf(PETSC_COMM_WORLD, "b norm: %f\n", norm);
    // VecNorm(*((*shell->multi->GetRList())[i]), NORM_2, &norm);
    // PetscPrintf(PETSC_COMM_WORLD, "r norm: %f\n", norm);

    VecScale(*((*shell->multi->GetRList())[i]), -1.0);

    Mat *R = &(*shell->multi->GetRestrictionList())[i]->__mat;
    Vec *v1 = (*shell->multi->GetRList())[i];
    Vec *v2 = (*shell->multi->GetBList())[i - 1];
    MatMult(*R, *v1, *v2);
  }

  // solve on coarest-level
  // stage 1
  // orthogonalize to constant vector
  // VecScatterBegin(*((*shell->multi->GetPressureScatterList())[0]),
  //                 *((*shell->multi->GetBList())[0]),
  //                 *((*shell->multi->GetXPressureList())[0]), INSERT_VALUES,
  //                 SCATTER_FORWARD);
  // VecScatterEnd(*((*shell->multi->GetPressureScatterList())[0]),
  //               *((*shell->multi->GetBList())[0]),
  //               *((*shell->multi->GetXPressureList())[0]), INSERT_VALUES,
  //               SCATTER_FORWARD);

  // VecSum(*((*shell->multi->GetXPressureList())[0]), &pressure_sum);
  // VecGetSize(*((*shell->multi->GetXPressureList())[0]), &size);
  // VecSet(*((*shell->multi->GetXPressureList())[0]), -pressure_sum / size);

  // VecScatterBegin(*((*shell->multi->GetPressureScatterList())[0]),
  //                 *((*shell->multi->GetXPressureList())[0]),
  //                 *((*shell->multi->GetBList())[0]), ADD_VALUES,
  //                 SCATTER_REVERSE);
  // VecScatterEnd(*((*shell->multi->GetPressureScatterList())[0]),
  //               *((*shell->multi->GetXPressureList())[0]),
  //               *((*shell->multi->GetBList())[0]), ADD_VALUES,
  //               SCATTER_REVERSE);

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

  // orthogonalize to constant vector
  // VecScatterBegin(*((*shell->multi->GetPressureScatterList())[0]),
  //                 *((*shell->multi->GetXList())[0]),
  //                 *((*shell->multi->GetXPressureList())[0]), INSERT_VALUES,
  //                 SCATTER_FORWARD);
  // VecScatterEnd(*((*shell->multi->GetPressureScatterList())[0]),
  //               *((*shell->multi->GetXList())[0]),
  //               *((*shell->multi->GetXPressureList())[0]), INSERT_VALUES,
  //               SCATTER_FORWARD);

  // VecSum(*((*shell->multi->GetXPressureList())[0]), &pressure_sum);
  // VecGetSize(*((*shell->multi->GetXPressureList())[0]), &size);
  // VecSet(*((*shell->multi->GetXPressureList())[0]), -pressure_sum / size);

  // VecScatterBegin(*((*shell->multi->GetPressureScatterList())[0]),
  //                 *((*shell->multi->GetXPressureList())[0]),
  //                 *((*shell->multi->GetXList())[0]), ADD_VALUES,
  //                 SCATTER_REVERSE);
  // VecScatterEnd(*((*shell->multi->GetPressureScatterList())[0]),
  //               *((*shell->multi->GetXPressureList())[0]),
  //               *((*shell->multi->GetXList())[0]), ADD_VALUES,
  //               SCATTER_REVERSE);

  // sweep up
  for (int i = 1; i < shell->adaptive_level; i++) {
    // interpolation
    Mat *I = &(*shell->multi->GetInterpolationList())[i]->__mat;
    Vec *v1 = (*shell->multi->GetTList())[i];
    Vec *v2 = (*shell->multi->GetXList())[i - 1];
    MatMult(*I, *v2, *v1);

    // post-smooth
    // orthogonalize to constant vector
    // VecScatterBegin(*((*shell->multi->GetPressureScatterList())[i]),
    //                 *((*shell->multi->GetXList())[i]),
    //                 *((*shell->multi->GetXPressureList())[i]), INSERT_VALUES,
    //                 SCATTER_FORWARD);
    // VecScatterEnd(*((*shell->multi->GetPressureScatterList())[i]),
    //               *((*shell->multi->GetXList())[i]),
    //               *((*shell->multi->GetXPressureList())[i]), INSERT_VALUES,
    //               SCATTER_FORWARD);

    // VecSum(*((*shell->multi->GetXPressureList())[i]), &pressure_sum);
    // VecGetSize(*((*shell->multi->GetXPressureList())[i]), &size);
    // VecSet(*((*shell->multi->GetXPressureList())[i]), -pressure_sum / size);

    // VecScatterBegin(*((*shell->multi->GetPressureScatterList())[i]),
    //                 *((*shell->multi->GetXPressureList())[i]),
    //                 *((*shell->multi->GetXList())[i]), ADD_VALUES,
    //                 SCATTER_REVERSE);
    // VecScatterEnd(*((*shell->multi->GetPressureScatterList())[i]),
    //               *((*shell->multi->GetXPressureList())[i]),
    //               *((*shell->multi->GetXList())[i]), ADD_VALUES,
    //               SCATTER_REVERSE);

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

    // orthogonalize to constant vector
    // VecScatterBegin(*((*shell->multi->GetPressureScatterList())[i]),
    //                 *((*shell->multi->GetXList())[i]),
    //                 *((*shell->multi->GetXPressureList())[i]), INSERT_VALUES,
    //                 SCATTER_FORWARD);
    // VecScatterEnd(*((*shell->multi->GetPressureScatterList())[i]),
    //               *((*shell->multi->GetXList())[i]),
    //               *((*shell->multi->GetXPressureList())[i]), INSERT_VALUES,
    //               SCATTER_FORWARD);

    // VecSum(*((*shell->multi->GetXPressureList())[i]), &pressure_sum);
    // VecGetSize(*((*shell->multi->GetXPressureList())[i]), &size);
    // VecSet(*((*shell->multi->GetXPressureList())[i]), -pressure_sum / size);

    // VecScatterBegin(*((*shell->multi->GetPressureScatterList())[i]),
    //                 *((*shell->multi->GetXPressureList())[i]),
    //                 *((*shell->multi->GetXList())[i]), ADD_VALUES,
    //                 SCATTER_REVERSE);
    // VecScatterEnd(*((*shell->multi->GetPressureScatterList())[i]),
    //               *((*shell->multi->GetXPressureList())[i]),
    //               *((*shell->multi->GetXList())[i]), ADD_VALUES,
    //               SCATTER_REVERSE);
  }

  VecCopy(*((*shell->multi->GetXList())[shell->adaptive_level - 1]), y);

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

  KSPSetUp(shell->ksp_hypre);

  shell->block_size = block_size;
  shell->offset = block_size - 1;

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

  int localsize = local_particle_num;
  MPI_Allreduce(MPI_IN_PLACE, &localsize, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

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

  return 0;
}

PetscErrorCode HypreConstConstraintPCDestroy(PC pc) {
  HypreConstConstraintPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  KSPDestroy(&shell->ksp_hypre);

  return 0;
}