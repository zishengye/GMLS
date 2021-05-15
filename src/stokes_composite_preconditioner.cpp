#include "stokes_composite_preconditioner.hpp"

using namespace std;

PetscErrorCode HypreLUShellPCCreate(HypreLUShellPC **shell) {
  HypreLUShellPC *newctx;

  PetscNew(&newctx);
  *shell = newctx;

  return 0;
}

PetscErrorCode HypreLUShellPCSetUp(PC pc, stokes_multilevel *multi, Vec x,
                                   PetscInt local_particle_num,
                                   PetscInt field_dof) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  shell->multi = multi;

  shell->refinement_level = shell->multi->get_interpolation_list().size();

  shell->local_particle_num = local_particle_num;
  shell->field_dof = field_dof;

  PetscInt global_particle_num;
  MPI_Allreduce(&local_particle_num, &global_particle_num, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  shell->global_particle_num = global_particle_num;

  shell->field_smooth_duration = new double[shell->refinement_level];
  shell->colloid_smooth_duration = new double[shell->refinement_level];
  shell->colloid_smooth_matmult_duration = new double[shell->refinement_level];
  shell->restriction_duration = new double[shell->refinement_level];
  shell->interpolation_duration = new double[shell->refinement_level];
  shell->level_iteration_duration = new double[shell->refinement_level];

  for (int i = 0; i < shell->refinement_level; i++) {
    shell->field_smooth_duration[i] = 0.0;
    shell->colloid_smooth_duration[i] = 0.0;
    shell->colloid_smooth_matmult_duration[i] = 0.0;
    shell->restriction_duration[i] = 0.0;
    shell->interpolation_duration[i] = 0.0;
    shell->level_iteration_duration[i] = 0.0;
  }
  shell->base_field_duration = 0.0;
  shell->base_colloid_duration = 0.0;

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

  VecScatterBegin(shell->multi->get_field_scatter_list()[0]->get_reference(), x,
                  shell->multi->get_x_field_list()[0]->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(shell->multi->get_field_scatter_list()[0]->get_reference(), x,
                shell->multi->get_x_field_list()[0]->get_reference(),
                INSERT_VALUES, SCATTER_FORWARD);

  KSPSolve(shell->multi->get_field_base()->get_reference(),
           shell->multi->get_x_field_list()[0]->get_reference(),
           shell->multi->get_y_field_list()[0]->get_reference());

  VecScatterBegin(shell->multi->get_field_scatter_list()[0]->get_reference(),
                  shell->multi->get_y_field_list()[0]->get_reference(), y,
                  INSERT_VALUES, SCATTER_REVERSE);
  VecScatterEnd(shell->multi->get_field_scatter_list()[0]->get_reference(),
                shell->multi->get_y_field_list()[0]->get_reference(), y,
                INSERT_VALUES, SCATTER_REVERSE);

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

  shell->multi->get_colloid_whole_mat(0)->get_reference();
  shell->multi->get_colloid_x()->get_reference();
  MatMult(shell->multi->get_colloid_whole_mat(0)->get_reference(), y,
          shell->multi->get_colloid_x()->get_reference());

  VecScatterBegin(shell->multi->get_colloid_scatter_list()[0]->get_reference(),
                  x, shell->multi->get_colloid_y()->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(shell->multi->get_colloid_scatter_list()[0]->get_reference(), x,
                shell->multi->get_colloid_y()->get_reference(), INSERT_VALUES,
                SCATTER_FORWARD);

  VecAXPY(shell->multi->get_colloid_y()->get_reference(), -1.0,
          shell->multi->get_colloid_x()->get_reference());

  KSPSolve(shell->multi->get_colloid_base()->get_reference(),
           shell->multi->get_colloid_y()->get_reference(),
           shell->multi->get_colloid_x()->get_reference());

  VecScatterBegin(shell->multi->get_colloid_scatter_list()[0]->get_reference(),
                  shell->multi->get_colloid_x()->get_reference(), y, ADD_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(shell->multi->get_colloid_scatter_list()[0]->get_reference(),
                shell->multi->get_colloid_x()->get_reference(), y, ADD_VALUES,
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

  return 0;
}

PetscErrorCode HypreLUShellPCApplyAdaptive(PC pc, Vec x, Vec y) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  PetscReal *a;
  PetscInt local_size, global_size;
  PetscReal pressure_sum;
  PetscInt size;

  double timer1, timer2;

  VecCopy(x,
          shell->multi->get_b_list()[shell->refinement_level]->get_reference());

  // sweep down
  for (int i = shell->refinement_level; i > 0; i--) {
    // pre smooth
    // orthogonalize to constant vector
    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_b_list()[i]->get_reference(),
        shell->multi->get_x_pressure_list()[i]->get_reference(), INSERT_VALUES,
        SCATTER_FORWARD);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_b_list()[i]->get_reference(),
                  shell->multi->get_x_pressure_list()[i]->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);

    VecSum(shell->multi->get_x_pressure_list()[i]->get_reference(),
           &pressure_sum);
    VecGetSize(shell->multi->get_x_pressure_list()[i]->get_reference(), &size);
    VecSet(shell->multi->get_x_pressure_list()[i]->get_reference(),
           -pressure_sum / size);

    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_x_pressure_list()[i]->get_reference(),
        shell->multi->get_b_list()[i]->get_reference(), ADD_VALUES,
        SCATTER_REVERSE);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_x_pressure_list()[i]->get_reference(),
                  shell->multi->get_b_list()[i]->get_reference(), ADD_VALUES,
                  SCATTER_REVERSE);

    // fluid part smoothing
    MPI_Barrier(MPI_COMM_WORLD);
    timer1 = MPI_Wtime();

    VecScatterBegin(shell->multi->get_field_scatter_list()[i]->get_reference(),
                    shell->multi->get_b_list()[i]->get_reference(),
                    shell->multi->get_b_field_list()[i]->get_reference(),
                    INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(shell->multi->get_field_scatter_list()[i]->get_reference(),
                  shell->multi->get_b_list()[i]->get_reference(),
                  shell->multi->get_b_field_list()[i]->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);

    KSPSolve(shell->multi->get_field_relaxation(i)->get_reference(),
             shell->multi->get_b_field_list()[i]->get_reference(),
             shell->multi->get_x_field_list()[i]->get_reference());

    VecSet(shell->multi->get_x_list()[i]->get_reference(), 0.0);

    VecScatterBegin(shell->multi->get_field_scatter_list()[i]->get_reference(),
                    shell->multi->get_x_field_list()[i]->get_reference(),
                    shell->multi->get_x_list()[i]->get_reference(),
                    INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(shell->multi->get_field_scatter_list()[i]->get_reference(),
                  shell->multi->get_x_field_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(), INSERT_VALUES,
                  SCATTER_REVERSE);

    MPI_Barrier(MPI_COMM_WORLD);
    timer2 = MPI_Wtime();
    shell->field_smooth_duration[i - 1] += timer2 - timer1;

    // orthogonalize to constant vector
    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_x_list()[i]->get_reference(),
        shell->multi->get_x_pressure_list()[i]->get_reference(), INSERT_VALUES,
        SCATTER_FORWARD);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(),
                  shell->multi->get_x_pressure_list()[i]->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);

    VecSum(shell->multi->get_x_pressure_list()[i]->get_reference(),
           &pressure_sum);
    VecGetSize(shell->multi->get_x_pressure_list()[i]->get_reference(), &size);
    VecSet(shell->multi->get_x_pressure_list()[i]->get_reference(),
           -pressure_sum / size);

    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_x_pressure_list()[i]->get_reference(),
        shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
        SCATTER_REVERSE);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_x_pressure_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
                  SCATTER_REVERSE);

    // neighbor part smoothing
    MPI_Barrier(MPI_COMM_WORLD);
    timer1 = MPI_Wtime();

    MatMult(shell->multi->getA(i)->get_shell_reference(),
            shell->multi->get_x_list()[i]->get_reference(),
            shell->multi->get_r_list()[i]->get_reference());

    VecAXPY(shell->multi->get_r_list()[i]->get_reference(), -1.0,
            shell->multi->get_b_list()[i]->get_reference());

    VecScale(shell->multi->get_r_list()[i]->get_reference(), -1.0);

    MPI_Barrier(MPI_COMM_WORLD);
    timer2 = MPI_Wtime();
    shell->colloid_smooth_matmult_duration[i - 1] += timer2 - timer1;

    MPI_Barrier(MPI_COMM_WORLD);
    timer1 = MPI_Wtime();

    VecScatterBegin(
        shell->multi->get_colloid_scatter_list()[i]->get_reference(),
        shell->multi->get_r_list()[i]->get_reference(),
        shell->multi->get_x_colloid_list()[i]->get_reference(), INSERT_VALUES,
        SCATTER_FORWARD);
    VecScatterEnd(shell->multi->get_colloid_scatter_list()[i]->get_reference(),
                  shell->multi->get_r_list()[i]->get_reference(),
                  shell->multi->get_x_colloid_list()[i]->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);

    KSPSolve(shell->multi->get_colloid_relaxation(i)->get_reference(),
             shell->multi->get_x_colloid_list()[i]->get_reference(),
             shell->multi->get_b_colloid_list()[i]->get_reference());

    VecScatterBegin(
        shell->multi->get_colloid_scatter_list()[i]->get_reference(),
        shell->multi->get_b_colloid_list()[i]->get_reference(),
        shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
        SCATTER_REVERSE);
    VecScatterEnd(shell->multi->get_colloid_scatter_list()[i]->get_reference(),
                  shell->multi->get_b_colloid_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
                  SCATTER_REVERSE);

    MPI_Barrier(MPI_COMM_WORLD);
    timer2 = MPI_Wtime();
    shell->colloid_smooth_duration[i - 1] += timer2 - timer1;

    // orthogonalize to constant vector
    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_x_list()[i]->get_reference(),
        shell->multi->get_x_pressure_list()[i]->get_reference(), INSERT_VALUES,
        SCATTER_FORWARD);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(),
                  shell->multi->get_x_pressure_list()[i]->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);

    VecSum(shell->multi->get_x_pressure_list()[i]->get_reference(),
           &pressure_sum);
    VecGetSize(shell->multi->get_x_pressure_list()[i]->get_reference(), &size);
    VecSet(shell->multi->get_x_pressure_list()[i]->get_reference(),
           -pressure_sum / size);

    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_x_pressure_list()[i]->get_reference(),
        shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
        SCATTER_REVERSE);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_x_pressure_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
                  SCATTER_REVERSE);

    // pressure part smoothing
    MatMult(shell->multi->get_pressure_whole_mat(i)->get_reference(),
            shell->multi->get_x_list()[i]->get_reference(),
            shell->multi->get_x_pressure_list()[i]->get_reference());

    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_b_list()[i]->get_reference(),
        shell->multi->get_y_pressure_list()[i]->get_reference(), INSERT_VALUES,
        SCATTER_FORWARD);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_b_list()[i]->get_reference(),
                  shell->multi->get_y_pressure_list()[i]->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);

    VecAXPY(shell->multi->get_y_pressure_list()[i]->get_reference(), -1.0,
            shell->multi->get_x_pressure_list()[i]->get_reference());

    KSPSolve(shell->multi->get_pressure_relaxation(i)->get_reference(),
             shell->multi->get_y_pressure_list()[i]->get_reference(),
             shell->multi->get_x_pressure_list()[i]->get_reference());

    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_x_pressure_list()[i]->get_reference(),
        shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
        SCATTER_REVERSE);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_x_pressure_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
                  SCATTER_REVERSE);

    // orthogonalize to constant vector
    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_x_list()[i]->get_reference(),
        shell->multi->get_x_pressure_list()[i]->get_reference(), INSERT_VALUES,
        SCATTER_FORWARD);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(),
                  shell->multi->get_x_pressure_list()[i]->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);

    VecSum(shell->multi->get_x_pressure_list()[i]->get_reference(),
           &pressure_sum);
    VecGetSize(shell->multi->get_x_pressure_list()[i]->get_reference(), &size);
    VecSet(shell->multi->get_x_pressure_list()[i]->get_reference(),
           -pressure_sum / size);

    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_x_pressure_list()[i]->get_reference(),
        shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
        SCATTER_REVERSE);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_x_pressure_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
                  SCATTER_REVERSE);

    // restriction
    MPI_Barrier(MPI_COMM_WORLD);
    timer1 = MPI_Wtime();

    MatMult(shell->multi->getA(i)->get_shell_reference(),
            shell->multi->get_x_list()[i]->get_reference(),
            shell->multi->get_r_list()[i]->get_reference());

    VecAXPY(shell->multi->get_r_list()[i]->get_reference(), -1.0,
            shell->multi->get_b_list()[i]->get_reference());

    MPI_Barrier(MPI_COMM_WORLD);
    timer2 = MPI_Wtime();
    shell->level_iteration_duration[i - 1] += timer2 - timer1;

    MPI_Barrier(MPI_COMM_WORLD);
    timer1 = MPI_Wtime();

    VecScale(shell->multi->get_r_list()[i]->get_reference(), -1.0);

    Mat &R = shell->multi->get_restriction_list()[i - 1]->get_reference();
    Vec &v1 = shell->multi->get_r_list()[i]->get_reference();
    Vec &v2 = shell->multi->get_b_list()[i - 1]->get_reference();
    MatMult(R, v1, v2);

    MPI_Barrier(MPI_COMM_WORLD);
    timer2 = MPI_Wtime();
    shell->restriction_duration[i - 1] += timer2 - timer1;
  }

  // solve on coarest-level
  // stage 1

  // orthogonalize to constant vector
  VecScatterBegin(shell->multi->get_pressure_scatter_list()[0]->get_reference(),
                  shell->multi->get_b_list()[0]->get_reference(),
                  shell->multi->get_x_pressure_list()[0]->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(shell->multi->get_pressure_scatter_list()[0]->get_reference(),
                shell->multi->get_b_list()[0]->get_reference(),
                shell->multi->get_x_pressure_list()[0]->get_reference(),
                INSERT_VALUES, SCATTER_FORWARD);

  VecSum(shell->multi->get_x_pressure_list()[0]->get_reference(),
         &pressure_sum);
  VecGetSize(shell->multi->get_x_pressure_list()[0]->get_reference(), &size);
  VecSet(shell->multi->get_x_pressure_list()[0]->get_reference(),
         -pressure_sum / size);

  VecScatterBegin(shell->multi->get_pressure_scatter_list()[0]->get_reference(),
                  shell->multi->get_x_pressure_list()[0]->get_reference(),
                  shell->multi->get_b_list()[0]->get_reference(), ADD_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(shell->multi->get_pressure_scatter_list()[0]->get_reference(),
                shell->multi->get_x_pressure_list()[0]->get_reference(),
                shell->multi->get_b_list()[0]->get_reference(), ADD_VALUES,
                SCATTER_REVERSE);

  MPI_Barrier(MPI_COMM_WORLD);
  timer1 = MPI_Wtime();

  VecSet(shell->multi->get_x_list()[0]->get_reference(), 0.0);
  VecScatterBegin(shell->multi->get_field_scatter_list()[0]->get_reference(),
                  shell->multi->get_b_list()[0]->get_reference(),
                  shell->multi->get_b_field_list()[0]->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(shell->multi->get_field_scatter_list()[0]->get_reference(),
                shell->multi->get_b_list()[0]->get_reference(),
                shell->multi->get_b_field_list()[0]->get_reference(),
                INSERT_VALUES, SCATTER_FORWARD);

  KSPSolve(shell->multi->get_field_base()->get_reference(),
           shell->multi->get_b_field_list()[0]->get_reference(),
           shell->multi->get_x_field_list()[0]->get_reference());

  VecScatterBegin(shell->multi->get_field_scatter_list()[0]->get_reference(),
                  shell->multi->get_x_field_list()[0]->get_reference(),
                  shell->multi->get_x_list()[0]->get_reference(), INSERT_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(shell->multi->get_field_scatter_list()[0]->get_reference(),
                shell->multi->get_x_field_list()[0]->get_reference(),
                shell->multi->get_x_list()[0]->get_reference(), INSERT_VALUES,
                SCATTER_REVERSE);

  MPI_Barrier(MPI_COMM_WORLD);
  timer2 = MPI_Wtime();
  shell->base_field_duration += timer2 - timer1;

  // orthogonalize to constant vector
  VecScatterBegin(shell->multi->get_pressure_scatter_list()[0]->get_reference(),
                  shell->multi->get_x_list()[0]->get_reference(),
                  shell->multi->get_x_pressure_list()[0]->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(shell->multi->get_pressure_scatter_list()[0]->get_reference(),
                shell->multi->get_x_list()[0]->get_reference(),
                shell->multi->get_x_pressure_list()[0]->get_reference(),
                INSERT_VALUES, SCATTER_FORWARD);

  VecSum(shell->multi->get_x_pressure_list()[0]->get_reference(),
         &pressure_sum);
  VecGetSize(shell->multi->get_x_pressure_list()[0]->get_reference(), &size);
  VecSet(shell->multi->get_x_pressure_list()[0]->get_reference(),
         -pressure_sum / size);

  VecScatterBegin(shell->multi->get_pressure_scatter_list()[0]->get_reference(),
                  shell->multi->get_x_pressure_list()[0]->get_reference(),
                  shell->multi->get_x_list()[0]->get_reference(), ADD_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(shell->multi->get_pressure_scatter_list()[0]->get_reference(),
                shell->multi->get_x_pressure_list()[0]->get_reference(),
                shell->multi->get_x_list()[0]->get_reference(), ADD_VALUES,
                SCATTER_REVERSE);

  // stage 2
  MPI_Barrier(MPI_COMM_WORLD);
  timer1 = MPI_Wtime();

  MatMult(shell->multi->getA(0)->get_shell_reference(),
          shell->multi->get_x_list()[0]->get_reference(),
          shell->multi->get_r_list()[0]->get_reference());

  VecAXPY(shell->multi->get_r_list()[0]->get_reference(), -1.0,
          shell->multi->get_b_list()[0]->get_reference());

  VecScale(shell->multi->get_r_list()[0]->get_reference(), -1.0);

  VecScatterBegin(shell->multi->get_colloid_scatter_list()[0]->get_reference(),
                  shell->multi->get_r_list()[0]->get_reference(),
                  shell->multi->get_colloid_y()->get_reference(), INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(shell->multi->get_colloid_scatter_list()[0]->get_reference(),
                shell->multi->get_r_list()[0]->get_reference(),
                shell->multi->get_colloid_y()->get_reference(), INSERT_VALUES,
                SCATTER_FORWARD);

  KSPSolve(shell->multi->get_colloid_base()->get_reference(),
           shell->multi->get_colloid_y()->get_reference(),
           shell->multi->get_colloid_x()->get_reference());

  VecScatterBegin(shell->multi->get_colloid_scatter_list()[0]->get_reference(),
                  shell->multi->get_colloid_x()->get_reference(),
                  shell->multi->get_x_list()[0]->get_reference(), ADD_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(shell->multi->get_colloid_scatter_list()[0]->get_reference(),
                shell->multi->get_colloid_x()->get_reference(),
                shell->multi->get_x_list()[0]->get_reference(), ADD_VALUES,
                SCATTER_REVERSE);

  MPI_Barrier(MPI_COMM_WORLD);
  timer2 = MPI_Wtime();
  shell->base_colloid_duration += timer2 - timer1;

  // orthogonalize to constant vector
  VecScatterBegin(shell->multi->get_pressure_scatter_list()[0]->get_reference(),
                  shell->multi->get_x_list()[0]->get_reference(),
                  shell->multi->get_x_pressure_list()[0]->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(shell->multi->get_pressure_scatter_list()[0]->get_reference(),
                shell->multi->get_x_list()[0]->get_reference(),
                shell->multi->get_x_pressure_list()[0]->get_reference(),
                INSERT_VALUES, SCATTER_FORWARD);

  VecSum(shell->multi->get_x_pressure_list()[0]->get_reference(),
         &pressure_sum);
  VecGetSize(shell->multi->get_x_pressure_list()[0]->get_reference(), &size);
  VecSet(shell->multi->get_x_pressure_list()[0]->get_reference(),
         -pressure_sum / size);

  VecScatterBegin(shell->multi->get_pressure_scatter_list()[0]->get_reference(),
                  shell->multi->get_x_pressure_list()[0]->get_reference(),
                  shell->multi->get_x_list()[0]->get_reference(), ADD_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(shell->multi->get_pressure_scatter_list()[0]->get_reference(),
                shell->multi->get_x_pressure_list()[0]->get_reference(),
                shell->multi->get_x_list()[0]->get_reference(), ADD_VALUES,
                SCATTER_REVERSE);

  // sweep up
  for (int i = 1; i <= shell->refinement_level; i++) {
    // interpolation
    MPI_Barrier(MPI_COMM_WORLD);
    timer1 = MPI_Wtime();

    Mat &I = shell->multi->get_interpolation_list()[i - 1]->get_reference();
    Vec &v1 = shell->multi->get_t_list()[i]->get_reference();
    Vec &v2 = shell->multi->get_x_list()[i - 1]->get_reference();
    MatMult(I, v2, v1);

    MPI_Barrier(MPI_COMM_WORLD);
    timer2 = MPI_Wtime();
    shell->interpolation_duration[i - 1] += timer2 - timer1;

    // post-smooth
    // orthogonalize to constant vector
    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_x_list()[i]->get_reference(),
        shell->multi->get_x_pressure_list()[i]->get_reference(), INSERT_VALUES,
        SCATTER_FORWARD);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(),
                  shell->multi->get_x_pressure_list()[i]->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);

    VecSum(shell->multi->get_x_pressure_list()[i]->get_reference(),
           &pressure_sum);
    VecGetSize(shell->multi->get_x_pressure_list()[i]->get_reference(), &size);
    VecSet(shell->multi->get_x_pressure_list()[i]->get_reference(),
           -pressure_sum / size);

    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_x_pressure_list()[i]->get_reference(),
        shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
        SCATTER_REVERSE);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_x_pressure_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
                  SCATTER_REVERSE);

    // fluid part smoothing
    MPI_Barrier(MPI_COMM_WORLD);
    timer1 = MPI_Wtime();

    VecAXPY(shell->multi->get_x_list()[i]->get_reference(), 1.0,
            shell->multi->get_t_list()[i]->get_reference());

    MatMult(shell->multi->getA(i)->get_shell_reference(),
            shell->multi->get_x_list()[i]->get_reference(),
            shell->multi->get_r_list()[i]->get_reference());

    MPI_Barrier(MPI_COMM_WORLD);
    timer2 = MPI_Wtime();
    shell->level_iteration_duration[i - 1] += timer2 - timer1;

    VecAXPY(shell->multi->get_r_list()[i]->get_reference(), -1.0,
            shell->multi->get_b_list()[i]->get_reference());

    VecScale(shell->multi->get_r_list()[i]->get_reference(), -1.0);

    MPI_Barrier(MPI_COMM_WORLD);
    timer1 = MPI_Wtime();

    VecScatterBegin(shell->multi->get_field_scatter_list()[i]->get_reference(),
                    shell->multi->get_r_list()[i]->get_reference(),
                    shell->multi->get_r_field_list()[i]->get_reference(),
                    INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(shell->multi->get_field_scatter_list()[i]->get_reference(),
                  shell->multi->get_r_list()[i]->get_reference(),
                  shell->multi->get_r_field_list()[i]->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);

    KSPSolve(shell->multi->get_field_relaxation(i)->get_reference(),
             shell->multi->get_r_field_list()[i]->get_reference(),
             shell->multi->get_x_field_list()[i]->get_reference());

    VecScatterBegin(shell->multi->get_field_scatter_list()[i]->get_reference(),
                    shell->multi->get_x_field_list()[i]->get_reference(),
                    shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
                    SCATTER_REVERSE);
    VecScatterEnd(shell->multi->get_field_scatter_list()[i]->get_reference(),
                  shell->multi->get_x_field_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
                  SCATTER_REVERSE);

    MPI_Barrier(MPI_COMM_WORLD);
    timer2 = MPI_Wtime();
    shell->field_smooth_duration[i - 1] += timer2 - timer1;

    // orthogonalize to constant vector
    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_x_list()[i]->get_reference(),
        shell->multi->get_x_pressure_list()[i]->get_reference(), INSERT_VALUES,
        SCATTER_FORWARD);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(),
                  shell->multi->get_x_pressure_list()[i]->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);

    VecSum(shell->multi->get_x_pressure_list()[i]->get_reference(),
           &pressure_sum);
    VecGetSize(shell->multi->get_x_pressure_list()[i]->get_reference(), &size);
    VecSet(shell->multi->get_x_pressure_list()[i]->get_reference(),
           -pressure_sum / size);

    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_x_pressure_list()[i]->get_reference(),
        shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
        SCATTER_REVERSE);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_x_pressure_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
                  SCATTER_REVERSE);

    // neighbor part smoothing
    MPI_Barrier(MPI_COMM_WORLD);
    timer1 = MPI_Wtime();

    MatMult(shell->multi->getA(i)->get_shell_reference(),
            shell->multi->get_x_list()[i]->get_reference(),
            shell->multi->get_r_list()[i]->get_reference());

    VecAXPY(shell->multi->get_r_list()[i]->get_reference(), -1.0,
            shell->multi->get_b_list()[i]->get_reference());

    VecScale(shell->multi->get_r_list()[i]->get_reference(), -1.0);

    MPI_Barrier(MPI_COMM_WORLD);
    timer2 = MPI_Wtime();
    shell->colloid_smooth_matmult_duration[i - 1] += timer2 - timer1;

    MPI_Barrier(MPI_COMM_WORLD);
    timer1 = MPI_Wtime();

    VecScatterBegin(
        shell->multi->get_colloid_scatter_list()[i]->get_reference(),
        shell->multi->get_r_list()[i]->get_reference(),
        shell->multi->get_x_colloid_list()[i]->get_reference(), INSERT_VALUES,
        SCATTER_FORWARD);
    VecScatterEnd(shell->multi->get_colloid_scatter_list()[i]->get_reference(),
                  shell->multi->get_r_list()[i]->get_reference(),
                  shell->multi->get_x_colloid_list()[i]->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);

    KSPSolve(shell->multi->get_colloid_relaxation(i)->get_reference(),
             shell->multi->get_x_colloid_list()[i]->get_reference(),
             shell->multi->get_b_colloid_list()[i]->get_reference());

    VecScatterBegin(
        shell->multi->get_colloid_scatter_list()[i]->get_reference(),
        shell->multi->get_b_colloid_list()[i]->get_reference(),
        shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
        SCATTER_REVERSE);
    VecScatterEnd(shell->multi->get_colloid_scatter_list()[i]->get_reference(),
                  shell->multi->get_b_colloid_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
                  SCATTER_REVERSE);

    MPI_Barrier(MPI_COMM_WORLD);
    timer2 = MPI_Wtime();
    shell->colloid_smooth_duration[i - 1] += timer2 - timer1;

    // orthogonalize to constant vector
    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_x_list()[i]->get_reference(),
        shell->multi->get_x_pressure_list()[i]->get_reference(), INSERT_VALUES,
        SCATTER_FORWARD);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(),
                  shell->multi->get_x_pressure_list()[i]->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);

    VecSum(shell->multi->get_x_pressure_list()[i]->get_reference(),
           &pressure_sum);
    VecGetSize(shell->multi->get_x_pressure_list()[i]->get_reference(), &size);
    VecSet(shell->multi->get_x_pressure_list()[i]->get_reference(),
           -pressure_sum / size);

    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_x_pressure_list()[i]->get_reference(),
        shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
        SCATTER_REVERSE);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_x_pressure_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
                  SCATTER_REVERSE);

    // pressure part smoothing
    MatMult(shell->multi->get_pressure_whole_mat(i)->get_reference(),
            shell->multi->get_x_list()[i]->get_reference(),
            shell->multi->get_x_pressure_list()[i]->get_reference());

    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_b_list()[i]->get_reference(),
        shell->multi->get_y_pressure_list()[i]->get_reference(), INSERT_VALUES,
        SCATTER_FORWARD);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_b_list()[i]->get_reference(),
                  shell->multi->get_y_pressure_list()[i]->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);

    VecAXPY(shell->multi->get_y_pressure_list()[i]->get_reference(), -1.0,
            shell->multi->get_x_pressure_list()[i]->get_reference());

    KSPSolve(shell->multi->get_pressure_relaxation(i)->get_reference(),
             shell->multi->get_y_pressure_list()[i]->get_reference(),
             shell->multi->get_x_pressure_list()[i]->get_reference());

    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_x_pressure_list()[i]->get_reference(),
        shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
        SCATTER_REVERSE);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_x_pressure_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
                  SCATTER_REVERSE);

    // orthogonalize to constant vector
    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_x_list()[i]->get_reference(),
        shell->multi->get_x_pressure_list()[i]->get_reference(), INSERT_VALUES,
        SCATTER_FORWARD);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(),
                  shell->multi->get_x_pressure_list()[i]->get_reference(),
                  INSERT_VALUES, SCATTER_FORWARD);

    VecSum(shell->multi->get_x_pressure_list()[i]->get_reference(),
           &pressure_sum);
    VecGetSize(shell->multi->get_x_pressure_list()[i]->get_reference(), &size);
    VecSet(shell->multi->get_x_pressure_list()[i]->get_reference(),
           -pressure_sum / size);

    VecScatterBegin(
        shell->multi->get_pressure_scatter_list()[i]->get_reference(),
        shell->multi->get_x_pressure_list()[i]->get_reference(),
        shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
        SCATTER_REVERSE);
    VecScatterEnd(shell->multi->get_pressure_scatter_list()[i]->get_reference(),
                  shell->multi->get_x_pressure_list()[i]->get_reference(),
                  shell->multi->get_x_list()[i]->get_reference(), ADD_VALUES,
                  SCATTER_REVERSE);
  }

  VecCopy(shell->multi->get_x_list()[shell->refinement_level]->get_reference(),
          y);

  return 0;
}

PetscErrorCode HypreLUShellPCDestroy(PC pc) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  PetscPrintf(PETSC_COMM_WORLD, "\nPreconditioner Log:\n");
  PetscPrintf(PETSC_COMM_WORLD, "Base field smooth duraction %fs\n",
              shell->base_field_duration);
  PetscPrintf(PETSC_COMM_WORLD, "Base colloid smooth duraction %fs\n",
              shell->base_colloid_duration);

  for (int i = 0; i < shell->refinement_level; i++) {
    PetscPrintf(PETSC_COMM_WORLD, "Field smooth level: %d, duraction %fs\n",
                i + 1, shell->field_smooth_duration[i]);
    PetscPrintf(PETSC_COMM_WORLD, "Colloid smooth level: %d, duraction %fs\n",
                i + 1, shell->colloid_smooth_duration[i]);
    PetscPrintf(PETSC_COMM_WORLD,
                "Colloid matmult smooth level: %d, duraction %fs\n", i + 1,
                shell->colloid_smooth_matmult_duration[i]);
    PetscPrintf(PETSC_COMM_WORLD, "Restriction level: %d, duraction %fs\n",
                i + 1, shell->restriction_duration[i]);
    PetscPrintf(PETSC_COMM_WORLD, "Interpolation level: %d, duraction %fs\n",
                i + 1, shell->interpolation_duration[i]);
    PetscPrintf(PETSC_COMM_WORLD, "Level iteration level: %d, duraction %fs\n",
                i + 1, shell->level_iteration_duration[i]);
    PetscPrintf(PETSC_COMM_WORLD, "\n");
  }

  delete[] shell->field_smooth_duration;
  delete[] shell->colloid_smooth_duration;
  delete[] shell->colloid_smooth_matmult_duration;
  delete[] shell->restriction_duration;
  delete[] shell->interpolation_duration;
  delete[] shell->level_iteration_duration;

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