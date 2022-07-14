#ifndef _STOKES_COMPOSITE_PRECONDITIONER_HPP_
#define _STOKES_COMPOSITE_PRECONDITIONER_HPP_

#include <petscksp.h>

#include <vector>

#include "StokesMultilevelPreconditioning.hpp"

struct HypreLUShellPC {
  StokesMultilevelPreconditioning *multi;

  int refinement_level, num_rigid_body;

  PetscInt local_particle_num, global_particle_num, field_dof;

  double *field_smooth_duration, *colloid_smooth_duration,
      *colloid_smooth_matmult_duration, *restriction_duration,
      *interpolation_duration, *level_iteration_duration, base_field_duration,
      base_colloid_duration;
};

PetscErrorCode HypreLUShellPCCreate(HypreLUShellPC **shell);

PetscErrorCode HypreLUShellPCSetUp(PC pc,
                                   StokesMultilevelPreconditioning *multi,
                                   Vec x, PetscInt local_particle_num,
                                   PetscInt field_dof, int num_rigid_body);

PetscErrorCode HypreLUShellPCApply(PC pc, Vec x, Vec y);

PetscErrorCode HypreLUShellPCApplyAdaptive(PC pc, Vec x, Vec y);

PetscErrorCode HypreLUShellPCDestroy(PC pc);

PetscErrorCode HypreLUShellPCDestroyAdaptive(PC pc);

struct HypreConstConstraintPC {
  KSP ksp_hypre;

  PetscInt block_size;
  PetscInt offset;
};

PetscErrorCode HypreConstConstraintPCCreate(HypreConstConstraintPC **pc);

PetscErrorCode HypreConstConstraintPCSetUp(PC pc, Mat *a, PetscInt block_size);

PetscErrorCode HypreConstConstraintPCApply(PC pc, Vec x, Vec y);

PetscErrorCode HypreConstConstraintPCDestroy(PC pc);

#endif