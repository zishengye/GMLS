#ifndef COMPOSITE_PRECONDITIONER_H
#define COMPOSITE_PRECONDITIONER_H

#include <petscksp.h>

#include <vector>

#include "multilevel.h"

struct HypreLUShellPC {
  multilevel *multi;

  int adaptive_level;

  PetscInt local_particle_num, global_particle_num, field_dof;
};

PetscErrorCode HypreLUShellPCCreate(HypreLUShellPC **shell);

PetscErrorCode HypreLUShellPCSetUp(PC pc, multilevel *multi, Vec x,
                                   PetscInt local_particle_num,
                                   PetscInt field_dof);

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