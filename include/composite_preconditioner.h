#pragma once

#include <petscksp.h>

#include <vector>

#include "multilevel.h"

struct HypreLUShellPC {
  multilevel *multi;

  int adaptive_level;

  PetscInt local_pressure_size, global_pressure_size;
};

PetscErrorCode HypreLUShellPCCreate(HypreLUShellPC **shell);

PetscErrorCode HypreLUShellPCSetUp(PC pc, multilevel *multi, Vec x);

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