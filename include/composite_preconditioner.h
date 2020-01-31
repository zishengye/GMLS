#pragma once

#include <petscksp.h>

struct HypreLUShellPC {
  KSP field;
  KSP nearField;

  IS isg0, isg1;

  Mat *A;
};

PetscErrorCode HypreLUShellPCCreate(HypreLUShellPC **shell);

PetscErrorCode HypreLUShellPCSetUp(PC pc, Mat *a, Mat *amat, Mat *cmat,
                                   IS *isg0, IS *isg1, IS *isg00, IS *isg01,
                                   Mat *asmat, Vec x);

PetscErrorCode HypreLUShellPCApply(PC pc, Vec x, Vec y);

PetscErrorCode HypreLUShellPCDestroy(PC pc);