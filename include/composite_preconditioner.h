#pragma once

#include <petscksp.h>

struct HypreLUShellPC {
  KSP field;
  KSP nearField;
  KSP globalSmoother;

  IS isg0, isg1;

  Mat *A;

  Vec x1, x2, y1, y2, z, z1, z2, t, t1, t2;
};

PetscErrorCode HypreLUShellPCCreate(HypreLUShellPC **shell);

PetscErrorCode HypreLUShellPCSetUp(PC pc, Mat *a, Mat *amat, Mat *cmat,
                                   IS *isg0, IS *isg1, Vec x);

PetscErrorCode HypreLUShellPCApply(PC pc, Vec x, Vec y);

PetscErrorCode HypreLUShellPCDestroy(PC pc);