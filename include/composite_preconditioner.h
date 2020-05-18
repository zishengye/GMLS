#pragma once

#include <petscksp.h>

#include <vector>

#include "multilevel.h"

struct HypreLUShellPC
{
  KSP field;
  KSP nearField;
  KSP globalSmoother;

  IS isg0, isg1;

  Mat *A;
  Mat *a;

  Mat *A_base;

  Mat stage1, stage2;

  Vec x1, x2, y1, y2, z, z1, z2, t, t1, t2;

  VecScatter ctx_scatter1, ctx_scatter2;

  std::vector<PetscSparseMatrix *> *interpolation;
  std::vector<PetscSparseMatrix *> *restriction;
  std::vector<Vec *> level_vec;
};

PetscErrorCode HypreLUShellPCCreate(HypreLUShellPC **shell);

PetscErrorCode HypreLUShellPCSetUp(PC pc, Mat *a, Mat *amat, Mat *cmat,
                                   IS *isg0, IS *isg1, Vec x);

PetscErrorCode HypreLUShellPCSetUpAdaptive(
    PC pc, Mat *a, Mat *amat, Mat *amat_base, Mat *cmat, IS *isg0, IS *isg1,
    std::vector<PetscSparseMatrix *> *interpolation,
    std::vector<PetscSparseMatrix *> *restriction, Vec x);

PetscErrorCode HypreLUShellPCApply(PC pc, Vec x, Vec y);

PetscErrorCode HypreLUShellPCApplyAdaptive(PC pc, Vec x, Vec y);

PetscErrorCode HypreLUShellPCDestroy(PC pc);

PetscErrorCode HypreLUShellPCDestroyAdaptive(PC pc);