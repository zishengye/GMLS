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

  return 0;
}

PetscErrorCode HypreLUShellPCApply(PC pc, Vec x, Vec y) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  // stage 1
  VecSet(y, 0.0);
  VecScatterBegin(*((*shell->multi->GetFieldScatterList())[0]), x,
                  *((*shell->multi->GetXSubList())[0]), INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(*((*shell->multi->GetFieldScatterList())[0]), x,
                *((*shell->multi->GetXSubList())[0]), INSERT_VALUES,
                SCATTER_FORWARD);

  KSPSolve(shell->multi->getFieldBase(), *((*shell->multi->GetXSubList())[0]),
           *((*shell->multi->GetYSubList())[0]));

  VecScatterBegin(*((*shell->multi->GetFieldScatterList())[0]),
                  *((*shell->multi->GetYSubList())[0]), y, INSERT_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(*((*shell->multi->GetFieldScatterList())[0]),
                *((*shell->multi->GetYSubList())[0]), y, INSERT_VALUES,
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

  return 0;
}

PetscErrorCode HypreLUShellPCApplyAdaptive(PC pc, Vec x, Vec y) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  // VecScatterBegin(shell->ctx_scatter1, x, shell->x1, INSERT_VALUES,
  //                 SCATTER_FORWARD);
  // VecScatterEnd(shell->ctx_scatter1, x, shell->x1, INSERT_VALUES,
  //               SCATTER_FORWARD);

  // VecScatterBegin(shell->ctx_scatter2, x, shell->x2, INSERT_VALUES,
  //                 SCATTER_FORWARD);
  // VecScatterEnd(shell->ctx_scatter2, x, shell->x2, INSERT_VALUES,
  //               SCATTER_FORWARD);

  // // stage 1
  // VecSet(y, 0.0);
  // VecScatterBegin(shell->ctx_scatter1, y, shell->y1, INSERT_VALUES,
  //                 SCATTER_FORWARD);
  // VecScatterEnd(shell->ctx_scatter1, y, shell->y1, INSERT_VALUES,
  //               SCATTER_FORWARD);

  // VecCopy(shell->x1, *((*shell->multi->GetBList())[shell->adaptive_level -
  // 1]));

  // // sweep down
  // for (int i = shell->adaptive_level - 1; i > 0; i--) {
  //   // pre-smooth
  //   KSPSolve(shell->multi->getRelaxation(i),
  //   *((*shell->multi->GetBList())[i]),
  //            *((*shell->multi->GetXList())[i]));

  //   MatMult(shell->multi->getFieldMat(i), *((*shell->multi->GetXList())[i]),
  //           *((*shell->multi->GetRList())[i]));

  //   VecAXPY(*((*shell->multi->GetRList())[i]), -1.0,
  //           *((*shell->multi->GetBList())[i]));

  //   VecScale(*((*shell->multi->GetRList())[i]), -1.0);

  //   Mat *R = &(*shell->multi->GetRestrictionList())[i]->__mat;
  //   Vec *v1 = (*shell->multi->GetRList())[i];
  //   Vec *v2 = (*shell->multi->GetBList())[i - 1];
  //   MatMult(*R, *v1, *v2);
  // }

  // // solve on coarest-level
  // Vec *x_base = (*shell->multi->GetBList())[0];
  // Vec y_base;
  // VecDuplicate(*x_base, &y_base);
  // KSPSolve(shell->field, *x_base, y_base);
  // VecCopy(y_base, *(*shell->multi->GetXList())[0]);

  // // sweep up
  // for (int i = 1; i < shell->adaptive_level; i++) {
  //   Mat *I = &(*shell->multi->GetInterpolationList())[i]->__mat;
  //   Vec *v1 = (*shell->multi->GetTList())[i];
  //   Vec *v2 = (*shell->multi->GetXList())[i - 1];
  //   MatMult(*I, *v2, *v1);

  //   VecAXPY(*((*shell->multi->GetXList())[i]), 1.0,
  //           *((*shell->multi->GetTList())[i]));

  //   MatMult(shell->multi->getFieldMat(i), *((*shell->multi->GetXList())[i]),
  //           *((*shell->multi->GetRList())[i]));

  //   VecAXPY(*((*shell->multi->GetRList())[i]), -1.0,
  //           *((*shell->multi->GetBList())[i]));

  //   KSPSolve(shell->multi->getRelaxation(i),
  //   *((*shell->multi->GetRList())[i]),
  //            *((*shell->multi->GetTList())[i]));

  //   VecAXPY(*((*shell->multi->GetXList())[i]), -1.0,
  //           *((*shell->multi->GetTList())[i]));
  // }

  // VecCopy(*((*shell->multi->GetXList())[shell->adaptive_level - 1]),
  // shell->y1);

  // VecScatterBegin(shell->ctx_scatter1, shell->y1, y, INSERT_VALUES,
  //                 SCATTER_REVERSE);
  // VecScatterEnd(shell->ctx_scatter1, shell->y1, y, INSERT_VALUES,
  //               SCATTER_REVERSE);

  // // stage 2
  // MatMultAdd(shell->stage2, y, shell->x2, shell->t2);

  // KSPSolve(shell->nearField, shell->t2, shell->z2);

  // VecScatterBegin(shell->ctx_scatter2, shell->z2, y, ADD_VALUES,
  //                 SCATTER_REVERSE);
  // VecScatterEnd(shell->ctx_scatter2, shell->z2, y, ADD_VALUES,
  // SCATTER_REVERSE);

  // stage 3
  // MatMultAdd(shell->stage1, y, shell->x1, shell->t1);

  // KSPSolve(shell->globalSmoother, shell->t1, shell->z1);

  // VecScatterBegin(shell->ctx_scatter1, shell->z1, y, ADD_VALUES,
  //                 SCATTER_REVERSE);
  // VecScatterEnd(shell->ctx_scatter1, shell->z1, y, ADD_VALUES,
  // SCATTER_REVERSE);

  return 0;
}

PetscErrorCode HypreLUShellPCDestroy(PC pc) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  PetscFree(shell);

  return 0;
}