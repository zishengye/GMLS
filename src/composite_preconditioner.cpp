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

  VecCopy(x, *((*shell->multi->GetBList())[shell->adaptive_level - 1]));

  // sweep down
  for (int i = shell->adaptive_level - 1; i > 0; i--) {
    // pre-smooth
    VecScatterBegin(*((*shell->multi->GetFieldScatterList())[i]),
                    *((*shell->multi->GetBList())[i]),
                    *((*shell->multi->GetBSubList())[i]), INSERT_VALUES,
                    SCATTER_FORWARD);
    VecScatterEnd(*((*shell->multi->GetFieldScatterList())[i]),
                  *((*shell->multi->GetBList())[i]),
                  *((*shell->multi->GetBSubList())[i]), INSERT_VALUES,
                  SCATTER_FORWARD);

    KSPSolve(shell->multi->getRelaxation(i),
             *((*shell->multi->GetBSubList())[i]),
             *((*shell->multi->GetXSubList())[i]));

    VecSet(*((*shell->multi->GetXList())[i]), 0.0);

    VecScatterBegin(*((*shell->multi->GetFieldScatterList())[i]),
                    *((*shell->multi->GetXSubList())[i]),
                    *((*shell->multi->GetXList())[i]), INSERT_VALUES,
                    SCATTER_REVERSE);
    VecScatterEnd(*((*shell->multi->GetFieldScatterList())[i]),
                  *((*shell->multi->GetXSubList())[i]),
                  *((*shell->multi->GetXList())[i]), INSERT_VALUES,
                  SCATTER_REVERSE);

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
                  *((*shell->multi->GetBSubList())[0]), INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(*((*shell->multi->GetFieldScatterList())[0]),
                *((*shell->multi->GetBList())[0]),
                *((*shell->multi->GetBSubList())[0]), INSERT_VALUES,
                SCATTER_FORWARD);

  KSPSolve(shell->multi->getFieldBase(), *((*shell->multi->GetBSubList())[0]),
           *((*shell->multi->GetXSubList())[0]));

  VecScatterBegin(*((*shell->multi->GetFieldScatterList())[0]),
                  *((*shell->multi->GetXSubList())[0]),
                  *((*shell->multi->GetXList())[0]), INSERT_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(*((*shell->multi->GetFieldScatterList())[0]),
                *((*shell->multi->GetXSubList())[0]),
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
    VecAXPY(*((*shell->multi->GetXList())[i]), 1.0,
            *((*shell->multi->GetTList())[i]));

    MatMult(shell->multi->getA(i).__mat, *((*shell->multi->GetXList())[i]),
            *((*shell->multi->GetRList())[i]));

    VecAXPY(*((*shell->multi->GetRList())[i]), -1.0,
            *((*shell->multi->GetBList())[i]));

    VecScale(*((*shell->multi->GetRList())[i]), -1.0);

    VecScatterBegin(*((*shell->multi->GetFieldScatterList())[i]),
                    *((*shell->multi->GetRList())[i]),
                    *((*shell->multi->GetRSubList())[i]), INSERT_VALUES,
                    SCATTER_FORWARD);
    VecScatterEnd(*((*shell->multi->GetFieldScatterList())[i]),
                  *((*shell->multi->GetRList())[i]),
                  *((*shell->multi->GetRSubList())[i]), INSERT_VALUES,
                  SCATTER_FORWARD);

    KSPSolve(shell->multi->getRelaxation(i),
             *((*shell->multi->GetRSubList())[i]),
             *((*shell->multi->GetXSubList())[i]));

    VecScatterBegin(*((*shell->multi->GetFieldScatterList())[i]),
                    *((*shell->multi->GetXSubList())[i]),
                    *((*shell->multi->GetXList())[i]), ADD_VALUES,
                    SCATTER_REVERSE);
    VecScatterEnd(*((*shell->multi->GetFieldScatterList())[i]),
                  *((*shell->multi->GetXSubList())[i]),
                  *((*shell->multi->GetXList())[i]), ADD_VALUES,
                  SCATTER_REVERSE);
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