#include "composite_preconditioner.h"

PetscErrorCode HypreLUShellPCCreate(HypreLUShellPC **shell) {
  HypreLUShellPC *newctx;

  PetscNew(&newctx);
  *shell = newctx;

  return 0;
}

PetscErrorCode HypreLUShellPCSetUp(PC pc, Mat *a, Mat *amat, Mat *cmat,
                                   IS *isg0, IS *isg1, IS *isg00, IS *isg01,
                                   Mat *asmat, Vec x) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  shell->A = a;

  KSPCreate(PETSC_COMM_WORLD, &shell->field);
  KSPCreate(PETSC_COMM_WORLD, &shell->nearField);
  KSPSetOperators(shell->field, *amat, *amat);
  KSPSetOperators(shell->nearField, *cmat, *cmat);
  ISDuplicate(*isg0, &shell->isg0);
  ISDuplicate(*isg1, &shell->isg1);
  KSPSetType(shell->field, KSPGMRES);
  KSPGMRESSetRestart(shell->field, 50);
  KSPSetType(shell->nearField, KSPPREONLY);
  KSPSetTolerances(shell->field, 1e-1, 1e-50, 1e5, 5000);
  KSPSetTolerances(shell->nearField, 1e-6, 1e-50, 1e5, 1);

  PC pcField;
  PC pcNearField;

  KSPGetPC(shell->field, &pcField);
  PCSetType(pcField, PCFIELDSPLIT);
  PCFieldSplitSetIS(pcField, "0", *isg00);
  PCFieldSplitSetIS(pcField, "1", *isg01);
  PCFieldSplitSetSchurPre(pcField, PC_FIELDSPLIT_SCHUR_PRE_USER, *asmat);
  PCSetFromOptions(pcField);
  PCSetUp(pcField);

  KSP *subKsp;
  PetscInt n = 1;
  PCFieldSplitGetSubKSP(pcField, &n, &subKsp);
  KSPSetOperators(subKsp[1], *asmat, *asmat);
  KSPSetFromOptions(subKsp[0]);
  KSPSetFromOptions(subKsp[1]);
  PetscFree(subKsp);

  KSPGetPC(shell->nearField, &pcNearField);
  PCSetType(pcNearField, PCLU);
  PCSetUp(pcNearField);

  KSPSetUp(shell->field);
  KSPSetUp(shell->nearField);

  return 0;
}

PetscErrorCode HypreLUShellPCApply(PC pc, Vec x, Vec y) {
  Vec x1, x2, y1, y2, z2, t, t2;

  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  // multiplicative - first field, then rigid
  VecSet(y, 0.0);

  VecGetSubVector(x, shell->isg0, &x1);
  VecGetSubVector(y, shell->isg0, &y1);
  KSPSolve(shell->field, x1, y1);
  VecRestoreSubVector(x, shell->isg0, &x1);
  VecRestoreSubVector(y, shell->isg0, &y1);

  VecDuplicate(x, &t);
  MatMult(*shell->A, y, t);
  VecAXPY(t, -1.0, x);
  VecGetSubVector(y, shell->isg1, &y2);
  VecGetSubVector(t, shell->isg1, &t2);
  VecDuplicate(y2, &z2);
  KSPSolve(shell->nearField, t2, z2);
  VecAXPY(y2, -1.0, z2);
  VecRestoreSubVector(t, shell->isg1, &t2);
  VecRestoreSubVector(y, shell->isg1, &y2);

  VecDestroy(&t);
  VecDestroy(&z2);

  return 0;
}

PetscErrorCode HypreLUShellPCDestroy(PC pc) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  KSPDestroy(&shell->field);
  KSPDestroy(&shell->nearField);

  ISDestroy(&shell->isg0);
  ISDestroy(&shell->isg1);

  PetscFree(shell);

  return 0;
}