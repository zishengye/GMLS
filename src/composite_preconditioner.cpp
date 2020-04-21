#include "composite_preconditioner.h"

PetscErrorCode HypreLUShellPCCreate(HypreLUShellPC **shell) {
  HypreLUShellPC *newctx;

  PetscNew(&newctx);
  *shell = newctx;

  return 0;
}

PetscErrorCode HypreLUShellPCSetUp(PC pc, Mat *a, Mat *amat, Mat *cmat,
                                   IS *isg0, IS *isg1, Vec x) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  shell->A = a;

  KSPCreate(PETSC_COMM_WORLD, &shell->field);
  KSPCreate(PETSC_COMM_WORLD, &shell->nearField);
  KSPCreate(PETSC_COMM_WORLD, &shell->globalSmoother);
  KSPSetOperators(shell->field, *amat, *amat);
  KSPSetOperators(shell->nearField, *cmat, *cmat);
  KSPSetOperators(shell->globalSmoother, *amat, *amat);
  ISDuplicate(*isg0, &shell->isg0);
  ISDuplicate(*isg1, &shell->isg1);
  KSPSetType(shell->field, KSPPREONLY);
  KSPSetTolerances(shell->field, 1e-3, 1e-50, 1e5, 1);
  KSPSetType(shell->nearField, KSPPREONLY);
  KSPSetTolerances(shell->nearField, 1e-6, 1e-50, 1e5, 1);
  KSPSetType(shell->globalSmoother, KSPPREONLY);
  KSPSetTolerances(shell->globalSmoother, 1e-3, 1e-50, 1e5, 1);

  PC pcField;
  PC pcNearField;
  PC pcGlobalSmoother;

  KSPGetPC(shell->field, &pcField);
  PCSetType(pcField, PCHYPRE);
  PCSetFromOptions(pcField);
  PCSetUp(pcField);

  KSPGetPC(shell->nearField, &pcNearField);
  PCSetType(pcNearField, PCLU);
  PCSetUp(pcNearField);

  KSPGetPC(shell->globalSmoother, &pcGlobalSmoother);
  PCSetType(pcGlobalSmoother, PCBJACOBI);
  PCSetFromOptions(pcGlobalSmoother);
  PCSetUp(pcGlobalSmoother);

  KSPSetUp(shell->field);
  KSPSetUp(shell->nearField);
  KSPSetUp(shell->globalSmoother);

  VecDuplicate(x, &shell->t);
  MatCreateVecs(*amat, &shell->z1, NULL);
  MatCreateVecs(*cmat, &shell->z2, NULL);

  return 0;
}

PetscErrorCode HypreLUShellPCApply(PC pc, Vec x, Vec y) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  VecSet(y, 0.0);

  MatMult(*shell->A, y, shell->t);
  VecAXPY(shell->t, -1.0, x);
  VecGetSubVector(shell->t, shell->isg0, &shell->t1);
  KSPSolve(shell->field, shell->t1, shell->z1);
  VecRestoreSubVector(shell->t, shell->isg0, &shell->t1);
  VecGetSubVector(y, shell->isg0, &shell->y1);
  VecAXPY(shell->y1, -1.0, shell->z1);
  VecRestoreSubVector(y, shell->isg0, &shell->y1);

  MatMult(*shell->A, y, shell->t);
  VecAXPY(shell->t, -1.0, x);
  VecGetSubVector(y, shell->isg1, &shell->y2);
  VecGetSubVector(shell->t, shell->isg1, &shell->t2);
  KSPSolve(shell->nearField, shell->t2, shell->z2);
  VecAXPY(shell->y2, -1.0, shell->z2);
  VecRestoreSubVector(shell->t, shell->isg1, &shell->t2);
  VecRestoreSubVector(y, shell->isg1, &shell->y2);

  MatMult(*shell->A, y, shell->t);
  VecAXPY(shell->t, -1.0, x);
  VecGetSubVector(shell->t, shell->isg0, &shell->t1);
  KSPSolve(shell->globalSmoother, shell->t1, shell->z1);
  VecRestoreSubVector(shell->t, shell->isg0, &shell->t1);
  VecGetSubVector(y, shell->isg0, &shell->y1);
  VecAXPY(shell->y1, -1.0, shell->z1);
  VecRestoreSubVector(y, shell->isg0, &shell->y1);

  return 0;
}

PetscErrorCode HypreLUShellPCDestroy(PC pc) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  KSPDestroy(&shell->field);
  KSPDestroy(&shell->nearField);
  KSPDestroy(&shell->globalSmoother);

  ISDestroy(&shell->isg0);
  ISDestroy(&shell->isg1);

  VecDestroy(&shell->t);
  VecDestroy(&shell->z1);
  VecDestroy(&shell->z2);

  PetscFree(shell);

  return 0;
}