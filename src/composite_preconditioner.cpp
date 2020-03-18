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
  KSPSetType(shell->field, KSPFGMRES);
  KSPSetTolerances(shell->field, 1e-3, 1e-50, 1e5, 3000);
  KSPGMRESSetOrthogonalization(shell->field,
                               KSPGMRESModifiedGramSchmidtOrthogonalization);
  KSPGMRESSetRestart(shell->field, 150);
  KSPSetType(shell->nearField, KSPPREONLY);
  KSPSetTolerances(shell->nearField, 1e-6, 1e-50, 1e5, 1);

  PC pcField;
  PC pcNearField;

  KSPGetPC(shell->field, &pcField);
  PCSetType(pcField, PCHYPRE);
  // PCSetType(pcField, PCFIELDSPLIT);
  // PCFieldSplitSetIS(pcField, "0", *isg00);
  // PCFieldSplitSetIS(pcField, "1", *isg01);
  // PCFieldSplitSetSchurPre(pcField, PC_FIELDSPLIT_SCHUR_PRE_USER, *asmat);
  PCSetFromOptions(pcField);
  PCSetUp(pcField);

  // KSP *subKsp;
  // PetscInt n = 1;
  // PCFieldSplitGetSubKSP(pcField, &n, &subKsp);
  // KSPSetOperators(subKsp[1], *asmat, *asmat);
  // KSPSetFromOptions(subKsp[0]);
  // KSPSetFromOptions(subKsp[1]);
  // PetscFree(subKsp);

  KSPGetPC(shell->nearField, &pcNearField);
  PCSetType(pcNearField, PCLU);
  PCSetUp(pcNearField);

  KSPSetUp(shell->field);
  KSPSetUp(shell->nearField);

  return 0;
}

PetscErrorCode HypreLUShellPCApply(PC pc, Vec x, Vec y) {
  Vec x1, x2, y1, y2, z1, z2, t, t1, t2;

  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  // multiplicative - first field, then rigid
  VecSet(y, 0.0);

  VecGetSubVector(x, shell->isg0, &x1);
  VecGetSubVector(y, shell->isg0, &y1);
  KSPSolve(shell->field, x1, y1);
  PetscInt its;
  KSPGetIterationNumber(shell->field, &its);
  PetscPrintf(PETSC_COMM_WORLD, "iteration number: %d\n", its);
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

  // multiplicative - first rigid, then field
  // VecSet(y, 0.0);

  // VecGetSubVector(x, shell->isg1, &x2);
  // VecGetSubVector(y, shell->isg1, &y2);
  // KSPSolve(shell->nearField, x2, y2);
  // VecRestoreSubVector(x, shell->isg1, &x2);
  // VecRestoreSubVector(y, shell->isg1, &y2);

  // VecDuplicate(x, &t);
  // MatMult(*shell->A, y, t);
  // VecAXPY(t, -1.0, x);
  // VecGetSubVector(y, shell->isg0, &y1);
  // VecGetSubVector(t, shell->isg0, &t1);
  // VecDuplicate(y1, &z1);
  // KSPSolve(shell->field, t1, z1);
  // VecAXPY(y1, -1.0, z1);
  // VecRestoreSubVector(t, shell->isg0, &t1);
  // VecRestoreSubVector(y, shell->isg0, &y1);

  // VecDestroy(&t);
  // VecDestroy(&z1);

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