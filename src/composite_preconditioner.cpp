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
  PCSetType(pcNearField, PCBJACOBI);
  PCSetUp(pcNearField);
  KSP *bjacobi_ksp;
  PCBJacobiGetSubKSP(pcNearField, NULL, NULL, &bjacobi_ksp);
  KSPSetType(bjacobi_ksp[0], KSPPREONLY);
  PC bjacobi_pc;
  KSPGetPC(bjacobi_ksp[0], &bjacobi_pc);
  PCSetType(bjacobi_pc, PCLU);
  PCFactorSetMatSolverType(bjacobi_pc, MATSOLVERMUMPS);
  PCSetUp(bjacobi_pc);

  KSPGetPC(shell->globalSmoother, &pcGlobalSmoother);
  PCSetType(pcGlobalSmoother, PCPBJACOBI);
  PCSetFromOptions(pcGlobalSmoother);
  PCSetUp(pcGlobalSmoother);

  KSPSetUp(shell->field);
  KSPSetUp(shell->nearField);
  KSPSetUp(shell->globalSmoother);

  MatCreateVecs(*amat, &shell->z1, NULL);
  MatCreateVecs(*cmat, &shell->z2, NULL);
  VecDuplicate(shell->z1, &shell->x1);
  VecDuplicate(shell->z1, &shell->y1);
  VecDuplicate(shell->z1, &shell->t1);
  VecDuplicate(shell->z2, &shell->x2);
  VecDuplicate(shell->z2, &shell->y2);
  VecDuplicate(shell->z2, &shell->t2);

  VecScatterCreate(x, *isg0, shell->z1, NULL, &shell->ctx_scatter1);
  VecScatterCreate(x, *isg1, shell->z2, NULL, &shell->ctx_scatter2);

  IS isg_col;
  MatGetOwnershipIS(*a, &isg_col, NULL);

  MatCreateSubMatrix(*a, *isg0, isg_col, MAT_INITIAL_MATRIX, &shell->stage1);
  MatCreateSubMatrix(*a, *isg1, isg_col, MAT_INITIAL_MATRIX, &shell->stage2);

  MatScale(shell->stage1, -1.0);
  MatScale(shell->stage2, -1.0);

  ISDestroy(&isg_col);

  return 0;
}

PetscErrorCode HypreLUShellPCSetUpAdaptive(PC pc, Mat *a, Mat *amat, Mat *cmat,
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
  PCSetType(pcNearField, PCBJACOBI);
  PCSetUp(pcNearField);
  KSP *bjacobi_ksp;
  PCBJacobiGetSubKSP(pcNearField, NULL, NULL, &bjacobi_ksp);
  KSPSetType(bjacobi_ksp[0], KSPPREONLY);
  PC bjacobi_pc;
  KSPGetPC(bjacobi_ksp[0], &bjacobi_pc);
  PCSetType(bjacobi_pc, PCLU);
  PCFactorSetMatSolverType(bjacobi_pc, MATSOLVERMUMPS);
  PCSetUp(bjacobi_pc);

  KSPGetPC(shell->globalSmoother, &pcGlobalSmoother);
  PCSetType(pcGlobalSmoother, PCJACOBI);
  PCSetFromOptions(pcGlobalSmoother);
  PCSetUp(pcGlobalSmoother);

  KSPSetUp(shell->field);
  KSPSetUp(shell->nearField);
  KSPSetUp(shell->globalSmoother);

  MatCreateVecs(*amat, &shell->z1, NULL);
  MatCreateVecs(*cmat, &shell->z2, NULL);
  VecDuplicate(shell->z1, &shell->x1);
  VecDuplicate(shell->z1, &shell->y1);
  VecDuplicate(shell->z1, &shell->t1);
  VecDuplicate(shell->z2, &shell->x2);
  VecDuplicate(shell->z2, &shell->y2);
  VecDuplicate(shell->z2, &shell->t2);

  VecScatterCreate(x, *isg0, shell->z1, NULL, &shell->ctx_scatter1);
  VecScatterCreate(x, *isg1, shell->z2, NULL, &shell->ctx_scatter2);

  IS isg_col;
  MatGetOwnershipIS(*a, &isg_col, NULL);

  MatCreateSubMatrix(*a, *isg0, isg_col, MAT_INITIAL_MATRIX, &shell->stage1);
  MatCreateSubMatrix(*a, *isg1, isg_col, MAT_INITIAL_MATRIX, &shell->stage2);

  MatScale(shell->stage1, -1.0);
  MatScale(shell->stage2, -1.0);

  ISDestroy(&isg_col);

  return 0;
}

PetscErrorCode HypreLUShellPCApply(PC pc, Vec x, Vec y) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  VecScatterBegin(shell->ctx_scatter1, x, shell->x1, INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(shell->ctx_scatter1, x, shell->x1, INSERT_VALUES,
                SCATTER_FORWARD);

  VecScatterBegin(shell->ctx_scatter2, x, shell->x2, INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(shell->ctx_scatter2, x, shell->x2, INSERT_VALUES,
                SCATTER_FORWARD);

  // stage 1
  VecSet(y, 0.0);
  VecScatterBegin(shell->ctx_scatter1, y, shell->y1, INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(shell->ctx_scatter1, y, shell->y1, INSERT_VALUES,
                SCATTER_FORWARD);

  KSPSolve(shell->field, shell->x1, shell->y1);

  VecScatterBegin(shell->ctx_scatter1, shell->y1, y, INSERT_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(shell->ctx_scatter1, shell->y1, y, INSERT_VALUES,
                SCATTER_REVERSE);

  // stage 2
  MatMultAdd(shell->stage2, y, shell->x2, shell->t2);

  KSPSolve(shell->nearField, shell->t2, shell->z2);

  VecScatterBegin(shell->ctx_scatter2, shell->z2, y, ADD_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(shell->ctx_scatter2, shell->z2, y, ADD_VALUES, SCATTER_REVERSE);

  // stage 3
  MatMultAdd(shell->stage1, y, shell->x1, shell->t1);

  KSPSolve(shell->globalSmoother, shell->t1, shell->z1);

  VecScatterBegin(shell->ctx_scatter1, shell->z1, y, ADD_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(shell->ctx_scatter1, shell->z1, y, ADD_VALUES, SCATTER_REVERSE);

  return 0;
}

PetscErrorCode HypreLUShellPCApplyAdaptive(PC pc, Vec x, Vec y) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  VecScatterBegin(shell->ctx_scatter1, x, shell->x1, INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(shell->ctx_scatter1, x, shell->x1, INSERT_VALUES,
                SCATTER_FORWARD);

  VecScatterBegin(shell->ctx_scatter2, x, shell->x2, INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(shell->ctx_scatter2, x, shell->x2, INSERT_VALUES,
                SCATTER_FORWARD);

  // stage 1
  VecSet(y, 0.0);
  VecScatterBegin(shell->ctx_scatter1, y, shell->y1, INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(shell->ctx_scatter1, y, shell->y1, INSERT_VALUES,
                SCATTER_FORWARD);

  KSPSolve(shell->field, shell->x1, shell->y1);

  VecScatterBegin(shell->ctx_scatter1, shell->y1, y, INSERT_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(shell->ctx_scatter1, shell->y1, y, INSERT_VALUES,
                SCATTER_REVERSE);

  // stage 2
  MatMultAdd(shell->stage1, y, shell->x1, shell->t1);

  KSPSolve(shell->globalSmoother, shell->t1, shell->z1);

  VecScatterBegin(shell->ctx_scatter1, shell->z1, y, ADD_VALUES,
                  SCATTER_REVERSE);
  VecScatterEnd(shell->ctx_scatter1, shell->z1, y, ADD_VALUES, SCATTER_REVERSE);

  return 0;
}

PetscErrorCode HypreLUShellPCDestroy(PC pc) {
  HypreLUShellPC *shell;
  PCShellGetContext(pc, (void **)&shell);

  KSPDestroy(&shell->field);
  KSPDestroy(&shell->nearField);
  KSPDestroy(&shell->globalSmoother);

  VecScatterDestroy(&shell->ctx_scatter1);
  VecScatterDestroy(&shell->ctx_scatter2);

  ISDestroy(&shell->isg0);
  ISDestroy(&shell->isg1);

  VecDestroy(&shell->z1);
  VecDestroy(&shell->z2);
  VecDestroy(&shell->x1);
  VecDestroy(&shell->x2);
  VecDestroy(&shell->y1);
  VecDestroy(&shell->y2);
  VecDestroy(&shell->t1);
  VecDestroy(&shell->t2);

  MatDestroy(&shell->stage1);
  MatDestroy(&shell->stage2);

  PetscFree(shell);

  return 0;
}