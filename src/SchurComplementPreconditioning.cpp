#include "SchurComplementPreconditioning.hpp"

PetscErrorCode SchurComplementIterationWrapper(PC pc, Vec x, Vec y) {
  SchurComplementPreconditioning *shellPtr;
  PCShellGetContext(pc, (void **)&shellPtr);

  return shellPtr->ApplyPreconditioningIteration(x, y);
}

SchurComplementPreconditioning::SchurComplementPreconditioning()
    : blockM_(2), blockN_(2), schurComplement_(PETSC_NULL), a00Ksp_(PETSC_NULL),
      a11Ksp_(PETSC_NULL) {}

SchurComplementPreconditioning::~SchurComplementPreconditioning() {
  if (schurComplement_ != PETSC_NULL)
    MatDestroy(&schurComplement_);
  for (PetscInt i = 0; i < blockM_; i++) {
    VecDestroy(&lhsVector_[i]);
  }
  for (PetscInt i = 0; i < blockN_; i++) {
    VecDestroy(&rhsVector_[i]);
  }

  if (a00Ksp_ != PETSC_NULL)
    KSPDestroy(&a00Ksp_);
  if (a11Ksp_ != PETSC_NULL)
    KSPDestroy(&a11Ksp_);
}

PetscErrorCode
SchurComplementPreconditioning::ApplyPreconditioningIteration(Vec x, Vec y) {
  PetscReal *a, *b;
  VecGetArray(x, &a);
  for (PetscInt i = 0; i < blockN_; i++) {
    VecGetArray(rhsVector_[i], &b);
    for (unsigned int j = localRhsVectorOffset_[i];
         j < localRhsVectorOffset_[i + 1]; j++) {
      b[j - localRhsVectorOffset_[i]] = a[j];
    }
    VecRestoreArray(rhsVector_[i], &b);
  }
  VecRestoreArray(x, &a);

  for (PetscInt i = 0; i < blockM_; i++) {
    VecSet(lhsVector_[i], 0.0);
  }

  Vec a0, a1;
  VecDuplicate(rhsVector_[0], &a0);
  VecDuplicate(rhsVector_[1], &a1);

  auto &a01 = *(std::static_pointer_cast<PetscMatrix>(
      linearSystemsPtr_->GetSubMat(0, 1)));
  auto &a10 = *(std::static_pointer_cast<PetscMatrix>(
      linearSystemsPtr_->GetSubMat(1, 0)));

  KSPSolve(a00Ksp_, rhsVector_[0], lhsVector_[0]);
  MatMult(a10.GetReference(), lhsVector_[0], a1);
  VecAXPY(rhsVector_[1], -1.0, a1);
  KSPSolve(a11Ksp_, rhsVector_[1], lhsVector_[1]);

  MatMult(a01.GetReference(), lhsVector_[1], a0);
  KSPSolve(a00Ksp_, a0, a0);
  VecAXPY(lhsVector_[0], -1.0, a0);

  VecDestroy(&a0);
  VecDestroy(&a1);

  VecGetArray(y, &a);
  for (PetscInt i = 0; i < blockM_; i++) {
    VecGetArray(lhsVector_[i], &b);
    for (unsigned int j = localLhsVectorOffset_[i];
         j < localLhsVectorOffset_[i + 1]; j++) {
      a[j] = b[j - localLhsVectorOffset_[i]];
    }
    VecRestoreArray(lhsVector_[i], &b);
  }
  VecRestoreArray(y, &a);

  return 0;
}

void SchurComplementPreconditioning::AddLinearSystem(
    std::shared_ptr<PetscBlockMatrix> mat) {
  linearSystemsPtr_ = mat;

  // prepare schur complement matrix
  Mat B, C;

  auto &a00 = *(std::static_pointer_cast<PetscMatrix>(
      linearSystemsPtr_->GetSubMat(0, 0)));
  auto &a01 = *(std::static_pointer_cast<PetscMatrix>(
      linearSystemsPtr_->GetSubMat(0, 1)));
  auto &a10 = *(std::static_pointer_cast<PetscMatrix>(
      linearSystemsPtr_->GetSubMat(1, 0)));
  auto &a11 = *(std::static_pointer_cast<PetscMatrix>(
      linearSystemsPtr_->GetSubMat(1, 1)));
  MatCreate(MPI_COMM_WORLD, &B);
  MatSetType(B, MATAIJMKL);
  MatInvertBlockDiagonalMat(a00.GetReference(), B);

  MatMatMult(B, a01.GetReference(), MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);
  MatMatMult(a10.GetReference(), C, MAT_INITIAL_MATRIX, PETSC_DEFAULT,
             &schurComplement_);
  MatScale(schurComplement_, -1.0);
  MatAXPY(schurComplement_, 1.0, a11.GetReference(), DIFFERENT_NONZERO_PATTERN);

  MatDestroy(&B);
  MatDestroy(&C);

  MatNullSpace nullSpace;
  MatNullSpaceCreate(MPI_COMM_WORLD, PETSC_TRUE, 0, PETSC_NULL, &nullSpace);
  MatSetNullSpace(schurComplement_, nullSpace);
  MatNullSpaceDestroy(&nullSpace);

  // prepare auxiliary vectors
  PetscInt localRow = 0, localCol = 0;
  lhsVector_.resize(blockM_);
  rhsVector_.resize(blockN_);
  localLhsVectorOffset_.resize(blockM_ + 1);
  localRhsVectorOffset_.resize(blockN_ + 1);
  localLhsVectorOffset_[0] = 0;
  localRhsVectorOffset_[0] = 0;
  for (PetscInt i = 0; i < blockM_; i++) {
    PetscInt tLocalRow = linearSystemsPtr_->GetSubMat(i, 0)->GetRowSize();
    localRow += tLocalRow;
    localLhsVectorOffset_[i + 1] = localRow;
    VecCreateMPI(MPI_COMM_WORLD, tLocalRow, PETSC_DECIDE, &lhsVector_[i]);
  }

  for (PetscInt i = 0; i < blockN_; i++) {
    PetscInt tLocalCol = linearSystemsPtr_->GetSubMat(0, i)->GetColSize();
    localCol += tLocalCol;
    localRhsVectorOffset_[i + 1] = localCol;
    VecCreateMPI(MPI_COMM_WORLD, tLocalCol, PETSC_DECIDE, &rhsVector_[i]);
  }

  // prepare preconditioner for diagonal blocks
  KSPCreate(MPI_COMM_WORLD, &a00Ksp_);
  KSPSetType(a00Ksp_, KSPGMRES);
  KSPSetTolerances(a00Ksp_, 1e-3, 1e-50, 1e20, 500);
  KSPSetOperators(a00Ksp_, a00.GetReference(), a00.GetReference());

  PC a00Pc;
  KSPGetPC(a00Ksp_, &a00Pc);
  PCSetType(a00Pc, PCSOR);

  KSPSetUp(a00Ksp_);

  KSPCreate(MPI_COMM_WORLD, &a11Ksp_);
  KSPSetType(a11Ksp_, KSPGMRES);
  KSPSetTolerances(a11Ksp_, 1e-3, 1e-50, 1e20, 100);
  KSPSetOperators(a11Ksp_, schurComplement_, schurComplement_);

  PC a11Pc;
  KSPGetPC(a11Ksp_, &a11Pc);
  PCSetType(a11Pc, PCSOR);

  KSPSetUp(a11Ksp_);
}