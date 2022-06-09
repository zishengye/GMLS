#include "SchurComplementPreconditioning.hpp"

PetscErrorCode SchurComplementIterationWrapper(PC pc, Vec x, Vec y) {
  SchurComplementPreconditioning *shellPtr;
  PCShellGetContext(pc, (void **)&shellPtr);

  return shellPtr->ApplyPreconditioningIteration(x, y);
}

SchurComplementPreconditioning::SchurComplementPreconditioning()
    : blockM_(2), blockN_(2), schurComplement_(PETSC_NULL) {}

SchurComplementPreconditioning::~SchurComplementPreconditioning() {
  if (schurComplement_ != PETSC_NULL)
    MatDestroy(&schurComplement_);
  for (PetscInt i = 0; i < blockM_; i++) {
    VecDestroy(&lhsVector_[i]);
  }
  for (PetscInt i = 0; i < blockN_; i++) {
    VecDestroy(&rhsVector_[i]);
  }
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

  PetscReal sum, average;
  PetscInt length;
  VecSum(rhsVector_[1], &sum);
  VecGetArray(rhsVector_[1], &a);
  VecGetSize(rhsVector_[1], &length);
  average = sum / length;
  for (unsigned int i = 0;
       i < localRhsVectorOffset_[2] - localRhsVectorOffset_[1]; i++) {
    a[i] -= average;
  }
  VecRestoreArray(rhsVector_[1], &a);

  KSPSolve(a00Ksp_, rhsVector_[0], lhsVector_[0]);
  KSPSolve(a11Ksp_, rhsVector_[1], lhsVector_[1]);

  VecSum(lhsVector_[1], &sum);
  VecGetArray(lhsVector_[1], &a);
  VecGetSize(lhsVector_[1], &length);
  average = sum / length;
  for (unsigned int i = 0;
       i < localLhsVectorOffset_[2] - localLhsVectorOffset_[1]; i++) {
    a[i] -= average;
  }
  VecRestoreArray(lhsVector_[1], &a);

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
  KSPSetType(a00Ksp_, KSPPREONLY);
  KSPSetOperators(a00Ksp_, a00.GetReference(), a00.GetReference());

  PC a00Pc;
  KSPGetPC(a00Ksp_, &a00Pc);
  PCSetType(a00Pc, PCSOR);

  KSPSetUp(a00Ksp_);

  KSPCreate(MPI_COMM_WORLD, &a11Ksp_);
  KSPSetType(a11Ksp_, KSPPREONLY);
  KSPSetOperators(a11Ksp_, schurComplement_, schurComplement_);

  PC a11Pc;
  KSPGetPC(a11Ksp_, &a11Pc);
  PCSetType(a11Pc, PCSOR);

  KSPSetUp(a11Ksp_);
}