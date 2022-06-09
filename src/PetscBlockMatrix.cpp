#include "PetscBlockMatrix.hpp"

PetscBlockMatrix::PetscBlockMatrix() : PetscMatrixBase() {}

PetscBlockMatrix::PetscBlockMatrix(const PetscInt blockM, const PetscInt blockN)
    : PetscMatrixBase(), blockM_(blockM), blockN_(blockN) {
  subMat_.resize(blockM * blockN);
}

PetscBlockMatrix::~PetscBlockMatrix() {
  for (PetscInt i = 0; i < blockM_; i++) {
    VecDestroy(&lhsVector_[i]);
  }
  for (PetscInt i = 0; i < blockN_; i++) {
    VecDestroy(&rhsVector_[i]);
  }
}

void PetscBlockMatrix::Resize(const PetscInt blockM, const PetscInt blockN) {
  blockM_ = blockM;
  blockN_ = blockN;

  subMat_.resize(blockM * blockN);
}

PetscInt PetscBlockMatrix::GetRowSize() { return blockM_; }

PetscInt PetscBlockMatrix::GetColSize() { return blockN_; }

std::shared_ptr<PetscMatrixBase>
PetscBlockMatrix::GetSubMat(const PetscInt blockI, const PetscInt blockJ) {
  return subMat_[blockI * blockN_ + blockJ];
}

void PetscBlockMatrix::SetSubMat(const PetscInt blockI, const PetscInt blockJ,
                                 std::shared_ptr<PetscMatrixBase> mat) {
  subMat_[blockI * blockN_ + blockJ] = mat;
}

unsigned long PetscBlockMatrix::Assemble() {
  unsigned long nnz = 0;
  for (unsigned int i = 0; i < subMat_.size(); i++) {
    nnz += subMat_[i]->Assemble();
  }

  PetscInt localRow = 0, localCol = 0;
  lhsVector_.resize(blockM_);
  rhsVector_.resize(blockN_);
  localLhsVectorOffset_.resize(blockM_ + 1);
  localRhsVectorOffset_.resize(blockN_ + 1);
  localLhsVectorOffset_[0] = 0;
  localRhsVectorOffset_[0] = 0;
  for (PetscInt i = 0; i < blockM_; i++) {
    PetscInt tLocalRow = subMat_[i * blockN_]->GetRowSize();
    localRow += tLocalRow;
    localLhsVectorOffset_[i + 1] = localRow;
    VecCreateMPI(MPI_COMM_WORLD, tLocalRow, PETSC_DECIDE, &lhsVector_[i]);
  }

  for (PetscInt i = 0; i < blockN_; i++) {
    PetscInt tLocalCol = subMat_[i]->GetColSize();
    localCol += tLocalCol;
    localRhsVectorOffset_[i + 1] = localCol;
    VecCreateMPI(MPI_COMM_WORLD, tLocalCol, PETSC_DECIDE, &rhsVector_[i]);
  }

  MatCreateShell(PETSC_COMM_WORLD, localRow, localCol, PETSC_DECIDE,
                 PETSC_DECIDE, this, &mat_);
  MatShellSetOperation(mat_, MATOP_MULT,
                       (void (*)(void))PetscBlockMatrixMatMultWrapper);

  return nnz;
}

PetscErrorCode PetscBlockMatrix::MatMult(Vec x, Vec y) {
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
  average = sum / (double)length;
  for (unsigned int i = 0;
       i < localRhsVectorOffset_[2] - localRhsVectorOffset_[1]; i++) {
    a[i] -= average;
  }
  VecRestoreArray(rhsVector_[1], &a);

  for (PetscInt i = 0; i < blockM_; i++) {
    for (PetscInt j = 0; j < blockN_; j++) {
      MatMultAdd(subMat_[i * blockN_ + j]->GetReference(), rhsVector_[j],
                 lhsVector_[i], lhsVector_[i]);
    }
  }

  VecSum(lhsVector_[1], &sum);
  VecGetArray(lhsVector_[1], &a);
  VecGetSize(lhsVector_[1], &length);
  average = sum / (double)length;
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

PetscErrorCode PetscBlockMatrixMatMultWrapper(Mat mat, Vec x, Vec y) {
  PetscBlockMatrix *ctx;
  MatShellGetContext(mat, &ctx);

  return ctx->MatMult(x, y);
}