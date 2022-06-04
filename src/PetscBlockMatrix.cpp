#include "PetscBlockMatrix.hpp"

PetscBlockMatrix::PetscBlockMatrix() : PetscMatrixBase() {}

PetscBlockMatrix::PetscBlockMatrix(const PetscInt blockM, const PetscInt blockN)
    : PetscMatrixBase(), blockM_(blockM), blockN_(blockN) {
  subMat_.resize(blockM * blockN);
}

PetscBlockMatrix::~PetscBlockMatrix() {}

void PetscBlockMatrix::Resize(const PetscInt blockM, const PetscInt blockN) {
  blockM_ = blockM;
  blockN_ = blockN;

  subMat_.resize(blockM * blockN);
}

PetscInt PetscBlockMatrix::GetRowSize() { return blockM_; }

PetscInt PetscBlockMatrix::GetColSize() { return blockN_; }

std::shared_ptr<PetscMatrixBase> &
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
  for (PetscInt i = 0; i < blockM_; i++) {
    localRow += subMat_[i * blockN_ + i]->GetRowSize();
    localCol += subMat_[i * blockN_ + i]->GetColSize();
  }

  MatCreateShell(PETSC_COMM_WORLD, localRow, localCol, PETSC_DECIDE,
                 PETSC_DECIDE, this, &mat_);

  return nnz;
}