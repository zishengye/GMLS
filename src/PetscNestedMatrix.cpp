#include "PetscNestedMatrix.hpp"

PetscNestedMatrix::PetscNestedMatrix()
    : nestedRowBlockSize_(0), nestedColBlockSize_(0), PetscMatrixBase() {}

PetscNestedMatrix::PetscNestedMatrix(const PetscInt nestedRowBlockSize,
                                     const PetscInt nestedColBlockSize)
    : nestedRowBlockSize_(nestedRowBlockSize),
      nestedColBlockSize_(nestedColBlockSize), PetscMatrixBase() {
  nestedMat_.resize(nestedRowBlockSize * nestedColBlockSize);
  nestedWrappedMat_.resize(nestedRowBlockSize * nestedColBlockSize);

  for (unsigned int i = 0; i < nestedMat_.size(); i++) {
    nestedMat_[i] = PETSC_NULL;
  }

  for (unsigned int i = 0; i < nestedWrappedMat_.size(); i++) {
    nestedWrappedMat_[i] = std::make_shared<PetscMatrix>();
  }
}

PetscNestedMatrix::~PetscNestedMatrix() {
  for (unsigned int i = 0; i < nestedMat_.size(); i++) {
    if (nestedMat_[i] != PETSC_NULL)
      MatDestroy(&nestedMat_[i]);
  }
}

void PetscNestedMatrix::Resize(const PetscInt nestedRowBlockSize,
                               const PetscInt nestedColBlockSize) {
  nestedRowBlockSize_ = nestedRowBlockSize;
  nestedColBlockSize_ = nestedColBlockSize;

  for (unsigned int i = 0; i < nestedMat_.size(); i++) {
    nestedMat_[i] = PETSC_NULL;
  }

  nestedMat_.resize(nestedRowBlockSize * nestedColBlockSize);
  nestedWrappedMat_.resize(nestedRowBlockSize * nestedColBlockSize);
}

std::shared_ptr<PetscMatrix> PetscNestedMatrix::GetMatrix(const PetscInt row,
                                                          const PetscInt col) {
  return nestedWrappedMat_[row * nestedColBlockSize_ + col];
}

unsigned long PetscNestedMatrix::GraphAssemble() {
  unsigned long nnz = 0;
  for (unsigned int i = 0; i < nestedWrappedMat_.size(); i++) {
    nnz += nestedWrappedMat_[i]->GraphAssemble();
  }
}

unsigned long PetscNestedMatrix::Assemble() {
  unsigned long nnz = 0;
  for (unsigned int i = 0; i < nestedWrappedMat_.size(); i++) {
    nnz += nestedWrappedMat_[i]->Assemble();
  }
}