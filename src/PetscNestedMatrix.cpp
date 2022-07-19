#include "PetscNestedMatrix.hpp"
#include "petscsystypes.h"

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
    nestedMat_[i] = PETSC_NULL;
  }
}

void PetscNestedMatrix::Resize(const PetscInt nestedRowBlockSize,
                               const PetscInt nestedColBlockSize) {
  nestedRowBlockSize_ = nestedRowBlockSize;
  nestedColBlockSize_ = nestedColBlockSize;

  for (unsigned int i = 0; i < nestedMat_.size(); i++) {
    if (nestedMat_[i] != PETSC_NULL)
      MatDestroy(&nestedMat_[i]);
  }

  nestedMat_.resize(nestedRowBlockSize * nestedColBlockSize);
  nestedWrappedMat_.resize(nestedRowBlockSize * nestedColBlockSize);

  for (unsigned int i = 0; i < nestedMat_.size(); i++) {
    nestedMat_[i] = PETSC_NULL;
  }

  for (unsigned int i = 0; i < nestedWrappedMat_.size(); i++) {
    nestedWrappedMat_[i] = std::make_shared<PetscMatrix>();
  }
}

std::shared_ptr<PetscMatrix> PetscNestedMatrix::GetMatrix(const PetscInt row,
                                                          const PetscInt col) {
  return nestedWrappedMat_[row * nestedColBlockSize_ + col];
}

unsigned long PetscNestedMatrix::GraphAssemble() {
  unsigned long nnz = 0;

  for (auto &it : nestedWrappedMat_)
    nnz += it->GraphAssemble();

  return nnz;
}

unsigned long PetscNestedMatrix::Assemble() {
  unsigned long nnz = 0;
  for (auto &it : nestedWrappedMat_)
    nnz += it->Assemble();

  nestedMat_[0] = nestedWrappedMat_[0]->GetReference();
  nestedMat_[1] = nestedWrappedMat_[1]->GetReference();
  nestedMat_[2] = nestedWrappedMat_[2]->GetReference();
  nestedMat_[3] = nestedWrappedMat_[3]->GetReference();

  MatCreateNest(MPI_COMM_WORLD, 2, PETSC_NULL, 2, PETSC_NULL, nestedMat_.data(),
                &mat_);

  return nnz;
}