#include "PetscNestedMatrix.hpp"
#include "petscsystypes.h"
#include "petscvec.h"

PetscErrorCode NestedMatMultWrapper(Mat mat, Vec x, Vec y) {
  PetscNestedMatrix *ctx;
  MatShellGetContext(mat, &ctx);

  return ctx->NestedMatrixMult(x, y);
}

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
  for (auto &it : nestedWrappedMat_) {
    nnz += it->Assemble();
  }

  for (unsigned int i = 0; i < nestedRowBlockSize_ * nestedColBlockSize_; i++) {
    nestedMat_[i] = nestedWrappedMat_[i]->GetReference();
  }

  PetscInt rowSize, colSize;
  rowSize = 0;
  colSize = 0;
  for (unsigned int i = 0; i < nestedRowBlockSize_; i++) {
    rowSize += nestedWrappedMat_[i * nestedColBlockSize_ + i]->GetRowSize();
    colSize += nestedWrappedMat_[i * nestedColBlockSize_ + i]->GetColSize();
  }

  MatCreateShell(MPI_COMM_WORLD, rowSize, colSize, PETSC_DECIDE, PETSC_DECIDE,
                 this, &mat_);
  MatShellSetOperation(mat_, MATOP_MULT, (void (*)(void))NestedMatMultWrapper);

  return nnz;
}

PetscErrorCode PetscNestedMatrix::NestedMatrixMult(Vec x, Vec y) {
  std::vector<PetscInt> nestedRowSize(nestedRowBlockSize_);
  std::vector<PetscInt> nestedRowOffset(nestedRowBlockSize_ + 1);
  std::vector<PetscInt> nestedColSize(nestedRowBlockSize_);
  std::vector<PetscInt> nestedColOffset(nestedRowBlockSize_ + 1);

  std::vector<Vec> xList(nestedRowBlockSize_);
  std::vector<Vec> yList(nestedRowBlockSize_);

  PetscReal *b;

  VecGetArray(x, &b);
  nestedRowOffset[0] = 0;
  nestedColOffset[0] = 0;
  for (int i = 0; i < nestedRowBlockSize_; i++) {
    MatGetLocalSize(nestedMat_[i * nestedColBlockSize_ + i], &nestedRowSize[i],
                    &nestedColSize[i]);
    VecCreateMPI(MPI_COMM_WORLD, nestedColSize[i], PETSC_DECIDE, &xList[i]);
    VecCreateMPI(MPI_COMM_WORLD, nestedRowSize[i], PETSC_DECIDE, &yList[i]);

    PetscReal *a;
    VecGetArray(xList[i], &a);
    for (int j = 0; j < nestedColSize[i]; j++) {
      a[j] = b[nestedColOffset[i] + j];
    }
    VecRestoreArray(xList[i], &a);

    nestedRowOffset[i + 1] = nestedRowOffset[i] + nestedRowSize[i];
    nestedColOffset[i + 1] = nestedColOffset[i] + nestedColSize[i];
  }
  VecRestoreArray(x, &b);

  for (int i = 0; i < nestedRowBlockSize_; i++) {
    MatMult(nestedMat_[i * nestedColBlockSize_], xList[0], yList[i]);
    for (int j = 1; j < nestedColBlockSize_; j++) {
      MatMultAdd(nestedMat_[i * nestedColBlockSize_ + j], xList[j], yList[i],
                 yList[i]);
    }
  }

  VecGetArray(y, &b);
  for (int i = 0; i < nestedRowBlockSize_; i++) {
    PetscReal *a;
    VecGetArray(yList[i], &a);
    for (int j = 0; j < nestedRowSize[i]; j++) {
      b[nestedRowOffset[i] + j] = a[j];
    }
    VecRestoreArray(yList[i], &a);

    VecDestroy(&xList[i]);
    VecDestroy(&yList[i]);
  }
  VecRestoreArray(y, &b);

  return 0;
}