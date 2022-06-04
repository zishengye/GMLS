#include "PetscMatrixBase.hpp"

PetscMatrixBase::PetscMatrixBase() : mat_(PETSC_NULL) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);
}

PetscMatrixBase::~PetscMatrixBase() {
  if (mat_ != PETSC_NULL) {
    MatDestroy(&mat_);
  }
}

void PetscMatrixBase::Clear() {
  if (mat_ != PETSC_NULL) {
    MatDestroy(&mat_);
  }
  mat_ = PETSC_NULL;
}

void PetscMatrixBase::Resize(const PetscInt m, const PetscInt n) {}

PetscInt PetscMatrixBase::GetRowSize() { return 0; }

PetscInt PetscMatrixBase::GetColSize() { return 0; }

unsigned long PetscMatrixBase::GraphAssemble() { return 0; }

unsigned long PetscMatrixBase::Assemble() { return 0; }

Mat &PetscMatrixBase::GetReference() { return mat_; }