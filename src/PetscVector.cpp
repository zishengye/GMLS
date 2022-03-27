#include "PetscVector.hpp"

PetscVector::PetscVector() : vec_(PETSC_NULL) {}

PetscVector::~PetscVector() {
  if (vec_ != PETSC_NULL)
    VecDestroy(&vec_);
}

void PetscVector::Create(std::vector<double> &vec) {
  VecCreateMPI(MPI_COMM_WORLD, vec.size(), PETSC_DECIDE, &vec_);

  PetscScalar *a;
  VecGetArray(vec_, &a);
  for (int i = 0; i < vec.size(); i++) {
    a[i] = vec[i];
  }
  VecRestoreArray(vec_, &a);
}

void PetscVector::Create(PetscVector &vec) {
  VecDuplicate(vec.GetReference(), &vec_);
}

void PetscVector::Create(HostRealVector &vec) {
  VecCreateMPI(MPI_COMM_WORLD, vec.extent(0), PETSC_DECIDE, &vec_);

  PetscScalar *a;
  VecGetArray(vec_, &a);
  for (int i = 0; i < vec.extent(0); i++) {
    a[i] = vec(i);
  }
  VecRestoreArray(vec_, &a);
}

void PetscVector::Clear() {
  if (vec_ != PETSC_NULL)
    VecDestroy(&vec_);

  vec_ = PETSC_NULL;
}

Vec &PetscVector::GetReference() { return vec_; }

Vec *PetscVector::GetPointer() { return &vec_; }