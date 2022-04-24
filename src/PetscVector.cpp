#include "PetscVector.hpp"

PetscVector::PetscVector() : vec_(PETSC_NULL) {}

PetscVector::PetscVector(const int localSize) {
  VecCreateMPI(MPI_COMM_WORLD, localSize, PETSC_DECIDE, &vec_);
}

PetscVector::~PetscVector() {
  if (vec_ != PETSC_NULL)
    VecDestroy(&vec_);
}

void PetscVector::Create(std::vector<double> &vec) {
  if (vec_ != PETSC_NULL)
    VecDestroy(&vec_);
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
  if (vec_ != PETSC_NULL)
    VecDestroy(&vec_);
  VecCreateMPI(MPI_COMM_WORLD, vec.extent(0), PETSC_DECIDE, &vec_);

  PetscScalar *a;
  VecGetArray(vec_, &a);
  for (int i = 0; i < vec.extent(0); i++) {
    a[i] = vec(i);
  }
  VecRestoreArray(vec_, &a);
}

void PetscVector::Copy(HostRealVector &vec) {
  PetscInt size;
  VecGetLocalSize(vec_, &size);
  Kokkos::resize(vec, size);

  PetscScalar *a;
  VecGetArray(vec_, &a);
  for (int i = 0; i < size; i++) {
    vec(i) = a[i];
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