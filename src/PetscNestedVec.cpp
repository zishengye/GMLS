#include "PetscNestedVec.hpp"

PetscNestedVec::PetscNestedVec(const PetscInt size) {
  nestedVec_.resize(size);
  for (int i = 0; i < size; i++) {
    nestedVec_[i] = PETSC_NULL;
  }
}

PetscNestedVec::~PetscNestedVec() {
  for (unsigned int i = 0; i < nestedVec_.size(); i++) {
    VecDestroy(&nestedVec_[i]);
  }
}

void PetscNestedVec::Create(const PetscInt index,
                            const std::vector<double> &vec) {
  VecCreateMPI(MPI_COMM_WORLD, vec.size(), PETSC_DECIDE, &nestedVec_[index]);

  PetscScalar *a;
  VecGetArray(nestedVec_[index], &a);
  for (int i = 0; i < vec.size(); i++) {
    a[i] = vec[i];
  }
  VecRestoreArray(nestedVec_[index], &a);
}

void PetscNestedVec::Create(const PetscInt index, const PetscInt size) {
  VecCreateMPI(MPI_COMM_WORLD, size, PETSC_DECIDE, &nestedVec_[index]);
}

void PetscNestedVec::Create() {
  VecCreateNest(MPI_COMM_WORLD, 2, PETSC_NULL, nestedVec_.data(), &vec_);
}

void PetscNestedVec::Copy(const PetscInt index, std::vector<double> &vec) {
  PetscScalar *a;
  VecGetArray(nestedVec_[index], &a);
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = a[i];
  }
  VecRestoreArray(nestedVec_[index], &a);
}

void PetscNestedVec::Duplicate(PetscNestedVec &vec) {
  for (unsigned int i = 0; i < nestedVec_.size(); i++) {
    VecDuplicate(vec.GetSubVector(i), &(nestedVec_[i]));
  }
  Create();
}

Vec &PetscNestedVec::GetSubVector(const PetscInt index) {
  return nestedVec_[index];
}