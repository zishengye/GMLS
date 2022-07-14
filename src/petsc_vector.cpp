#include "petsc_vector.hpp"

using namespace std;

void petsc_vector::create(const vector<double> &vec) {
  VecCreateMPI(MPI_COMM_WORLD, vec.size(), PETSC_DECIDE, &vec_);

  PetscScalar *a;
  VecGetArray(vec_, &a);
  for (int i = 0; i < vec.size(); i++) {
    a[i] = vec[i];
  }
  VecRestoreArray(vec_, &a);
}

void petsc_vector::create(petsc_vector &vec) {
  VecDuplicate(vec.GetReference(), &vec_);
}

void petsc_vector::copy(vector<double> &vec) {
  PetscScalar *a;
  PetscInt local_size;

  VecGetLocalSize(vec_, &local_size);
  VecGetArray(vec_, &a);
  for (PetscInt i = 0; i < local_size; i++) {
    vec[i] = a[i];
  }
  VecRestoreArray(vec_, &a);
}