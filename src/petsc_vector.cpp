#include "petsc_vector.hpp"

using namespace std;

void petsc_vector::create(vector<double> &_vec) {
  VecCreateMPI(MPI_COMM_WORLD, _vec.size(), PETSC_DECIDE, &vec);

  PetscScalar *a;
  VecGetArray(vec, &a);
  for (int i = 0; i < _vec.size(); i++) {
    a[i] = _vec[i];
  }
  VecRestoreArray(vec, &a);
}

void petsc_vector::create(petsc_vector &_vec) {
  VecDuplicate(_vec.get_reference(), &vec);
}

void petsc_vector::copy(vector<double> &_vec) {
  PetscScalar *a;
  PetscInt local_size;

  VecGetLocalSize(vec, &local_size);
  VecGetArray(vec, &a);
  for (PetscInt i = 0; i < local_size; i++) {
    _vec[i] = a[i];
  }
  VecRestoreArray(vec, &a);
}