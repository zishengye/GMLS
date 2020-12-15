#include "petsc_vector.hpp"

using namespace std;

void petsc_vector::create(vector<double> &_vec) {
  VecCreateMPIWithArray(MPI_COMM_WORLD, 1, _vec.size(), PETSC_DECIDE,
                        _vec.data(), &vec);
  is_created = true;
}

void petsc_vector::create(petsc_vector &_vec) {
  VecDuplicate(_vec.get_reference(), &vec);
  is_created = true;
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