#include "petsc_nest_vector.hpp"

using namespace std;

petsc_nest_vector::~petsc_nest_vector() {
  if (vec != PETSC_NULL) {
    VecDestroy(&vec);
  }
  if (vec_list[0] != PETSC_NULL) {
    VecDestroy(&vec_list[0]);
  }
  if (vec_list[1] != PETSC_NULL) {
    VecDestroy(&vec_list[1]);
  }
}

void petsc_nest_vector::create(vector<double> &_vec1, vector<double> &_vec2) {
  VecCreateMPIWithArray(MPI_COMM_WORLD, 1, _vec1.size(), PETSC_DECIDE,
                        _vec1.data(), &vec_list[0]);
  VecCreateMPIWithArray(MPI_COMM_WORLD, 1, _vec2.size(), PETSC_DECIDE,
                        _vec2.data(), &vec_list[1]);

  VecCreateNest(PETSC_COMM_WORLD, 2, NULL, vec_list, &vec);
}

void petsc_nest_vector::copy(vector<double> &_vec1, vector<double> &_vec2) {
  Vec sub_vec;
  PetscInt size;
  PetscReal *a;

  VecNestGetSubVec(vec, 0, &sub_vec);
  VecGetArray(sub_vec, &a);
  VecGetLocalSize(sub_vec, &size);
  _vec1.resize(size);
  for (int i = 0; i < size; i++) {
    _vec1[i] = a[i];
  }
  VecRestoreArray(sub_vec, &a);

  VecNestGetSubVec(vec, 1, &sub_vec);
  VecGetArray(sub_vec, &a);
  VecGetLocalSize(sub_vec, &size);
  _vec2.resize(size);
  for (int i = 0; i < size; i++) {
    _vec2[i] = a[i];
  }
  VecRestoreArray(sub_vec, &a);
}