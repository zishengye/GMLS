#include "petsc_ksp.hpp"

void petsc_ksp::setup(petsc_sparse_matrix &mat, KSPType type) {
  KSPCreate(MPI_COMM_WORLD, &__ksp);
  KSPSetType(__ksp, type);
  KSPSetOperators(__ksp, mat.get_operator_reference(),
                  mat.get_operator_reference());
  KSPSetFromOptions(__ksp);
  KSPSetUp(__ksp);

  __is_setup = true;
}

void petsc_ksp::setup(petsc_sparse_matrix &mat) {
  KSPCreate(MPI_COMM_WORLD, &__ksp);
  KSPSetOperators(__ksp, mat.get_operator_reference(),
                  mat.get_operator_reference());
  KSPSetFromOptions(__ksp);
  KSPSetUp(__ksp);

  __is_setup = true;
}

void petsc_ksp::setup(Mat &mat) {
  KSPCreate(MPI_COMM_WORLD, &__ksp);
  KSPSetOperators(__ksp, mat, mat);
  KSPSetFromOptions(__ksp);
  KSPSetUp(__ksp);

  __is_setup = true;
}

void petsc_ksp::solve(petsc_vector &rhs, petsc_vector &x) {
  if (__is_setup) {
    KSPSolve(__ksp, rhs.get_reference(), x.get_reference());
  }
}