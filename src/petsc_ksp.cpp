#include "petsc_ksp.hpp"

void petsc_ksp::setup(petsc_sparse_matrix &mat, KSPType type) {
  KSPCreate(MPI_COMM_WORLD, &ksp);
  KSPSetType(ksp, type);
  KSPSetOperators(ksp, mat.get_operator(), mat.get_operator());
  KSPSetFromOptions(ksp);
  KSPSetUp(ksp);
}

void petsc_ksp::setup(petsc_sparse_matrix &mat) {
  KSPCreate(MPI_COMM_WORLD, &ksp);
  KSPSetOperators(ksp, mat.get_operator(), mat.get_operator());
  KSPSetFromOptions(ksp);
  KSPSetUp(ksp);
}

void petsc_ksp::setup(Mat &mat) {
  KSPCreate(MPI_COMM_WORLD, &ksp);
  KSPSetOperators(ksp, mat, mat);
  KSPSetFromOptions(ksp);
  KSPSetUp(ksp);
}

void petsc_ksp::solve(petsc_vector &rhs, petsc_vector &x) {
  if (ksp != PETSC_NULL) {
    KSPSolve(ksp, rhs.get_reference(), x.get_reference());
  }
}