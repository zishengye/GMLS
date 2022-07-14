#ifndef _PETSC_KSP_HPP_
#define _PETSC_KSP_HPP_

#include <petscksp.h>

#include "petsc_sparse_matrix.hpp"
#include "petsc_vector.hpp"

class petsc_ksp {
private:
  KSP ksp;

public:
  petsc_ksp() : ksp(PETSC_NULL) {}

  ~petsc_ksp() {
    if (ksp != PETSC_NULL) {
      KSPDestroy(&ksp);
    }
  }

  void setup(petsc_sparse_matrix &mat, KSPType type);
  void setup(petsc_sparse_matrix &mat);
  void setup(Mat &mat);

  void solve(petsc_vector &rhs, petsc_vector &p);

  KSP &GetReference() { return ksp; }

  KSP *GetPointer() { return &ksp; }
};

#endif