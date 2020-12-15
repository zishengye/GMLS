#ifndef _PETSC_KSP_HPP_
#define _PETSC_KSP_HPP_

#include <petscksp.h>

#include "petsc_sparse_matrix.hpp"
#include "petsc_vector.hpp"

class petsc_ksp {
private:
  KSP __ksp;

  bool __is_setup;

public:
  petsc_ksp() : __is_setup(false) {}

  ~petsc_ksp() {
    if (__is_setup) {
      KSPDestroy(&__ksp);
    }
  }

  void setup(petsc_sparse_matrix &mat, KSPType type);
  void setup(petsc_sparse_matrix &mat);
  void setup(Mat &mat);

  void solve(petsc_vector &rhs, petsc_vector &p);

  KSP &get_reference() { return __ksp; }
};

#endif