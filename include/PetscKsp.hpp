#ifndef _PetscKsp_HPP_
#define _PetscKsp_HPP_

#include <petscksp.h>

class PetscKsp {
private:
  KSP ksp;

public:
  PetscKsp() : ksp(PETSC_NULL) {}

  ~PetscKsp() {
    if (ksp != PETSC_NULL) {
      KSPDestroy(&ksp);
    }
  }

  KSP &GetReference() { return ksp; }

  KSP *GetPointer() { return &ksp; }
};

#endif