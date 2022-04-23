#ifndef _PetscKsp_Hpp_
#define _PetscKsp_Hpp_

#include <petscksp.h>

#include "PetscMatrix.hpp"
#include "PetscVector.hpp"

class PetscKsp {
private:
  KSP ksp_;

public:
  PetscKsp() : ksp_(PETSC_NULL) {}

  ~PetscKsp() {
    if (ksp_ != PETSC_NULL)
      KSPDestroy(&ksp_);
  }

  void SetUp(PetscMatrix &mat, KSPType type) {
    if (ksp_ != PETSC_NULL)
      KSPDestroy(&ksp_);
    KSPCreate(MPI_COMM_WORLD, &ksp_);
    KSPSetType(ksp_, type);
    KSPSetOperators(ksp_, mat.GetReference(), mat.GetReference());
    KSPSetFromOptions(ksp_);
    KSPSetUp(ksp_);
  }

  void Solve(Vec &b, Vec &x) { KSPSolve(ksp_, b, x); }

  KSP &GetReference() { return ksp_; }

  KSP *GetPointer() { return &ksp_; }
};

#endif