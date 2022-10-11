#include "LinearAlgebra/Impl/Petsc/PetscBackend.hpp"
#include "petscsys.h"

LinearAlgebra::Impl::PetscBackend::PetscBackend() {}

LinearAlgebra::Impl::PetscBackend::~PetscBackend() {}

Void LinearAlgebra::Impl::PetscBackend::LinearAlgebraInitialize(
    int *argc, char ***args, const char file[], const char help[]) {
  PetscInitialize(argc, args, file, help);
}

Void LinearAlgebra::Impl::PetscBackend::LinearAlgebraFinalize() {
  PetscFinalize();
}