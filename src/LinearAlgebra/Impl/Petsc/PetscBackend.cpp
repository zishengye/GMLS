#include "LinearAlgebra/Impl/Petsc/PetscBackend.hpp"
#include "petscsys.h"

LinearAlgebra::Impl::PetscBackend::PetscBackend() {}

LinearAlgebra::Impl::PetscBackend::~PetscBackend() {}

Void LinearAlgebra::Impl::PetscBackend::LinearAlgebraInitialize(
    int *argc, char ***args, std::string &fileName) {
  PetscInitialize(argc, args, fileName.c_str(), PETSC_NULL);
}

Void LinearAlgebra::Impl::PetscBackend::LinearAlgebraFinalize() {
  PetscFinalize();
}