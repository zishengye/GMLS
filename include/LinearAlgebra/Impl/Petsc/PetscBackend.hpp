#ifndef _LinearAlgebra_Impl_Petsc_PetscBackend_Hpp_
#define _LinearAlgebra_Impl_Petsc_PetscBackend_Hpp_

#include <petscksp.h>

#include "LinearAlgebra/Impl/Petsc/PetscMatrix.hpp"
#include "LinearAlgebra/Impl/Petsc/PetscVector.hpp"

class PetscBackend {
public:
  typedef LinearAlgebra::Impl::PetscMatrix Matrix;
  typedef LinearAlgebra::Impl::PetscVector Vector;

  typedef PetscInt DefaultInteger;
  typedef PetscReal DefaultReal;
};

#endif