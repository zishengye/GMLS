#ifndef _SchurComplementPreconditioning_Hpp_
#define _SchurComplementPreconditioning_Hpp_

#include <vector>

#include "PetscBlockMatrix.hpp"

PetscErrorCode SchurComplementIterationWrapper(PC pc, Vec x, Vec y);

class SchurComplementPreconditioning {
protected:
  std::shared_ptr<PetscBlockMatrix> linearSystemsPtr_;

  PetscInt blockM_, blockN_;

  Mat schurComplement_;
  std::vector<Vec> lhsVector_, rhsVector_;
  std::vector<unsigned long> localLhsVectorOffset_;
  std::vector<unsigned long> localRhsVectorOffset_;

  KSP a00Ksp_, a11Ksp_;

public:
  SchurComplementPreconditioning();
  ~SchurComplementPreconditioning();

  PetscErrorCode ApplyPreconditioningIteration(Vec x, Vec y);

  void AddLinearSystem(std::shared_ptr<PetscBlockMatrix> mat);
};

#endif