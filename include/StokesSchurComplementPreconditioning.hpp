#ifndef _StokesSchurComplementPreconditioning_Hpp_
#define _StokesSchurComplementPreconditioning_Hpp_

#include "StokesMatrix.hpp"

PetscErrorCode StokesSchurComplementIterationWrapper(PC pc, Vec x, Vec y);

class StokesSchurComplementPreconditioning {
protected:
  std::shared_ptr<StokesMatrix> stokesLinearSystemsPtr_;

  PetscInt blockM_, blockN_;

  Mat schurComplement_;
  std::vector<Vec> lhsVector_, rhsVector_;
  std::vector<unsigned long> localLhsVectorOffset_;
  std::vector<unsigned long> localRhsVectorOffset_;

  KSP a00Ksp_, a11Ksp_;

public:
  StokesSchurComplementPreconditioning();
  ~StokesSchurComplementPreconditioning();

  virtual PetscErrorCode ApplyPreconditioningIteration(Vec x, Vec y);

  void AddLinearSystem(std::shared_ptr<StokesMatrix> mat);
};

#endif