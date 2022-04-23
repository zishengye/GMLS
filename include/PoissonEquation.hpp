#ifndef _PoissonEquation_Hpp_
#define _PoissonEquation_Hpp_

#include "Equation.hpp"
#include "PoissonEquationPreconditioning.hpp"

class PoissonEquation : public Equation {
protected:
  virtual void InitLinearSystem();
  virtual void ConstructLinearSystem();
  virtual void ConstructRhs();

  virtual void SolveEquation();
  virtual void CalculateError();

  virtual void Output();

  HostRealMatrix recoveredGradientChunk_;
  HostRealVector field_;

public:
  PoissonEquation();
  ~PoissonEquation();

  virtual void Init();

  HostRealVector &GetField();
};

#endif