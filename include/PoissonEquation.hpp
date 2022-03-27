#ifndef _POISSON_EQUATION_HPP_
#define _POISSON_EQUATION_HPP_

#include "Equation.hpp"

class PoissonEquation : public Equation {
protected:
  virtual void InitLinearSystem();
  virtual void ConstructLinearSystem();
  virtual void ConstructRhs();

  HostRealVector field_;

public:
  PoissonEquation();
  ~PoissonEquation();

  virtual void Init();

  HostRealVector &GetField();
};

#endif