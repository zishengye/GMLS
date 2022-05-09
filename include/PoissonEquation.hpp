#ifndef _PoissonEquation_Hpp_
#define _PoissonEquation_Hpp_

#include "Equation.hpp"
#include "PoissonEquationPreconditioning.hpp"

#include <functional>

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

  std::function<double(double, double, double)> interiorRhs_;
  std::function<double(double, double, double)> boundaryRhs_;

public:
  PoissonEquation();
  ~PoissonEquation();

  virtual void Init();

  HostRealVector &GetField();

  virtual void
  SetInteriorRhs(const std::function<double(double, double, double)> &func);
  virtual void
  SetBoundaryRhs(const std::function<double(double, double, double)> &func);
};

#endif