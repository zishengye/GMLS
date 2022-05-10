#ifndef _StokesEquation_Hpp_
#define _StokesEquation_Hpp_

#include "Equation.hpp"

class StokesEquation : public Equation {
protected:
  virtual void InitLinearSystem();
  virtual void ConstructLinearSystem();
  virtual void ConstructRhs();

  virtual void SolveEquation();
  virtual void CalculateError();

  virtual void Output();

  HostRealMatrix recoveredGradientChunk_;
  HostRealMatrix velocity_;
  HostRealVector pressure_;

public:
  StokesEquation();
  ~StokesEquation();

  virtual void Init();

  HostRealMatrix &GetVelocity();
  HostRealVector &GetPressure();
};

#endif