#ifndef _StokesEquation_Hpp_
#define _StokesEquation_Hpp_

#include "Equation.hpp"
#include "StokesEquationPreconditioning.hpp"

#include <functional>

class StokesEquation : public Equation {
protected:
  virtual void InitLinearSystem();
  virtual void ConstructLinearSystem();
  virtual void ConstructRhs();

  virtual void SolveEquation();
  virtual void CalculateError();

  virtual void Output();

  HostRealMatrix recoveredGradientChunk_;
  HostRealMatrix velocity_, oldVelocity_;
  HostRealVector pressure_, oldPressure_;
  HostRealVector bi_;

  std::function<double(const double, const double, const double,
                       const unsigned int)>
      interiorVelocityRhs_;
  std::function<double(const double, const double, const double,
                       const unsigned int)>
      boundaryVelocityRhs_;
  std::function<double(const double, const double, const double)>
      interiorPressureRhs_;

public:
  StokesEquation();
  ~StokesEquation();

  virtual void Init();

  HostRealMatrix &GetVelocity();
  HostRealVector &GetPressure();

  virtual void SetVelocityInteriorRhs(
      const std::function<double(const double, const double, const double,
                                 const unsigned int)> &func);
  virtual void SetVelocityBoundaryRhs(
      const std::function<double(const double, const double, const double,
                                 const unsigned int)> &func);

  virtual void SetPressureInteriorRhs(
      const std::function<double(const double, const double, const double)>
          &func);
};

#endif