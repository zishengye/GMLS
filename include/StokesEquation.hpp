#ifndef _StokesEquation_Hpp_
#define _StokesEquation_Hpp_

#include "Equation.hpp"

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
  HostRealMatrix velocity_;
  HostRealVector pressure_;

  std::function<double(const double, const double, const double,
                       const unsigned int)>
      interiorVelocityRhs_;
  std::function<double(const double, const double, const double,
                       const unsigned int)>
      boundaryVelocityRhs_;
  std::function<double(const double, const double, const double)>
      interiorPressureRhs_;
  std::function<double(const double, const double, const double)>
      boundaryPressureRhs_;

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
  virtual void SetPressureBoundaryRhs(
      const std::function<double(const double, const double, const double)>
          &func);
};

#endif