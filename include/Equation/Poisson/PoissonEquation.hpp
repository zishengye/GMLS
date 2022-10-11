#ifndef _Equation_PoissonEquation_Hpp_
#define _Equation_PoissonEquation_Hpp_

#include "Equation/Equation.hpp"
#include "Equation/Poisson/PoissonPreconditioner.hpp"

#include <functional>

namespace Equation {
class PoissonEquation : public Equation {
protected:
  virtual Void InitLinearSystem();
  virtual Void ConstructLinearSystem();
  virtual Void ConstructRhs();

  virtual Void SolveEquation();
  virtual Void CalculateError();

  virtual Void Output();

  HostRealMatrix recoveredGradientChunk_;
  HostRealVector field_, oldField_, kappa_;

  std::function<double(const double, const double, const double)> interiorRhs_;
  std::function<double(const double, const double, const double)> boundaryRhs_;
  std::function<double(const double, const double, const double)>
      analyticalFieldSolution_;
  std::function<double(const double, const double, const double,
                       const unsigned int)>
      analyticalFieldGradientSolution_;
  std::function<double(const double, const double, const double)> kappaFunc_;

  bool isFieldAnalyticalSolutionSet_;
  bool isFieldGradientAnalyticalSolutionSet_;

public:
  PoissonEquation();
  ~PoissonEquation();

  virtual Void Init();

  HostRealVector &GetField();

  virtual Void SetInteriorRhs(
      const std::function<double(const double, const double, const double)>
          &func);
  virtual Void SetBoundaryRhs(
      const std::function<double(const double, const double, const double)>
          &func);
  virtual Void SetAnalyticalFieldSolution(
      const std::function<double(const double, const double, const double)>
          &func);
  virtual Void SetAnalyticalFieldGradientSolution(
      const std::function<double(const double, const double, const double,
                                 const unsigned int)> &func);
  virtual Void SetKappa(const std::function<double(const double, const double,
                                                   const double)> &func);
};
} // namespace Equation

#endif