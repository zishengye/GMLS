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
  HostRealVector field_, oldField_;

  std::function<double(const double, const double, const double)> interiorRhs_;
  std::function<double(const double, const double, const double)> boundaryRhs_;
  std::function<double(const double, const double, const double)>
      analyticalFieldSolution_;
  std::function<double(const double, const double, const double,
                       const unsigned int)>
      analyticalFieldGradientSolution_;

  bool isFieldAnalyticalSolutionSet_;
  bool isFieldGradientAnalyticalSolutionSet_;

public:
  PoissonEquation();
  ~PoissonEquation();

  virtual void Init();

  HostRealVector &GetField();

  virtual void SetInteriorRhs(
      const std::function<double(const double, const double, const double)>
          &func);
  virtual void SetBoundaryRhs(
      const std::function<double(const double, const double, const double)>
          &func);
  virtual void SetAnalyticalFieldSolution(
      const std::function<double(const double, const double, const double)>
          &func);
  virtual void SetAnalyticalFieldGradientSolution(
      const std::function<double(const double, const double, const double,
                                 const unsigned int)> &func);
};

#endif