#ifndef _Equation_PoissonEquation_Hpp_
#define _Equation_PoissonEquation_Hpp_

#include "Core/Typedef.hpp"
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
  virtual Void SolveAdjointEquation();
  virtual Void CalculateError();

  virtual Void Output();

  HostRealMatrix recoveredGradientChunk_;
  HostRealMatrix gradientChunk_;
  HostRealVector field_, oldField_, kappa_, bi_, adjoint_, oldAdjoint_;

  std::function<bool(const double, const double, const double)> boundaryType_;
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

  virtual Void Init();
  virtual Void CalculateSensitivity(DefaultParticleManager &particleMgr,
                                    HostRealVector &sensitivity);

  HostRealVector &GetField();
  HostRealVector &GetRhs();

  virtual Void SetBoundaryType(
      const std::function<bool(const double, const double, const double)>
          &func);
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
  Void SetAdjointEquation();
};
} // namespace Equation

#endif