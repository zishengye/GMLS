#ifndef _Equation_StokesEquation_Hpp_
#define _Equation_StokesEquation_Hpp_

#include "Core/Typedef.hpp"
#include "Equation/Equation.hpp"
#include "Equation/Stokes/StokesMatrix.hpp"
#include "Equation/Stokes/StokesPreconditioner.hpp"

#include <functional>

#define VELOCITY_ERROR_EST 1
#define PRESSURE_ERROR_EST 2

namespace Equation {
class StokesEquation : public Equation {
protected:
  virtual Void InitLinearSystem();
  virtual Void ConstructLinearSystem();
  virtual Void ConstructRhs();

  virtual Void SolveEquation();
  virtual Void CalculateError();

  HostRealMatrix recoveredGradientChunk_;
  HostRealMatrix gradientChunk_;
  HostRealMatrix velocity_, oldVelocity_;
  HostRealVector pressure_, oldPressure_;
  HostRealVector field_, oldField_;
  HostRealVector kappa_, bi_;

  std::function<bool(const double, const double, const double)>
      velocityBoundaryType_;
  std::function<int(const double, const double, const double)>
      pressureBoundaryType_;
  std::function<double(const double, const double, const double,
                       const unsigned int)>
      velocityInteriorRhs_;
  std::function<double(const double, const double, const double,
                       const unsigned int)>
      velocityBoundaryRhs_;
  std::function<double(const double, const double, const double)>
      pressureInteriorRhs_;
  std::function<double(const double, const double, const double)>
      pressureBoundaryRhs_;
  std::function<double(const double, const double, const double,
                       const unsigned int)>
      analyticalVelocitySolution_;
  std::function<double(const double, const double, const double)>
      analyticalPressureSolution_;
  std::function<double(const double, const double, const double,
                       const unsigned int)>
      analyticalVelocityGradientSolution_;
  std::function<double(const double, const double, const double,
                       const unsigned int)>
      analyticalPressureGradientSolution_;

  bool isVelocityAnalyticalSolutionSet_, isPressureAnalyticalSolutionSet_;
  bool isVelocityGradientAnalyticalSolutionSet_,
      isPressureGradientAnalyticalSolutionSet_;

public:
  StokesEquation();
  ~StokesEquation();

  virtual Void Output();
  virtual Void Output(String &outputFileName);

  virtual Void Init();
  virtual Void CalculateSensitivity(DefaultParticleManager &particleMgr,
                                    HostRealVector &sensitivity);

  virtual Scalar GetObjFunc();

  virtual Void SetVelocityBoundaryType(
      const std::function<bool(const double, const double, const double)>
          &func);
  virtual Void SetPressureBoundaryType(
      const std::function<int(const double, const double, const double)> &func);
  virtual Void SetVelocityInteriorRhs(
      const std::function<double(const double, const double, const double,
                                 const unsigned int)> &func);
  virtual Void SetVelocityBoundaryRhs(
      const std::function<double(const double, const double, const double,
                                 const unsigned int)> &func);
  virtual Void SetPressureInteriorRhs(
      const std::function<double(const double, const double, const double)>
          &func);
  virtual Void SetPressureBoundaryRhs(
      const std::function<double(const double, const double, const double)>
          &func);
  virtual Void SetAnalyticalVelocitySolution(
      const std::function<double(const double, const double, const double,
                                 const unsigned int)> &func);
  virtual Void SetAnalyticalPressureSolution(
      const std::function<double(const double, const double, const double)>
          &func);
  virtual Void SetAnalyticalVelocityGradientSolution(
      const std::function<double(const double, const double, const double,
                                 const unsigned int)> &func);
  virtual Void SetAnalyticalPressureGradientSolution(
      const std::function<double(const double, const double, const double,
                                 const unsigned int)> &func);
};
} // namespace Equation

#endif