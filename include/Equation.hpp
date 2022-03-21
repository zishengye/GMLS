#ifndef _EQUATION_HPP_
#define _EQUATION_HPP_

#include "ParticleManager.hpp"
#include "Typedef.hpp"

class BoundaryCondition {};

class Field {};

enum RefinementMethod { UniformRefinement, AdaptiveRefinement };

class Equation {
protected:
  double errorTolerance_, error_;
  int maxRefinementIteration_, refinementIteration_;
  RefinementMethod refinementMethod_;

  void InitLinearSystem();
  void ConstructLinearSystem();
  void ConstructRhs();

  void DiscretizeEquation();
  void InitPreconditioner();
  void SolveEquation();
  void CalculateError();
  void Refine();

  HierarchicalParticleManager particleMgr_;

public:
  Equation();

  void SetErrorTolerance(const double errorTolerance);
  void SetRefinementMethod(const RefinementMethod refinementMethod);
  void SetMaxRefinementIteration(const int maxRefinementIteration);

  void Update();
  const double GetError();
  const int GetRefinementIteration();
};

#endif