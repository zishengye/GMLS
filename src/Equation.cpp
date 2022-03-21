#include "Equation.hpp"

void Equation::DiscretizeEquation() {
  InitLinearSystem();
  ConstructLinearSystem();
  ConstructRhs();
}

void Equation::Update() {
  refinementIteration_ = 0;
  error_ = 1e9;

  while (error > errorTolerance_ &&
         refinementIteration < maxRefinementIteration) {
    DiscretizeEquation();
    InitPreconditioner();
    SolveEquation();
    CalculateError();
    Refine();
  }
}