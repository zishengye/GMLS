#include "gmls_solver.h"

bool GMLS_Solver::NeedRefinement() {
  if (__adaptive_step > 0) {
  }

  __adaptive_step++;

  return true;
}