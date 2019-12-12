#include "gmls_solver.h"

bool GMLS_Solver::NeedRefinement() {
  if (__adaptive_step > 0) {
    return false;
  }

  __adaptive_step++;

  return true;
}