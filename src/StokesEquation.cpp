#include "StokesEquation.hpp"
#include "PolyBasis.hpp"

#include <Compadre_Evaluator.hpp>
#include <Compadre_GMLS.hpp>
#include <Compadre_PointCloudSearch.hpp>

void StokesEquation::InitLinearSystem() {}

void StokesEquation::ConstructLinearSystem() {}

void StokesEquation::ConstructRhs() {}

void StokesEquation::SolveEquation() { Equation::SolveEquation(); }

void StokesEquation::CalculateError() {}

void StokesEquation::Output() {}

StokesEquation::StokesEquation() : Equation() {}

StokesEquation::~StokesEquation() {}

void StokesEquation::Init() { Equation::Init(); }

HostRealMatrix &StokesEquation::GetVelocity() { return velocity_; }

HostRealVector &StokesEquation::GetPressure() { return pressure_; }