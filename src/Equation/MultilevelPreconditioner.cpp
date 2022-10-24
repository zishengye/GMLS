#include "Equation/MultilevelPreconditioner.hpp"
#include <cstdio>
#include <mpi.h>

Equation::MultilevelPreconditioner::MultilevelPreconditioner() {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);
}

Equation::MultilevelPreconditioner::~MultilevelPreconditioner() {}

Void Equation::MultilevelPreconditioner::ApplyPreconditioningIteration(
    DefaultVector &x, DefaultVector &y) {
  const int numLevel = linearSystemsPtr_.size();

  // sweep down
  auxiliaryVectorBPtr_[numLevel - 1] = x;
  for (int i = numLevel - 1; i > 0; i--) {
    // pre-smooth
    smootherPtr_[i].Solve(auxiliaryVectorBPtr_[i], auxiliaryVectorXPtr_[i]);

    // get residual
    auxiliaryVectorRPtr_[i] = (*linearSystemsPtr_[i]) * auxiliaryVectorXPtr_[i];
    auxiliaryVectorRPtr_[i] -= auxiliaryVectorBPtr_[i];
    auxiliaryVectorRPtr_[i] *= -1.0;

    // restrict
    auxiliaryVectorBPtr_[i - 1] = restrictionPtr_[i] * auxiliaryVectorRPtr_[i];
  }

  // smooth on the base level
  smootherPtr_[0].Solve(auxiliaryVectorBPtr_[0], auxiliaryVectorXPtr_[0]);

  // sweep up
  for (int i = 1; i < numLevel; i++) {
    // interpolate
    auxiliaryVectorRPtr_[i] =
        interpolationPtr_[i] * auxiliaryVectorXPtr_[i - 1];
    auxiliaryVectorXPtr_[i] += auxiliaryVectorRPtr_[i];

    // get residual
    auxiliaryVectorRPtr_[i] = (*linearSystemsPtr_[i]) * auxiliaryVectorXPtr_[i];
    auxiliaryVectorRPtr_[i] -= auxiliaryVectorBPtr_[i];
    auxiliaryVectorRPtr_[i] *= -1.0;

    // post smooth
    smootherPtr_[i].Solve(auxiliaryVectorRPtr_[i], auxiliaryVectorBPtr_[i]);
    auxiliaryVectorXPtr_[i] += auxiliaryVectorBPtr_[i];
  }
  y = auxiliaryVectorXPtr_[numLevel - 1];
}

Void Equation::MultilevelPreconditioner::ApplyAdjointPreconditioningIteration(
    DefaultVector &x, DefaultVector &y) {
  const int numLevel = linearSystemsPtr_.size();

  // sweep down
  auxiliaryVectorBPtr_[numLevel - 1] = x;
  for (int i = numLevel - 1; i > 0; i--) {
    // pre-smooth
    adjointSmootherPtr_[i].Solve(auxiliaryVectorBPtr_[i],
                                 auxiliaryVectorXPtr_[i]);

    // get residual
    auxiliaryVectorRPtr_[i] = (*linearSystemsPtr_[i]) * auxiliaryVectorXPtr_[i];
    auxiliaryVectorRPtr_[i] -= auxiliaryVectorBPtr_[i];
    auxiliaryVectorRPtr_[i] *= -1.0;

    // restrict
    auxiliaryVectorBPtr_[i - 1] = restrictionPtr_[i] * auxiliaryVectorRPtr_[i];
  }

  // smooth on the base level
  adjointSmootherPtr_[0].Solve(auxiliaryVectorBPtr_[0],
                               auxiliaryVectorXPtr_[0]);

  // sweep up
  for (int i = 1; i < numLevel; i++) {
    // interpolate
    auxiliaryVectorRPtr_[i] =
        interpolationPtr_[i] * auxiliaryVectorXPtr_[i - 1];
    auxiliaryVectorXPtr_[i] += auxiliaryVectorRPtr_[i];

    // get residual
    auxiliaryVectorRPtr_[i] = (*linearSystemsPtr_[i]) * auxiliaryVectorXPtr_[i];
    auxiliaryVectorRPtr_[i] -= auxiliaryVectorBPtr_[i];
    auxiliaryVectorRPtr_[i] *= -1.0;

    // post smooth
    adjointSmootherPtr_[i].Solve(auxiliaryVectorRPtr_[i],
                                 auxiliaryVectorBPtr_[i]);
    auxiliaryVectorXPtr_[i] += auxiliaryVectorBPtr_[i];
  }
  y = auxiliaryVectorXPtr_[numLevel - 1];
}

Equation::MultilevelPreconditioner::DefaultMatrix &
Equation::MultilevelPreconditioner::GetInterpolation(const Size level) {
  return interpolationPtr_[level];
}

Equation::MultilevelPreconditioner::DefaultMatrix &
Equation::MultilevelPreconditioner::GetRestriction(const Size level) {
  return restrictionPtr_[level];
}

Equation::MultilevelPreconditioner::DefaultLinearSolver &
Equation::MultilevelPreconditioner::GetSmoother(const Size level) {
  return smootherPtr_[level];
}

Equation::MultilevelPreconditioner::DefaultLinearSolver &
Equation::MultilevelPreconditioner::GetAdjointSmoother(const Size level) {
  return adjointSmootherPtr_[level];
}

Void Equation::MultilevelPreconditioner::AddLinearSystem(
    std::shared_ptr<DefaultMatrix> &mat) {
  linearSystemsPtr_.push_back(mat);
}

Void Equation::MultilevelPreconditioner::AddAdjointLinearSystem(
    std::shared_ptr<DefaultMatrix> &mat) {
  adjointLinearSystemsPtr_.push_back(mat);
}

Void Equation::MultilevelPreconditioner::PrepareVectors(const Size localSize) {
  auxiliaryVectorXPtr_.emplace_back(localSize);
  auxiliaryVectorRPtr_.emplace_back(localSize);
  auxiliaryVectorBPtr_.emplace_back(localSize);
}

Void Equation::MultilevelPreconditioner::ConstructInterpolation(
    DefaultParticleManager &particleMgr) {
  interpolationPtr_.emplace_back();
  const Size currentLevel = linearSystemsPtr_.size() - 1;
  // build ghost
  if (currentLevel > 0) {
    interpolationGhost_.Init(
        particleMgr.GetParticleCoordsByLevel(currentLevel),
        particleMgr.GetParticleSizeByLevel(currentLevel),
        particleMgr.GetParticleCoordsByLevel(currentLevel - 1), 8.0,
        particleMgr.GetDimension());
  }
}

Void Equation::MultilevelPreconditioner::ConstructRestriction(
    DefaultParticleManager &particleMgr) {
  restrictionPtr_.emplace_back();
  const Size currentLevel = linearSystemsPtr_.size() - 1;
  // build ghost
  if (currentLevel > 0) {
    restrictionGhost_.Init(
        particleMgr.GetParticleCoordsByLevel(currentLevel - 1),
        particleMgr.GetParticleSizeByLevel(currentLevel - 1),
        particleMgr.GetParticleCoordsByLevel(currentLevel), 8.0,
        particleMgr.GetDimension());
  }
}

Void Equation::MultilevelPreconditioner::ConstructSmoother() {
  smootherPtr_.emplace_back();
}

Void Equation::MultilevelPreconditioner::ConstructAdjointSmoother() {
  adjointSmootherPtr_.emplace_back();
}