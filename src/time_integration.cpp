#include "GMLS_solver.h"

void GMLS_Solver::TimeIntegration() {
  if (__manifoldOrder == 0) {
    SetBoundingBox();
    SetBoundingBoxBoundary();

    InitColloid();

    InitDomainDecomposition();
  } else {
    SetBoundingBoxManifold();
    SetBoundingBoxBoundaryManifold();

    InitColloid();

    InitDomainDecompositionManifold();
  }

  // equation type selection
  if (__equationType == "Stokes" && __manifoldOrder == 0) {
    __equationSolver = &GMLS_Solver::StokesEquation;
  }

  if (__equationType == "Poisson" && __manifoldOrder == 0) {
    __equationSolver = &GMLS_Solver::PoissonEquation;
  }

  if (__equationType == "Poisson" && __manifoldOrder > 0) {
    __equationSolver = &GMLS_Solver::PoissonEquationManifold;
  }

  if (__timeIntegrationMethod == "ForwardEuler") {
    ForwardEulerIntegration();
  }
}

void GMLS_Solver::ForwardEulerIntegration() {
  for (double t = 0; t < __finalTime; t += __dt) {
    PetscPrintf(PETSC_COMM_WORLD, "===================================\n");
    PetscPrintf(PETSC_COMM_WORLD, "==== Start of time integration ====\n");
    PetscPrintf(PETSC_COMM_WORLD, "===================================\n");
    PetscPrintf(PETSC_COMM_WORLD, "==> Current time: %f s\n", t);
    PetscPrintf(PETSC_COMM_WORLD, "==> current time step: %f s\n", __dt);

    PetscPrintf(PETSC_COMM_WORLD, "\nGenerating uniform particle field...\n");
    if (__manifoldOrder > 0) {
      InitUniformParticleManifoldField();

      EmposeBoundaryCondition();

      BuildNeighborListManifold();
    } else {
      InitUniformParticleField();

      EmposeBoundaryCondition();

      BuildNeighborList();
    }

    (this->*__equationSolver)();

    PetscPrintf(PETSC_COMM_WORLD, "\n=================================\n");
    PetscPrintf(PETSC_COMM_WORLD, "==== End of time integration ====\n");
    PetscPrintf(PETSC_COMM_WORLD, "=================================\n\n");

    if (__writeData != 0) {
      WriteData();
    }
  }
}