#include "GMLS_solver.h"

#include <iostream>

using namespace std;

template <typename T>
int SearchCommand(int argc, char **argv, const string &commandName, T &res) {
  int i;
  for (i = 1; i < argc; i++) {
    if (commandName.compare(argv[i]) == 0) {
      break;
    }
  }

  if (i != argc) {
    stringstream converter(argv[i + 1]);
    converter >> res;

    return 0;
  } else {
    return 1;
  }
}

GMLS_Solver::GMLS_Solver(int argc, char **argv) {
  __particle.scalarBasis = nullptr;
  __particle.vectorBasis = nullptr;
  __particle.scalarNeumannBoundaryBasis = nullptr;
  __particle.vectorNeumannBoundaryBasis = nullptr;
  // [default setup]
  __successInitialized = false;

  // MPI setup
  MPI_Comm_size(MPI_COMM_WORLD, &__MPISize);
  MPI_Comm_rank(MPI_COMM_WORLD, &__myID);

  // default dimension is 3
  if (SearchCommand<int>(argc, argv, "-Dim", __dim) == 1) {
    __dim = 3;
  } else {
    if (__dim > 3 || __dim < 1) {
      PetscPrintf(PETSC_COMM_WORLD, "Wrong dimension!\n");
      return;
    }
  }

  // default time integration method is forward euler method
  if (SearchCommand<string>(argc, argv, "-TimeIntegration",
                            __timeIntegrationMethod) == 1) {
    __timeIntegrationMethod = "ForwardEuler";
  } else {
    // TODO: check the correctness of the command
  }

  // default governing equation is Navier-Stokes equation
  if (SearchCommand<string>(argc, argv, "-EquationType", __equationType) == 1) {
    __equationType = "Navier-Stokes";
  } else {
    // TODO: check the correctness of the command
  }

  // default particle scheme is Eulerian particles
  if ((SearchCommand<string>(argc, argv, "-Scheme", __schemeType)) == 1) {
    __schemeType = "Eulerian";
  } else {
    // TODO: check the correctness of the command
  }

  // defalut discretization order is 2
  if ((SearchCommand<int>(argc, argv, "-PolynomialOrder", __polynomialOrder)) ==
      1) {
    __polynomialOrder = 2;
  } else {
    // TODO: check the correctness of the command
  }

  // default recovery_UerrorTolerance is 1e-1
  if ((SearchCommand<double>(argc, argv, "-Recovery_UerrorTolerance",
                             __recoveryGradUerrorTolerance)) == 1) {
    __recoveryGradUerrorTolerance = 1e-1;
  } else {
    // TODO: check the correctness of the command
  }

  // default serial output
  if ((SearchCommand<int>(argc, argv, "-WriteData", __writeData)) == 1) {
    __writeData = 1;
  }

  // default manifold flag is off
  if ((SearchCommand<int>(argc, argv, "-ManifoldOrder", __manifoldOrder)) ==
      1) {
    __manifoldOrder = 0;
  } else {
    if (__manifoldOrder < 0) {
      return;
    }
  }

  // [parameter must appear in command]

  // discretization parameter
  if (__dim == 3) {
    int xCheck = SearchCommand<int>(argc, argv, "-Mx", __boundingBoxCount[0]);
    int yCheck = SearchCommand<int>(argc, argv, "-My", __boundingBoxCount[1]);
    int zCheck = SearchCommand<int>(argc, argv, "-Mz", __boundingBoxCount[2]);
    if ((xCheck == 1) && (yCheck == 1) && (zCheck == 1)) {
      return;
    } else {
      // TODO: check the correctness of the command
    }
  } else if (__dim == 2) {
    int xCheck = SearchCommand<int>(argc, argv, "-Mx", __boundingBoxCount[0]);
    int yCheck = SearchCommand<int>(argc, argv, "-My", __boundingBoxCount[1]);
    if ((xCheck == 1) && (yCheck == 1)) {
      return;
    } else {
      // TODO: check the correctness of the command
    }
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "Please specify discretization parameter!\n");
    return;
  }

  // final time
  if ((SearchCommand<double>(argc, argv, "-ft", __finalTime)) == 1) {
    return;
  } else if (__finalTime < 0.0) {
    return;
  }

  // time step
  if ((SearchCommand<double>(argc, argv, "-dt", __dt)) == 1) {
    return;
  } else if (__dt < 0.0) {
    return;
  }

  // kinetic viscosity distance
  if ((SearchCommand<double>(argc, argv, "-eta", __eta)) == 1) {
    return;
  } else if (__eta < 0.0) {
    return;
  }

  // [summary of problem setup]

  PetscPrintf(PETSC_COMM_WORLD, "===============================\n");
  PetscPrintf(PETSC_COMM_WORLD, "==== Problem setup summary ====\n");
  PetscPrintf(PETSC_COMM_WORLD, "===============================\n");
  PetscPrintf(PETSC_COMM_WORLD, "==> Dimension: %d\n", __dim);
  PetscPrintf(PETSC_COMM_WORLD, "==> Governing equation: %s\n",
              __equationType.c_str());
  PetscPrintf(PETSC_COMM_WORLD, "==> Time interval: %fs\n", __dt);
  PetscPrintf(PETSC_COMM_WORLD, "==> Final time: %fs\n", __finalTime);
  PetscPrintf(PETSC_COMM_WORLD, "==> Polynomial order: %d\n",
              __polynomialOrder);
  if (__dim == 3) {
    PetscPrintf(PETSC_COMM_WORLD, "==> Particle count in X axis: %d\n",
                __boundingBoxCount[0]);
    PetscPrintf(PETSC_COMM_WORLD, "==> Particle count in Y axis: %d\n",
                __boundingBoxCount[1]);
    PetscPrintf(PETSC_COMM_WORLD, "==> Particle count in Z axis: %d\n",
                __boundingBoxCount[2]);
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "==> Particle count in X axis: %d\n",
                __boundingBoxCount[0]);
    PetscPrintf(PETSC_COMM_WORLD, "==> Particle count in Y axis: %d\n",
                __boundingBoxCount[1]);
  }
  PetscPrintf(PETSC_COMM_WORLD, "==> Kinetic viscosity: %f\n", __eta);

  __successInitialized = true;
}

void GMLS_Solver::ClearMemory() {
  __particle.X.clear();
  __particle.particleType.clear();
  __particle.normal.clear();
  __particle.globalIndex.clear();
  __particle.vol.clear();
}