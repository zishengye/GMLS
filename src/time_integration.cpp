#include "gmls_solver.h"

using namespace std;
using namespace Compadre;

void GMLS_Solver::TimeIntegration() {
  InitParticle();

  if (__manifoldOrder == 0) {
    SetBoundingBox();
    SetBoundingBoxBoundary();

    InitRigidBody();

    InitDomainDecomposition();

    if (__adaptiveRefinement) {
      __field.vector.Register("old coord");
      __background.vector.Register("old source coord");
      __background.index.Register("old source index");
    }
  } else {
    SetBoundingBoxManifold();
    SetBoundingBoxBoundaryManifold();

    InitRigidBody();

    InitDomainDecompositionManifold();
  }

  // equation type selection and initialization
  if (__equationType == "Stokes" && __manifoldOrder == 0) {
    __equationSolverInitialization = &GMLS_Solver::StokesEquationInitialization;
    __equationSolver = &GMLS_Solver::StokesEquation;
  }

  if (__equationType == "Poisson" && __manifoldOrder == 0) {
    __equationSolver = &GMLS_Solver::PoissonEquation;
  }

  if (__equationType == "Poisson" && __manifoldOrder > 0) {
    __equationSolver = &GMLS_Solver::PoissonEquationManifold;
  }

  if (__equationType == "Diffusion" && __manifoldOrder > 0) {
    __equationSolver = &GMLS_Solver::DiffusionEquationManifold;
  }

  if (__timeIntegrationMethod == "ForwardEuler") {
    ForwardEulerIntegration();
  }

  if (__timeIntegrationMethod == "RK4") {
    RungeKuttaIntegration();
  }
}

void GMLS_Solver::ForwardEulerIntegration() {
  (this->*__equationSolverInitialization)();

  for (double t = 0; t < __finalTime + 1e-5; t += __dt) {
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

    // if (t == 0) {
    //   InitialCondition();
    // }

    do {
      if (__writeData)
        WriteDataAdaptiveGeometry();
      PetscPrintf(PETSC_COMM_WORLD, "Adaptive level: %d\n", __adaptive_step);
      (this->*__equationSolver)();
    } while (NeedRefinement());

    PetscPrintf(PETSC_COMM_WORLD, "\n=================================\n");
    PetscPrintf(PETSC_COMM_WORLD, "==== End of time integration ====\n");
    PetscPrintf(PETSC_COMM_WORLD, "=================================\n\n");

    if (__writeData != 0) {
      WriteDataTimeStep();
    }
  }
}

void GMLS_Solver::RungeKuttaIntegration() {
  (this->*__equationSolverInitialization)();

  vector<vec3> &rigidBodyPosition = __rigidBody.vector.GetHandle("position");
  vector<vec3> &rigidBodyOrientation =
      __rigidBody.vector.GetHandle("orientation");
  vector<vec3> &rigidBodyVelocity = __rigidBody.vector.GetHandle("velocity");
  vector<vec3> &rigidBodyAngularVelocity =
      __rigidBody.vector.GetHandle("angular velocity");

  int numRigidBody = rigidBodyVelocity.size();

  vector<vec3> position0(numRigidBody);
  vector<vec3> velocity_k1(numRigidBody);
  vector<vec3> velocity_k2(numRigidBody);
  vector<vec3> velocity_k3(numRigidBody);
  vector<vec3> velocity_k4(numRigidBody);

  vector<vec3> orientation0(numRigidBody);
  vector<vec3> angularVelocity_k1(numRigidBody);
  vector<vec3> angularVelocity_k2(numRigidBody);
  vector<vec3> angularVelocity_k3(numRigidBody);
  vector<vec3> angularVelocity_k4(numRigidBody);

  ofstream output;
  if (__myID == 0) {
    output.open("traj.txt", ios::trunc);
    output.close();
  }

  for (double t = 0; t < __finalTime + 1e-5; t += __dt) {
    PetscPrintf(PETSC_COMM_WORLD, "===================================\n");
    PetscPrintf(PETSC_COMM_WORLD, "==== Start of time integration ====\n");
    PetscPrintf(PETSC_COMM_WORLD, "===================================\n");
    PetscPrintf(PETSC_COMM_WORLD, "==> Current time: %f s\n", t);
    PetscPrintf(PETSC_COMM_WORLD, "==> current time step: %f s\n", __dt);

    for (int num = 0; num < numRigidBody; num++) {
      for (int j = 0; j < 3; j++) {
        position0[num][j] = rigidBodyPosition[num][j];
        orientation0[num][j] = rigidBodyOrientation[num][j];
      }
    }

    for (int i = 0; i < 4; i++) {
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

      PetscPrintf(PETSC_COMM_WORLD, "start of adaptive step\n");

      __adaptive_step = 0;
      do {
        if (__writeData)
          WriteDataAdaptiveGeometry();
        PetscPrintf(PETSC_COMM_WORLD, "Adaptive level: %d\n", __adaptive_step);
        (this->*__equationSolver)();
      } while (NeedRefinement());

      switch (i) {
      case 0:
        for (int num = 0; num < numRigidBody; num++) {
          for (int j = 0; j < 3; j++) {
            velocity_k1[num][j] = rigidBodyVelocity[num][j];
            angularVelocity_k1[num][j] = rigidBodyAngularVelocity[num][j];
          }
        }

        for (int num = 0; num < numRigidBody; num++) {
          for (int j = 0; j < 3; j++) {
            rigidBodyPosition[num][j] =
                position0[num][j] + velocity_k1[num][j] * __dt * 0.5;
            rigidBodyOrientation[num][j] =
                orientation0[num][j] + angularVelocity_k1[num][j] * __dt * 0.5;
          }
        }
        break;
      case 1:
        for (int num = 0; num < numRigidBody; num++) {
          for (int j = 0; j < 3; j++) {
            velocity_k2[num][j] = rigidBodyVelocity[num][j];
            angularVelocity_k2[num][j] = rigidBodyAngularVelocity[num][j];
          }
        }

        for (int num = 0; num < numRigidBody; num++) {
          for (int j = 0; j < 3; j++) {
            rigidBodyPosition[num][j] =
                position0[num][j] + velocity_k2[num][j] * __dt * 0.5;
            rigidBodyOrientation[num][j] =
                orientation0[num][j] + angularVelocity_k2[num][j] * __dt * 0.5;
          }
        }
        break;
      case 2:
        for (int num = 0; num < numRigidBody; num++) {
          for (int j = 0; j < 3; j++) {
            velocity_k3[num][j] = rigidBodyVelocity[num][j];
            angularVelocity_k3[num][j] = rigidBodyAngularVelocity[num][j];
          }
        }

        for (int num = 0; num < numRigidBody; num++) {
          for (int j = 0; j < 3; j++) {
            rigidBodyPosition[num][j] =
                position0[num][j] + velocity_k3[num][j] * __dt;
            rigidBodyOrientation[num][j] =
                orientation0[num][j] + angularVelocity_k3[num][j] * __dt;
          }
        }
        break;
      case 3:
        for (int num = 0; num < numRigidBody; num++) {
          for (int j = 0; j < 3; j++) {
            velocity_k4[num][j] = rigidBodyVelocity[num][j];
            angularVelocity_k4[num][j] = rigidBodyAngularVelocity[num][j];
          }
        }

        for (int num = 0; num < numRigidBody; num++) {
          for (int j = 0; j < 3; j++) {
            rigidBodyPosition[num][j] =
                position0[num][j] +
                (velocity_k1[num][j] + 2.0 * velocity_k2[num][j] +
                 2.0 * velocity_k3[num][j] + velocity_k4[num][j]) *
                    __dt / 6.0;
            rigidBodyOrientation[num][j] =
                orientation0[num][j] +
                (angularVelocity_k1[num][j] + 2.0 * angularVelocity_k2[num][j] +
                 2.0 * angularVelocity_k3[num][j] +
                 angularVelocity_k4[num][j]) *
                    __dt / 6.0;
          }
        }
        break;
      }
    }

    PetscPrintf(PETSC_COMM_WORLD, "\n=================================\n");
    PetscPrintf(PETSC_COMM_WORLD, "==== End of time integration ====\n");
    PetscPrintf(PETSC_COMM_WORLD, "=================================\n\n");

    if (__myID == 0) {
      output.open("traj.txt", ios::app);
      for (int num = 0; num < numRigidBody; num++) {
        for (int j = 0; j < 3; j++) {
          output << rigidBodyPosition[num][j] << '\t';
        }
        for (int j = 0; j < 3; j++) {
          output << rigidBodyOrientation[num][j] << '\t';
        }
      }
      output << endl;
      output.close();
    }

    if (__writeData != 0) {
      WriteDataTimeStep();
    }
  }
}