#include "gmls_solver.h"

using namespace std;

void GMLS_Solver::InitialGuessFromPreviousAdaptiveStep(
    PetscSparseMatrix &I, vector<double> &initial_guess) {
  // set initial guess for field values
  static vector<vec3> &velocity = __field.vector.GetHandle("fluid velocity");
  static vector<double> &pressure = __field.scalar.GetHandle("fluid pressure");

  static auto &coord = __field.vector.GetHandle("coord");

  vector<double> previous_result;

  int field_dof = __dim + 1;
  int velocity_dof = __dim;
  int pressure_dof = 1;

  int old_local_particle_num = pressure.size();
  int new_local_particle_num = coord.size();

  if (__myID == __MPISize - 1) {
    static vector<vec3> &rigid_body_velocity =
        __rigidBody.vector.GetHandle("velocity");
    static vector<vec3> &rigid_body_angular_velocity =
        __rigidBody.vector.GetHandle("angular velocity");

    int local_rigid_body_offset = old_local_particle_num * field_dof;
    int num_rigid_body = rigid_body_velocity.size();

    int translation_dof = __dim;
    int rotation_dof = (__dim == 3) ? 3 : 1;
    int rigid_body_dof = translation_dof + rotation_dof;

    previous_result.resize(field_dof * old_local_particle_num +
                           num_rigid_body * rigid_body_dof);

    for (int i = 0; i < num_rigid_body; i++) {
      for (int j = 0; j < translation_dof; j++) {
        previous_result[local_rigid_body_offset + i * rigid_body_dof + j] =
            rigid_body_velocity[i][j];
      }
      for (int j = 0; j < rotation_dof; j++) {
        previous_result[local_rigid_body_offset + i * rigid_body_dof +
                        translation_dof + j] =
            rigid_body_angular_velocity[i][j];
      }
    }

  } else {
    previous_result.resize(field_dof * old_local_particle_num);
  }

  for (int i = 0; i < old_local_particle_num; i++) {
    for (int j = 0; j < velocity_dof; j++) {
      previous_result[i * field_dof + j] = velocity[i][j];
    }

    previous_result[i * field_dof + velocity_dof] = pressure[i];
  }

  Vec previous_result_vec, initial_guess_vec;
  VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, previous_result.size(),
                        PETSC_DECIDE, previous_result.data(),
                        &previous_result_vec);

  MatCreateVecs(I.__mat, NULL, &initial_guess_vec);
  MatMult(I.__mat, previous_result_vec, initial_guess_vec);

  velocity.resize(new_local_particle_num);
  pressure.resize(new_local_particle_num);

  double *ptr;
  VecGetArray(initial_guess_vec, &ptr);
  for (int i = 0; i < initial_guess.size(); i++) {
    initial_guess[i] = ptr[i];
  }
  VecRestoreArray(initial_guess_vec, &ptr);

  VecDestroy(&initial_guess_vec);
  VecDestroy(&previous_result_vec);
}