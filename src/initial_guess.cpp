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

  previous_result.resize(field_dof * old_local_particle_num);

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
  pressure.resize(pressure_dof * new_local_particle_num);

  double *ptr;
  VecGetArray(initial_guess_vec, &ptr);
  for (int i = 0; i < new_local_particle_num; i++) {
    for (int j = 0; j < velocity_dof; j++) {
      initial_guess[i * field_dof + j] = ptr[i * field_dof + j];
    }

    initial_guess[i * field_dof + velocity_dof] =
        ptr[i * field_dof + velocity_dof];
  }
  VecRestoreArray(initial_guess_vec, &ptr);

  VecDestroy(&initial_guess_vec);
  VecDestroy(&previous_result_vec);

  // set initial value for lagrange multiplier
  double pressure_sum = 0.0;
  for (int i = 0; i < new_local_particle_num; i++)
    pressure_sum += initial_guess[i * field_dof + velocity_dof];
  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  int new_global_particle_num;
  MPI_Allreduce(&new_local_particle_num, &new_global_particle_num, 1, MPI_INT,
                MPI_SUM, MPI_COMM_WORLD);

  double lagrange_multiplier;

  if (__myID == __MPISize - 1) {
    initial_guess[new_local_particle_num * field_dof + velocity_dof] =
        -pressure_sum / new_global_particle_num;
    lagrange_multiplier = -pressure_sum / new_global_particle_num;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&lagrange_multiplier, 1, MPI_DOUBLE, __MPISize - 1, MPI_COMM_WORLD);

  for (int i = 0; i < new_local_particle_num; i++) {
    initial_guess[i * field_dof + velocity_dof] -= lagrange_multiplier;
  }

  // set initial value for rigid body dofs
  if (__myID == __MPISize - 1) {
    static vector<vec3> &rigid_body_velocity =
        __rigidBody.vector.GetHandle("velocity");
    static vector<vec3> &rigid_body_angular_velocity =
        __rigidBody.vector.GetHandle("angular velocity");

    int local_rigid_body_offset = (new_local_particle_num + 1) * field_dof;
    int num_rigid_body = rigid_body_velocity.size();

    int translation_dof = __dim;
    int rotation_dof = (__dim == 3) ? 3 : 1;
    int rigid_body_dof = translation_dof + rotation_dof;

    for (int i = 0; i < num_rigid_body; i++) {
      for (int j = 0; j < translation_dof; j++) {
        initial_guess[local_rigid_body_offset + i * rigid_body_dof + j] =
            rigid_body_velocity[i][j];
      }
      for (int j = 0; j < rotation_dof; j++) {
        initial_guess[local_rigid_body_offset + i * rigid_body_dof +
                      translation_dof + j] = rigid_body_angular_velocity[i][j];
      }
    }
  }
}