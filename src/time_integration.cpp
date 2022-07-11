#include "gmls_solver.hpp"

#include <petscdmredundant.h>
#include <petscsnes.h>

using namespace std;
using namespace Compadre;

Vec3 Cross(const Vec3 &v1, const Vec3 &v2) {
  Vec3 res(v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2],
           v1[0] * v2[1] - v1[1] * v2[0]);
  return res;
}

Vec3 Bracket(const Vec3 &v1, const Vec3 &v2) { return Cross(v1, v2) * 2.0; }

Vec3 dexpinv(const Vec3 &u, const Vec3 &k) {
  Vec3 res;
  Vec3 bracket_res = Bracket(u, k);

  res = k - bracket_res * 0.5;
  bracket_res = Bracket(u, bracket_res);
  res = res + bracket_res * static_cast<double>(1.0 / 12.0);
  bracket_res = Bracket(u, Bracket(u, bracket_res));
  res = res - bracket_res * static_cast<double>(1.0 / 720.0);
  bracket_res = Bracket(u, Bracket(u, bracket_res));
  res = res + bracket_res * static_cast<double>(1.0 / 30240.0);
  bracket_res = Bracket(u, Bracket(u, bracket_res));
  res = res - bracket_res * static_cast<double>(1.0 / 1209600.0);

  return res;
}

PetscErrorCode implicit_midpoint_integration_sub_wrapper(SNES, Vec, Vec,
                                                         void *);

inline double correct_radius(double x) {
  while (x > 2.0 * M_PI)
    x -= 2.0 * M_PI;
  while (x < 0.0)
    x += 2.0 * M_PI;

  return x;
}

void gmls_solver::time_integration() {
  if (time_integration_method == "ForwardEuler") {
    foward_euler_integration();
  }

  if (time_integration_method == "RK4") {
    adaptive_runge_kutta_intagration();
  }

  if (time_integration_method == "ImplicitMidpoint") {
    implicit_midpoint_integration();
  }
}

void gmls_solver::foward_euler_integration() {
  for (double t = 0; t < final_time + 1e-5; t += max_dt) {
    PetscPrintf(PETSC_COMM_WORLD, "===================================\n");
    PetscPrintf(PETSC_COMM_WORLD, "==== Start of time integration ====\n");
    PetscPrintf(PETSC_COMM_WORLD, "===================================\n");
    PetscPrintf(PETSC_COMM_WORLD, "==> Current time: %f s\n", t);
    PetscPrintf(PETSC_COMM_WORLD, "==> current time step: %f s\n", max_dt);

    PetscPrintf(PETSC_COMM_WORLD, "\nGenerating uniform particle field...\n");
    geo_mgr->generate_uniform_particle();

    // if (t == 0) {
    //   InitialCondition();
    // }

    current_refinement_step = 0;
    equation_mgr->Reset();
    do {
      if (write_data == 1 || write_data == 4)
        write_refinement_data_geometry_only();
      PetscPrintf(PETSC_COMM_WORLD, "refinement level: %d\n",
                  current_refinement_step);
      equation_mgr->Update();
    } while (refinement());

    PetscPrintf(PETSC_COMM_WORLD, "\n=================================\n");
    PetscPrintf(PETSC_COMM_WORLD, "==== End of time integration ====\n");
    PetscPrintf(PETSC_COMM_WORLD, "=================================\n\n");

    if (write_data == 2 || write_data == 4) {
      write_time_step_data();
    }
  }
}

void gmls_solver::adaptive_runge_kutta_intagration() {
  vector<Vec3> &rigidBodyPosition = rb_mgr->get_position();
  vector<Vec3> &rigidBodyOrientation = rb_mgr->get_orientation();
  vector<quaternion> &rigidBodyQuaternion = rb_mgr->get_quaternion();
  vector<Vec3> &rigidBodyVelocity = rb_mgr->get_velocity();
  vector<Vec3> &rigidBodyAngularVelocity = rb_mgr->get_angular_velocity();
  vector<Vec3> &rigidBodyForce = rb_mgr->get_force();
  vector<Vec3> &rigidBodyTorque = rb_mgr->get_torque();

  int numRigidBody = rb_mgr->get_rigid_body_num();

  position0.resize(numRigidBody);
  orientation0.resize(numRigidBody);
  quaternion0.resize(numRigidBody);

  vector<Vec3> velocity_k1(numRigidBody);
  vector<Vec3> velocity_k2(numRigidBody);
  vector<Vec3> velocity_k3(numRigidBody);
  vector<Vec3> velocity_k4(numRigidBody);
  vector<Vec3> velocity_k5(numRigidBody);
  vector<Vec3> velocity_k6(numRigidBody);
  vector<Vec3> velocity_k7(numRigidBody);

  vector<Vec3> angularVelocity_k1(numRigidBody);
  vector<Vec3> angularVelocity_k2(numRigidBody);
  vector<Vec3> angularVelocity_k3(numRigidBody);
  vector<Vec3> angularVelocity_k4(numRigidBody);
  vector<Vec3> angularVelocity_k5(numRigidBody);
  vector<Vec3> angularVelocity_k6(numRigidBody);
  vector<Vec3> angularVelocity_k7(numRigidBody);

  vector<Vec3> modified_angularVelocity_k1(numRigidBody);
  vector<Vec3> modified_angularVelocity_k2(numRigidBody);
  vector<Vec3> modified_angularVelocity_k3(numRigidBody);
  vector<Vec3> modified_angularVelocity_k4(numRigidBody);
  vector<Vec3> modified_angularVelocity_k5(numRigidBody);
  vector<Vec3> modified_angularVelocity_k6(numRigidBody);
  vector<Vec3> modified_angularVelocity_k7(numRigidBody);

  vector<quaternion> intermediate_quaternion1(numRigidBody);
  vector<quaternion> intermediate_quaternion2(numRigidBody);

  vector<double> sychronize_velocity(numRigidBody * 3);
  vector<double> sychronize_angularVelocity(numRigidBody * 3);

  // ode45 algorithm parameter
  double t, dt, dtMin, rtol, atol, err, norm_y;
  rtol = 1e-3;
  atol = 1e-10;
  dt = max_dt;
  t = 0;
  dtMin = 1e-10;

  // constants for integration
  const double a21 = static_cast<double>(1) / static_cast<double>(5);

  const double a31 = static_cast<double>(3) / static_cast<double>(40);
  const double a32 = static_cast<double>(9) / static_cast<double>(40);

  const double a41 = static_cast<double>(44) / static_cast<double>(45);
  const double a42 = static_cast<double>(-56) / static_cast<double>(15);
  const double a43 = static_cast<double>(32) / static_cast<double>(9);

  const double a51 = static_cast<double>(19372) / static_cast<double>(6561);
  const double a52 = static_cast<double>(-25360) / static_cast<double>(2187);
  const double a53 = static_cast<double>(64448) / static_cast<double>(6561);
  const double a54 = static_cast<double>(-212) / static_cast<double>(729);

  const double a61 = static_cast<double>(9017) / static_cast<double>(3168);
  const double a62 = static_cast<double>(-355) / static_cast<double>(33);
  const double a63 = static_cast<double>(46732) / static_cast<double>(5247);
  const double a64 = static_cast<double>(49) / static_cast<double>(176);
  const double a65 = static_cast<double>(-5103) / static_cast<double>(18656);

  const double b1 = static_cast<double>(35) / static_cast<double>(384);
  const double b3 = static_cast<double>(500) / static_cast<double>(1113);
  const double b4 = static_cast<double>(125) / static_cast<double>(192);
  const double b5 = static_cast<double>(-2187) / static_cast<double>(6784);
  const double b6 = static_cast<double>(11) / static_cast<double>(84);

  const double dc1 =
      b1 - static_cast<double>(5179) / static_cast<double>(57600);
  const double dc3 =
      b3 - static_cast<double>(7571) / static_cast<double>(16695);
  const double dc4 = b4 - static_cast<double>(393) / static_cast<double>(640);
  const double dc5 =
      b5 - static_cast<double>(-92097) / static_cast<double>(339200);
  const double dc6 = b6 - static_cast<double>(187) / static_cast<double>(2100);
  const double dc7 = static_cast<double>(-1) / static_cast<double>(40);

  // ode45 algorithm counter
  long funcEvalCount = 1;

  PetscPrintf(PETSC_COMM_WORLD, "\n==================\n");
  PetscPrintf(PETSC_COMM_WORLD, "initial evaluation\n");
  PetscPrintf(PETSC_COMM_WORLD, "start of refinement step\n");

  // get initial value
  if (!geo_mgr->generate_uniform_particle()) {
    PetscPrintf(PETSC_COMM_WORLD, "\nWrong colloids placement\n");
    return;
  }

  current_refinement_step = 0;
  equation_mgr->Reset();
  do {
    if (write_data == 1 || write_data == 4) {
      write_refinement_data_geometry_only();
    }
    PetscPrintf(PETSC_COMM_WORLD, "refinement level: %d\n",
                current_refinement_step);
    equation_mgr->Update();
  } while (refinement());
  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = 0; i < numRigidBody; i++) {
    for (int j = 0; j < 3; j++) {
      sychronize_velocity[i * 3 + j] = rigidBodyVelocity[i][j];
      sychronize_angularVelocity[i * 3 + j] = rigidBodyAngularVelocity[i][j];
    }
  }

  MPI_Bcast(sychronize_velocity.data(), numRigidBody * 3, MPI_DOUBLE, 0,
            MPI_COMM_WORLD);
  MPI_Bcast(sychronize_angularVelocity.data(), numRigidBody * 3, MPI_DOUBLE, 0,
            MPI_COMM_WORLD);

  for (int i = 0; i < numRigidBody; i++) {
    for (int j = 0; j < 3; j++) {
      rigidBodyVelocity[i][j] = sychronize_velocity[i * 3 + j];
      rigidBodyAngularVelocity[i][j] = sychronize_angularVelocity[i * 3 + j];
    }
  }

  if (write_data == 2 || write_data == 4) {
    write_time_step_data();
  }

  // average
  // for (int j = 0; j < 3; j++) {
  //   double average;
  //   average = (rigidBodyVelocity[0][j] + rigidBodyVelocity[1][j]) / 2.0;
  //   rigidBodyVelocity[0][j] -= average;
  //   rigidBodyVelocity[1][j] -= average;
  //   average =
  //       (rigidBodyAngularVelocity[0][j] + rigidBodyAngularVelocity[1][j])
  //       / 2.0;
  //   rigidBodyAngularVelocity[0][j] = average;
  //   rigidBodyAngularVelocity[1][j] = average;
  // }

  for (int num = 0; num < numRigidBody; num++) {
    for (int j = 0; j < 3; j++) {
      velocity_k1[num][j] = rigidBodyVelocity[num][j];
      angularVelocity_k1[num][j] = rigidBodyAngularVelocity[num][j];
    }
    modified_angularVelocity_k1[num] = rigidBodyAngularVelocity[num];
  }

  // setup output file
  ofstream output;
  ofstream output_runge_kutta;
  ofstream outputVelocity;
  ofstream outputForce;
  if (rank == 0) {
    output.open(trajectory_output_file_name, ios::trunc);
    output << t << '\t';
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

    output_runge_kutta.open("traj_runge_kutta.txt", ios::trunc);
    output_runge_kutta.close();

    outputVelocity.open(velocity_output_file_name, ios::trunc);
    outputVelocity << t << '\t';
    for (int num = 0; num < numRigidBody; num++) {
      for (int j = 0; j < 3; j++) {
        outputVelocity << rigidBodyVelocity[num][j] << '\t';
      }
      for (int j = 0; j < 3; j++) {
        outputVelocity << rigidBodyAngularVelocity[num][j] << '\t';
      }
    }
    outputVelocity << endl;
    outputVelocity.close();

    outputForce.open(force_output_file_name, ios::trunc);
    outputForce << t << '\t';
    for (int num = 0; num < numRigidBody; num++) {
      for (int j = 0; j < 3; j++) {
        outputForce << rigidBodyForce[num][j] << '\t';
      }
      for (int j = 0; j < 3; j++) {
        outputForce << rigidBodyTorque[num][j] << '\t';
      }
    }
    outputForce << endl;
    outputForce.close();
  }

  // main loop
  while (t < final_time - 1e-5) {
    bool noFail = true;
    bool acceptableTrial = false;

    // ensure end exactly at final time
    dt = min(dt, final_time - t);

    for (int num = 0; num < numRigidBody; num++) {
      for (int j = 0; j < 3; j++) {
        position0[num][j] = rigidBodyPosition[num][j];
        if (dim == 2)
          orientation0[num][j] = correct_radius(rigidBodyOrientation[num][j]);
      }
      quaternion0[num] = rigidBodyQuaternion[num];
      if (dim == 3)
        rigidBodyQuaternion[num].ToEulerAngle(rigidBodyOrientation[num][0],
                                              rigidBodyOrientation[num][1],
                                              rigidBodyOrientation[num][2]);
    }

    err = 100;
    while (err > rtol) {
      PetscPrintf(PETSC_COMM_WORLD, "===================================\n");
      PetscPrintf(PETSC_COMM_WORLD, "==== Start of time integration ====\n");
      PetscPrintf(PETSC_COMM_WORLD, "===================================\n");
      PetscPrintf(PETSC_COMM_WORLD, "==> Current time: %f s\n", t);
      PetscPrintf(PETSC_COMM_WORLD, "==> current test time step: %f s\n", dt);

      for (int i = 1; i < 7; i++) {
        PetscPrintf(PETSC_COMM_WORLD, "=============================\n");
        PetscPrintf(PETSC_COMM_WORLD, "Current Runge-Kutta step: %d\n", i);
        PetscPrintf(PETSC_COMM_WORLD, "=============================\n");
        PetscPrintf(PETSC_COMM_WORLD,
                    "\nGenerating uniform particle field...\n");

        switch (i) {
        case 1:
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              rigidBodyPosition[num][j] =
                  position0[num][j] + dt * velocity_k1[num][j] * a21;
              if (dim == 2)
                rigidBodyOrientation[num][j] =
                    correct_radius(orientation0[num][j] +
                                   dt * angularVelocity_k1[num][j] * a21);
            }
            rigidBodyQuaternion[num].Cross(
                quaternion0[num],
                quaternion(modified_angularVelocity_k1[num], a21 * dt));
            if (dim == 3)
              rigidBodyQuaternion[num].ToEulerAngle(
                  rigidBodyOrientation[num][0], rigidBodyOrientation[num][1],
                  rigidBodyOrientation[num][2]);
          }
          break;
        case 2:
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              rigidBodyPosition[num][j] =
                  position0[num][j] +
                  dt * (velocity_k1[num][j] * a31 + velocity_k2[num][j] * a32);
              if (dim == 2)
                rigidBodyOrientation[num][j] =
                    correct_radius(orientation0[num][j] +
                                   dt * (angularVelocity_k1[num][j] * a31 +
                                         angularVelocity_k2[num][j] * a32));
            }
            rigidBodyQuaternion[num].Cross(
                quaternion0[num],
                quaternion((modified_angularVelocity_k1[num] * a31 +
                            modified_angularVelocity_k2[num] * a32),
                           dt));
            if (dim == 3)
              rigidBodyQuaternion[num].ToEulerAngle(
                  rigidBodyOrientation[num][0], rigidBodyOrientation[num][1],
                  rigidBodyOrientation[num][2]);
          }
          break;
        case 3:
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              rigidBodyPosition[num][j] =
                  position0[num][j] +
                  dt * (velocity_k1[num][j] * a41 + velocity_k2[num][j] * a42 +
                        velocity_k3[num][j] * a43);
              if (dim == 2)
                rigidBodyOrientation[num][j] =
                    correct_radius(orientation0[num][j] +
                                   dt * (angularVelocity_k1[num][j] * a41 +
                                         angularVelocity_k2[num][j] * a42 +
                                         angularVelocity_k3[num][j] * a43));
            }
            rigidBodyQuaternion[num].Cross(
                quaternion0[num],
                quaternion((modified_angularVelocity_k1[num] * a41 +
                            modified_angularVelocity_k2[num] * a42 +
                            modified_angularVelocity_k3[num] * a43),
                           dt));
            if (dim == 3)
              rigidBodyQuaternion[num].ToEulerAngle(
                  rigidBodyOrientation[num][0], rigidBodyOrientation[num][1],
                  rigidBodyOrientation[num][2]);
          }
          break;
        case 4:
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              rigidBodyPosition[num][j] =
                  position0[num][j] +
                  dt * (velocity_k1[num][j] * a51 + velocity_k2[num][j] * a52 +
                        velocity_k3[num][j] * a53 + velocity_k4[num][j] * a54);
              if (dim == 2)
                rigidBodyOrientation[num][j] =
                    correct_radius(orientation0[num][j] +
                                   dt * (angularVelocity_k1[num][j] * a51 +
                                         angularVelocity_k2[num][j] * a52 +
                                         angularVelocity_k3[num][j] * a53 +
                                         angularVelocity_k4[num][j] * a54));
            }
            rigidBodyQuaternion[num].Cross(
                quaternion0[num],
                quaternion((modified_angularVelocity_k1[num] * a51 +
                            modified_angularVelocity_k2[num] * a52 +
                            modified_angularVelocity_k3[num] * a53 +
                            modified_angularVelocity_k4[num] * a54),
                           dt));
            if (dim == 3)
              rigidBodyQuaternion[num].ToEulerAngle(
                  rigidBodyOrientation[num][0], rigidBodyOrientation[num][1],
                  rigidBodyOrientation[num][2]);
          }
          break;
        case 5:
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              rigidBodyPosition[num][j] =
                  position0[num][j] +
                  dt * (velocity_k1[num][j] * a61 + velocity_k2[num][j] * a62 +
                        velocity_k3[num][j] * a63 + velocity_k4[num][j] * a64 +
                        velocity_k5[num][j] * a65);
              if (dim == 2)
                rigidBodyOrientation[num][j] =
                    correct_radius(orientation0[num][j] +
                                   dt * (angularVelocity_k1[num][j] * a61 +
                                         angularVelocity_k2[num][j] * a62 +
                                         angularVelocity_k3[num][j] * a63 +
                                         angularVelocity_k4[num][j] * a64 +
                                         angularVelocity_k5[num][j] * a65));
            }
            rigidBodyQuaternion[num].Cross(
                quaternion0[num],
                quaternion((modified_angularVelocity_k1[num] * a61 +
                            modified_angularVelocity_k2[num] * a62 +
                            modified_angularVelocity_k3[num] * a63 +
                            modified_angularVelocity_k4[num] * a64 +
                            modified_angularVelocity_k5[num] * a65),
                           dt));
            if (dim == 3)
              rigidBodyQuaternion[num].ToEulerAngle(
                  rigidBodyOrientation[num][0], rigidBodyOrientation[num][1],
                  rigidBodyOrientation[num][2]);
          }
          break;
        case 6:
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              rigidBodyPosition[num][j] =
                  position0[num][j] +
                  dt * (velocity_k1[num][j] * b1 + velocity_k3[num][j] * b3 +
                        velocity_k4[num][j] * b4 + velocity_k5[num][j] * b5 +
                        velocity_k6[num][j] * b6);
              if (dim == 2)
                rigidBodyOrientation[num][j] =
                    correct_radius(orientation0[num][j] +
                                   dt * (angularVelocity_k1[num][j] * b1 +
                                         angularVelocity_k3[num][j] * b3 +
                                         angularVelocity_k4[num][j] * b4 +
                                         angularVelocity_k5[num][j] * b5 +
                                         angularVelocity_k6[num][j] * b6));
            }
            rigidBodyQuaternion[num].Cross(
                quaternion0[num],
                quaternion((modified_angularVelocity_k1[num] * b1 +
                            modified_angularVelocity_k3[num] * b3 +
                            modified_angularVelocity_k4[num] * b4 +
                            modified_angularVelocity_k5[num] * b5 +
                            modified_angularVelocity_k6[num] * b6),
                           dt));
            if (dim == 3)
              rigidBodyQuaternion[num].ToEulerAngle(
                  rigidBodyOrientation[num][0], rigidBodyOrientation[num][1],
                  rigidBodyOrientation[num][2]);
          }
          break;
        }

        if (rank == 0) {
          output_runge_kutta.open("traj_runge_kutta.txt", ios::app);
          output_runge_kutta << t << '\t';
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              output_runge_kutta << rigidBodyPosition[num][j] << '\t';
            }
            for (int j = 0; j < 3; j++) {
              output_runge_kutta << rigidBodyOrientation[num][j] << '\t';
            }
            for (int j = 0; j < 3; j++) {
              output_runge_kutta << rigidBodyVelocity[num][j] << '\t';
            }
            for (int j = 0; j < 3; j++) {
              output_runge_kutta << rigidBodyAngularVelocity[num][j] << '\t';
            }
          }
          output_runge_kutta << endl;
          output_runge_kutta.close();
        }

        // Check if the colloids contact with each other or move out of the
        // domain
        if (!geo_mgr->generate_uniform_particle()) {
          // halve the time step and restart the time integration
          dt = 0.5 * dt;
          acceptableTrial = false;
          err = 100;
          break;
        } else {
          acceptableTrial = true;
        }

        PetscPrintf(PETSC_COMM_WORLD, "start of refinement step\n");

        // refinement loop
        current_refinement_step = 0;
        equation_mgr->Reset();
        do {
          if (write_data == 1 || write_data == 4)
            write_refinement_data_geometry_only();
          PetscPrintf(PETSC_COMM_WORLD, "refinement level: %d\n",
                      current_refinement_step);
          equation_mgr->Update();
        } while (refinement());
        MPI_Barrier(MPI_COMM_WORLD);

        // average
        // for (int j = 0; j < 3; j++) {
        //   double average;
        //   average = (rigidBodyVelocity[0][j] + rigidBodyVelocity[1][j])
        //   / 2.0; rigidBodyVelocity[0][j] -= average; rigidBodyVelocity[1][j]
        //   -= average; average = (rigidBodyAngularVelocity[0][j] +
        //              rigidBodyAngularVelocity[1][j]) /
        //             2.0;
        //   rigidBodyAngularVelocity[0][j] = average;
        //   rigidBodyAngularVelocity[1][j] = average;
        // }

        switch (i) {
        case 1:
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              velocity_k2[num][j] = rigidBodyVelocity[num][j];
              angularVelocity_k2[num][j] = rigidBodyAngularVelocity[num][j];
            }
            modified_angularVelocity_k2[num] =
                dexpinv(modified_angularVelocity_k1[num] * a21 * dt,
                        rigidBodyAngularVelocity[num]);
          }
          break;
        case 2:
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              velocity_k3[num][j] = rigidBodyVelocity[num][j];
              angularVelocity_k3[num][j] = rigidBodyAngularVelocity[num][j];
            }
            modified_angularVelocity_k3[num] =
                dexpinv((modified_angularVelocity_k1[num] * a31 +
                         modified_angularVelocity_k2[num] * a32) *
                            dt,
                        rigidBodyAngularVelocity[num]);
          }
          break;
        case 3:
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              velocity_k4[num][j] = rigidBodyVelocity[num][j];
              angularVelocity_k4[num][j] = rigidBodyAngularVelocity[num][j];
            }
            modified_angularVelocity_k4[num] =
                dexpinv((modified_angularVelocity_k1[num] * a41 +
                         modified_angularVelocity_k2[num] * a42 +
                         modified_angularVelocity_k3[num] * a43) *
                            dt,
                        rigidBodyAngularVelocity[num]);
          }
          break;
        case 4:
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              velocity_k5[num][j] = rigidBodyVelocity[num][j];
              angularVelocity_k5[num][j] = rigidBodyAngularVelocity[num][j];
            }
            modified_angularVelocity_k5[num] =
                dexpinv((modified_angularVelocity_k1[num] * a51 +
                         modified_angularVelocity_k2[num] * a52 +
                         modified_angularVelocity_k3[num] * a53 +
                         modified_angularVelocity_k4[num] * a54) *
                            dt,
                        rigidBodyAngularVelocity[num]);
          }
          break;
        case 5:
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              velocity_k6[num][j] = rigidBodyVelocity[num][j];
              angularVelocity_k6[num][j] = rigidBodyAngularVelocity[num][j];
            }
            modified_angularVelocity_k6[num] =
                dexpinv((modified_angularVelocity_k1[num] * a61 +
                         modified_angularVelocity_k2[num] * a62 +
                         modified_angularVelocity_k3[num] * a63 +
                         modified_angularVelocity_k4[num] * a64 +
                         modified_angularVelocity_k5[num] * a65) *
                            dt,
                        rigidBodyAngularVelocity[num]);
          }
          break;
        case 6:
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              velocity_k7[num][j] = rigidBodyVelocity[num][j];
              angularVelocity_k7[num][j] = rigidBodyAngularVelocity[num][j];
            }
            modified_angularVelocity_k7[num] =
                dexpinv((modified_angularVelocity_k1[num] * b1 +
                         modified_angularVelocity_k3[num] * b3 +
                         modified_angularVelocity_k4[num] * b4 +
                         modified_angularVelocity_k5[num] * b5 +
                         modified_angularVelocity_k6[num] * b6) *
                            dt,
                        rigidBodyAngularVelocity[num]);
          }
          break;
        }
      }

      // estimate local error
      if (acceptableTrial) {
        double norm = 0.0;
        err = 0.0;
        for (int num = 0; num < numRigidBody; num++) {
          for (int j = 0; j < 2; j++) {
            double velocity_err =
                dt * (dc1 * velocity_k1[num][j] + dc3 * velocity_k3[num][j] +
                      dc4 * velocity_k4[num][j] + dc5 * velocity_k5[num][j] +
                      dc6 * velocity_k6[num][j] + dc7 * velocity_k7[num][j]);

            err += velocity_err * velocity_err;
            norm += rigidBodyPosition[num][j] * rigidBodyPosition[num][j];
          }
          // double angularVelocity_err =
          //     dt * (dc1 * angularVelocity_k1[num][0] +
          //           dc3 * angularVelocity_k3[num][0] +
          //           dc4 * angularVelocity_k4[num][0] +
          //           dc5 * angularVelocity_k5[num][0] +
          //           dc6 * angularVelocity_k6[num][0] +
          //           dc7 * angularVelocity_k7[num][0]);

          // err += angularVelocity_err * angularVelocity_err;
          // norm += rigidBodyOrientation[num][0] *
          // rigidBodyOrientation[num][0];
        }
        err = sqrt(err / norm);

        if (err > rtol) {
          noFail = false;

          PetscPrintf(PETSC_COMM_WORLD,
                      "Current time step test failed. Error is %f\n", err);
          // increase time step
          dt = dt * max(0.8 * pow(err / rtol, -0.2), 0.1);
          dt = max(dt, dtMin);
        } else {
          PetscPrintf(PETSC_COMM_WORLD,
                      "Current time step test succeeded. Error is %f\n", err);
        }
      }
    }

    PetscPrintf(PETSC_COMM_WORLD, "\n=================================\n");
    PetscPrintf(PETSC_COMM_WORLD, "==== End of time integration ====\n");
    PetscPrintf(PETSC_COMM_WORLD, "=================================\n\n");

    // output current time step result
    if (rank == 0) {
      output.open(trajectory_output_file_name, ios::app);

      const double bi12 = -static_cast<double>(183) / static_cast<double>(64);
      const double bi13 = static_cast<double>(37) / static_cast<double>(12);
      const double bi14 = -static_cast<double>(145) / static_cast<double>(128);
      const double bi32 = static_cast<double>(1500) / static_cast<double>(371);
      const double bi33 = -static_cast<double>(1000) / static_cast<double>(159);
      const double bi34 = static_cast<double>(1000) / static_cast<double>(371);
      const double bi42 = -static_cast<double>(125) / static_cast<double>(32);
      const double bi43 = static_cast<double>(125) / static_cast<double>(12);
      const double bi44 = -static_cast<double>(375) / static_cast<double>(64);
      const double bi52 = static_cast<double>(9477) / static_cast<double>(3392);
      const double bi53 = -static_cast<double>(729) / static_cast<double>(106);
      const double bi54 =
          static_cast<double>(25515) / static_cast<double>(6784);
      const double bi62 = -static_cast<double>(11) / static_cast<double>(7);
      const double bi63 = static_cast<double>(11) / static_cast<double>(3);
      const double bi64 = -static_cast<double>(55) / static_cast<double>(28);
      const double bi72 = static_cast<double>(3) / static_cast<double>(2);
      const double bi73 = -static_cast<double>(4);
      const double bi74 = static_cast<double>(5) / static_cast<double>(2);

      // refined output
      int refine_num = 4;
      for (int ite = 1; ite < refine_num; ite++) {
        double sj = double(ite) / double(refine_num);
        double sj2 = sj * sj;
        double bs1 = (sj + sj2 * (bi12 + sj * (bi13 + bi14 * sj)));
        double bs3 = (sj2 * (bi32 + sj * (bi33 + bi34 * sj)));
        double bs4 = (sj2 * (bi42 + sj * (bi43 + bi44 * sj)));
        double bs5 = (sj2 * (bi52 + sj * (bi53 + bi54 * sj)));
        double bs6 = (sj2 * (bi62 + sj * (bi63 + bi64 * sj)));
        double bs7 = (sj2 * (bi72 + sj * (bi73 + bi74 * sj)));

        vector<Vec3> refinedRigidBodyPosition(numRigidBody);
        vector<Vec3> refinedRigidBodyOrientation(numRigidBody);
        vector<quaternion> refinedRigidBodyQuaternion(numRigidBody);

        for (int num = 0; num < numRigidBody; num++) {
          for (int j = 0; j < 3; j++) {
            refinedRigidBodyPosition[num][j] =
                position0[num][j] +
                dt * (velocity_k1[num][j] * bs1 + velocity_k3[num][j] * bs3 +
                      velocity_k4[num][j] * bs4 + velocity_k5[num][j] * bs5 +
                      velocity_k6[num][j] * bs6 + velocity_k7[num][j] * bs7);
            if (dim == 2)
              refinedRigidBodyOrientation[num][j] =
                  correct_radius(orientation0[num][j] +
                                 dt * (angularVelocity_k1[num][j] * bs1 +
                                       angularVelocity_k3[num][j] * bs3 +
                                       angularVelocity_k4[num][j] * bs4 +
                                       angularVelocity_k5[num][j] * bs5 +
                                       angularVelocity_k6[num][j] * bs6 +
                                       angularVelocity_k7[num][j] * bs7));
          }
          refinedRigidBodyQuaternion[num].Cross(
              quaternion0[num],
              quaternion((modified_angularVelocity_k1[num] * bs1 +
                          modified_angularVelocity_k3[num] * bs3 +
                          modified_angularVelocity_k4[num] * bs4 +
                          modified_angularVelocity_k5[num] * bs5 +
                          modified_angularVelocity_k6[num] * bs6 +
                          modified_angularVelocity_k7[num] * bs7),
                         dt));
          if (dim == 3)
            refinedRigidBodyQuaternion[num].ToEulerAngle(
                refinedRigidBodyOrientation[num][0],
                refinedRigidBodyOrientation[num][1],
                refinedRigidBodyOrientation[num][2]);
        }

        output << t + dt * sj << '\t';
        for (int num = 0; num < numRigidBody; num++) {
          for (int j = 0; j < 3; j++) {
            output << refinedRigidBodyPosition[num][j] << '\t';
          }
          for (int j = 0; j < 3; j++) {
            output << refinedRigidBodyOrientation[num][j] << '\t';
          }
        }
        output << endl;
      }

      output << t + dt << '\t';
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

      outputVelocity.open(velocity_output_file_name, ios::app);
      outputVelocity << t + dt << '\t';
      for (int num = 0; num < numRigidBody; num++) {
        for (int j = 0; j < 3; j++) {
          outputVelocity << rigidBodyVelocity[num][j] << '\t';
        }
        for (int j = 0; j < 3; j++) {
          outputVelocity << rigidBodyAngularVelocity[num][j] << '\t';
        }
      }
      outputVelocity << endl;
      outputVelocity.close();
    }

    // increase time
    t += dt;

    // increase time step
    if (noFail) {
      double temp = 1.25 * pow(err / rtol, 0.2);
      if (temp > 0.2) {
        dt = dt / temp;
      } else {
        dt *= 5.0;
      }
    }

    dt = min(dt, max_dt);

    // reset k1 for next step
    for (int num = 0; num < numRigidBody; num++) {
      for (int j = 0; j < 3; j++) {
        velocity_k1[num][j] = velocity_k7[num][j];
        angularVelocity_k1[num][j] = angularVelocity_k7[num][j];
      }
      modified_angularVelocity_k1[num] = angularVelocity_k7[num];
    }

    if (write_data == 2 || write_data == 4) {
      write_time_step_data();
    }
  }
}

void gmls_solver::implicit_midpoint_integration() {
  vector<Vec3> &rigidBodyPosition = rb_mgr->get_position();
  vector<Vec3> &rigidBodyOrientation = rb_mgr->get_orientation();
  vector<quaternion> &rigidBodyQuaternion = rb_mgr->get_quaternion();
  vector<Vec3> &rigidBodyVelocity = rb_mgr->get_velocity();
  vector<Vec3> &rigidBodyAngularVelocity = rb_mgr->get_angular_velocity();
  vector<Vec3> &rigidBodyForce = rb_mgr->get_force();
  vector<Vec3> &rigidBodyTorque = rb_mgr->get_torque();

  int numRigidBody = rb_mgr->get_rigid_body_num();

  int rigid_body_dof = (dim == 3) ? 6 : 3;
  int translation_dof = dim;
  int rotation_dof = (dim == 3) ? 3 : 1;

  // setup snes
  SNES snes;
  SNESCreate(MPI_COMM_WORLD, &snes);
  Vec x, y, z;
  Mat J;

  int local_vec_size;
  if (rank == size - 1)
    local_vec_size = numRigidBody * rigid_body_dof;
  else
    local_vec_size = 0;

  VecCreate(PETSC_COMM_WORLD, &x);
  VecSetSizes(x, local_vec_size, numRigidBody * rigid_body_dof);
  VecSetFromOptions(x);
  VecCreate(PETSC_COMM_WORLD, &y);
  VecSetSizes(y, local_vec_size, numRigidBody * rigid_body_dof);
  VecSetFromOptions(y);

  MatCreate(PETSC_COMM_WORLD, &J);
  MatSetSizes(J, local_vec_size, local_vec_size, numRigidBody * rigid_body_dof,
              numRigidBody * rigid_body_dof);
  MatSetUp(J);

  SNESSetFunction(snes, y, implicit_midpoint_integration_sub_wrapper, this);
  SNESSetJacobian(snes, J, J, SNESComputeJacobianDefault, this);
  SNESSetConvergenceTest(snes, SNESConvergedDefault, NULL, NULL);

  KSP ksp;
  PC pc;
  SNESGetKSP(snes, &ksp);
  KSPGetPC(ksp, &pc);
  KSPSetType(ksp, KSPGMRES);

  DM dm;
  DMRedundantCreate(MPI_COMM_WORLD, size - 1, numRigidBody * rigid_body_dof,
                    &dm);
  DMSetUp(dm);
  SNESSetDM(snes, dm);

  SNESSetFromOptions(snes);
  SNESSetUp(snes);

  rtol = 1e-5;
  atol = 1e-10;
  dt = max_dt;
  t = 0;
  dtMin = 1e-10;

  position0.resize(numRigidBody);
  orientation0.resize(numRigidBody);
  quaternion0.resize(numRigidBody);

  ofstream output;
  if (rank == 0) {
    output.open(trajectory_output_file_name, ios::trunc);
    output << t << '\t';
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

  int timeStepCounter = 0;
  int totalNFunc = 0;
  int totalIteration = 0;
  while (t < final_time - 1e-5) {
    PetscReal *a;

    dt = min(dt, final_time - t);
    timeStepCounter++;

    PetscPrintf(PETSC_COMM_WORLD, "===================================\n");
    PetscPrintf(PETSC_COMM_WORLD, "==== Start of time integration ====\n");
    PetscPrintf(PETSC_COMM_WORLD, "===================================\n");
    PetscPrintf(PETSC_COMM_WORLD, "==> Current time: %f s\n", t);
    PetscPrintf(PETSC_COMM_WORLD, "==> current test time step: %f s\n", dt);

    for (int i = 0; i < numRigidBody; i++) {
      position0[i] = rigidBodyPosition[i];
    }
    for (int i = 0; i < numRigidBody; i++) {
      orientation0[i] = rigidBodyOrientation[i];
    }
    for (int i = 0; i < numRigidBody; i++) {
      quaternion0[i] = rigidBodyQuaternion[i];
    }

    // get the velocity at the local step
    geo_mgr->generate_uniform_particle();
    current_refinement_step = 0;
    equation_mgr->Reset();
    do {
      if (write_data == 1 || write_data == 4)
        write_refinement_data_geometry_only();
      PetscPrintf(PETSC_COMM_WORLD, "refinement level: %d\n",
                  current_refinement_step);
      equation_mgr->Update();
    } while (refinement());
    MPI_Barrier(MPI_COMM_WORLD);

    // move a half step
    for (int num = 0; num < numRigidBody; num++) {
      for (int j = 0; j < 3; j++) {
        rigidBodyPosition[num][j] =
            position0[num][j] + dt * rigidBodyVelocity[num][j] * 0.5;
      }
      if (dim == 2) {
        rigidBodyOrientation[num][2] =
            orientation0[num][2] + dt * rigidBodyAngularVelocity[num][0] * 0.5;
      }
      if (dim == 3) {
        rigidBodyQuaternion[num].Cross(
            quaternion0[num],
            quaternion(rigidBodyAngularVelocity[num], 0.5 * dt));
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // get an initial guess
    geo_mgr->generate_uniform_particle();
    current_refinement_step = 0;
    equation_mgr->Reset();
    do {
      if (write_data == 1 || write_data == 4)
        write_refinement_data_geometry_only();
      PetscPrintf(PETSC_COMM_WORLD, "refinement level: %d\n",
                  current_refinement_step);
      equation_mgr->Update();
    } while (refinement());
    MPI_Barrier(MPI_COMM_WORLD);

    // set the initial guess
    VecGetArray(x, &a);
    if (rank == size - 1) {
      for (int i = 0; i < numRigidBody; i++) {
        for (int j = 0; j < translation_dof; j++) {
          a[i * rigid_body_dof + j] = rigidBodyVelocity[i][j];
        }
        for (int j = 0; j < rotation_dof; j++) {
          a[i * rigid_body_dof + translation_dof + j] =
              rigidBodyAngularVelocity[i][j];
        }
      }
    }
    VecRestoreArray(x, &a);

    SNESSolve(snes, NULL, x);
    PetscInt iter, nfuncs;
    SNESGetIterationNumber(snes, &iter);
    SNESGetNumberFunctionEvals(snes, &nfuncs);

    totalNFunc += nfuncs;
    totalIteration += iter;

    PetscPrintf(
        PETSC_COMM_WORLD,
        "snes iteration number: %d,  number of function evaluation %d\n", iter,
        nfuncs);
    PetscPrintf(PETSC_COMM_WORLD,
                "snes average iteration number: %d,  average "
                "number of function evaluation %d\n",
                totalIteration / timeStepCounter, totalNFunc / timeStepCounter);

    vector<double> sychronize_velocity(numRigidBody * 3);
    vector<double> sychronize_angularVelocity(numRigidBody * 3);

    // sychronize velocity
    VecGetArray(x, &a);
    if (rank == size - 1) {
      for (int i = 0; i < numRigidBody; i++) {
        for (int j = 0; j < translation_dof; j++) {
          sychronize_velocity[i * 3 + j] = a[i * rigid_body_dof + j];
        }
        for (int j = 0; j < rotation_dof; j++) {
          sychronize_angularVelocity[i * 3 + j] =
              a[i * rigid_body_dof + translation_dof + j];
        }
      }
    }
    VecRestoreArray(x, &a);

    MPI_Bcast(sychronize_velocity.data(), numRigidBody * 3, MPI_DOUBLE,
              size - 1, MPI_COMM_WORLD);
    MPI_Bcast(sychronize_angularVelocity.data(), numRigidBody * 3, MPI_DOUBLE,
              size - 1, MPI_COMM_WORLD);

    for (int i = 0; i < numRigidBody; i++) {
      for (int j = 0; j < translation_dof; j++) {
        rigidBodyVelocity[i][j] = sychronize_velocity[i * 3 + j];
      }
      for (int j = 0; j < rotation_dof; j++) {
        rigidBodyAngularVelocity[i][j] = sychronize_angularVelocity[i * 3 + j];
      }
    }

    for (int num = 0; num < numRigidBody; num++) {
      for (int j = 0; j < 3; j++) {
        rigidBodyPosition[num][j] =
            position0[num][j] + dt * rigidBodyVelocity[num][j];
      }
      if (dim == 2) {
        rigidBodyOrientation[num][2] =
            orientation0[num][2] + dt * rigidBodyAngularVelocity[num][0];
      }
      if (dim == 3) {
        rigidBodyQuaternion[num].Cross(
            quaternion0[num], quaternion(rigidBodyAngularVelocity[num], dt));
        rigidBodyQuaternion[num].ToEulerAngle(rigidBodyOrientation[num][0],
                                              rigidBodyOrientation[num][1],
                                              rigidBodyOrientation[num][2]);
      }
    }

    // output current time step result
    if (rank == 0) {
      output.open(trajectory_output_file_name, ios::app);

      vector<Vec3> refinedRigidBodyPosition(numRigidBody);
      vector<Vec3> refinedRigidBodyOrientation(numRigidBody);
      vector<quaternion> refinedRigidBodyQuaternion(numRigidBody);

      for (int num = 0; num < numRigidBody; num++) {
        for (int j = 0; j < 3; j++) {
          refinedRigidBodyPosition[num][j] =
              position0[num][j] + 0.5 * dt * rigidBodyVelocity[num][j];
        }
        if (dim == 2) {
          refinedRigidBodyOrientation[num][2] =
              orientation0[num][2] +
              dt * rigidBodyAngularVelocity[num][0] * 0.5;
        }
        if (dim == 3) {
          refinedRigidBodyQuaternion[num].Cross(
              quaternion0[num],
              quaternion(rigidBodyAngularVelocity[num], 0.5 * dt));
          refinedRigidBodyQuaternion[num].ToEulerAngle(
              refinedRigidBodyOrientation[num][0],
              refinedRigidBodyOrientation[num][1],
              refinedRigidBodyOrientation[num][2]);
        }
      }

      output << t + 0.5 * dt << '\t';
      for (int num = 0; num < numRigidBody; num++) {
        for (int j = 0; j < 3; j++) {
          output << refinedRigidBodyPosition[num][j] << '\t';
        }
        for (int j = 0; j < 3; j++) {
          output << refinedRigidBodyOrientation[num][j] << '\t';
        }
      }
      output << endl;

      output << t + dt << '\t';
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

    PetscPrintf(PETSC_COMM_WORLD, "\n=================================\n");
    PetscPrintf(PETSC_COMM_WORLD, "==== End of time integration ====\n");
    PetscPrintf(PETSC_COMM_WORLD, "=================================\n\n");
    t += dt;
  }

  MatDestroy(&J);
  VecDestroy(&x);
  VecDestroy(&y);
  DMDestroy(&dm);
  SNESDestroy(&snes);
}

void gmls_solver::implicit_midpoint_integration_sub(Vec x, Vec y) {
  vector<Vec3> &rigidBodyPosition = rb_mgr->get_position();
  vector<Vec3> &rigidBodyOrientation = rb_mgr->get_orientation();
  vector<quaternion> &rigidBodyQuaternion = rb_mgr->get_quaternion();
  vector<Vec3> &rigidBodyVelocity = rb_mgr->get_velocity();
  vector<Vec3> &rigidBodyAngularVelocity = rb_mgr->get_angular_velocity();
  vector<Vec3> &rigidBodyForce = rb_mgr->get_force();
  vector<Vec3> &rigidBodyTorque = rb_mgr->get_torque();

  PetscReal norm1, norm2;
  VecNorm(x, NORM_2, &norm1);
  PetscPrintf(PETSC_COMM_WORLD, "snes norm before refinement x: %.10e\n",
              norm1);

  MPI_Barrier(MPI_COMM_WORLD);

  int numRigidBody = rb_mgr->get_rigid_body_num();

  int rigid_body_dof = (dim == 3) ? 6 : 3;
  int translation_dof = dim;
  int rotation_dof = (dim == 3) ? 3 : 1;

  vector<double> sychronize_velocity(numRigidBody * 3);
  vector<double> sychronize_angularVelocity(numRigidBody * 3);

  const PetscReal *a;
  PetscReal *b;
  // sychronize velocity
  VecGetArrayRead(x, &a);
  if (rank == size - 1) {
    for (int i = 0; i < numRigidBody; i++) {
      for (int j = 0; j < 2; j++) {
        sychronize_velocity[i * 3 + j] = a[i * rigid_body_dof + j];
      }
      for (int j = 0; j < rotation_dof; j++) {
        sychronize_angularVelocity[i * 3 + j] =
            a[i * rigid_body_dof + translation_dof + j];
      }
    }
  }
  VecRestoreArrayRead(x, &a);

  MPI_Bcast(sychronize_velocity.data(), numRigidBody * 3, MPI_DOUBLE, size - 1,
            MPI_COMM_WORLD);
  MPI_Bcast(sychronize_angularVelocity.data(), numRigidBody * 3, MPI_DOUBLE,
            size - 1, MPI_COMM_WORLD);

  for (int i = 0; i < numRigidBody; i++) {
    for (int j = 0; j < translation_dof; j++) {
      rigidBodyVelocity[i][j] = sychronize_velocity[i * 3 + j];
    }
    for (int j = 0; j < rotation_dof; j++) {
      rigidBodyAngularVelocity[i][j] = sychronize_angularVelocity[i * 3 + j];
    }
  }

  for (int num = 0; num < numRigidBody; num++) {
    for (int j = 0; j < 3; j++) {
      rigidBodyPosition[num][j] =
          position0[num][j] + dt * rigidBodyVelocity[num][j] * 0.5;
    }
    if (dim == 2) {
      rigidBodyOrientation[num][2] =
          orientation0[num][2] + dt * rigidBodyAngularVelocity[num][0] * 0.5;
    }
    if (dim == 3) {
      rigidBodyQuaternion[num].Cross(
          quaternion0[num],
          quaternion(rigidBodyAngularVelocity[num], 0.5 * dt));
    }
  }

  geo_mgr->generate_uniform_particle();
  current_refinement_step = 0;
  equation_mgr->Reset();
  do {
    if (write_data == 1 || write_data == 4)
      write_refinement_data_geometry_only();
    PetscPrintf(PETSC_COMM_WORLD, "refinement level: %d\n",
                current_refinement_step);
    equation_mgr->Update();
  } while (refinement());
  MPI_Barrier(MPI_COMM_WORLD);

  VecGetArray(y, &b);
  if (rank == size - 1) {
    for (int i = 0; i < numRigidBody; i++) {
      for (int j = 0; j < translation_dof; j++) {
        b[i * rigid_body_dof + j] = rigidBodyVelocity[i][j];
      }
      for (int j = 0; j < rotation_dof; j++) {
        b[i * rigid_body_dof + translation_dof + j] =
            rigidBodyAngularVelocity[i][j];
      }
    }
  }
  VecRestoreArray(y, &b);

  PetscReal norm;
  VecNorm(y, NORM_2, &norm);
  VecAXPY(y, -1.0, x);
  // VecScale(y, 1.0 / norm);
}

PetscErrorCode implicit_midpoint_integration_sub_wrapper(SNES snes, Vec x,
                                                         Vec y, void *ctx) {
  gmls_solver *solver = (gmls_solver *)ctx;

  solver->implicit_midpoint_integration_sub(x, y);

  PetscReal norm1, norm2;
  VecNorm(x, NORM_2, &norm1);
  VecNorm(y, NORM_2, &norm2);
  PetscPrintf(PETSC_COMM_WORLD, "snes norm x: %.10e, norm y: %.10e\n", norm1,
              norm2);

  return 0;
}