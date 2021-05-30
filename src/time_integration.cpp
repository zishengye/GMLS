#include "gmls_solver.hpp"

using namespace std;
using namespace Compadre;

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
    equation_mgr->reset();
    do {
      if (write_data == 1 || write_data == 4)
        write_refinement_data_geometry_only();
      PetscPrintf(PETSC_COMM_WORLD, "refinement level: %d\n",
                  current_refinement_step);
      equation_mgr->update();
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
  vector<vec3> &rigidBodyPosition = rb_mgr->get_position();
  vector<vec3> &rigidBodyOrientation = rb_mgr->get_orientation();
  vector<vec3> &rigidBodyVelocity = rb_mgr->get_velocity();
  vector<vec3> &rigidBodyAngularVelocity = rb_mgr->get_angular_velocity();

  int numRigidBody = rb_mgr->get_rigid_body_num();

  vector<vec3> position0(numRigidBody);
  vector<vec3> orientation0(numRigidBody);

  vector<vec3> velocity_k1(numRigidBody);
  vector<vec3> velocity_k2(numRigidBody);
  vector<vec3> velocity_k3(numRigidBody);
  vector<vec3> velocity_k4(numRigidBody);
  vector<vec3> velocity_k5(numRigidBody);
  vector<vec3> velocity_k6(numRigidBody);
  vector<vec3> velocity_k7(numRigidBody);

  vector<vec3> angularVelocity_k1(numRigidBody);
  vector<vec3> angularVelocity_k2(numRigidBody);
  vector<vec3> angularVelocity_k3(numRigidBody);
  vector<vec3> angularVelocity_k4(numRigidBody);
  vector<vec3> angularVelocity_k5(numRigidBody);
  vector<vec3> angularVelocity_k6(numRigidBody);
  vector<vec3> angularVelocity_k7(numRigidBody);

  vector<double> sychronize_velocity(numRigidBody * 3);
  vector<double> sychronize_angularVelocity(numRigidBody * 3);

  // ode45 algorithm parameter
  double t, dt, dtMin, rtol, atol, err, norm_y;
  rtol = 1e-5;
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
  equation_mgr->reset();
  do {
    if (write_data == 1 || write_data == 4) {
      write_refinement_data_geometry_only();
    }
    PetscPrintf(PETSC_COMM_WORLD, "refinement level: %d\n",
                current_refinement_step);
    equation_mgr->update();
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
  }

  // setup output file
  ofstream output;
  ofstream output_runge_kutta;
  ofstream outputVelocity;
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

  // main loop
  while (t < final_time - 1e-5) {
    bool noFail = true;
    bool acceptableTrial = false;

    // ensure end exactly at final time
    dt = min(dt, final_time - t);

    for (int num = 0; num < numRigidBody; num++) {
      for (int j = 0; j < 3; j++) {
        position0[num][j] = rigidBodyPosition[num][j];
        orientation0[num][j] = correct_radius(rigidBodyOrientation[num][j]);
      }
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
              rigidBodyOrientation[num][j] = correct_radius(
                  orientation0[num][j] + dt * angularVelocity_k1[num][j] * a21);
            }
          }
          break;
        case 2:
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              rigidBodyPosition[num][j] =
                  position0[num][j] +
                  dt * (velocity_k1[num][j] * a31 + velocity_k2[num][j] * a32);
              rigidBodyOrientation[num][j] =
                  correct_radius(orientation0[num][j] +
                                 dt * (angularVelocity_k1[num][j] * a31 +
                                       angularVelocity_k2[num][j] * a32));
            }
          }
          break;
        case 3:
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              rigidBodyPosition[num][j] =
                  position0[num][j] +
                  dt * (velocity_k1[num][j] * a41 + velocity_k2[num][j] * a42 +
                        velocity_k3[num][j] * a43);
              rigidBodyOrientation[num][j] =
                  correct_radius(orientation0[num][j] +
                                 dt * (angularVelocity_k1[num][j] * a41 +
                                       angularVelocity_k2[num][j] * a42 +
                                       angularVelocity_k3[num][j] * a43));
            }
          }
          break;
        case 4:
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              rigidBodyPosition[num][j] =
                  position0[num][j] +
                  dt * (velocity_k1[num][j] * a51 + velocity_k2[num][j] * a52 +
                        velocity_k3[num][j] * a53 + velocity_k4[num][j] * a54);
              rigidBodyOrientation[num][j] =
                  correct_radius(orientation0[num][j] +
                                 dt * (angularVelocity_k1[num][j] * a51 +
                                       angularVelocity_k2[num][j] * a52 +
                                       angularVelocity_k3[num][j] * a53 +
                                       angularVelocity_k4[num][j] * a54));
            }
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
              rigidBodyOrientation[num][j] =
                  correct_radius(orientation0[num][j] +
                                 dt * (angularVelocity_k1[num][j] * a61 +
                                       angularVelocity_k2[num][j] * a62 +
                                       angularVelocity_k3[num][j] * a63 +
                                       angularVelocity_k4[num][j] * a64 +
                                       angularVelocity_k5[num][j] * a65));
            }
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
              rigidBodyOrientation[num][j] =
                  correct_radius(orientation0[num][j] +
                                 dt * (angularVelocity_k1[num][j] * b1 +
                                       angularVelocity_k3[num][j] * b3 +
                                       angularVelocity_k4[num][j] * b4 +
                                       angularVelocity_k5[num][j] * b5 +
                                       angularVelocity_k6[num][j] * b6));
            }
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
        equation_mgr->reset();
        do {
          if (write_data == 1 || write_data == 4)
            write_refinement_data_geometry_only();
          PetscPrintf(PETSC_COMM_WORLD, "refinement level: %d\n",
                      current_refinement_step);
          equation_mgr->update();
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
          }
          break;
        case 2:
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              velocity_k3[num][j] = rigidBodyVelocity[num][j];
              angularVelocity_k3[num][j] = rigidBodyAngularVelocity[num][j];
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
          break;
        case 4:
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              velocity_k5[num][j] = rigidBodyVelocity[num][j];
              angularVelocity_k5[num][j] = rigidBodyAngularVelocity[num][j];
            }
          }
          break;
        case 5:
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              velocity_k6[num][j] = rigidBodyVelocity[num][j];
              angularVelocity_k6[num][j] = rigidBodyAngularVelocity[num][j];
            }
          }
          break;
        case 6:
          for (int num = 0; num < numRigidBody; num++) {
            for (int j = 0; j < 3; j++) {
              velocity_k7[num][j] = rigidBodyVelocity[num][j];
              angularVelocity_k7[num][j] = rigidBodyAngularVelocity[num][j];
            }
          }
          break;
        }
      }

      // estimate local error
      if (acceptableTrial) {
        double norm = 0.0;
        err = 0.0;
        for (int num = 0; num < numRigidBody; num++) {
          if (dim == 2) {
            for (int j = 0; j < 2; j++) {
              double velocity_err =
                  dt * (dc1 * velocity_k1[num][j] + dc3 * velocity_k3[num][j] +
                        dc4 * velocity_k4[num][j] + dc5 * velocity_k5[num][j] +
                        dc6 * velocity_k6[num][j] + dc7 * velocity_k7[num][j]);

              err += velocity_err * velocity_err;
              norm += rigidBodyPosition[num][j] * rigidBodyPosition[num][j];
            }
            double angularVelocity_err =
                dt * (dc1 * angularVelocity_k1[num][0] +
                      dc3 * angularVelocity_k3[num][0] +
                      dc4 * angularVelocity_k4[num][0] +
                      dc5 * angularVelocity_k5[num][0] +
                      dc6 * angularVelocity_k6[num][0] +
                      dc7 * angularVelocity_k7[num][0]);

            err += angularVelocity_err * angularVelocity_err;
            norm += rigidBodyOrientation[num][0] * rigidBodyOrientation[num][0];
          }
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

        vector<vec3> refinedRigidBodyPosition(numRigidBody);
        vector<vec3> refinedRigidBodyOrientation(numRigidBody);

        for (int num = 0; num < numRigidBody; num++) {
          for (int j = 0; j < 3; j++) {
            refinedRigidBodyPosition[num][j] =
                position0[num][j] +
                dt * (velocity_k1[num][j] * bs1 + velocity_k3[num][j] * bs3 +
                      velocity_k4[num][j] * bs4 + velocity_k5[num][j] * bs5 +
                      velocity_k6[num][j] * bs6 + velocity_k7[num][j] * bs7);
            refinedRigidBodyOrientation[num][j] = correct_radius(
                orientation0[num][j] + dt * (angularVelocity_k1[num][j] * bs1 +
                                             angularVelocity_k3[num][j] * bs3 +
                                             angularVelocity_k4[num][j] * bs4 +
                                             angularVelocity_k5[num][j] * bs5 +
                                             angularVelocity_k6[num][j] * bs6 +
                                             angularVelocity_k7[num][j] * bs7));
          }
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
    }

    if (write_data == 2 || write_data == 4) {
      write_time_step_data();
    }
  }
}