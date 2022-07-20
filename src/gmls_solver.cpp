#include "gmls_solver.hpp"

#include <iostream>

using namespace std;

gmls_solver::gmls_solver(int argc, char **argv) {
  current_refinement_step = 0;
  // [default setup]
  initialization_status = false;

  // change stdout to log file

  MPI_Barrier(MPI_COMM_WORLD);

  // MPI setup
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  // serial_operation([processor_name, this]() {
  //   cout << "[Process " << __myID << "], on " << processor_name << endl;
  // });

  string outputFileName;
  if (SearchCommand<string>(argc, argv, "-OutputFile", outputFileName) == 0) {
    if (rank == 0) {
      freopen(outputFileName.data(), "w", stdout);
    }
  }

  // default dimension is 3
  if (SearchCommand<int>(argc, argv, "-Dim", dim) == 1) {
    dim = 3;
  } else {
    if (dim > 3 || dim < 1) {
      PetscPrintf(PETSC_COMM_WORLD, "Wrong dimension!\n");
      return;
    }
  }

  // default spacing is 0.1
  if (SearchCommand<double>(argc, argv, "-Spacing", spacing) == 1) {
    spacing = 0.1;
  }

  // default time integration method is forward euler method
  if (SearchCommand<string>(argc, argv, "-TimeIntegration",
                            time_integration_method) == 1) {
    time_integration_method = "ForwardEuler";
  } else {
    // TODO: check the correctness of the command
  }

  // default governing equation is Navier-Stokes equation
  if (SearchCommand<string>(argc, argv, "-EquationType", equation_type) == 1) {
    equation_type = "Navier-Stokes";
  } else {
    // TODO: check the correctness of the command
  }

  // default particle scheme is Eulerian particles
  if ((SearchCommand<string>(argc, argv, "-Scheme", scheme_type)) == 1) {
    scheme_type = "Eulerian";
  } else {
    // TODO: check the correctness of the command
  }

  // defalut discretization order is 2
  if ((SearchCommand<int>(argc, argv, "-PolynomialOrder", polynomial_order)) ==
      1) {
    polynomial_order = 2;
  } else {
    // TODO: check the correctness of the command
  }

  if ((SearchCommand<int>(argc, argv, "-WeightingFunctionOrder",
                          weight_func_order)) == 1) {
    weight_func_order = 4;
  } else {
    // TODO: check the correctness of the command
  }

  // default serial output
  if ((SearchCommand<int>(argc, argv, "-WriteData", write_data)) == 1) {
    write_data = 0;
  }

  string refinement_field_name;
  if ((SearchCommand<int>(argc, argv, "-Refinement", refinement_method)) == 1) {
    refinement_method = 0;
  } else {
    if ((SearchCommand<double>(argc, argv, "-RefinementTolerance",
                               refinement_tolerance)) == 1) {
      refinement_tolerance = 1e-3;
    }
    if ((SearchCommand<int>(argc, argv, "-MaxRefinementLevel",
                            max_refinement_level)) == 1) {
      max_refinement_level = 4;
    }

    if ((SearchCommand<string>(argc, argv, "-RefinementField",
                               refinement_field_name)) == 1) {
      refinement_field = 1;
    } else {
      refinement_field = 1;
      if (refinement_field_name == "Velocity") {
        refinement_field = 1;
      }
      if (refinement_field_name == "Pressure") {
        refinement_field = 2;
      }
    }
  }

  int min_count, max_count, stride;
  if (refinement_method == UNIFORM_REFINE) {
    if ((SearchCommand<int>(argc, argv, "-MinCount", min_count)) == 1) {
      min_count = 0;
    }
    if ((SearchCommand<int>(argc, argv, "-MaxCount", max_count)) == 1) {
      max_count = 0;
    }
    if ((SearchCommand<int>(argc, argv, "-Stride", stride)) == 1) {
      stride = 0;
    }
  } else {
    stride = 0;
  }

  if ((SearchCommand<double>(argc, argv, "-MaxParticleNum",
                             max_particle_num)) == 1) {
    max_particle_num = 1e6;
  }

  // [optional command]
  if (SearchCommand<string>(argc, argv, "-rigid_body_input",
                            rigid_body_input_file_name) == 0) {
    rigid_body_inclusion = true;
    if (SearchCommand<string>(argc, argv, "-traj_output",
                              trajectory_output_file_name) == 1) {
      trajectory_output_file_name = "traj";
    }
    if (SearchCommand<string>(argc, argv, "-vel_output",
                              velocity_output_file_name) == 1) {
      velocity_output_file_name = "vel";
    }
    if (SearchCommand<string>(argc, argv, "-force_output",
                              force_output_file_name) == 1) {
      force_output_file_name = "force";
    }
  } else {
    rigid_body_inclusion = false;
  }

  // [parameter must appear in command]
  // final time
  if ((SearchCommand<double>(argc, argv, "-ft", final_time)) == 1) {
    return;
  } else if (final_time < 0.0) {
    return;
  }

  // time step
  if ((SearchCommand<double>(argc, argv, "-dt", max_dt)) == 1) {
    return;
  } else if (max_dt < 0.0) {
    return;
  }

  // kinetic viscosity distance
  if ((SearchCommand<double>(argc, argv, "-eta", eta)) == 1) {
    eta = 1.0;
    return;
  } else if (eta < 0.0) {
    return;
  }

  if ((SearchCommand<int>(argc, argv, "-BatchSize", batch_size)) == 1) {
    return;
  } else if (batch_size < 0) {
    return;
  }

  int compress_memory = 0;
  if ((SearchCommand<int>(argc, argv, "-CompressMemory", compress_memory)) ==
      1) {
    compress_memory = 0;
  } else if (compress_memory < 0) {
    return;
  }

  if ((SearchCommand<int>(argc, argv, "-Viewer", use_viewer)) == 1) {
    use_viewer = 0;
  }

  if ((SearchCommand<double>(argc, argv, "-EpsilonMultiplier",
                             epsilon_multiplier)) == 1) {
    epsilon_multiplier = 0.0;
  }

  string geometry_input_file_name;
  if ((SearchCommand<string>(argc, argv, "-GeometryInput",
                             geometry_input_file_name)) == 1) {
    geometry_input_file_name = "";
  }

  initialization_status = true;

  MPI_Barrier(MPI_COMM_WORLD);

  geo_mgr = make_shared<ParticleGeometry>();
  rb_mgr = make_shared<rigid_body_manager>();

  geo_mgr->init(dim, STANDARD_PROBLEM, refinement_method, spacing,
                epsilon_multiplier, min_count, max_count, stride,
                geometry_input_file_name);
  rb_mgr->init(rigid_body_input_file_name, dim);
  geo_mgr->init_rigid_body(rb_mgr);

  // equation type selection and initialization
  if (equation_type == "Stokes") {
    equation_mgr = make_shared<StokesEquation>();
  }

  if (equation_type == "Poisson") {
  }

  if (equation_type == "Poisson") {
  }

  if (equation_type == "Diffusion") {
  }

  equation_mgr->Init(geo_mgr, rb_mgr, polynomial_order, dim, refinement_field,
                     epsilon_multiplier, eta);

  if (use_viewer == 1)
    equation_mgr->SetViewer();

  // [summary of problem setup]

  PetscPrintf(PETSC_COMM_WORLD, "===============================\n");
  PetscPrintf(PETSC_COMM_WORLD, "==== Problem setup summary ====\n");
  PetscPrintf(PETSC_COMM_WORLD, "===============================\n");
  PetscPrintf(PETSC_COMM_WORLD, "==> Dimension: %d\n", dim);
  PetscPrintf(PETSC_COMM_WORLD, "==> Governing equation: %s\n",
              equation_type.c_str());
  PetscPrintf(PETSC_COMM_WORLD, "==> Time integration scheme: %s\n",
              time_integration_method.c_str());
  PetscPrintf(PETSC_COMM_WORLD, "==> Time interval: %fs\n", max_dt);
  PetscPrintf(PETSC_COMM_WORLD, "==> Final time: %fs\n", final_time);
  PetscPrintf(PETSC_COMM_WORLD, "==> Polynomial order: %d\n", polynomial_order);
  PetscPrintf(PETSC_COMM_WORLD, "==> Kinetic viscosity: %f\n", eta);
  if (refinement_method != 0) {
    PetscPrintf(PETSC_COMM_WORLD, "==> Refinement: on\n");
    PetscPrintf(PETSC_COMM_WORLD, "==> Refinement method:  %d\n",
                refinement_method);
    PetscPrintf(PETSC_COMM_WORLD, "==> Refinement tolerance:  %f\n",
                refinement_tolerance);
    PetscPrintf(PETSC_COMM_WORLD, "==> Refinement field: %s(%d)\n",
                refinement_field_name.c_str(), refinement_field);
    PetscPrintf(PETSC_COMM_WORLD, "==> Maximum refinement level: %d\n",
                max_refinement_level);
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "==> Refinement: off\n");
  }
  PetscPrintf(PETSC_COMM_WORLD, "==> Total number of rigid bodys: %d\n",
              rb_mgr->get_rigid_body_num());
}