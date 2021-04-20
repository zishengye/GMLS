#include "gmls_solver.hpp"

#include <fstream>
#include <string>

using namespace std;

void gmls_solver::write_time_step_data() {
  static int write_step = 0;
  vector<vec3> &coord = *(geo_mgr->get_current_work_particle_coord());
  vector<vec3> &normal = *(geo_mgr->get_current_work_particle_normal());
  vector<double> &spacing = *(geo_mgr->get_current_work_particle_spacing());
  vector<int> &particle_type = *(geo_mgr->get_current_work_particle_type());
  vector<int> &num_neighbor =
      *(geo_mgr->get_current_work_particle_num_neighbor());

  int local_particle_num;
  int global_particle_num;

  local_particle_num = coord.size();
  MPI_Allreduce(&local_particle_num, &global_particle_num, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  // int thread = 3;

  // master_operation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_step" + to_string(write_step) + ".vtk",
  //   ios::trunc); file << "# vtk DataFile Version 2.0" << endl; file <<
  //   "particlePositions" << endl; file << "ASCII" << endl; file << "DATASET
  //   POLYDATA " << endl; file << " POINTS " <<
  //   this->__backgroundParticle.coord.size() << " float"
  //        << endl;
  //   file.close();
  // });

  // master_operation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_step" + to_string(write_step) + ".vtk",
  //   ios::app); for (size_t i = 0; i <
  //   this->__backgroundParticle.coord.size(); i++) {
  //     file << __backgroundParticle.coord[i][0] << ' '
  //          << __backgroundParticle.coord[i][1] << ' '
  //          << __backgroundParticle.coord[i][2] << endl;
  //   }
  //   file.close();
  // });

  // master_operation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_step" + to_string(write_step) + ".vtk",
  //   ios::app); file << "POINT_DATA " <<
  //   this->__backgroundParticle.coord.size() << endl; file.close();
  // });

  // master_operation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_step" + to_string(write_step) + ".vtk",
  //   ios::app); file << "SCALARS index int 1" << endl; file << "LOOKUP_TABLE
  //   default" << endl; file.close();
  // });

  // master_operation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_step" + to_string(write_step) + ".vtk",
  //   ios::app); for (size_t i = 0; i <
  //   this->__backgroundParticle.coord.size(); i++) {
  //     file << __backgroundParticle.index[i] << endl;
  //   }
  //   file.close();
  // });

  // master_operation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_local_step" + to_string(write_step) + ".vtk",
  //             ios::trunc);
  //   file << "# vtk DataFile Version 2.0" << endl;
  //   file << "particlePositions" << endl;
  //   file << "ASCII" << endl;
  //   file << "DATASET POLYDATA " << endl;
  //   file << " POINTS " << this->__particle.localParticleNum << " float" <<
  //   endl; file.close();
  // });

  // master_operation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_local_step" + to_string(write_step) + ".vtk",
  //             ios::app);
  //   for (size_t i = 0; i < this->__particle.localParticleNum; i++) {
  //     file << __particle.X[i][0] << ' ' << __particle.X[i][1] << ' '
  //          << __particle.X[i][2] << endl;
  //   }
  //   file.close();
  // });

  // master_operation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_local_step" + to_string(write_step) + ".vtk",
  //             ios::app);
  //   file << "POINT_DATA " << this->__particle.localParticleNum << endl;
  //   file.close();
  // });

  // master_operation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_local_step" + to_string(write_step) + ".vtk",
  //             ios::app);
  //   file << "SCALARS ID int 1" << endl;
  //   file << "LOOKUP_TABLE default" << endl;
  //   file.close();
  // });

  // master_operation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_local_step" + to_string(write_step) + ".vtk",
  //             ios::app);
  //   for (size_t i = 0; i < this->__particle.localParticleNum; i++) {
  //     file << __particle.particleType[i] << endl;
  //   }
  //   file.close();
  // });

  master_operation(0, [global_particle_num]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(write_step) + ".vtk", ios::trunc);
    file << "# vtk DataFile Version 2.0" << endl;
    file << "particlePositions" << endl;
    file << "ASCII" << endl;
    file << "DATASET POLYDATA " << endl;
    file << " POINTS " << global_particle_num << " float" << endl;
    file.close();
  });

  serial_operation([coord]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(write_step) + ".vtk", ios::app);
    for (size_t i = 0; i < coord.size(); i++) {
      file << fixed << setprecision(10) << coord[i][0] << ' ' << coord[i][1]
           << ' ' << coord[i][2] << endl;
    }
    file.close();
  });

  master_operation(0, [global_particle_num]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(write_step) + ".vtk", ios::app);
    file << "POINT_DATA " << global_particle_num << endl;
    file.close();
  });

  master_operation(0, []() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(write_step) + ".vtk", ios::app);
    file << "SCALARS ID int 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  serial_operation([particle_type]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(write_step) + ".vtk", ios::app);
    for (size_t i = 0; i < particle_type.size(); i++) {
      file << particle_type[i] << endl;
    }
    file.close();
  });

  master_operation(0, []() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(write_step) + ".vtk", ios::app);
    file << "SCALARS nn int 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  serial_operation([num_neighbor]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(write_step) + ".vtk", ios::app);
    for (size_t i = 0; i < num_neighbor.size(); i++) {
      file << num_neighbor[i] << endl;
    }
    file.close();
  });

  master_operation(0, []() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(write_step) + ".vtk", ios::app);
    file << "SCALARS d float 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  serial_operation([spacing]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(write_step) + ".vtk", ios::app);
    for (size_t i = 0; i < spacing.size(); i++) {
      file << spacing[i] << endl;
    }
    file.close();
  });

  master_operation(0, []() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(write_step) + ".vtk", ios::app);
    file << "SCALARS n float 3" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  serial_operation([normal]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(write_step) + ".vtk", ios::app);
    for (size_t i = 0; i < normal.size(); i++) {
      file << normal[i][0] << ' ' << normal[i][1] << ' ' << normal[i][2]
           << endl;
    }
    file.close();
  });

  master_operation(0, [this]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(write_step) + ".vtk", ios::app);
    file << "SCALARS domain int 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  serial_operation([particle_type, this]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(write_step) + ".vtk", ios::app);
    for (size_t i = 0; i < particle_type.size(); i++) {
      file << rank << endl;
    }
    file.close();
  });

  // physical data output, equation type depedent
  if (equation_type == "Diffusion") {
  }

  if (equation_type == "Stokes") {
    vector<vec3> &velocity = equation_mgr->get_velocity();
    vector<double> &pressure = equation_mgr->get_pressure();
    vector<double> &error = equation_mgr->get_error();

    master_operation(0, []() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(write_step) + ".vtk", ios::app);
      file << "SCALARS p float 1" << endl;
      file << "LOOKUP_TABLE default " << endl;
      file.close();
    });

    serial_operation([pressure]() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(write_step) + ".vtk", ios::app);
      for (size_t i = 0; i < pressure.size(); i++) {
        file << ((abs(pressure[i]) > 1e-10) ? pressure[i] : 0.0) << endl;
      }
      file.close();
    });

    master_operation(0, [this]() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(write_step) + ".vtk", ios::app);
      file << "SCALARS u float " + to_string(dim) << endl;
      file << "LOOKUP_TABLE default" << endl;
      file.close();
    });

    serial_operation([velocity, this]() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(write_step) + ".vtk", ios::app);
      for (size_t i = 0; i < velocity.size(); i++) {
        for (int axes = 0; axes < dim; axes++) {
          file << ((abs(velocity[i][axes]) > 1e-10) ? velocity[i][axes] : 0.0)
               << ' ';
        }
        file << endl;
      }
      file.close();
    });

    master_operation(0, []() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(write_step) + ".vtk", ios::app);
      file << "SCALARS err float 1" << endl;
      file << "LOOKUP_TABLE default " << endl;
      file.close();
    });

    serial_operation([error]() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(write_step) + ".vtk", ios::app);
      for (size_t i = 0; i < error.size(); i++) {
        file << ((abs(error[i]) > 1e-10) ? error[i] : 0.0) << endl;
      }
      file.close();
    });

    if (refinement_field == 1) {
      // gradient of velocity
      auto &gradient = equation_mgr->get_gradient();
      master_operation(0, [this]() {
        ofstream file;
        file.open("./vtk/output_step" + to_string(write_step) + ".vtk",
                  ios::app);
        file << "SCALARS dUX float " + to_string(dim) << endl;
        file << "LOOKUP_TABLE default" << endl;
        file.close();
      });

      serial_operation([gradient, this]() {
        ofstream file;
        file.open("./vtk/output_step" + to_string(write_step) + ".vtk",
                  ios::app);
        for (size_t i = 0; i < gradient.size(); i++) {
          for (int axes = 0; axes < dim; axes++) {
            file << ((abs(gradient[i][axes]) > 1e-10) ? gradient[i][axes] : 0.0)
                 << ' ';
          }
          file << endl;
        }
        file.close();
      });

      if (dim > 1) {
        master_operation(0, [this]() {
          ofstream file;
          file.open("./vtk/output_step" + to_string(write_step) + ".vtk",
                    ios::app);
          file << "SCALARS dUY float " + to_string(dim) << endl;
          file << "LOOKUP_TABLE default" << endl;
          file.close();
        });

        serial_operation([gradient, this]() {
          ofstream file;
          file.open("./vtk/output_step" + to_string(write_step) + ".vtk",
                    ios::app);
          for (size_t i = 0; i < gradient.size(); i++) {
            for (int axes = 0; axes < dim; axes++) {
              file << ((abs(gradient[i][axes + dim]) > 1e-10)
                           ? gradient[i][axes]
                           : 0.0)
                   << ' ';
            }
            file << endl;
          }
          file.close();
        });
      }

      if (dim > 2) {
        master_operation(0, [this]() {
          ofstream file;
          file.open("./vtk/output_step" + to_string(write_step) + ".vtk",
                    ios::app);
          file << "SCALARS dUZ float " + to_string(dim) << endl;
          file << "LOOKUP_TABLE default" << endl;
          file.close();
        });

        serial_operation([gradient, this]() {
          ofstream file;
          file.open("./vtk/output_step" + to_string(write_step) + ".vtk",
                    ios::app);
          for (size_t i = 0; i < gradient.size(); i++) {
            for (int axes = 0; axes < dim; axes++) {
              file << ((abs(gradient[i][axes + dim * 2]) > 1e-10)
                           ? gradient[i][axes]
                           : 0.0)
                   << ' ';
            }
            file << endl;
          }
          file.close();
        });
      }
    }

    if (refinement_field == 2) {
      // gradient of pressure
      auto &gradient = equation_mgr->get_gradient();
      master_operation(0, [this]() {
        ofstream file;
        file.open("./vtk/output_step" + to_string(write_step) + ".vtk",
                  ios::app);
        file << "SCALARS dP float " + to_string(dim) << endl;
        file << "LOOKUP_TABLE default" << endl;
        file.close();
      });

      serial_operation([gradient, this]() {
        ofstream file;
        file.open("./vtk/output_step" + to_string(write_step) + ".vtk",
                  ios::app);
        for (size_t i = 0; i < gradient.size(); i++) {
          for (int axes = 0; axes < dim; axes++) {
            file << ((abs(gradient[i][axes]) > 1e-10) ? gradient[i][axes] : 0.0)
                 << ' ';
          }
          file << endl;
        }
        file.close();
      });
    }
  }

  write_step++;
}

void gmls_solver::write_refinement_data() {
  vector<vec3> &coord = *(geo_mgr->get_current_work_particle_coord());
  vector<vec3> &normal = *(geo_mgr->get_current_work_particle_normal());
  vector<double> &spacing = *(geo_mgr->get_current_work_particle_spacing());
  vector<double> &volume = *(geo_mgr->get_current_work_particle_volume());
  vector<int> &particle_type = *(geo_mgr->get_current_work_particle_type());
  vector<int> &adaptive_level =
      *(geo_mgr->get_current_work_particle_adaptive_level());
  vector<int> &num_neighbor =
      *(geo_mgr->get_current_work_particle_num_neighbor());

  auto &epsilon = equation_mgr->get_epsilon();

  int local_particle_num;
  int global_particle_num;

  local_particle_num = coord.size();
  MPI_Allreduce(&local_particle_num, &global_particle_num, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  PetscPrintf(PETSC_COMM_WORLD, "writing adaptive step output\n");

  master_operation(0, [global_particle_num, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                  ".vtk",
              ios::trunc);
    if (!file.is_open()) {
      cout << "adaptive step output file open failed\n";
    }
    file << "# vtk DataFile Version 2.0" << endl;
    file << "particlePositions" << endl;
    file << "ASCII" << endl;
    file << "DATASET POLYDATA " << endl;
    file << " POINTS " << global_particle_num << " float" << endl;
    file.close();
  });

  serial_operation([coord, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                  ".vtk",
              ios::app);
    for (size_t i = 0; i < coord.size(); i++) {
      file << fixed << setprecision(10) << coord[i][0] << ' ' << coord[i][1]
           << ' ' << coord[i][2] << endl;
    }
    file.close();
  });

  master_operation(0, [global_particle_num, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                  ".vtk",
              ios::app);
    file << "POINT_DATA " << global_particle_num << endl;
    file.close();
  });

  master_operation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                  ".vtk",
              ios::app);
    file << "SCALARS ID int 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  serial_operation([particle_type, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                  ".vtk",
              ios::app);
    for (size_t i = 0; i < particle_type.size(); i++) {
      file << particle_type[i] << endl;
    }
    file.close();
  });

  master_operation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                  ".vtk",
              ios::app);
    file << "SCALARS nn int 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  serial_operation([num_neighbor, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                  ".vtk",
              ios::app);
    for (size_t i = 0; i < num_neighbor.size(); i++) {
      file << num_neighbor[i] << endl;
    }
    file.close();
  });

  master_operation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                  ".vtk",
              ios::app);
    file << "SCALARS l int 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  serial_operation([adaptive_level, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                  ".vtk",
              ios::app);
    for (size_t i = 0; i < adaptive_level.size(); i++) {
      file << adaptive_level[i] << endl;
    }
    file.close();
  });

  master_operation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                  ".vtk",
              ios::app);
    file << "SCALARS d float 1" << endl;
    file << "LOOKUP_TABLE default " << endl;
    file.close();
  });

  serial_operation([spacing, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                  ".vtk",
              ios::app);
    for (size_t i = 0; i < spacing.size(); i++) {
      file << spacing[i] << endl;
    }
    file.close();
  });

  master_operation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                  ".vtk",
              ios::app);
    file << "SCALARS vol float 1" << endl;
    file << "LOOKUP_TABLE default " << endl;
    file.close();
  });

  serial_operation([volume, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                  ".vtk",
              ios::app);
    for (size_t i = 0; i < volume.size(); i++) {
      file << volume[i] << endl;
    }
    file.close();
  });

  master_operation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                  ".vtk",
              ios::app);
    file << "SCALARS n float " + to_string(dim) << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  serial_operation([normal, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                  ".vtk",
              ios::app);
    for (size_t i = 0; i < normal.size(); i++) {
      for (int axes = 0; axes < dim; axes++) {
        file << ((abs(normal[i][axes]) > 1e-10) ? normal[i][axes] : 0.0) << ' ';
      }
      file << endl;
    }
    file.close();
  });

  master_operation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                  ".vtk",
              ios::app);
    file << "SCALARS domain int 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  serial_operation([particle_type, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                  ".vtk",
              ios::app);
    for (size_t i = 0; i < particle_type.size(); i++) {
      file << rank << endl;
    }
    file.close();
  });

  master_operation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                  ".vtk",
              ios::app);
    file << "SCALARS epsilon float 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  serial_operation([epsilon, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                  ".vtk",
              ios::app);
    for (size_t i = 0; i < epsilon.size(); i++) {
      file << epsilon[i] << endl;
    }
    file.close();
  });

  if (equation_type == "Stokes") {
    vector<vec3> &velocity = equation_mgr->get_velocity();
    vector<double> &pressure = equation_mgr->get_pressure();
    vector<double> &error = equation_mgr->get_error();

    master_operation(0, [this]() {
      ofstream file;
      file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                    ".vtk",
                ios::app);
      file << "SCALARS err float 1" << endl;
      file << "LOOKUP_TABLE default " << endl;
      file.close();
    });

    serial_operation([error, this]() {
      ofstream file;
      file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                    ".vtk",
                ios::app);
      for (size_t i = 0; i < error.size(); i++) {
        file << error[i] << endl;
      }
      file.close();
    });

    master_operation(0, [this]() {
      ofstream file;
      file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                    ".vtk",
                ios::app);
      file << "SCALARS p float 1" << endl;
      file << "LOOKUP_TABLE default " << endl;
      file.close();
    });

    serial_operation([pressure, this]() {
      ofstream file;
      file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                    ".vtk",
                ios::app);
      for (size_t i = 0; i < pressure.size(); i++) {
        file << ((abs(pressure[i]) > 1e-10) ? pressure[i] : 0.0) << endl;
      }
      file.close();
    });

    master_operation(0, [this]() {
      ofstream file;
      file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                    ".vtk",
                ios::app);
      file << "SCALARS u float " + to_string(dim) << endl;
      file << "LOOKUP_TABLE default" << endl;
      file.close();
    });

    serial_operation([velocity, this]() {
      ofstream file;
      file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                    ".vtk",
                ios::app);
      for (size_t i = 0; i < velocity.size(); i++) {
        for (int axes = 0; axes < dim; axes++) {
          file << ((abs(velocity[i][axes]) > 1e-10) ? velocity[i][axes] : 0.0)
               << ' ';
        }
        file << endl;
      }
      file.close();
    });

    if (refinement_field == 1) {
      // gradient of velocity
      auto &gradient = equation_mgr->get_gradient();
      master_operation(0, [this]() {
        ofstream file;
        file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                      ".vtk",
                  ios::app);
        file << "SCALARS dUX float " + to_string(dim) << endl;
        file << "LOOKUP_TABLE default" << endl;
        file.close();
      });

      serial_operation([gradient, this]() {
        ofstream file;
        file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                      ".vtk",
                  ios::app);
        for (size_t i = 0; i < gradient.size(); i++) {
          for (int axes = 0; axes < dim; axes++) {
            file << ((abs(gradient[i][axes]) > 1e-10) ? gradient[i][axes] : 0.0)
                 << ' ';
          }
          file << endl;
        }
        file.close();
      });

      if (dim > 1) {
        master_operation(0, [this]() {
          ofstream file;
          file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                        ".vtk",
                    ios::app);
          file << "SCALARS dUY float " + to_string(dim) << endl;
          file << "LOOKUP_TABLE default" << endl;
          file.close();
        });

        serial_operation([gradient, this]() {
          ofstream file;
          file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                        ".vtk",
                    ios::app);
          for (size_t i = 0; i < gradient.size(); i++) {
            for (int axes = 0; axes < dim; axes++) {
              file << ((abs(gradient[i][axes + dim]) > 1e-10)
                           ? gradient[i][axes]
                           : 0.0)
                   << ' ';
            }
            file << endl;
          }
          file.close();
        });
      }

      if (dim > 2) {
        master_operation(0, [this]() {
          ofstream file;
          file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                        ".vtk",
                    ios::app);
          file << "SCALARS dUZ float " + to_string(dim) << endl;
          file << "LOOKUP_TABLE default" << endl;
          file.close();
        });

        serial_operation([gradient, this]() {
          ofstream file;
          file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                        ".vtk",
                    ios::app);
          for (size_t i = 0; i < gradient.size(); i++) {
            for (int axes = 0; axes < dim; axes++) {
              file << ((abs(gradient[i][axes + dim * 2]) > 1e-10)
                           ? gradient[i][axes]
                           : 0.0)
                   << ' ';
            }
            file << endl;
          }
          file.close();
        });
      }
    }

    if (refinement_field == 2) {
      // gradient of pressure
      auto &gradient = equation_mgr->get_gradient();
      master_operation(0, [this]() {
        ofstream file;
        file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                      ".vtk",
                  ios::app);
        file << "SCALARS dP float " + to_string(dim) << endl;
        file << "LOOKUP_TABLE default" << endl;
        file.close();
      });

      serial_operation([gradient, this]() {
        ofstream file;
        file.open("./vtk/adaptive_step" + to_string(current_refinement_step) +
                      ".vtk",
                  ios::app);
        for (size_t i = 0; i < gradient.size(); i++) {
          for (int axes = 0; axes < dim; axes++) {
            file << ((abs(gradient[i][axes]) > 1e-10) ? gradient[i][axes] : 0.0)
                 << ' ';
          }
          file << endl;
        }
        file.close();
      });
    }
  }
}

void gmls_solver::write_refinement_data_geometry_only() {
  vector<vec3> &coord = *(geo_mgr->get_current_work_particle_coord());
  vector<vec3> &normal = *(geo_mgr->get_current_work_particle_normal());
  vector<double> &spacing = *(geo_mgr->get_current_work_particle_spacing());
  vector<int> &particle_type = *(geo_mgr->get_current_work_particle_type());
  vector<int> &local_idx = *(geo_mgr->get_current_work_particle_local_index());
  vector<vec3> &p_spacing = *(geo_mgr->get_current_work_particle_p_spacing());

  int local_particle_num;
  int global_particle_num;

  local_particle_num = coord.size();
  MPI_Allreduce(&local_particle_num, &global_particle_num, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  master_operation(0, [global_particle_num, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::trunc);
    if (!file.is_open()) {
      cout << "adaptive step output file open failed\n";
    }
    file << "# vtk DataFile Version 2.0" << endl;
    file << "particlePositions" << endl;
    file << "ASCII" << endl;
    file << "DATASET POLYDATA " << endl;
    file << " POINTS " << global_particle_num << " float" << endl;
    file.close();
  });

  serial_operation([coord, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < coord.size(); i++) {
      file << fixed << setprecision(10) << coord[i][0] << ' ' << coord[i][1]
           << ' ' << coord[i][2] << endl;
    }
    file.close();
  });

  master_operation(0, [global_particle_num, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    file << "POINT_DATA " << global_particle_num << endl;
    file.close();
  });

  master_operation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    file << "SCALARS ID int 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  serial_operation([particle_type, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < particle_type.size(); i++) {
      file << particle_type[i] << endl;
    }
    file.close();
  });

  master_operation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    file << "SCALARS index int 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  serial_operation([local_idx, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < local_idx.size(); i++) {
      file << local_idx[i] << endl;
    }
    file.close();
  });

  master_operation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    file << "SCALARS d float 1" << endl;
    file << "LOOKUP_TABLE default " << endl;
    file.close();
  });

  serial_operation([spacing, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < spacing.size(); i++) {
      file << spacing[i] << endl;
    }
    file.close();
  });

  master_operation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    file << "SCALARS n float 3" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  serial_operation([normal, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < normal.size(); i++) {
      file << normal[i][0] << ' ' << normal[i][1] << ' ' << normal[i][2]
           << endl;
    }
    file.close();
  });

  master_operation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    file << "SCALARS p_spacing float 3" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  serial_operation([p_spacing, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < p_spacing.size(); i++) {
      file << p_spacing[i][0] << ' ' << p_spacing[i][1] << ' '
           << p_spacing[i][2] << endl;
    }
    file.close();
  });

  master_operation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    file << "SCALARS domain int 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  serial_operation([particle_type, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < particle_type.size(); i++) {
      file << rank << endl;
    }
    file.close();
  });

  vector<vec3> &gap_coord = *(geo_mgr->get_local_gap_particle_coord());
  vector<double> &gap_spacing = *(geo_mgr->get_local_gap_particle_spacing());
  vector<int> &adaptive_level =
      *(geo_mgr->get_local_gap_particle_adaptive_level());

  int global_gap_particle_num = gap_coord.size();
  MPI_Allreduce(MPI_IN_PLACE, &global_gap_particle_num, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  master_operation(0, [global_gap_particle_num, this]() {
    ofstream file;
    file.open("./vtk/adaptive_gap_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::trunc);
    if (!file.is_open()) {
      cout << "adaptive step output file open failed\n";
    }
    file << "# vtk DataFile Version 2.0" << endl;
    file << "particlePositions" << endl;
    file << "ASCII" << endl;
    file << "DATASET POLYDATA " << endl;
    file << " POINTS " << global_gap_particle_num << " float" << endl;
    file.close();
  });

  serial_operation([gap_coord, this]() {
    ofstream file;
    file.open("./vtk/adaptive_gap_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < gap_coord.size(); i++) {
      file << fixed << setprecision(10) << gap_coord[i][0] << ' '
           << gap_coord[i][1] << ' ' << gap_coord[i][2] << endl;
    }
    file.close();
  });

  master_operation(0, [global_gap_particle_num, this]() {
    ofstream file;
    file.open("./vtk/adaptive_gap_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    file << "POINT_DATA " << global_gap_particle_num << endl;
    file.close();
  });

  master_operation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_gap_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    file << "SCALARS d float 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  serial_operation([gap_spacing, this]() {
    ofstream file;
    file.open("./vtk/adaptive_gap_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < gap_spacing.size(); i++) {
      file << gap_spacing[i] << endl;
    }
    file.close();
  });

  master_operation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_gap_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    file << "SCALARS l int 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  serial_operation([adaptive_level, this]() {
    ofstream file;
    file.open("./vtk/adaptive_gap_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < adaptive_level.size(); i++) {
      file << adaptive_level[i] << endl;
    }
    file.close();
  });

  master_operation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_gap_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    file << "SCALARS domain int 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  serial_operation([adaptive_level, this]() {
    ofstream file;
    file.open("./vtk/adaptive_gap_geometry" +
                  to_string(current_refinement_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < adaptive_level.size(); i++) {
      file << rank << endl;
    }
    file.close();
  });
}