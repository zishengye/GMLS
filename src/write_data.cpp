#include "gmls_solver.h"

#include <fstream>
#include <string>

using namespace std;

void GMLS_Solver::WriteDataTimeStep() {
  static int writeStep = 0;
  vector<vec3> &coord = __field.vector.GetHandle("coord");
  vector<vec3> &particleSize = __field.vector.GetHandle("size");
  vector<vec3> &normal = __field.vector.GetHandle("normal");
  vector<int> &particleType = __field.index.GetHandle("particle type");
  vector<int> &particleNum = __field.index.GetHandle("particle number");
  int &globalParticleNum = particleNum[1];

  // int thread = 3;

  // MasterOperation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_step" + to_string(writeStep) + ".vtk",
  //   ios::trunc); file << "# vtk DataFile Version 2.0" << endl; file <<
  //   "particlePositions" << endl; file << "ASCII" << endl; file << "DATASET
  //   POLYDATA " << endl; file << " POINTS " <<
  //   this->__backgroundParticle.coord.size() << " float"
  //        << endl;
  //   file.close();
  // });

  // MasterOperation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_step" + to_string(writeStep) + ".vtk", ios::app);
  //   for (size_t i = 0; i < this->__backgroundParticle.coord.size(); i++) {
  //     file << __backgroundParticle.coord[i][0] << ' '
  //          << __backgroundParticle.coord[i][1] << ' '
  //          << __backgroundParticle.coord[i][2] << endl;
  //   }
  //   file.close();
  // });

  // MasterOperation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_step" + to_string(writeStep) + ".vtk", ios::app);
  //   file << "POINT_DATA " << this->__backgroundParticle.coord.size() << endl;
  //   file.close();
  // });

  // MasterOperation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_step" + to_string(writeStep) + ".vtk", ios::app);
  //   file << "SCALARS index int 1" << endl;
  //   file << "LOOKUP_TABLE default" << endl;
  //   file.close();
  // });

  // MasterOperation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_step" + to_string(writeStep) + ".vtk", ios::app);
  //   for (size_t i = 0; i < this->__backgroundParticle.coord.size(); i++) {
  //     file << __backgroundParticle.index[i] << endl;
  //   }
  //   file.close();
  // });

  // MasterOperation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_local_step" + to_string(writeStep) + ".vtk",
  //             ios::trunc);
  //   file << "# vtk DataFile Version 2.0" << endl;
  //   file << "particlePositions" << endl;
  //   file << "ASCII" << endl;
  //   file << "DATASET POLYDATA " << endl;
  //   file << " POINTS " << this->__particle.localParticleNum << " float" <<
  //   endl; file.close();
  // });

  // MasterOperation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_local_step" + to_string(writeStep) + ".vtk",
  //             ios::app);
  //   for (size_t i = 0; i < this->__particle.localParticleNum; i++) {
  //     file << __particle.X[i][0] << ' ' << __particle.X[i][1] << ' '
  //          << __particle.X[i][2] << endl;
  //   }
  //   file.close();
  // });

  // MasterOperation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_local_step" + to_string(writeStep) + ".vtk",
  //             ios::app);
  //   file << "POINT_DATA " << this->__particle.localParticleNum << endl;
  //   file.close();
  // });

  // MasterOperation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_local_step" + to_string(writeStep) + ".vtk",
  //             ios::app);
  //   file << "SCALARS ID int 1" << endl;
  //   file << "LOOKUP_TABLE default" << endl;
  //   file.close();
  // });

  // MasterOperation(thread, [this]() {
  //   ofstream file;
  //   file.open("./vtk/domain_local_step" + to_string(writeStep) + ".vtk",
  //             ios::app);
  //   for (size_t i = 0; i < this->__particle.localParticleNum; i++) {
  //     file << __particle.particleType[i] << endl;
  //   }
  //   file.close();
  // });

  MasterOperation(0, [globalParticleNum]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::trunc);
    file << "# vtk DataFile Version 2.0" << endl;
    file << "particlePositions" << endl;
    file << "ASCII" << endl;
    file << "DATASET POLYDATA " << endl;
    file << " POINTS " << globalParticleNum << " float" << endl;
    file.close();
  });

  SerialOperation([coord]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    for (size_t i = 0; i < coord.size(); i++) {
      file << coord[i][0] << ' ' << coord[i][1] << ' ' << coord[i][2] << endl;
    }
    file.close();
  });

  MasterOperation(0, [globalParticleNum]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    file << "POINT_DATA " << globalParticleNum << endl;
    file.close();
  });

  MasterOperation(0, []() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    file << "SCALARS ID int 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  SerialOperation([particleType]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    for (size_t i = 0; i < particleType.size(); i++) {
      file << particleType[i] << endl;
    }
    file.close();
  });

  MasterOperation(0, []() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    file << "SCALARS d float 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  SerialOperation([particleSize]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    for (size_t i = 0; i < particleSize.size(); i++) {
      file << particleSize[i][0] << endl;
    }
    file.close();
  });

  MasterOperation(0, []() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    file << "SCALARS n float 3" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  SerialOperation([normal]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    for (size_t i = 0; i < normal.size(); i++) {
      file << normal[i][0] << ' ' << normal[i][1] << ' ' << normal[i][2]
           << endl;
    }
    file.close();
  });

  // physical data output, equation type depedent
  if (__equationType == "Diffusion") {
    vector<double> &us = __field.scalar.GetHandle("us");

    MasterOperation(0, []() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
      file << "SCALARS us float 1" << endl;
      file << "LOOKUP_TABLE default " << endl;
      file.close();
    });

    SerialOperation([us]() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
      for (size_t i = 0; i < us.size(); i++) {
        file << us[i] << endl;
      }
      file.close();
    });
  }

  if (__equationType == "Stokes") {
    vector<vec3> &velocity = __field.vector.GetHandle("fluid velocity");
    vector<double> &pressure = __field.scalar.GetHandle("fluid pressure");
    auto &error = __field.scalar.GetHandle("error");
    vector<vec3> &rigidPos = __rigidBody.vector.GetHandle("position");
    vector<vec3> &rigidTheta = __rigidBody.vector.GetHandle("orientation");
    vector<vec3> &rigidVelocity = __rigidBody.vector.GetHandle("velocity");
    vector<vec3> &rigidAngularVelocity =
        __rigidBody.vector.GetHandle("angular velocity");

    MasterOperation(0, []() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
      file << "SCALARS p float 1" << endl;
      file << "LOOKUP_TABLE default " << endl;
      file.close();
    });

    SerialOperation([pressure]() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
      for (size_t i = 0; i < pressure.size(); i++) {
        file << ((abs(pressure[i]) > 1e-10) ? pressure[i] : 0.0) << endl;
      }
      file.close();
    });

    MasterOperation(0, [this]() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
      file << "SCALARS u float " + to_string(__dim) << endl;
      file << "LOOKUP_TABLE default" << endl;
      file.close();
    });

    SerialOperation([velocity, this]() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
      for (size_t i = 0; i < velocity.size(); i++) {
        for (int axes = 0; axes < __dim; axes++) {
          file << ((abs(velocity[i][axes]) > 1e-10) ? velocity[i][axes] : 0.0)
               << ' ';
        }
        file << endl;
      }
      file.close();
    });

    MasterOperation(0, [this]() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
      file << "SCALARS err float 1" << endl;
      file << "LOOKUP_TABLE default " << endl;
      file.close();
    });

    SerialOperation([error, this]() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
      for (size_t i = 0; i < error.size(); i++) {
        file << sqrt(error[i]) << endl;
      }
      file.close();
    });

    MasterOperation(
        0, [rigidPos, rigidTheta, rigidVelocity, rigidAngularVelocity]() {
          ofstream file;
          file.open("./txt/output_step" + to_string(writeStep) + ".txt",
                    ios::trunc);
          for (size_t i = 0; i < rigidPos.size(); i++) {
            for (int j = 0; j < 3; j++) {
              file << rigidPos[i][j] << ' ';
            }
            for (int j = 0; j < 3; j++) {
              file << rigidTheta[i][j] << ' ';
            }
            for (int j = 0; j < 3; j++) {
              file << rigidVelocity[i][j] << ' ';
            }
            for (int j = 0; j < 3; j++) {
              file << rigidAngularVelocity[i][j] << ' ';
            }
            file << endl;
          }
          file.close();
        });
  }

  writeStep++;
}

void GMLS_Solver::WriteDataAdaptiveStep() {
  auto &coord = __field.vector.GetHandle("coord");
  auto &particleSize = __field.vector.GetHandle("size");
  auto &normal = __field.vector.GetHandle("normal");
  auto &particleType = __field.index.GetHandle("particle type");
  auto &particleNum = __field.index.GetHandle("particle number");
  auto &adaptive_level = __field.index.GetHandle("adaptive level");
  int &globalParticleNum = particleNum[1];

  auto &velocity = __field.vector.GetHandle("fluid velocity");
  auto &pressure = __field.scalar.GetHandle("fluid pressure");
  auto &error = __field.scalar.GetHandle("error");
  auto &volume = __field.scalar.GetHandle("volume");

  PetscPrintf(PETSC_COMM_WORLD, "writing adaptive step output\n");

  MasterOperation(0, [globalParticleNum, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::trunc);
    if (!file.is_open()) {
      cout << "adaptive step output file open failed\n";
    }
    file << "# vtk DataFile Version 2.0" << endl;
    file << "particlePositions" << endl;
    file << "ASCII" << endl;
    file << "DATASET POLYDATA " << endl;
    file << " POINTS " << globalParticleNum << " float" << endl;
    file.close();
  });

  SerialOperation([coord, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < coord.size(); i++) {
      file << coord[i][0] << ' ' << coord[i][1] << ' ' << coord[i][2] << endl;
    }
    file.close();
  });

  MasterOperation(0, [globalParticleNum, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::app);
    file << "POINT_DATA " << globalParticleNum << endl;
    file.close();
  });

  MasterOperation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::app);
    file << "SCALARS ID int 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  SerialOperation([particleType, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < particleType.size(); i++) {
      file << particleType[i] << endl;
    }
    file.close();
  });

  MasterOperation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::app);
    file << "SCALARS l int 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  SerialOperation([adaptive_level, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < adaptive_level.size(); i++) {
      file << adaptive_level[i] << endl;
    }
    file.close();
  });

  MasterOperation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::app);
    file << "SCALARS d float 1" << endl;
    file << "LOOKUP_TABLE default " << endl;
    file.close();
  });

  SerialOperation([particleSize, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < particleSize.size(); i++) {
      file << particleSize[i][0] << endl;
    }
    file.close();
  });

  MasterOperation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::app);
    file << "SCALARS vol float 1" << endl;
    file << "LOOKUP_TABLE default " << endl;
    file.close();
  });

  SerialOperation([volume, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < volume.size(); i++) {
      file << volume[i] << endl;
    }
    file.close();
  });

  MasterOperation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::app);
    file << "SCALARS err float 1" << endl;
    file << "LOOKUP_TABLE default " << endl;
    file.close();
  });

  SerialOperation([error, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < error.size(); i++) {
      file << sqrt(error[i]) << endl;
    }
    file.close();
  });

  MasterOperation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::app);
    file << "SCALARS p float 1" << endl;
    file << "LOOKUP_TABLE default " << endl;
    file.close();
  });

  SerialOperation([pressure, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < pressure.size(); i++) {
      file << ((abs(pressure[i]) > 1e-10) ? pressure[i] : 0.0) << endl;
    }
    file.close();
  });

  MasterOperation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::app);
    file << "SCALARS u float " + to_string(__dim) << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  SerialOperation([velocity, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < velocity.size(); i++) {
      for (int axes = 0; axes < __dim; axes++) {
        file << ((abs(velocity[i][axes]) > 1e-10) ? velocity[i][axes] : 0.0)
             << ' ';
      }
      file << endl;
    }
    file.close();
  });

  MasterOperation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::app);
    file << "SCALARS n float " + to_string(__dim) << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  SerialOperation([normal, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < normal.size(); i++) {
      for (int axes = 0; axes < __dim; axes++) {
        file << ((abs(normal[i][axes]) > 1e-10) ? normal[i][axes] : 0.0) << ' ';
      }
      file << endl;
    }
    file.close();
  });
}

void GMLS_Solver::WriteDataAdaptiveGeometry() {
  auto &coord = __field.vector.GetHandle("coord");
  auto &particleSize = __field.vector.GetHandle("size");
  auto &normal = __field.vector.GetHandle("normal");
  auto &particleType = __field.index.GetHandle("particle type");
  auto &particleNum = __field.index.GetHandle("particle number");
  auto &globalParticleNum = particleNum[1];

  auto &velocity = __field.vector.GetHandle("fluid velocity");
  auto &pressure = __field.scalar.GetHandle("fluid pressure");
  auto &error = __field.scalar.GetHandle("error");
  auto &volume = __field.scalar.GetHandle("volume");

  auto &_gapCoord = __gap.vector.GetHandle("coord");
  auto &_gapNormal = __gap.vector.GetHandle("normal");
  auto &_gapParticleSize = __gap.vector.GetHandle("size");
  auto &_gapParticleType = __gap.index.GetHandle("particle type");

  static auto &gapRigidBodyCoord =
      __gap.vector.GetHandle("rigid body surface coord");
  static auto &gapRigidBodyNormal =
      __gap.vector.GetHandle("rigid body surface normal");
  static auto &gapRigidBodySize =
      __gap.vector.GetHandle("rigid body surface size");
  static auto &gapRigidBodyParticleType =
      __gap.index.GetHandle("rigid body surface particle type");

  MasterOperation(0, [globalParticleNum, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" + to_string(__adaptive_step) +
                  ".vtk",
              ios::trunc);
    if (!file.is_open()) {
      cout << "adaptive step output file open failed\n";
    }
    file << "# vtk DataFile Version 2.0" << endl;
    file << "particlePositions" << endl;
    file << "ASCII" << endl;
    file << "DATASET POLYDATA " << endl;
    file << " POINTS " << globalParticleNum << " float" << endl;
    file.close();
  });

  SerialOperation([coord, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" + to_string(__adaptive_step) +
                  ".vtk",
              ios::app);
    for (size_t i = 0; i < coord.size(); i++) {
      file << coord[i][0] << ' ' << coord[i][1] << ' ' << coord[i][2] << endl;
    }
    file.close();
  });

  MasterOperation(0, [globalParticleNum, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" + to_string(__adaptive_step) +
                  ".vtk",
              ios::app);
    file << "POINT_DATA " << globalParticleNum << endl;
    file.close();
  });

  MasterOperation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" + to_string(__adaptive_step) +
                  ".vtk",
              ios::app);
    file << "SCALARS ID int 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  SerialOperation([particleType, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" + to_string(__adaptive_step) +
                  ".vtk",
              ios::app);
    for (size_t i = 0; i < particleType.size(); i++) {
      file << particleType[i] << endl;
    }
    file.close();
  });

  MasterOperation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" + to_string(__adaptive_step) +
                  ".vtk",
              ios::app);
    file << "SCALARS d float 1" << endl;
    file << "LOOKUP_TABLE default " << endl;
    file.close();
  });

  SerialOperation([particleSize, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" + to_string(__adaptive_step) +
                  ".vtk",
              ios::app);
    for (size_t i = 0; i < particleSize.size(); i++) {
      file << particleSize[i][0] << endl;
    }
    file.close();
  });

  MasterOperation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" + to_string(__adaptive_step) +
                  ".vtk",
              ios::app);
    file << "SCALARS vol float 1" << endl;
    file << "LOOKUP_TABLE default " << endl;
    file.close();
  });

  SerialOperation([volume, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step_geometry" + to_string(__adaptive_step) +
                  ".vtk",
              ios::app);
    for (size_t i = 0; i < volume.size(); i++) {
      file << volume[i] << endl;
    }
    file.close();
  });

  int globalGapParticleNum = _gapCoord.size() + gapRigidBodyCoord.size();
  MPI_Allreduce(MPI_IN_PLACE, &globalGapParticleNum, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  MasterOperation(0, [globalGapParticleNum, this]() {
    ofstream file;
    file.open("./vtk/adaptive_gap_geometry" + to_string(__adaptive_step) +
                  ".vtk",
              ios::trunc);
    if (!file.is_open()) {
      cout << "adaptive step output file open failed\n";
    }
    file << "# vtk DataFile Version 2.0" << endl;
    file << "particlePositions" << endl;
    file << "ASCII" << endl;
    file << "DATASET POLYDATA " << endl;
    file << " POINTS " << globalGapParticleNum << " float" << endl;
    file.close();
  });

  SerialOperation([_gapCoord, this]() {
    ofstream file;
    file.open("./vtk/adaptive_gap_geometry" + to_string(__adaptive_step) +
                  ".vtk",
              ios::app);
    for (size_t i = 0; i < _gapCoord.size(); i++) {
      file << _gapCoord[i][0] << ' ' << _gapCoord[i][1] << ' '
           << _gapCoord[i][2] << endl;
    }
    file.close();
  });

  SerialOperation([gapRigidBodyCoord, this]() {
    ofstream file;
    file.open("./vtk/adaptive_gap_geometry" + to_string(__adaptive_step) +
                  ".vtk",
              ios::app);
    for (size_t i = 0; i < gapRigidBodyCoord.size(); i++) {
      file << gapRigidBodyCoord[i][0] << ' ' << gapRigidBodyCoord[i][1] << ' '
           << gapRigidBodyCoord[i][2] << endl;
    }
    file.close();
  });

  MasterOperation(0, [globalGapParticleNum, this]() {
    ofstream file;
    file.open("./vtk/adaptive_gap_geometry" + to_string(__adaptive_step) +
                  ".vtk",
              ios::app);
    file << "POINT_DATA " << globalGapParticleNum << endl;
    file.close();
  });

  MasterOperation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_gap_geometry" + to_string(__adaptive_step) +
                  ".vtk",
              ios::app);
    file << "SCALARS ID int 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  SerialOperation([_gapParticleType, this]() {
    ofstream file;
    file.open("./vtk/adaptive_gap_geometry" + to_string(__adaptive_step) +
                  ".vtk",
              ios::app);
    for (size_t i = 0; i < _gapParticleType.size(); i++) {
      file << _gapParticleType[i] << endl;
    }
    file.close();
  });

  SerialOperation([gapRigidBodyParticleType, this]() {
    ofstream file;
    file.open("./vtk/adaptive_gap_geometry" + to_string(__adaptive_step) +
                  ".vtk",
              ios::app);
    for (size_t i = 0; i < gapRigidBodyParticleType.size(); i++) {
      file << gapRigidBodyParticleType[i] << endl;
    }
    file.close();
  });

  MasterOperation(0, [this]() {
    ofstream file;
    file.open("./vtk/adaptive_gap_geometry" + to_string(__adaptive_step) +
                  ".vtk",
              ios::app);
    file << "SCALARS d float 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  SerialOperation([_gapParticleSize, this]() {
    ofstream file;
    file.open("./vtk/adaptive_gap_geometry" + to_string(__adaptive_step) +
                  ".vtk",
              ios::app);
    for (size_t i = 0; i < _gapParticleSize.size(); i++) {
      file << _gapParticleSize[i][0] << endl;
    }
    file.close();
  });

  SerialOperation([gapRigidBodySize, this]() {
    ofstream file;
    file.open("./vtk/adaptive_gap_geometry" + to_string(__adaptive_step) +
                  ".vtk",
              ios::app);
    for (size_t i = 0; i < gapRigidBodySize.size(); i++) {
      file << gapRigidBodySize[i][0] << endl;
    }
    file.close();
  });
}