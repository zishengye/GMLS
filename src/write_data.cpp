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
        file << pressure[i] << endl;
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
          file << velocity[i][axes] << ' ';
        }
        file << endl;
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
  vector<vec3> &coord = __field.vector.GetHandle("coord");
  vector<vec3> &particleSize = __field.vector.GetHandle("size");
  vector<vec3> &normal = __field.vector.GetHandle("normal");
  vector<int> &particleType = __field.index.GetHandle("particle type");
  vector<int> &particleNum = __field.index.GetHandle("particle number");
  int &globalParticleNum = particleNum[1];

  vector<vec3> &velocity = __field.vector.GetHandle("fluid velocity");
  vector<double> &pressure = __field.scalar.GetHandle("fluid pressure");

  MasterOperation(0, [globalParticleNum, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::trunc);
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
    file << "SCALARS p float 1" << endl;
    file << "LOOKUP_TABLE default " << endl;
    file.close();
  });

  SerialOperation([pressure, this]() {
    ofstream file;
    file.open("./vtk/adaptive_step" + to_string(__adaptive_step) + ".vtk",
              ios::app);
    for (size_t i = 0; i < pressure.size(); i++) {
      file << pressure[i] << endl;
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
        file << velocity[i][axes] << ' ';
      }
      file << endl;
    }
    file.close();
  });
}