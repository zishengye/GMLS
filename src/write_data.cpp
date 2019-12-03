#include "GMLS_solver.h"

#include <fstream>
#include <string>

using namespace std;

void GMLS_Solver::WriteData() {
  static int writeStep = 0;

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

  MasterOperation(0, [this]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::trunc);
    file << "# vtk DataFile Version 2.0" << endl;
    file << "particlePositions" << endl;
    file << "ASCII" << endl;
    file << "DATASET POLYDATA " << endl;
    file << " POINTS " << this->__particle.globalParticleNum << " float"
         << endl;
    file.close();
  });

  SerialOperation([this]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    for (size_t i = 0; i < this->__particle.X.size(); i++) {
      file << __particle.X[i][0] << ' ' << __particle.X[i][1] << ' '
           << __particle.X[i][2] << endl;
    }
    file.close();
  });

  MasterOperation(0, [this]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    file << "POINT_DATA " << this->__particle.globalParticleNum << endl;
    file.close();
  });

  MasterOperation(0, []() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    file << "SCALARS Vol float 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  SerialOperation([this]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    for (size_t i = 0; i < this->__particle.X.size(); i++) {
      file << __particle.vol[i] << endl;
    }
    file.close();
  });

  MasterOperation(0, []() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    file << "SCALARS ID int 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  SerialOperation([this]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    for (size_t i = 0; i < this->__particle.X.size(); i++) {
      file << __particle.particleType[i] << endl;
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

  SerialOperation([this]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    for (size_t i = 0; i < this->__particle.X.size(); i++) {
      file << __particle.d[i] << endl;
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

  SerialOperation([this]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    for (size_t i = 0; i < this->__particle.X.size(); i++) {
      file << __particle.normal[i][0] << ' ' << __particle.normal[i][1] << ' '
           << __particle.normal[i][2] << endl;
    }
    file.close();
  });

  // physical data output, equation type depedent
  if (__equationType == "Diffusion") {
    MasterOperation(0, []() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
      file << "SCALARS us float 1" << endl;
      file << "LOOKUP_TABLE default " << endl;
      file.close();
    });

    SerialOperation([this]() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
      for (size_t i = 0; i < this->__particle.X.size(); i++) {
        file << __particle.us[i] << endl;
      }
      file.close();
    });
  }

  if (__equationType == "Stokes") {
    MasterOperation(0, []() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
      file << "SCALARS p float 1" << endl;
      file << "LOOKUP_TABLE default " << endl;
      file.close();
    });

    SerialOperation([this]() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
      for (size_t i = 0; i < this->__particle.X.size(); i++) {
        file << __particle.pressure[i] << endl;
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

    SerialOperation([this]() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
      for (size_t i = 0; i < this->__particle.X.size(); i++) {
        for (int axes = 0; axes < __dim; axes++)
          file << __particle.velocity[__dim * i + axes] << ' ';
        file << endl;
      }
      file.close();
    });

    MasterOperation(0, [this]() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
      file << "SCALARS rhs_u float " + to_string(__dim) << endl;
      file << "LOOKUP_TABLE default" << endl;
      file.close();
    });

    SerialOperation([this]() {
      ofstream file;
      file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
      for (size_t i = 0; i < this->__particle.X.size(); i++) {
        for (int axes = 0; axes < __dim; axes++)
          file << __eq.rhsVector[__dim * i + axes] << ' ';
        file << endl;
      }
      file.close();
    });

    MasterOperation(0, [this]() {
      ofstream file;
      file.open("./txt/output_step" + to_string(writeStep) + ".txt", ios::app);
      for (int i = 0; i < this->__rigidBody.Ci_X.size(); i++) {
        for (int j = 0; j < this->__dim; j++) {
          cout << __rigidBody.Ci_X[i][j] << ' ';
        }
        for (int j = 0; j < this->__dim; j++) {
          cout << __rigidBody.Ci_Theta[i][j] << ' ';
        }
        for (int j = 0; j < this->__dim; j++) {
          cout << __rigidBody.Ci_V[i][j] << ' ';
        }
        for (int j = 0; j < this->__dim; j++) {
          cout << __rigidBody.Ci_Omega[i][j] << ' ';
        }
      }
      file.close();
    });
  }

  writeStep++;
}