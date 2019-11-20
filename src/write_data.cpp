#include "GMLS_solver.h"

#include <fstream>
#include <string>

using namespace std;

void GMLS_Solver::WriteData() {
  static int writeStep = 0;
  PetscPrintf(PETSC_COMM_WORLD,
              ("writing to ./vtk/output_step" + to_string(writeStep) + ".vtk\n")
                  .c_str());
  MasterOperation(0, [this]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::trunc);
    file << "# vtk DataFile Version 2.0" << endl;
    file << "particlePositions" << endl;
    file << "ASCII" << endl;
    file << "DATASET POLYDATA" << endl;
    file << "POINTS " << this->__particle.globalParticleNum << " float" << endl;
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
      file << "LOOKUP_TABLE default" << endl;
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
      file << "LOOKUP_TABLE default" << endl;
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
  }

  writeStep++;
}