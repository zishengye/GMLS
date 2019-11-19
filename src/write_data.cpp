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

  MasterOperation(0, []() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    file << "SCALARS rhs float 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  SerialOperation([this]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    for (size_t i = 0; i < this->__particle.X.size(); i++) {
      file << __eq.rhs[i] << endl;
    }
    file.close();
  });

  MasterOperation(0, []() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    file << "SCALARS flux float 3" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  SerialOperation([this]() {
    ofstream file;
    file.open("./vtk/output_step" + to_string(writeStep) + ".vtk", ios::app);
    for (size_t i = 0; i < this->__particle.X.size(); i++) {
      file << __particle.flux[i][0] << ' ' << __particle.flux[i][1] << ' '
           << __particle.flux[i][2] << endl;
    }
    file.close();
  });

  writeStep++;
}