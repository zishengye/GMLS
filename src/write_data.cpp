#include "GMLS_solver.h"

#include <fstream>
#include <string>

using namespace std;

void GMLS_Solver::WriteData() {
  MasterOperation(0, [this]() {
    ofstream file;
    file.open("test.vtk", ios::trunc);
    file << "# vtk DataFile Version 2.0" << endl;
    file << "particlePositions" << endl;
    file << "ASCII" << endl;
    file << "DATASET POLYDATA" << endl;
    file << "POINTS " << this->__fluid.globalParticleNum << " float" << endl;
    file.close();
  });

  SerialOperation([this]() {
    ofstream file;
    file.open("test.vtk", ios::app);
    for (size_t i = 0; i < this->__fluid.X.size(); i++) {
      file << __fluid.X[i][0] << ' ' << __fluid.X[i][1] << ' '
           << __fluid.X[i][2] << endl;
    }
    file.close();
  });

  MasterOperation(0, [this]() {
    ofstream file;
    file.open("test.vtk", ios::app);
    file << "POINT_DATA " << this->__fluid.globalParticleNum << endl;
    file.close();
  });

  MasterOperation(0, []() {
    ofstream file;
    file.open("test.vtk", ios::app);
    file << "SCALARS Vol float 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  SerialOperation([this]() {
    ofstream file;
    file.open("test.vtk", ios::app);
    for (size_t i = 0; i < this->__fluid.X.size(); i++) {
      file << __fluid.vol[i] << endl;
    }
    file.close();
  });

  MasterOperation(0, []() {
    ofstream file;
    file.open("test.vtk", ios::app);
    file << "SCALARS ID int 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  SerialOperation([this]() {
    ofstream file;
    file.open("test.vtk", ios::app);
    for (size_t i = 0; i < this->__fluid.X.size(); i++) {
      file << __fluid.particleType[i] << endl;
    }
    file.close();
  });

  MasterOperation(0, []() {
    ofstream file;
    file.open("test.vtk", ios::app);
    file << "SCALARS d float 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  SerialOperation([this]() {
    ofstream file;
    file.open("test.vtk", ios::app);
    for (size_t i = 0; i < this->__fluid.X.size(); i++) {
      file << __fluid.d[i] << endl;
    }
    file.close();
  });

  MasterOperation(0, []() {
    ofstream file;
    file.open("test.vtk", ios::app);
    file << "SCALARS P float 1" << endl;
    file << "LOOKUP_TABLE default" << endl;
    file.close();
  });

  SerialOperation([this]() {
    ofstream file;
    file.open("test.vtk", ios::app);
    for (size_t i = 0; i < this->__fluid.X.size(); i++) {
      file << __fluid.pressure[i] << endl;
    }
    file.close();
  });
}