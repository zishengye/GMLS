#include "gmls_solver.hpp"

#include <memory>
#include <string>

using namespace std;

void GMLS_Solver::InitNeighborListManifold() {
  vector<int> &neighborFlag = __neighbor.index.GetHandle("neighbor flag");

  int neighborNum = pow(3, 2);

  neighborFlag.resize(neighborNum);
  for (int i = 0; i < neighborNum; i++) {
    neighborFlag[i] = false;
  }

  int offsetX[3] = {-1, 0, 1};
  int offsetY[3] = {-1, 0, 1};

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (__nI + offsetX[i] >= 0 && __nI + offsetX[i] < __nX) {
        if (__nJ + offsetY[j] >= 0 && __nJ + offsetY[j] < __nY) {
          int destination = (__nI + offsetX[i]) + (__nJ + offsetY[j]) * __nX;
          if (destination >= 0 && destination != __myID &&
              destination < __MPISize) {
            neighborFlag[i + j * 3] = true;
          }
        }
      }
    }
  }
}

void GMLS_Solver::BuildNeighborListManifold() {
  static vector<int> &neighborFlag =
      __neighbor.index.GetHandle("neighbor flag");
  static vector<vec3> &backgroundSourceCoord =
      __background.vector.GetHandle("source coord");
  static vector<int> &backgroundSourceIndex =
      __background.index.GetHandle("source index");
  static vector<vec3> &coord = __field.vector.GetHandle("coord");
  static vector<int> &globalIndex = __field.index.GetHandle("global index");
  static vector<int> &particleNum = __field.index.GetHandle("particle number");

  PetscPrintf(PETSC_COMM_WORLD, "\nBuilding neighbor list...\n");
  // set up neighbor list for communication
  backgroundSourceCoord.clear();
  vector<vector<vec3>> neighborSendParticleCoord;
  vector<vector<int>> neighborSendParticleIndex;

  vector<bool> isNeighbor;
  int neighborNum = pow(3, __dim);
  isNeighbor.resize(neighborNum);
  neighborSendParticleCoord.resize(neighborNum);
  neighborSendParticleIndex.resize(neighborNum);

  int &localParticleNum = particleNum[0];
  for (int i = 0; i < localParticleNum; i++) {
    backgroundSourceCoord.push_back(coord[i]);
    backgroundSourceIndex.push_back(globalIndex[i]);
    double xPos, yPos;
    xPos = coord[i][0];
    yPos = coord[i][1];

    isNeighbor[0] = PutParticleInNeighborList(0, [=]() {
      return xPos < __domainBoundingBox[0][0] + __cutoffDistance &&
             yPos < __domainBoundingBox[0][1] + __cutoffDistance;
    });

    isNeighbor[1] = PutParticleInNeighborList(1, [=]() {
      return yPos < __domainBoundingBox[0][1] + __cutoffDistance;
    });

    isNeighbor[2] = PutParticleInNeighborList(2, [=]() {
      return xPos > __domainBoundingBox[1][0] - __cutoffDistance &&
             yPos < __domainBoundingBox[0][1] + __cutoffDistance;
    });

    isNeighbor[3] = PutParticleInNeighborList(3, [=]() {
      return xPos < __domainBoundingBox[0][0] + __cutoffDistance;
    });

    isNeighbor[5] = PutParticleInNeighborList(5, [=]() {
      return xPos > __domainBoundingBox[1][0] - __cutoffDistance;
    });

    isNeighbor[6] = PutParticleInNeighborList(6, [=]() {
      return xPos < __domainBoundingBox[0][0] + __cutoffDistance &&
             yPos > __domainBoundingBox[1][1] - __cutoffDistance;
    });

    isNeighbor[7] = PutParticleInNeighborList(7, [=]() {
      return yPos > __domainBoundingBox[1][1] - __cutoffDistance;
    });

    isNeighbor[8] = PutParticleInNeighborList(8, [=]() {
      return xPos > __domainBoundingBox[1][0] - __cutoffDistance &&
             yPos > __domainBoundingBox[1][1] - __cutoffDistance;
    });

    for (size_t j = 0; j < isNeighbor.size(); j++) {
      if (isNeighbor[j] == true) {
        neighborSendParticleCoord[j].push_back(coord[i]);
        neighborSendParticleIndex[j].push_back(globalIndex[i]);
      }
    }
  }

  SerialOperation([=]() {
    if (backgroundSourceCoord.size() != backgroundSourceIndex.size())
      cout << "[Proc " << __myID
           << "]: wrong generation of background particles" << endl;
    else
      cout << "[Proc " << __myID << "]: generated "
           << backgroundSourceCoord.size() << " background particles." << endl;
  });
}