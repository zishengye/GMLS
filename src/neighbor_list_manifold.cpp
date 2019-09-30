#include "GMLS_solver.h"

#include <memory>
#include <string>

using namespace std;

void GMLS_Solver::InitNeighborListManifold() {
  int neighborNum = pow(3, 2);

  __neighborFlag.resize(neighborNum);
  for (int i = 0; i < neighborNum; i++) {
    __neighborFlag[i] = false;
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
            __neighborFlag[i + j * 3] = true;
          }
        }
      }
    }
  }
}

void GMLS_Solver::BuildNeighborListManifold() {
  PetscPrintf(PETSC_COMM_WORLD, "\nBuilding neighbor list...\n");
  // set up neighbor list for communication
  __backgroundParticle.clear();
  for (size_t i = 0; i < __neighborSendParticle.size(); i++) {
    __neighborSendParticle[i].clear();
  }
  vector<bool> isNeighbor;
  int neighborNum = pow(3, 2);
  isNeighbor.resize(neighborNum);
  __neighborSendParticle.resize(neighborNum);

  for (int i = 0; i < __fluid.localParticleNum; i++) {
    __backgroundParticle.coord.push_back(__fluid.X[i]);
    __backgroundParticle.index.push_back(__fluid.globalIndex[i]);
  }

  SerialOperation([=]() {
    if (__backgroundParticle.coord.size() != __backgroundParticle.index.size())
      cout << "[Proc " << __myID
           << "]: wrong generation of background particles" << endl;
    else
      cout << "[Proc " << __myID << "]: generated "
           << __backgroundParticle.coord.size() << " background particles."
           << endl;
  });
}