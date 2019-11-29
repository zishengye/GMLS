#include "GMLS_solver.h"

#include <memory>
#include <string>

using namespace std;

void GMLS_Solver::InitNeighborList() {
  int neighborNum = pow(3, __dim);

  __neighborFlag.resize(neighborNum);
  for (int i = 0; i < neighborNum; i++) {
    __neighborFlag[i] = false;
  }

  if (__dim == 2) {
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
  if (__dim == 3) {
    int offsetX[3] = {-1, 0, 1};
    int offsetY[3] = {-1, 0, 1};
    int offsetZ[3] = {-1, 0, 1};

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          if (__nI + offsetX[i] >= 0 && __nI + offsetX[i] < __nX) {
            if (__nJ + offsetY[j] >= 0 && __nJ + offsetY[j] < __nY) {
              if (__nK + offsetZ[k] >= 0 && __nK + offsetZ[k] < __nZ) {
                int destination = (__nI + offsetX[i]) +
                                  (__nJ + offsetY[j]) * __nX +
                                  (__nK + offsetZ[k]) * (__nX * __nY);
                if (destination >= 0 && destination != __myID &&
                    destination < __MPISize) {
                  __neighborFlag[i + j * 3 + k * 9] = true;
                }
              }
            }
          }
        }
      }
    }
  }
}

void GMLS_Solver::BuildNeighborList() {
  PetscPrintf(PETSC_COMM_WORLD, "\nBuilding neighbor list...\n");
  // set up neighbor list for communication
  __backgroundParticle.clear();
  for (size_t i = 0; i < __neighborSendParticle.size(); i++) {
    __neighborSendParticle[i].clear();
  }
  vector<bool> isNeighbor;
  int neighborNum = pow(3, __dim);
  isNeighbor.resize(neighborNum);
  __neighborSendParticle.resize(neighborNum);

  for (int i = 0; i < __particle.localParticleNum; i++) {
    __backgroundParticle.coord.push_back(__particle.X[i]);
    __backgroundParticle.index.push_back(__particle.globalIndex[i]);
    double xPos, yPos, zPos;
    xPos = __particle.X[i][0];
    yPos = __particle.X[i][1];
    zPos = __particle.X[i][2];
    for (auto it = isNeighbor.begin(); it != isNeighbor.end(); it++) {
      *it = false;
    }
    if (__dim == 2) {
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
    }
    if (__dim == 3) {
      isNeighbor[0] = PutParticleInNeighborList(0, [=]() {
        return xPos < __domainBoundingBox[0][0] + __cutoffDistance &&
               yPos < __domainBoundingBox[0][1] + __cutoffDistance &&
               zPos < __domainBoundingBox[0][2] + __cutoffDistance;
      });

      isNeighbor[1] = PutParticleInNeighborList(1, [=]() {
        return yPos < __domainBoundingBox[0][1] + __cutoffDistance &&
               zPos < __domainBoundingBox[0][1] + __cutoffDistance;
      });

      isNeighbor[2] = PutParticleInNeighborList(2, [=]() {
        return xPos > __domainBoundingBox[1][0] - __cutoffDistance &&
               yPos < __domainBoundingBox[0][1] + __cutoffDistance &&
               zPos < __domainBoundingBox[0][2] + __cutoffDistance;
      });

      isNeighbor[3] = PutParticleInNeighborList(3, [=]() {
        return xPos < __domainBoundingBox[0][0] + __cutoffDistance &&
               zPos < __domainBoundingBox[0][2] + __cutoffDistance;
      });

      isNeighbor[4] = PutParticleInNeighborList(4, [=]() {
        return zPos < __domainBoundingBox[0][2] + __cutoffDistance;
      });

      isNeighbor[5] = PutParticleInNeighborList(5, [=]() {
        return xPos > __domainBoundingBox[1][0] - __cutoffDistance &&
               zPos < __domainBoundingBox[0][2] + __cutoffDistance;
      });

      isNeighbor[6] = PutParticleInNeighborList(6, [=]() {
        return xPos < __domainBoundingBox[0][0] + __cutoffDistance &&
               yPos > __domainBoundingBox[1][1] - __cutoffDistance &&
               zPos < __domainBoundingBox[0][2] + __cutoffDistance;
      });

      isNeighbor[7] = PutParticleInNeighborList(7, [=]() {
        return yPos > __domainBoundingBox[1][1] - __cutoffDistance &&
               zPos < __domainBoundingBox[0][2] + __cutoffDistance;
      });

      isNeighbor[8] = PutParticleInNeighborList(8, [=]() {
        return xPos > __domainBoundingBox[1][0] - __cutoffDistance &&
               yPos > __domainBoundingBox[1][1] - __cutoffDistance &&
               zPos < __domainBoundingBox[0][2] + __cutoffDistance;
      });

      isNeighbor[9] = PutParticleInNeighborList(9, [=]() {
        return xPos < __domainBoundingBox[0][0] + __cutoffDistance &&
               yPos < __domainBoundingBox[0][1] + __cutoffDistance;
      });

      isNeighbor[10] = PutParticleInNeighborList(10, [=]() {
        return yPos < __domainBoundingBox[0][1] + __cutoffDistance;
      });

      isNeighbor[11] = PutParticleInNeighborList(11, [=]() {
        return xPos > __domainBoundingBox[1][0] - __cutoffDistance &&
               yPos < __domainBoundingBox[0][1] + __cutoffDistance;
      });

      isNeighbor[12] = PutParticleInNeighborList(12, [=]() {
        return xPos < __domainBoundingBox[0][0] + __cutoffDistance;
      });

      isNeighbor[14] = PutParticleInNeighborList(14, [=]() {
        return xPos > __domainBoundingBox[1][0] - __cutoffDistance;
      });

      isNeighbor[15] = PutParticleInNeighborList(15, [=]() {
        return xPos < __domainBoundingBox[0][0] + __cutoffDistance &&
               yPos > __domainBoundingBox[1][1] - __cutoffDistance;
      });

      isNeighbor[16] = PutParticleInNeighborList(16, [=]() {
        return yPos > __domainBoundingBox[1][1] - __cutoffDistance;
      });

      isNeighbor[17] = PutParticleInNeighborList(17, [=]() {
        return xPos > __domainBoundingBox[1][0] - __cutoffDistance &&
               yPos > __domainBoundingBox[1][1] - __cutoffDistance;
      });

      isNeighbor[18] = PutParticleInNeighborList(18, [=]() {
        return xPos < __domainBoundingBox[0][0] + __cutoffDistance &&
               yPos < __domainBoundingBox[0][1] + __cutoffDistance &&
               zPos > __domainBoundingBox[1][2] - __cutoffDistance;
      });

      isNeighbor[19] = PutParticleInNeighborList(19, [=]() {
        return yPos < __domainBoundingBox[0][1] + __cutoffDistance &&
               zPos > __domainBoundingBox[1][2] - __cutoffDistance;
      });

      isNeighbor[20] = PutParticleInNeighborList(20, [=]() {
        return xPos > __domainBoundingBox[1][0] - __cutoffDistance &&
               yPos < __domainBoundingBox[0][1] + __cutoffDistance &&
               zPos > __domainBoundingBox[1][2] - __cutoffDistance;
      });

      isNeighbor[21] = PutParticleInNeighborList(21, [=]() {
        return xPos < __domainBoundingBox[0][0] + __cutoffDistance &&
               zPos > __domainBoundingBox[1][2] - __cutoffDistance;
      });

      isNeighbor[22] = PutParticleInNeighborList(22, [=]() {
        return zPos > __domainBoundingBox[1][2] - __cutoffDistance;
      });

      isNeighbor[23] = PutParticleInNeighborList(23, [=]() {
        return xPos > __domainBoundingBox[1][0] - __cutoffDistance &&
               zPos > __domainBoundingBox[1][2] - __cutoffDistance;
      });

      isNeighbor[24] = PutParticleInNeighborList(24, [=]() {
        return xPos < __domainBoundingBox[0][0] + __cutoffDistance &&
               yPos > __domainBoundingBox[1][1] - __cutoffDistance &&
               zPos > __domainBoundingBox[1][2] - __cutoffDistance;
      });

      isNeighbor[25] = PutParticleInNeighborList(25, [=]() {
        return yPos > __domainBoundingBox[1][1] - __cutoffDistance &&
               zPos > __domainBoundingBox[1][2] - __cutoffDistance;
      });

      isNeighbor[26] = PutParticleInNeighborList(26, [=]() {
        return xPos > __domainBoundingBox[1][0] - __cutoffDistance &&
               yPos > __domainBoundingBox[1][1] - __cutoffDistance &&
               zPos > __domainBoundingBox[1][2] - __cutoffDistance;
      });
    }

    for (size_t j = 0; j < isNeighbor.size(); j++) {
      if (isNeighbor[j] == true) {
        __neighborSendParticle[j].coord.push_back(__particle.X[i]);
        __neighborSendParticle[j].index.push_back(__particle.globalIndex[i]);
      }
    }
  }

  if (__MPISize > 1) {
    vector<int> neighborCount, neighborIndex, neighborOffset, destinationIndex,
        sendOffset, sendCount;
    neighborCount.resize(neighborNum);
    neighborIndex.resize(neighborNum);
    neighborOffset.resize(neighborNum);
    destinationIndex.resize(neighborNum);
    sendOffset.resize(neighborNum);
    sendCount.resize(neighborNum);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_create(neighborCount.data(), neighborNum * sizeof(int), sizeof(int),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &__neighborWinCount);
    MPI_Win_create(neighborIndex.data(), neighborNum * sizeof(int), sizeof(int),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &__neighborWinIndex);
    MPI_Win_create(neighborOffset.data(), neighborNum * sizeof(int),
                   sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD,
                   &__neighborWinOffset);

    if (__dim == 2) {
      int offsetX[3] = {-1, 0, 1};
      int offsetY[3] = {-1, 0, 1};

      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          int index = i + j * 3;
          if (__neighborFlag[index] == true) {
            int destination = (__nI + offsetX[i]) + (__nJ + offsetY[j]) * __nX;
            int xScalar = (i == 1) ? 1 : (i + 2) % 4;
            int y = (j == 1) ? 1 : (j + 2) % 4;
            sendOffset[index] = xScalar + y * 3;
            sendCount[index] = __neighborSendParticle[index].coord.size();
            destinationIndex[index] = destination;

            MPI_Win_lock(MPI_LOCK_SHARED, destination, 0, __neighborWinIndex);
            MPI_Put(&__myID, 1, MPI_INT, destination, sendOffset[index], 1,
                    MPI_INT, __neighborWinIndex);

            MPI_Win_unlock(destination, __neighborWinIndex);
            MPI_Win_flush(destinationIndex[index], __neighborWinIndex);

            MPI_Win_lock(MPI_LOCK_SHARED, destination, 0, __neighborWinCount);
            MPI_Put(&sendCount[index], 1, MPI_INT, destination,
                    sendOffset[index], 1, MPI_INT, __neighborWinCount);
            MPI_Win_unlock(destination, __neighborWinCount);
            MPI_Win_flush(destinationIndex[i + j * 3], __neighborWinCount);
          }
        }
      }
    }
    if (__dim == 3) {
      int offsetX[3] = {-1, 0, 1};
      int offsetY[3] = {-1, 0, 1};
      int offsetZ[3] = {-1, 0, 1};

      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          for (int k = 0; k < 3; k++) {
            int index = i + j * 3 + k * 9;
            if (__neighborFlag[index] == true) {
              int destination = (__nI + offsetX[i]) +
                                (__nJ + offsetY[j]) * __nX +
                                (__nK + offsetZ[k]) * (__nX * __nY);
              int xScalar = (i == 1) ? 1 : (i + 2) % 4;
              int y = (j == 1) ? 1 : (j + 2) % 4;
              int z = (k == 1) ? 1 : (k + 2) % 4;
              sendOffset[index] = xScalar + y * 3 + z * 9;
              sendCount[index] = __neighborSendParticle[index].coord.size();
              destinationIndex[index] = destination;

              MPI_Win_lock(MPI_LOCK_SHARED, destination, 0, __neighborWinIndex);
              MPI_Put(&__myID, 1, MPI_INT, destination, sendOffset[index], 1,
                      MPI_INT, __neighborWinIndex);

              MPI_Win_unlock(destination, __neighborWinIndex);
              MPI_Win_flush(destinationIndex[index], __neighborWinIndex);

              MPI_Win_lock(MPI_LOCK_SHARED, destination, 0, __neighborWinCount);
              MPI_Put(&sendCount[index], 1, MPI_INT, destination,
                      sendOffset[index], 1, MPI_INT, __neighborWinCount);
              MPI_Win_unlock(destination, __neighborWinCount);
              MPI_Win_flush(destinationIndex[index], __neighborWinCount);
            }
          }
        }
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    vector<int> offset;
    offset.resize(neighborNum + 1);
    for (int i = 0; i < neighborNum; i++) {
      offset[i + 1] = offset[i] + neighborCount[i];

      if (__neighborFlag[i] == true) {
        MPI_Win_lock(MPI_LOCK_SHARED, destinationIndex[i], 0,
                     __neighborWinOffset);
        MPI_Put(&offset[i], 1, MPI_INT, destinationIndex[i], sendOffset[i], 1,
                MPI_INT, __neighborWinOffset);
        MPI_Win_unlock(destinationIndex[i], __neighborWinOffset);
      }
      MPI_Win_flush_all(__neighborWinOffset);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int totalNeighborParticleNum = offset[neighborNum];

    vector<double> sendParticleCoord;
    vector<double> recvParticleCoord;
    vector<int> recvParticleIndex;
    recvParticleCoord.resize(totalNeighborParticleNum * 3);
    recvParticleIndex.resize(totalNeighborParticleNum);
    MPI_Win_create(recvParticleCoord.data(),
                   totalNeighborParticleNum * 3 * sizeof(double),
                   sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD,
                   &__neighborWinParticleCoord);
    MPI_Win_create(recvParticleIndex.data(),
                   totalNeighborParticleNum * sizeof(int), sizeof(int),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &__neighborWinParticleIndex);
    for (int i = 0; i < neighborNum; i++) {
      if (__neighborFlag[i] == true) {
        sendParticleCoord.clear();
        int sendCount = __neighborSendParticle[i].coord.size();
        for (int j = 0; j < sendCount; j++) {
          for (int k = 0; k < 3; k++) {
            sendParticleCoord.push_back(__neighborSendParticle[i].coord[j][k]);
          }
        }
        MPI_Win_lock(MPI_LOCK_SHARED, destinationIndex[i], 0,
                     __neighborWinParticleCoord);
        MPI_Win_lock(MPI_LOCK_SHARED, destinationIndex[i], 0,
                     __neighborWinParticleIndex);

        MPI_Put(sendParticleCoord.data(), sendCount * 3, MPI_DOUBLE,
                destinationIndex[i], neighborOffset[i] * 3, sendCount * 3,
                MPI_DOUBLE, __neighborWinParticleCoord);
        MPI_Put(__neighborSendParticle[i].index.data(), sendCount, MPI_INT,
                destinationIndex[i], neighborOffset[i], sendCount, MPI_INT,
                __neighborWinParticleIndex);

        MPI_Win_unlock(destinationIndex[i], __neighborWinParticleCoord);
        MPI_Win_unlock(destinationIndex[i], __neighborWinParticleIndex);
      }
      MPI_Win_flush_all(__neighborWinParticleIndex);
      MPI_Win_flush_all(__neighborWinParticleIndex);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < totalNeighborParticleNum; i++) {
      __backgroundParticle.coord.push_back(vec3(recvParticleCoord[i * 3],
                                                recvParticleCoord[i * 3 + 1],
                                                recvParticleCoord[i * 3 + 2]));
      __backgroundParticle.index.push_back(recvParticleIndex[i]);
    }

    MPI_Win_free(&__neighborWinCount);
    MPI_Win_free(&__neighborWinIndex);
    MPI_Win_free(&__neighborWinOffset);
    MPI_Win_free(&__neighborWinParticleCoord);
    MPI_Win_free(&__neighborWinParticleIndex);

    MPI_Barrier(MPI_COMM_WORLD);
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