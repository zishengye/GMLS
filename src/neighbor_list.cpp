#include "gmls_solver.h"

#include <memory>
#include <string>

using namespace std;

void GMLS_Solver::InitNeighborList() {
  vector<int> &neighborFlag = __neighbor.index.Register("neighbor flag");

  int neighborNum = pow(3, __dim);

  neighborFlag.resize(neighborNum);
  for (int i = 0; i < neighborNum; i++) {
    neighborFlag[i] = false;
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
              neighborFlag[i + j * 3] = true;
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
                  neighborFlag[i + j * 3 + k * 9] = true;
                }
              }
            }
          }
        }
      }
    }
  }

  vector<int> &neighborCount = __neighbor.index.Register("neighbor count");
  vector<int> &neighborIndex = __neighbor.index.Register("neighbor index");
  vector<int> &neighborOffset = __neighbor.index.Register("neighbor offset");
  vector<int> &destinationIndex =
      __neighbor.index.Register("destination index");
  vector<int> &sendOffset = __neighbor.index.Register("send offset");
  vector<int> &sendCount = __neighbor.index.Register("send count");
  vector<int> &offset = __neighbor.index.Register("offset");

  neighborCount.resize(neighborNum);
  neighborIndex.resize(neighborNum);
  neighborOffset.resize(neighborNum);
  destinationIndex.resize(neighborNum);
  sendOffset.resize(neighborNum);
  sendCount.resize(neighborNum);
  offset.resize(neighborNum + 1);

  MPI_Win_create(neighborCount.data(), neighborNum * sizeof(int), sizeof(int),
                 MPI_INFO_NULL, MPI_COMM_WORLD, &__neighborWinCount);
  MPI_Win_create(neighborIndex.data(), neighborNum * sizeof(int), sizeof(int),
                 MPI_INFO_NULL, MPI_COMM_WORLD, &__neighborWinIndex);
  MPI_Win_create(neighborOffset.data(), neighborNum * sizeof(int), sizeof(int),
                 MPI_INFO_NULL, MPI_COMM_WORLD, &__neighborWinOffset);
}

void GMLS_Solver::BuildNeighborList() {
  static vector<int> &neighborFlag =
      __neighbor.index.GetHandle("neighbor flag");
  static vector<vec3> &backgroundSourceCoord =
      __background.vector.GetHandle("source coord");
  static vector<int> &backgroundSourceIndex =
      __background.index.GetHandle("source index");
  static vector<vec3> &coord = __field.vector.GetHandle("coord");
  static vector<int> &globalIndex = __field.index.GetHandle("global index");
  static vector<int> &particleNum = __field.index.GetHandle("particle number");

  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "\nBuilding neighbor list...\n");
  // set up neighbor list for communication
  backgroundSourceCoord.clear();
  backgroundSourceIndex.clear();

  vector<bool> isNeighbor;
  int neighborNum = pow(3, __dim);
  isNeighbor.resize(neighborNum);
  __neighborSendParticleIndex.resize(neighborNum);

  for (int i = 0; i < neighborNum; i++) {
    __neighborSendParticleIndex.clear();
  }

  int &localParticleNum = particleNum[0];
  for (int i = 0; i < localParticleNum; i++) {
    backgroundSourceCoord.push_back(coord[i]);
    backgroundSourceIndex.push_back(globalIndex[i]);
    double xPos, yPos, zPos;
    xPos = coord[i][0];
    yPos = coord[i][1];
    zPos = coord[i][2];
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
               zPos < __domainBoundingBox[0][2] + __cutoffDistance;
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
        __neighborSendParticleIndex[j].push_back(i);
      }
    }
  }

  vector<int> &neighborCount = __neighbor.index.GetHandle("neighbor count");
  vector<int> &destinationIndex =
      __neighbor.index.GetHandle("destination index");
  vector<int> &sendOffset = __neighbor.index.GetHandle("send offset");
  vector<int> &sendCount = __neighbor.index.GetHandle("send count");

  MPI_Barrier(MPI_COMM_WORLD);

  if (__dim == 2) {
    int offsetX[3] = {-1, 0, 1};
    int offsetY[3] = {-1, 0, 1};

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        int index = i + j * 3;
        if (neighborFlag[index] == true) {
          int destination = (__nI + offsetX[i]) + (__nJ + offsetY[j]) * __nX;
          int xScalar = (i == 1) ? 1 : (i + 2) % 4;
          int y = (j == 1) ? 1 : (j + 2) % 4;
          sendOffset[index] = xScalar + y * 3;
          sendCount[index] = __neighborSendParticleIndex[index].size();
          destinationIndex[index] = destination;

          MPI_Win_lock(MPI_LOCK_SHARED, destination, 0, __neighborWinIndex);
          MPI_Put(&__myID, 1, MPI_INT, destination, sendOffset[index], 1,
                  MPI_INT, __neighborWinIndex);

          MPI_Win_unlock(destination, __neighborWinIndex);
          MPI_Win_flush(destinationIndex[index], __neighborWinIndex);

          MPI_Win_lock(MPI_LOCK_SHARED, destination, 0, __neighborWinCount);
          MPI_Put(&sendCount[index], 1, MPI_INT, destination, sendOffset[index],
                  1, MPI_INT, __neighborWinCount);
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
          if (neighborFlag[index] == true) {
            int destination = (__nI + offsetX[i]) + (__nJ + offsetY[j]) * __nX +
                              (__nK + offsetZ[k]) * (__nX * __nY);
            int xScalar = (i == 1) ? 1 : (i + 2) % 4;
            int y = (j == 1) ? 1 : (j + 2) % 4;
            int z = (k == 1) ? 1 : (k + 2) % 4;
            sendOffset[index] = xScalar + y * 3 + z * 9;
            sendCount[index] = __neighborSendParticleIndex[index].size();
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

  static vector<int> &offset = __neighbor.index.GetHandle("offset");

  for (int i = 0; i < neighborNum; i++) {
    offset[i + 1] = offset[i] + neighborCount[i];

    if (neighborFlag[i] == true) {
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

  vector<double> recvParticleCoord;
  vector<int> recvParticleIndex;

  DataSwapAmongNeighbor(globalIndex, recvParticleIndex);
  DataSwapAmongNeighbor(coord, recvParticleCoord);

  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = 0; i < totalNeighborParticleNum; i++) {
    backgroundSourceCoord.push_back(vec3(recvParticleCoord[i * 3],
                                         recvParticleCoord[i * 3 + 1],
                                         recvParticleCoord[i * 3 + 2]));
    backgroundSourceIndex.push_back(recvParticleIndex[i]);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  SerialOperation([=]() {
    if (backgroundSourceCoord.size() != backgroundSourceIndex.size())
      cout << "[Proc " << __myID
           << "]: wrong generation of background particles "
           << backgroundSourceCoord.size() << ' '
           << backgroundSourceIndex.size() << endl;
    else
      cout << "[Proc " << __myID << "]: generated "
           << backgroundSourceCoord.size() << " background particles." << endl;
  });
}

void GMLS_Solver::DataSwapAmongNeighbor(vector<int> &sendData,
                                        vector<int> &recvData) {
  static vector<int> &neighborFlag =
      __neighbor.index.GetHandle("neighbor flag");

  static vector<int> &offset = __neighbor.index.GetHandle("offset");
  static const int neighborNum = pow(3, __dim);
  int totalNeighborParticleNum = offset[neighborNum];

  recvData.resize(totalNeighborParticleNum);

  vector<int> &destinationIndex =
      __neighbor.index.GetHandle("destination index");
  vector<int> &neighborOffset = __neighbor.index.GetHandle("neighbor offset");

  MPI_Win_create(recvData.data(), totalNeighborParticleNum * sizeof(int),
                 sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD,
                 &__neighborWinParticleSwap);

  vector<int> sendDataBlock;

  for (int i = 0; i < neighborNum; i++) {
    if (neighborFlag[i] == true) {
      sendDataBlock.clear();
      int sendCount = __neighborSendParticleIndex[i].size();
      for (int j = 0; j < sendCount; j++) {
        sendDataBlock.push_back(sendData[__neighborSendParticleIndex[i][j]]);
      }
      MPI_Win_lock(MPI_LOCK_SHARED, destinationIndex[i], 0,
                   __neighborWinParticleSwap);

      MPI_Put(sendDataBlock.data(), sendCount, MPI_INT, destinationIndex[i],
              neighborOffset[i], sendCount, MPI_INT, __neighborWinParticleSwap);

      MPI_Win_flush_all(__neighborWinParticleSwap);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Win_free(&__neighborWinParticleSwap);
}

void GMLS_Solver::DataSwapAmongNeighbor(vector<vec3> &sendData,
                                        vector<double> &recvData) {
  static vector<int> &neighborFlag =
      __neighbor.index.GetHandle("neighbor flag");

  static vector<int> &offset = __neighbor.index.GetHandle("offset");
  static const int neighborNum = pow(3, __dim);
  int totalNeighborParticleNum = offset[neighborNum];

  recvData.resize(totalNeighborParticleNum * 3);

  vector<int> &destinationIndex =
      __neighbor.index.GetHandle("destination index");
  vector<int> &neighborOffset = __neighbor.index.GetHandle("neighbor offset");

  MPI_Win_create(recvData.data(), totalNeighborParticleNum * sizeof(double) * 3,
                 sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD,
                 &__neighborWinParticleSwap);

  vector<double> sendDataBlock;

  for (int i = 0; i < neighborNum; i++) {
    if (neighborFlag[i] == true) {
      sendDataBlock.clear();
      int sendCount = __neighborSendParticleIndex[i].size();
      for (int j = 0; j < sendCount; j++) {
        for (int k = 0; k < 3; k++) {
          sendDataBlock.push_back(
              sendData[__neighborSendParticleIndex[i][j]][k]);
        }
      }
      MPI_Win_lock(MPI_LOCK_SHARED, destinationIndex[i], 0,
                   __neighborWinParticleSwap);

      MPI_Put(sendDataBlock.data(), sendCount * 3, MPI_DOUBLE,
              destinationIndex[i], neighborOffset[i] * 3, sendCount * 3,
              MPI_DOUBLE, __neighborWinParticleSwap);

      MPI_Win_flush_all(__neighborWinParticleSwap);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Win_free(&__neighborWinParticleSwap);
}

void GMLS_Solver::DataSwapAmongNeighbor(vector<vector<double>> &sendData,
                                        vector<double> &recvData,
                                        const int unitLength = 1) {
  static vector<int> &neighborFlag =
      __neighbor.index.GetHandle("neighbor flag");

  static vector<int> &offset = __neighbor.index.GetHandle("offset");
  static const int neighborNum = pow(3, __dim);
  int totalNeighborParticleNum = offset[neighborNum];

  recvData.resize(totalNeighborParticleNum * unitLength);

  vector<int> &destinationIndex =
      __neighbor.index.GetHandle("destination index");
  vector<int> &neighborOffset = __neighbor.index.GetHandle("neighbor offset");

  MPI_Win_create(recvData.data(),
                 totalNeighborParticleNum * sizeof(double) * unitLength,
                 sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD,
                 &__neighborWinParticleSwap);

  vector<double> sendDataBlock;

  for (int i = 0; i < neighborNum; i++) {
    if (neighborFlag[i] == true) {
      sendDataBlock.clear();
      int sendCount = __neighborSendParticleIndex[i].size();
      for (int j = 0; j < sendCount; j++) {
        for (int k = 0; k < unitLength; k++) {
          sendDataBlock.push_back(
              sendData[__neighborSendParticleIndex[i][j]][k]);
        }
      }
      MPI_Win_lock(MPI_LOCK_SHARED, destinationIndex[i], 0,
                   __neighborWinParticleSwap);

      MPI_Put(sendDataBlock.data(), sendCount * unitLength, MPI_DOUBLE,
              destinationIndex[i], neighborOffset[i] * unitLength,
              sendCount * unitLength, MPI_DOUBLE, __neighborWinParticleSwap);

      MPI_Win_flush_all(__neighborWinParticleSwap);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Win_free(&__neighborWinParticleSwap);
}