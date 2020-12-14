#include "gmls_solver.hpp"

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

  vector<int> &destinationIndex =
      __neighbor.index.Register("destination index");
  vector<int> &sendCount = __neighbor.index.Register("send count");
  vector<int> &recvOffset = __neighbor.index.Register("recv offset");
  vector<int> &recvCount = __neighbor.index.Register("recv count");

  destinationIndex.resize(neighborNum);
  sendCount.resize(neighborNum);
  recvOffset.resize(neighborNum + 1);
  recvCount.resize(neighborNum);

  for (int i = 0; i < neighborNum; i++)
    recvCount[i] = 0;
  for (int i = 0; i < neighborNum + 1; i++)
    recvOffset[i] = 0;
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

  for (int i = 0; i < __neighborSendParticleIndex.size(); i++) {
    __neighborSendParticleIndex[i].clear();
  }
  __neighborSendParticleIndex.resize(neighborNum);
  for (int i = 0; i < neighborNum; i++) {
    __neighborSendParticleIndex[i].clear();
  }

  int localParticleNum = coord.size();
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

  static vector<int> &destinationIndex =
      __neighbor.index.GetHandle("destination index");
  vector<int> &sendCount = __neighbor.index.GetHandle("send count");
  vector<int> &recvOffset = __neighbor.index.GetHandle("recv offset");
  vector<int> &recvCount = __neighbor.index.GetHandle("recv count");

  MPI_Barrier(MPI_COMM_WORLD);

  if (__dim == 2) {
    int offsetX[3] = {-1, 0, 1};
    int offsetY[3] = {-1, 0, 1};

    int count = 0;
    MPI_Request send_request[9];
    MPI_Request recv_request[9];
    MPI_Status status[9];

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        int index = i + j * 3;
        if (neighborFlag[index] == true) {
          int destination = (__nI + offsetX[i]) + (__nJ + offsetY[j]) * __nX;

          sendCount[index] = __neighborSendParticleIndex[index].size();
          destinationIndex[index] = destination;

          MPI_Isend(sendCount.data() + index, 1, MPI_INT,
                    destinationIndex[index], 0, MPI_COMM_WORLD,
                    send_request + count);
          MPI_Irecv(recvCount.data() + index, 1, MPI_INT,
                    destinationIndex[index], 0, MPI_COMM_WORLD,
                    recv_request + count);

          count++;
        }
      }
    }

    MPI_Waitall(count, send_request, status);
    MPI_Waitall(count, recv_request, status);
  }
  if (__dim == 3) {
    int offsetX[3] = {-1, 0, 1};
    int offsetY[3] = {-1, 0, 1};
    int offsetZ[3] = {-1, 0, 1};

    int count = 0;
    MPI_Request send_request[27];
    MPI_Request recv_request[27];
    MPI_Status status[27];

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          int index = i + j * 3 + k * 9;
          if (neighborFlag[index] == true) {
            int destination = (__nI + offsetX[i]) + (__nJ + offsetY[j]) * __nX +
                              (__nK + offsetZ[k]) * (__nX * __nY);

            sendCount[index] = __neighborSendParticleIndex[index].size();
            destinationIndex[index] = destination;

            MPI_Isend(sendCount.data() + index, 1, MPI_INT,
                      destinationIndex[index], 0, MPI_COMM_WORLD,
                      send_request + count);
            MPI_Irecv(recvCount.data() + index, 1, MPI_INT,
                      destinationIndex[index], 0, MPI_COMM_WORLD,
                      recv_request + count);

            count++;
          }
        }
      }
    }

    MPI_Waitall(count, send_request, status);
    MPI_Waitall(count, recv_request, status);
  }

  recvOffset[0] = 0;
  for (int i = 0; i < neighborNum; i++)
    recvOffset[i + 1] = recvOffset[i] + recvCount[i];

  MPI_Barrier(MPI_COMM_WORLD);
  int totalNeighborParticleNum = recvOffset[neighborNum];

  vector<vec3> recvParticleCoord;
  vector<int> recvParticleIndex;

  DataSwapAmongNeighbor(globalIndex, recvParticleIndex);
  DataSwapAmongNeighbor(coord, recvParticleCoord);

  for (int i = 0; i < totalNeighborParticleNum; i++) {
    backgroundSourceCoord.push_back(recvParticleCoord[i]);
    backgroundSourceIndex.push_back(recvParticleIndex[i]);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "finish of building neighbor list\n");
}

void GMLS_Solver::DataSwapAmongNeighbor(vector<int> &sendData,
                                        vector<int> &recvData) {
  MPI_Barrier(MPI_COMM_WORLD);

  static vector<int> &neighborFlag =
      __neighbor.index.GetHandle("neighbor flag");

  vector<int> &destinationIndex =
      __neighbor.index.GetHandle("destination index");
  vector<int> &sendCount = __neighbor.index.GetHandle("send count");
  vector<int> &recvOffset = __neighbor.index.GetHandle("recv offset");
  vector<int> &recvCount = __neighbor.index.GetHandle("recv count");

  static const int neighborNum = pow(3, __dim);
  int totalNeighborParticleNum = recvOffset[neighborNum];

  recvData.resize(totalNeighborParticleNum);

  vector<vector<int>> sendDataBlock(neighborNum);

  for (int i = 0; i < neighborNum; i++) {
    if (neighborFlag[i] == true) {
      sendDataBlock[i].resize(sendCount[i]);
      for (int j = 0; j < sendCount[i]; j++) {
        sendDataBlock[i][j] = sendData[__neighborSendParticleIndex[i][j]];
      }
    }
  }

  vector<MPI_Request> send_request(neighborNum);
  vector<MPI_Request> recv_request(neighborNum);
  vector<MPI_Status> status(neighborNum);

  int count = 0;

  for (int i = 0; i < neighborNum; i++) {
    if (neighborFlag[i] == true) {
      MPI_Isend(sendDataBlock[i].data(), sendCount[i], MPI_INT,
                destinationIndex[i], 0, MPI_COMM_WORLD,
                send_request.data() + count);
      MPI_Irecv(recvData.data() + recvOffset[i], recvCount[i], MPI_INT,
                destinationIndex[i], 0, MPI_COMM_WORLD,
                recv_request.data() + count);
      count++;
    }
  }

  MPI_Waitall(count, send_request.data(), status.data());
  MPI_Waitall(count, recv_request.data(), status.data());

  MPI_Barrier(MPI_COMM_WORLD);
}

void GMLS_Solver::DataSwapAmongNeighbor(vector<double> &sendData,
                                        vector<double> &recvData) {
  MPI_Barrier(MPI_COMM_WORLD);

  static vector<int> &neighborFlag =
      __neighbor.index.GetHandle("neighbor flag");

  vector<int> &destinationIndex =
      __neighbor.index.GetHandle("destination index");
  vector<int> &sendCount = __neighbor.index.GetHandle("send count");
  vector<int> &recvOffset = __neighbor.index.GetHandle("recv offset");
  vector<int> &recvCount = __neighbor.index.GetHandle("recv count");

  static const int neighborNum = pow(3, __dim);
  int totalNeighborParticleNum = recvOffset[neighborNum];

  recvData.resize(totalNeighborParticleNum);

  vector<vector<double>> sendDataBlock(neighborNum);

  for (int i = 0; i < neighborNum; i++) {
    if (neighborFlag[i] == true) {
      sendDataBlock[i].resize(sendCount[i]);
      for (int j = 0; j < sendCount[i]; j++) {
        sendDataBlock[i][j] = sendData[__neighborSendParticleIndex[i][j]];
      }
    }
  }

  vector<MPI_Request> send_request(neighborNum);
  vector<MPI_Request> recv_request(neighborNum);
  vector<MPI_Status> status(neighborNum);

  int count = 0;

  for (int i = 0; i < neighborNum; i++) {
    if (neighborFlag[i] == true) {
      MPI_Isend(sendDataBlock[i].data(), sendCount[i], MPI_DOUBLE,
                destinationIndex[i], 0, MPI_COMM_WORLD,
                send_request.data() + count);
      MPI_Irecv(recvData.data() + recvOffset[i], recvCount[i], MPI_DOUBLE,
                destinationIndex[i], 0, MPI_COMM_WORLD,
                recv_request.data() + count);
      count++;
    }
  }

  MPI_Waitall(count, send_request.data(), status.data());
  MPI_Waitall(count, recv_request.data(), status.data());

  MPI_Barrier(MPI_COMM_WORLD);
}

void GMLS_Solver::DataSwapAmongNeighbor(vector<vec3> &sendData,
                                        vector<vec3> &recvData) {
  MPI_Barrier(MPI_COMM_WORLD);

  static vector<int> &neighborFlag =
      __neighbor.index.GetHandle("neighbor flag");

  vector<int> &destinationIndex =
      __neighbor.index.GetHandle("destination index");
  vector<int> &sendCount = __neighbor.index.GetHandle("send count");
  vector<int> &recvOffset = __neighbor.index.GetHandle("recv offset");
  vector<int> &recvCount = __neighbor.index.GetHandle("recv count");

  static const int neighborNum = pow(3, __dim);
  int totalNeighborParticleNum = recvOffset[neighborNum];

  vector<double> tempRecvData(totalNeighborParticleNum * 3);

  vector<vector<double>> sendDataBlock(neighborNum);

  for (int i = 0; i < neighborNum; i++) {
    if (neighborFlag[i] == true) {
      sendDataBlock[i].resize(sendCount[i] * 3);
      for (int j = 0; j < sendCount[i]; j++) {
        for (int k = 0; k < 3; k++) {
          sendDataBlock[i][j * 3 + k] =
              sendData[__neighborSendParticleIndex[i][j]][k];
        }
      }
    }
  }

  vector<MPI_Request> send_request(neighborNum);
  vector<MPI_Request> recv_request(neighborNum);
  vector<MPI_Status> status(neighborNum);

  int count = 0;

  for (int i = 0; i < neighborNum; i++) {
    if (neighborFlag[i] == true) {
      MPI_Isend(sendDataBlock[i].data(), sendCount[i] * 3, MPI_DOUBLE,
                destinationIndex[i], 0, MPI_COMM_WORLD,
                send_request.data() + count);
      MPI_Irecv(tempRecvData.data() + recvOffset[i] * 3, recvCount[i] * 3,
                MPI_DOUBLE, destinationIndex[i], 0, MPI_COMM_WORLD,
                recv_request.data() + count);
      count++;
    }
  }

  MPI_Waitall(count, send_request.data(), status.data());
  MPI_Waitall(count, recv_request.data(), status.data());

  recvData.resize(totalNeighborParticleNum);
  for (int i = 0; i < totalNeighborParticleNum; i++) {
    for (int j = 0; j < 3; j++) {
      recvData[i][j] = tempRecvData[i * 3 + j];
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

void GMLS_Solver::DataSwapAmongNeighbor(vector<vector<double>> &sendData,
                                        vector<vector<double>> &recvData,
                                        const int unitLength = 1) {
  MPI_Barrier(MPI_COMM_WORLD);

  static vector<int> &neighborFlag =
      __neighbor.index.GetHandle("neighbor flag");

  vector<int> &destinationIndex =
      __neighbor.index.GetHandle("destination index");
  vector<int> &sendCount = __neighbor.index.GetHandle("send count");
  vector<int> &recvOffset = __neighbor.index.GetHandle("recv offset");
  vector<int> &recvCount = __neighbor.index.GetHandle("recv count");

  static const int neighborNum = pow(3, __dim);
  int totalNeighborParticleNum = recvOffset[neighborNum];

  vector<double> tempRecvData(totalNeighborParticleNum * unitLength);

  vector<vector<double>> sendDataBlock(neighborNum);

  for (int i = 0; i < neighborNum; i++) {
    if (neighborFlag[i] == true) {
      sendDataBlock[i].resize(sendCount[i] * unitLength);
      for (int j = 0; j < sendCount[i]; j++) {
        for (int k = 0; k < unitLength; k++) {
          sendDataBlock[i][j * unitLength + k] =
              sendData[__neighborSendParticleIndex[i][j]][k];
        }
      }
    }
  }

  vector<MPI_Request> send_request(neighborNum);
  vector<MPI_Request> recv_request(neighborNum);
  vector<MPI_Status> status(neighborNum);

  int count = 0;

  for (int i = 0; i < neighborNum; i++) {
    if (neighborFlag[i] == true) {
      MPI_Isend(sendDataBlock[i].data(), sendCount[i] * unitLength, MPI_DOUBLE,
                destinationIndex[i], 0, MPI_COMM_WORLD,
                send_request.data() + count);
      MPI_Irecv(tempRecvData.data() + recvOffset[i] * unitLength,
                recvCount[i] * unitLength, MPI_DOUBLE, destinationIndex[i], 0,
                MPI_COMM_WORLD, recv_request.data() + count);
      count++;
    }
  }

  MPI_Waitall(count, send_request.data(), status.data());
  MPI_Waitall(count, recv_request.data(), status.data());

  recvData.resize(totalNeighborParticleNum);
  for (int i = 0; i < totalNeighborParticleNum; i++) {
    recvData[i].resize(unitLength);
    for (int j = 0; j < unitLength; j++) {
      recvData[i][j] = tempRecvData[i * unitLength + j];
    }
  }
}