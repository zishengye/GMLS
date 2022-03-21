#include "Ghost.hpp"

Ghost::Ghost() {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);
}

void Ghost::Init(const HostRealMatrix &targetCoords,
                 const HostRealVector &targetSpacing,
                 const HostRealMatrix &sourceCoords, const double multiplier,
                 const int dimension) {
  double domainLow[3], domainHigh[3];

  for (int i = 0; i < 3; i++) {
    domainLow[i] = 1e9;
    domainHigh[i] = -1e9;
  }

  const int localSourceNum = sourceCoords.extent(0);
  const int localTargetNum = targetCoords.extent(0);

  for (int i = 0; i < localTargetNum; i++) {
    const double offset = multiplier * targetSpacing(i);
    if (domainHigh[0] < targetCoords(i, 0) + offset) {
      domainHigh[0] = targetCoords(i, 0) + offset;
    }
    if (domainHigh[1] < targetCoords(i, 1) + offset) {
      domainHigh[1] = targetCoords(i, 1) + offset;
    }
    if (domainHigh[2] < targetCoords(i, 2) + offset) {
      domainHigh[2] = targetCoords(i, 2) + offset;
    }

    if (domainLow[0] > targetCoords(i, 0) - offset) {
      domainLow[0] = targetCoords(i, 0) - offset;
    }
    if (domainLow[1] > targetCoords(i, 1) - offset) {
      domainLow[1] = targetCoords(i, 1) - offset;
    }
    if (domainLow[2] > targetCoords(i, 2) - offset) {
      domainLow[2] = targetCoords(i, 2) - offset;
    }
  }

  std::vector<double> rankDomain(mpiSize_ * 6);
  MPI_Allgather(&domainLow[0], 1, MPI_DOUBLE, rankDomain.data(), 1, MPI_DOUBLE,
                MPI_COMM_WORLD);
  MPI_Allgather(&domainLow[1], 1, MPI_DOUBLE, rankDomain.data() + mpiSize_, 1,
                MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Allgather(&domainLow[2], 1, MPI_DOUBLE, rankDomain.data() + mpiSize_ * 2,
                1, MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Allgather(&domainHigh[0], 1, MPI_DOUBLE, rankDomain.data() + mpiSize_ * 3,
                1, MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Allgather(&domainHigh[1], 1, MPI_DOUBLE, rankDomain.data() + mpiSize_ * 4,
                1, MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Allgather(&domainHigh[2], 1, MPI_DOUBLE, rankDomain.data() + mpiSize_ * 5,
                1, MPI_DOUBLE, MPI_COMM_WORLD);

  std::vector<double[3]> rankGhostDomainLow(mpiSize_);
  std::vector<double[3]> rankGhostDomainHigh(mpiSize_);
  for (int i = 0; i < mpiSize_; i++) {
    for (int j = 0; j < 3; j++) {
      rankGhostDomainLow[i][j] = rankDomain[i + j * mpiSize_];
      rankGhostDomainHigh[i][j] = rankDomain[i + (j + 3) * mpiSize_];
    }
  }

  std::vector<std::vector<int>> rankGhostOutMap;
  rankGhostOutMap.resize(mpiSize_);
  for (int i = 0; i < localSourceNum; i++) {
    for (int j = 0; j < mpiSize_; j++) {
      if (dimension == 2) {
        if (sourceCoords(i, 0) > rankGhostDomainLow[j][0] &&
            sourceCoords(i, 1) > rankGhostDomainLow[j][1] &&
            sourceCoords(i, 0) < rankGhostDomainHigh[j][0] &&
            sourceCoords(i, 1) < rankGhostDomainHigh[j][1]) {
          rankGhostOutMap[j].push_back(i);
        }
      } else if (dimension == 3) {
        if (sourceCoords(i, 0) > rankGhostDomainLow[j][0] &&
            sourceCoords(i, 1) > rankGhostDomainLow[j][1] &&
            sourceCoords(i, 2) > rankGhostDomainLow[j][2] &&
            sourceCoords(i, 0) < rankGhostDomainHigh[j][0] &&
            sourceCoords(i, 1) < rankGhostDomainHigh[j][1] &&
            sourceCoords(i, 2) < rankGhostDomainHigh[j][2]) {
          rankGhostOutMap[j].push_back(i);
        }
      }
    }
  }

  std::vector<int> rankGhostInNum(mpiSize_);
  for (int i = 0; i < mpiSize_; i++) {
    int outNum = (i != mpiRank_) ? (int)(rankGhostOutMap[i].size()) : 0;
    MPI_Gather(&outNum, 1, MPI_INT, rankGhostInNum.data(), 1, MPI_INT, i,
               MPI_COMM_WORLD);
  }

  ghostOutGraph_.clear();
  ghostInGraph_.clear();
  ghostOutNum_.clear();
  ghostInNum_.clear();

  for (int i = 0; i < mpiSize_; i++) {
    if (rankGhostOutMap[i].size() != 0) {
      if (i != mpiRank_) {
        ghostOutGraph_.push_back(i);
        ghostOutNum_.push_back(rankGhostOutMap[i].size());
      } else {
        localReserveNum_ = rankGhostOutMap[i].size();
      }
    }
  }

  remoteInNum_ = 0;
  for (int i = 0; i < mpiSize_; i++) {
    if (rankGhostInNum[i] != 0) {
      ghostInGraph_.push_back(i);
      ghostInNum_.push_back(rankGhostInNum[i]);
      remoteInNum_ += rankGhostInNum[i];
    }
  }

  ghostOutOffset_.resize(ghostOutGraph_.size() + 1);
  ghostInOffset_.resize(ghostInGraph_.size() + 1);
  ghostOutOffset_[0] = 0;
  for (int i = 0; i < ghostOutGraph_.size(); i++) {
    ghostOutOffset_[i + 1] = ghostOutOffset_[i] + ghostOutNum_[i];
  }
  ghostInOffset_[0] = 0;
  for (int i = 0; i < ghostInGraph_.size(); i++) {
    ghostInOffset_[i + 1] = ghostInOffset_[i] + ghostInNum_[i];
  }

  ghostMap_.resize(ghostOutOffset_[ghostOutNum_.size()]);
  reserveMap_.resize(rankGhostOutMap[mpiRank_].size());
  for (int i = 0; i < mpiSize_; i++) {
    auto ite = (size_t)(
        std::lower_bound(ghostOutGraph_.begin(), ghostOutGraph_.end(), i) -
        ghostOutGraph_.begin());
    if (i != mpiRank_) {
      for (int j = 0; j < rankGhostOutMap[i].size(); j++) {
        ghostMap_[ghostOutOffset_[ite] + j] = rankGhostOutMap[i][j];
      }
    } else {
      for (int j = 0; j < rankGhostOutMap[i].size(); j++) {
        reserveMap_[j] = rankGhostOutMap[i][j];
      }
    }
  }
}

void Ghost::ApplyGhost(const HostRealMatrix &sourceData,
                       HostRealMatrix &ghostData) {
  unsigned int unitLength = sourceData.extent(1);

  std::vector<MPI_Request> sendRequest(ghostOutGraph_.size());
  std::vector<MPI_Request> recvRequest(ghostInGraph_.size());
  std::vector<MPI_Status> sendStatus(ghostOutGraph_.size());
  std::vector<MPI_Status> recvStatus(ghostInGraph_.size());

  std::vector<double> flattenedSendData(ghostOutOffset_[ghostOutGraph_.size()] *
                                        unitLength);
  std::vector<double> flattenedRecvData(ghostInOffset_[ghostInGraph_.size()] *
                                        unitLength);

  // prepare send data
  for (int i = 0; i < ghostMap_.size(); i++) {
    for (int j = 0; j < unitLength; j++) {
      flattenedSendData[i * unitLength + j] = sourceData(ghostMap_[i], j);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // send and recv data
  for (int i = 0; i < ghostOutGraph_.size(); i++) {
    MPI_Isend(flattenedSendData.data() + ghostOutOffset_[i] * unitLength,
              ghostOutNum_[i] * unitLength, MPI_DOUBLE, ghostOutGraph_[i], 0,
              MPI_COMM_WORLD, sendRequest.data() + i);
  }

  for (int i = 0; i < ghostInGraph_.size(); i++) {
    MPI_Irecv(flattenedRecvData.data() + ghostInOffset_[i] * unitLength,
              ghostInNum_[i] * unitLength, MPI_DOUBLE, ghostInGraph_[i], 0,
              MPI_COMM_WORLD, recvRequest.data() + i);
  }

  MPI_Waitall(sendRequest.size(), sendRequest.data(), sendStatus.data());
  MPI_Waitall(recvRequest.size(), recvRequest.data(), recvStatus.data());
  MPI_Barrier(MPI_COMM_WORLD);

  Kokkos::resize(ghostData, localReserveNum_ + remoteInNum_, unitLength);
  // store reserved local data
  for (int i = 0; i < localReserveNum_; i++) {
    for (int j = 0; j < unitLength; j++) {
      ghostData(i, j) = sourceData(reserveMap_[i], j);
    }
  }

  // store remote data
  for (int i = 0; i < remoteInNum_; i++) {
    for (int j = 0; j < unitLength; j++) {
      ghostData(localReserveNum_ + i, j) =
          flattenedRecvData[i * unitLength + j];
    }
  }
}

void Ghost::ApplyGhost(const HostRealVector &sourceData,
                       HostRealVector &ghostData) {
  std::vector<MPI_Request> sendRequest(ghostOutGraph_.size());
  std::vector<MPI_Request> recvRequest(ghostInGraph_.size());
  std::vector<MPI_Status> sendStatus(ghostOutGraph_.size());
  std::vector<MPI_Status> recvStatus(ghostInGraph_.size());

  std::vector<double> flattenedSendData(ghostOutOffset_[ghostOutGraph_.size()]);
  std::vector<double> flattenedRecvData(ghostInOffset_[ghostInGraph_.size()]);

  // prepare send data
  for (int i = 0; i < ghostMap_.size(); i++) {
    flattenedSendData[i] = sourceData(ghostMap_[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // send and recv data
  for (int i = 0; i < ghostOutGraph_.size(); i++) {
    MPI_Isend(flattenedSendData.data() + ghostOutOffset_[i], ghostOutNum_[i],
              MPI_DOUBLE, ghostOutGraph_[i], 0, MPI_COMM_WORLD,
              sendRequest.data() + i);
  }

  for (int i = 0; i < ghostInGraph_.size(); i++) {
    MPI_Irecv(flattenedRecvData.data() + ghostInOffset_[i], ghostInNum_[i],
              MPI_DOUBLE, ghostInGraph_[i], 0, MPI_COMM_WORLD,
              recvRequest.data() + i);
  }

  MPI_Waitall(sendRequest.size(), sendRequest.data(), sendStatus.data());
  MPI_Waitall(recvRequest.size(), recvRequest.data(), recvStatus.data());
  MPI_Barrier(MPI_COMM_WORLD);

  Kokkos::resize(ghostData, localReserveNum_ + remoteInNum_);
  // store reserved local data
  for (int i = 0; i < localReserveNum_; i++) {
    ghostData(i) = sourceData(reserveMap_[i]);
  }

  // store remote data
  for (int i = 0; i < remoteInNum_; i++) {
    ghostData(localReserveNum_ + i) = flattenedRecvData[i];
  }
}

void Ghost::ApplyGhost(const HostIntVector &sourceData,
                       HostIntVector &ghostData) {
  std::vector<MPI_Request> sendRequest(ghostOutGraph_.size());
  std::vector<MPI_Request> recvRequest(ghostInGraph_.size());
  std::vector<MPI_Status> sendStatus(ghostOutGraph_.size());
  std::vector<MPI_Status> recvStatus(ghostInGraph_.size());

  std::vector<int> flattenedSendData(ghostOutOffset_[ghostOutGraph_.size()]);
  std::vector<int> flattenedRecvData(ghostInOffset_[ghostInGraph_.size()]);

  // prepare send data
  for (int i = 0; i < ghostMap_.size(); i++) {
    flattenedSendData[i] = sourceData(ghostMap_[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // send and recv data
  for (int i = 0; i < ghostOutGraph_.size(); i++) {
    MPI_Isend(flattenedSendData.data() + ghostOutOffset_[i], ghostOutNum_[i],
              MPI_INT, ghostOutGraph_[i], 0, MPI_COMM_WORLD,
              sendRequest.data() + i);
  }

  for (int i = 0; i < ghostInGraph_.size(); i++) {
    MPI_Irecv(flattenedRecvData.data() + ghostInOffset_[i], ghostInNum_[i],
              MPI_INT, ghostInGraph_[i], 0, MPI_COMM_WORLD,
              recvRequest.data() + i);
  }

  MPI_Waitall(sendRequest.size(), sendRequest.data(), sendStatus.data());
  MPI_Waitall(recvRequest.size(), recvRequest.data(), recvStatus.data());
  MPI_Barrier(MPI_COMM_WORLD);

  Kokkos::resize(ghostData, localReserveNum_ + remoteInNum_);
  // store reserved local data
  for (int i = 0; i < localReserveNum_; i++) {
    ghostData(i) = sourceData(reserveMap_[i]);
  }

  // store remote data
  for (int i = 0; i < remoteInNum_; i++) {
    ghostData(localReserveNum_ + i) = flattenedRecvData[i];
  }
}