#include "Partition.hpp"

Partition::Partition() {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);
}

void Partition::ConstructPartition(Kokkos::View<Scalar **> coords,
                                   Kokkos::View<GlobalIndex *> index) {
  // use Zoltan2 to partition
  Teuchos::ParameterList params("zoltan2 params");
  params.set("algorithm", "multijagged");
  params.set("mj_keep_part_boxes", true);
  params.set("rectilinear", true);
  params.set("num_global_parts", mpiSize_);

  std::vector<double> x, y, z;
  x.resize(coords.extent(0));
  y.resize(coords.extent(0));
  z.resize(coords.extent(0));

  for (int i = 0; i < coords.extent(0); i++) {
    x[i] = coords(i, 0);
    y[i] = coords(i, 1);
    z[i] = coords(i, 2);
  }

  std::vector<long long> idx(index.extent(0));
  for (int i = 0; i < index.extent(0); i++) {
    idx[i] = index(i);
  }

  InputAdapter *ia = new InputAdapter(coords.extent(0), idx.data(), x.data(),
                                      y.data(), z.data(), 1, 1, 1);

  Zoltan2::PartitioningProblem<InputAdapter> *problem =
      new Zoltan2::PartitioningProblem<InputAdapter>(ia, &params);

  problem->solve();

  // get solution
  auto &solution = problem->getSolution();
  const int *ptr = solution.getPartListView();

  std::vector<int> result(coords.extent(0));
  for (int i = 0; i < coords.extent(0); i++) {
    result[i] = ptr[i];
  }

  delete problem;
  delete ia;

  // construct migration map
  migrationOutNum_.resize(mpiSize_);
  migrationInNum_.resize(mpiSize_);
  for (int i = 0; i < mpiSize_; i++) {
    migrationOutNum_[i] = 0;
  }

  int localParticleNum = result.size();
  for (int i = 0; i < localParticleNum; i++) {
    if (result[i] != mpiRank_) {
      migrationOutNum_[result[i]]++;
    }
  }

  for (int i = 0; i < mpiSize_; i++) {
    int outNum = migrationOutNum_[i];
    MPI_Gather(&outNum, 1, MPI_INT, migrationInNum_.data(), 1, MPI_INT, i,
               MPI_COMM_WORLD);
  }

  migrationInGraph_.clear();
  migrationInGraphNum_.clear();
  for (int i = 0; i < mpiSize_; i++) {
    if (migrationInNum_[i] != 0) {
      migrationInGraph_.push_back(i);
      migrationInGraphNum_.push_back(migrationInNum_[i]);
    }
  }

  migrationOutGraph_.clear();
  migrationOutGraphNum_.clear();
  for (int i = 0; i < mpiSize_; i++) {
    if (migrationOutNum_[i] != 0) {
      migrationOutGraph_.push_back(i);
      migrationOutGraphNum_.push_back(migrationOutNum_[i]);
    }
  }

  migrationInOffset_.resize(migrationInGraph_.size() + 1);
  migrationOutOffset_.resize(migrationOutGraph_.size() + 1);

  migrationInOffset_[0] = 0;
  for (int i = 0; i < migrationInGraphNum_.size(); i++) {
    migrationInOffset_[i + 1] = migrationInOffset_[i] + migrationInGraphNum_[i];
  }

  migrationOutOffset_[0] = 0;
  for (int i = 0; i < migrationOutGraphNum_.size(); i++) {
    migrationOutOffset_[i + 1] =
        migrationOutOffset_[i] + migrationOutGraphNum_[i];
  }

  localMigrationMap_.resize(migrationOutOffset_[migrationOutGraphNum_.size()]);
  migrationMapIdx_.resize(migrationOutGraphNum_.size());
  localReserveMap_.clear();
  for (int i = 0; i < migrationOutGraphNum_.size(); i++) {
    migrationMapIdx_[i] = migrationOutOffset_[i];
  }
  for (int i = 0; i < localParticleNum; i++) {
    if (result[i] != mpiRank_) {
      auto ite = (size_t)(lower_bound(migrationOutGraph_.begin(),
                                      migrationOutGraph_.end(), result[i]) -
                          migrationOutGraph_.begin());
      localMigrationMap_[migrationMapIdx_[ite]] = i;
      migrationMapIdx_[ite]++;
    } else {
      localReserveMap_.push_back(i);
    }
  }
}

void Partition::ApplyPartition(Kokkos::View<Scalar **> data) {
  // migrate particles
  int newDataSize = data.extent(0);
  int unitLength = data.extent(1);
  for (int i = 0; i < migrationInGraphNum_.size(); i++) {
    newDataSize += migrationInGraphNum_[i];
  }
  for (int i = 0; i < migrationOutGraphNum_.size(); i++) {
    newDataSize -= migrationOutGraphNum_[i];
  }
  std::vector<MPI_Request> sendRequest(migrationOutGraph_.size());
  std::vector<MPI_Request> recvRequest(migrationInGraph_.size());
  std::vector<MPI_Status> sendStatus(migrationOutGraph_.size());
  std::vector<MPI_Status> recvStatus(migrationInGraph_.size());

  // migrate data
  const int outNum = migrationOutOffset_[migrationOutGraph_.size()];
  const int inNum = migrationInOffset_[migrationInGraph_.size()];
  std::vector<double> flattenedSendData(outNum * unitLength);
  std::vector<double> flattenedRecvData(inNum * unitLength);
  for (int i = 0; i < localMigrationMap_.size(); i++) {
    for (int j = 0; j < unitLength; j++) {
      flattenedSendData[i * unitLength + j] = data(localMigrationMap_[i], j);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // send and recv data
  for (int i = 0; i < migrationOutGraph_.size(); i++) {
    MPI_Isend(flattenedSendData.data() + migrationOutOffset_[i] * unitLength,
              migrationOutGraphNum_[i] * unitLength, MPI_DOUBLE,
              migrationOutGraph_[i], 0, MPI_COMM_WORLD, sendRequest.data() + i);
  }

  for (int i = 0; i < migrationInGraph_.size(); i++) {
    MPI_Irecv(flattenedRecvData.data() + migrationInOffset_[i] * unitLength,
              migrationInGraphNum_[i] * unitLength, MPI_DOUBLE,
              migrationInGraph_[i], 0, MPI_COMM_WORLD, recvRequest.data() + i);
  }

  MPI_Waitall(sendRequest.size(), sendRequest.data(), sendStatus.data());
  MPI_Waitall(recvRequest.size(), recvRequest.data(), recvStatus.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store reserved local data
  const int localReserveNum = localReserveMap_.size();

  std::vector<double> flattenedReservedData(localReserveNum * unitLength);
  for (int i = 0; i < localReserveNum; i++) {
    for (int j = 0; j < unitLength; j++) {
      flattenedReservedData[i * unitLength + j] = data(localReserveMap_[i], j);
    }
  }

  Kokkos::resize(data, newDataSize, unitLength);
  // move reserved local data to new array
  for (int i = 0; i < localReserveNum; i++) {
    for (int j = 0; j < unitLength; j++) {
      data(i, j) = flattenedReservedData[i * unitLength + j];
    }
  }

  // move remote data to new array
  for (int i = 0; i < inNum; i++) {
    for (int j = 0; j < unitLength; j++) {
      data(localReserveNum + i, j) = flattenedRecvData[i * unitLength + j];
    }
  }
}

void Partition::ApplyPartition(Kokkos::View<Scalar *> data) {
  // migrate particles
  int newDataSize = data.extent(0);
  for (int i = 0; i < migrationInGraphNum_.size(); i++) {
    newDataSize += migrationInGraphNum_[i];
  }
  for (int i = 0; i < migrationOutGraphNum_.size(); i++) {
    newDataSize -= migrationOutGraphNum_[i];
  }
  std::vector<MPI_Request> sendRequest(migrationOutGraph_.size());
  std::vector<MPI_Request> recvRequest(migrationInGraph_.size());
  std::vector<MPI_Status> sendStatus(migrationOutGraph_.size());
  std::vector<MPI_Status> recvStatus(migrationInGraph_.size());

  // migrate coords
  const int outNum = migrationOutOffset_[migrationOutGraph_.size()];
  const int inNum = migrationInOffset_[migrationInGraph_.size()];
  std::vector<double> flattenedSendData(outNum);
  std::vector<double> flattenedRecvData(inNum);
  for (int i = 0; i < localMigrationMap_.size(); i++) {
    flattenedSendData[i] = data(localMigrationMap_[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // send and recv data
  for (int i = 0; i < migrationOutGraph_.size(); i++) {
    MPI_Isend(flattenedSendData.data() + migrationOutOffset_[i],
              migrationOutGraphNum_[i], MPI_DOUBLE, migrationOutGraph_[i], 0,
              MPI_COMM_WORLD, sendRequest.data() + i);
  }

  for (int i = 0; i < migrationInGraph_.size(); i++) {
    MPI_Irecv(flattenedRecvData.data() + migrationInOffset_[i],
              migrationInGraphNum_[i], MPI_DOUBLE, migrationInGraph_[i], 0,
              MPI_COMM_WORLD, recvRequest.data() + i);
  }

  MPI_Waitall(sendRequest.size(), sendRequest.data(), sendStatus.data());
  MPI_Waitall(recvRequest.size(), recvRequest.data(), recvStatus.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store reserved local data
  const int localReserveNum = localReserveMap_.size();

  std::vector<double> flattenedReservedData(localReserveNum);
  for (int i = 0; i < localReserveNum; i++) {
    flattenedReservedData[i] = data(localReserveMap_[i]);
  }

  Kokkos::resize(data, newDataSize);
  // move reserved local data to new array
  for (int i = 0; i < localReserveNum; i++) {
    data(i) = flattenedReservedData[i];
  }

  // move remote data to new array
  for (int i = 0; i < inNum; i++) {
    data(localReserveNum + i) = flattenedRecvData[i];
  }
}

void Partition::ApplyPartition(Kokkos::View<LocalIndex *> data) {
  // migrate particles
  int newDataSize = data.extent(0);
  for (int i = 0; i < migrationInGraphNum_.size(); i++) {
    newDataSize += migrationInGraphNum_[i];
  }
  for (int i = 0; i < migrationOutGraphNum_.size(); i++) {
    newDataSize -= migrationOutGraphNum_[i];
  }
  std::vector<MPI_Request> sendRequest(migrationOutGraph_.size());
  std::vector<MPI_Request> recvRequest(migrationInGraph_.size());
  std::vector<MPI_Status> sendStatus(migrationOutGraph_.size());
  std::vector<MPI_Status> recvStatus(migrationInGraph_.size());

  // migrate coords
  const int outNum = migrationOutOffset_[migrationOutGraph_.size()];
  const int inNum = migrationInOffset_[migrationInGraph_.size()];
  std::vector<int> flattenedSendData(outNum);
  std::vector<int> flattenedRecvData(inNum);
  for (int i = 0; i < localMigrationMap_.size(); i++) {
    flattenedSendData[i] = data(localMigrationMap_[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // send and recv data
  for (int i = 0; i < migrationOutGraph_.size(); i++) {
    MPI_Isend(flattenedSendData.data() + migrationOutOffset_[i],
              migrationOutGraphNum_[i], MPI_INT, migrationOutGraph_[i], 0,
              MPI_COMM_WORLD, sendRequest.data() + i);
  }

  for (int i = 0; i < migrationInGraph_.size(); i++) {
    MPI_Irecv(flattenedRecvData.data() + migrationInOffset_[i],
              migrationInGraphNum_[i], MPI_INT, migrationInGraph_[i], 0,
              MPI_COMM_WORLD, recvRequest.data() + i);
  }

  MPI_Waitall(sendRequest.size(), sendRequest.data(), sendStatus.data());
  MPI_Waitall(recvRequest.size(), recvRequest.data(), recvStatus.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store reserved local data
  const int localReserveNum = localReserveMap_.size();

  std::vector<int> flattenedReservedData(localReserveNum);
  for (int i = 0; i < localReserveNum; i++) {
    flattenedReservedData[i] = data(localReserveMap_[i]);
  }

  Kokkos::resize(data, newDataSize);
  // move reserved local data to new array
  for (int i = 0; i < localReserveNum; i++) {
    data(i) = flattenedReservedData[i];
  }

  // move remote data to new array
  for (int i = 0; i < inNum; i++) {
    data(localReserveNum + i) = flattenedRecvData[i];
  }
}