#include "LinearAlgebra/Impl/Default/DefaultMatrix.hpp"

#include "Core/Typedef.hpp"
#include "LinearAlgebra/Impl/Default/Default.hpp"
#include "LinearAlgebra/Impl/Default/DefaultVector.hpp"

#include <Kokkos_CopyViews.hpp>
#include <Kokkos_Core.hpp>

#include <Kokkos_Core_fwd.hpp>
#include <algorithm>
#include <execution>
#include <memory>

#include <mpi.h>

Void LinearAlgebra::Impl::SequentialMatrixMultiplication(DeviceIndexVector &ia,
                                                         DeviceIndexVector &ja,
                                                         DeviceRealVector &a,
                                                         DeviceRealVector &x,
                                                         DeviceRealVector &y) {
  LocalIndex rowPerTeam = 20;
  LocalIndex numRow = ia.extent(0) - 1;
  LocalIndex rowByTeam = static_cast<LocalIndex>(
      std::ceil(static_cast<Scalar>(numRow) / static_cast<Scalar>(rowPerTeam)));

  int mpiRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

  Kokkos::parallel_for(
      Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(rowByTeam,
                                                        Kokkos::AUTO()),
      KOKKOS_LAMBDA(
          const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
              &teamMember) {
        const LocalIndex teamWork = teamMember.league_rank() * rowPerTeam;

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, rowPerTeam),
            [&](const LocalIndex j) {
              const LocalIndex row = teamWork + j;
              if (row >= numRow)
                return;

              y(row) = 0.0;
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(teamMember, ia(row), ia(row + 1)),
                  [&](const GlobalIndex k, Scalar &tSum) {
                    tSum += a(k) * x(ja(k));
                  },
                  Kokkos::Sum<Scalar>(y(row)));
            });
        teamMember.team_barrier();
      });
  Kokkos::fence();
}

Void LinearAlgebra::Impl::SequentialMatrixMultiplicationAddition(
    DeviceIndexVector &ia, DeviceIndexVector &ja, DeviceRealVector &a,
    DeviceRealVector &x, DeviceRealVector &y) {
  LocalIndex rowPerTeam = 20;
  LocalIndex numRow = ia.extent(0) - 1;
  LocalIndex rowByTeam = static_cast<LocalIndex>(
      std::ceil(static_cast<Scalar>(numRow) / static_cast<Scalar>(rowPerTeam)));

  int mpiRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

  Kokkos::parallel_for(
      Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(rowByTeam,
                                                        Kokkos::AUTO()),
      KOKKOS_LAMBDA(
          const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
              &teamMember) {
        const LocalIndex teamWork = teamMember.league_rank() * rowPerTeam;

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, rowPerTeam),
            [&](const LocalIndex j) {
              const LocalIndex row = teamWork + j;
              if (row >= numRow)
                return;

              Scalar sum = 0.0;
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(teamMember, ia(row), ia(row + 1)),
                  [&](const GlobalIndex k, Scalar &tSum) {
                    tSum += a(k) * x(ja(k));
                  },
                  Kokkos::Sum<Scalar>(sum));
              y(row) += sum;
            });
        teamMember.team_barrier();
      });
  Kokkos::fence();
}

Void LinearAlgebra::Impl::SequentialOffDiagMatrixMultiplication(
    DeviceIndexVector &ia, DeviceIndexVector &ja, DeviceRealVector &a,
    DeviceRealVector &x, DeviceRealVector &y) {
  LocalIndex rowSize = ia.extent(0) - 1;

  Kokkos::parallel_for(
      Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(rowSize,
                                                        Kokkos::AUTO()),
      KOKKOS_LAMBDA(
          const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
              &teamMember) {
        const GlobalIndex i = teamMember.league_rank();
        const GlobalIndex colSize = ia(i + 1) - ia(i);

        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(teamMember, colSize),
            [&](const GlobalIndex j, Scalar &tSum) {
              GlobalIndex offset = ia(i) + j;
              if (ja(offset) != i)
                tSum += a(offset) * x(ja(offset));
            },
            Kokkos::Sum<Scalar>(y(i)));
        teamMember.team_barrier();
      });
  Kokkos::fence();
}

Void LinearAlgebra::Impl::DefaultMatrix::CommunicateOffDiagVectorBegin(
    const DeviceRealVector &vec) const {
  DeviceRealVector::HostMirror hostVec;
  hostVec = Kokkos::create_mirror_view(vec);
  Kokkos::deep_copy(hostVec, vec);

  auto &sendRequest = *sendRequestPtr_;
  auto &recvRequest = *recvRequestPtr_;

  auto &sendVectorColIndex = *sendVectorColIndexPtr_;

  HostRealVector sendVec;
  Kokkos::resize(sendVec, sendVectorColIndex.size());

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
          0, sendVectorColIndex.size()),
      [&](const int i) { sendVec(i) = hostVec(sendVectorColIndex[i]); });
  Kokkos::fence();

  for (GlobalIndex i = 0; i < sendRank_.size(); i++) {
    MPI_Isend(sendVec.data() + sendOffsetByRank_[i], sendCountByRank_[i],
              MPI_DOUBLE, sendRank_[i], 0, MPI_COMM_WORLD,
              sendRequest.data() + i);
  }

  auto &hostOffDiagVector = *hostOffDiagVectorPtr_;
  for (GlobalIndex i = 0; i < recvRank_.size(); i++) {
    MPI_Irecv(hostOffDiagVector.data() + recvOffsetByRank_[i],
              recvCountByRank_[i], MPI_DOUBLE, recvRank_[i], 0, MPI_COMM_WORLD,
              recvRequest.data() + i);
  }
}

Void LinearAlgebra::Impl::DefaultMatrix::CommunicateOffDiagVectorEnd() const {
  auto &sendRequest = *sendRequestPtr_;
  auto &recvRequest = *recvRequestPtr_;

  auto &sendStatus = *sendStatusPtr_;
  auto &recvStatus = *recvStatusPtr_;

  MPI_Waitall(sendRequest.size(), sendRequest.data(), recvStatus.data());
  MPI_Waitall(recvRequest.size(), recvRequest.data(), recvStatus.data());
  MPI_Barrier(MPI_COMM_WORLD);

  auto &hostOffDiagVector = *hostOffDiagVectorPtr_;
  auto &deviceOffDiagVector = *deviceOffDiagVectorPtr_;
  Kokkos::deep_copy(deviceOffDiagVector, hostOffDiagVector);
}

Void LinearAlgebra::Impl::DefaultMatrix::FindSeqDiagOffset() {
  deviceSeqDiagOffsetPtr_ = std::make_shared<DeviceIndexVector>();
  hostSeqDiagOffsetPtr_ = std::make_shared<DeviceIndexVector::HostMirror>();

  auto &deviceSeqDiagOffset = *deviceSeqDiagOffsetPtr_;
  auto &hostSeqDiagOffset = *hostSeqDiagOffsetPtr_;

  auto &hostDiagMatrixRowIndex = *hostDiagMatrixRowIndexPtr_;
  auto &hostDiagMatrixColIndex = *hostDiagMatrixColIndexPtr_;

  Kokkos::resize(deviceSeqDiagOffset, localRowSize_);
  hostSeqDiagOffset = Kokkos::create_mirror_view(deviceSeqDiagOffset);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, localRowSize_),
      [&](const unsigned int row) {
        std::size_t low = hostDiagMatrixRowIndex(row);
        std::size_t high = hostDiagMatrixRowIndex(row + 1);
        std::size_t offset = (low + high) / 2;
        GlobalIndex target = row;
        while (true) {
          if (hostDiagMatrixColIndex(offset) == target || offset == low ||
              offset == high)
            break;
          if (hostDiagMatrixColIndex(offset) < target) {
            low = offset;
            offset = (offset + high) / 2;
          }
          if (hostDiagMatrixColIndex(offset) > target) {
            high = offset;
            offset = (low + offset) / 2;
          }
        }

        hostSeqDiagOffset(row) = offset;
      });
  Kokkos::fence();

  Kokkos::deep_copy(deviceSeqDiagOffset, hostSeqDiagOffset);
}

Void LinearAlgebra::Impl::DefaultMatrix::PrepareJacobiPreconditioning() {
  FindSeqDiagOffset();
  isJacobiPreconditioningPrepared_ = true;
}

Void LinearAlgebra::Impl::DefaultMatrix::PrepareSorPreconditioning() {
  FindSeqDiagOffset();

  // multi-coloring
  // TODO: sequential implementation only, parallel implementation needed
  std::vector<int> nodeColor;
  nodeColor.resize(localRowSize_);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, localRowSize_),
      [&](const unsigned int i) { nodeColor[i] = -1; });
  Kokkos::fence();

  auto &hostDiagMatrixRowIndex = *hostDiagMatrixRowIndexPtr_;
  auto &hostDiagMatrixColIndex = *hostDiagMatrixColIndexPtr_;

  for (int i = 0; i < localRowSize_; i++) {
    std::vector<int> adjColor(hostDiagMatrixRowIndex(i + 1) -
                              hostDiagMatrixRowIndex(i));
    for (GlobalIndex j = hostDiagMatrixRowIndex(i);
         j < hostDiagMatrixRowIndex(i + 1); j++) {
      adjColor[j - hostDiagMatrixRowIndex(i)] =
          nodeColor[hostDiagMatrixColIndex(j)];
    }

    std::sort(adjColor.begin(), adjColor.end());

    int minColor = 0;
    while (true) {
      auto it = std::lower_bound(adjColor.begin(), adjColor.end(), minColor);
      if (it == adjColor.end() || *it != minColor)
        break;
      else
        minColor++;
    }

    nodeColor[i] = minColor;
  }

  int maxColor = -1;
  for (int i = 0; i < localRowSize_; i++) {
    if (maxColor < nodeColor[i])
      maxColor = nodeColor[i];
  }
  int numColor = maxColor + 1;

  deviceMultiColorReorderingPtr_ = std::make_shared<DeviceIntVector>();
  hostMultiColorReorderingPtr_ =
      std::make_shared<DeviceIntVector::HostMirror>();

  auto &deviceMultiColorReordering = *deviceMultiColorReorderingPtr_;
  auto &hostMultiColorReordering = *hostMultiColorReorderingPtr_;

  Kokkos::resize(deviceMultiColorReordering, numColor + 1);
  hostMultiColorReordering =
      Kokkos::create_mirror_view(deviceMultiColorReordering);

  for (LocalIndex i = 0; i <= numColor; i++)
    hostMultiColorReordering(i) = 0;
  for (LocalIndex i = 0; i < localRowSize_; i++)
    hostMultiColorReordering(nodeColor[i] + 1)++;
  hostMultiColorReordering(0) = 0;
  for (LocalIndex i = 0; i < numColor; i++)
    hostMultiColorReordering(i + 1) += hostMultiColorReordering(i);

  Kokkos::deep_copy(deviceMultiColorReordering, hostMultiColorReordering);

  deviceMultiColorReorderingRowPtr_ = std::make_shared<DeviceIntVector>();
  hostMultiColorReorderingRowPtr_ =
      std::make_shared<DeviceIntVector::HostMirror>();

  auto &deviceMultiColorReorderingRow = *deviceMultiColorReorderingRowPtr_;
  auto &hostMultiColorReorderingRow = *hostMultiColorReorderingRowPtr_;

  Kokkos::resize(deviceMultiColorReorderingRow, localRowSize_);
  hostMultiColorReorderingRow =
      Kokkos::create_mirror_view(deviceMultiColorReorderingRow);

  std::vector<int> offset(numColor);
  for (LocalIndex i = 0; i < numColor; i++)
    offset[i] = 0;

  for (LocalIndex i = 0; i < localRowSize_; i++) {
    hostMultiColorReorderingRow(hostMultiColorReordering(nodeColor[i]) +
                                offset[nodeColor[i]]) = i;
    offset[nodeColor[i]]++;
  }

  Kokkos::deep_copy(deviceMultiColorReorderingRow, hostMultiColorReorderingRow);

  isSorPreconditioningPrepared_ = true;
}

LinearAlgebra::Impl::DefaultMatrix::DefaultMatrix() {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);

  localRowSize_ = 0;
  localColSize_ = 0;
  globalRowSize_ = 0;
  globalColSize_ = 0;

  colRangeLow_ = 0;
  colRangeHigh_ = 0;
  rowRangeLow_ = 0;
  rowRangeHigh_ = 0;

  diagMatrixGraphPtr_ =
      std::make_shared<std::vector<std::vector<GlobalIndex>>>();
  offDiagMatrixGraphPtr_ =
      std::make_shared<std::vector<std::vector<GlobalIndex>>>();

  hostDiagMatrixRowIndexPtr_ =
      std::make_shared<DeviceIndexVector::HostMirror>();
  hostOffDiagMatrixRowIndexPtr_ =
      std::make_shared<DeviceIndexVector::HostMirror>();

  hostDiagMatrixColIndexPtr_ =
      std::make_shared<DeviceIndexVector::HostMirror>();
  hostOffDiagMatrixColIndexPtr_ =
      std::make_shared<DeviceIndexVector::HostMirror>();

  hostDiagMatrixValuePtr_ = std::make_shared<DeviceRealVector::HostMirror>();
  hostOffDiagMatrixValuePtr_ = std::make_shared<DeviceRealVector::HostMirror>();

  deviceDiagMatrixRowIndexPtr_ = std::make_shared<DeviceIndexVector>();
  deviceOffDiagMatrixRowIndexPtr_ = std::make_shared<DeviceIndexVector>();

  deviceDiagMatrixColIndexPtr_ = std::make_shared<DeviceIndexVector>();
  deviceOffDiagMatrixColIndexPtr_ = std::make_shared<DeviceIndexVector>();

  deviceDiagMatrixValuePtr_ = std::make_shared<DeviceRealVector>();
  deviceOffDiagMatrixValuePtr_ = std::make_shared<DeviceRealVector>();

  hostOffDiagVectorPtr_ = std::make_shared<DeviceRealVector::HostMirror>();
  deviceOffDiagVectorPtr_ = std::make_shared<DeviceRealVector>();

  isAssembled_ = false;
  isHostDeviceSynchronized_ = false;
  isDeviceHostSynchronized_ = false;

  isJacobiPreconditioningPrepared_ = false;
  isSorPreconditioningPrepared_ = false;
}

LinearAlgebra::Impl::DefaultMatrix::~DefaultMatrix() {}

Void LinearAlgebra::Impl::DefaultMatrix::Resize(const GlobalIndex m,
                                                const GlobalIndex n,
                                                const LocalIndex blockSize) {
  localRowSize_ = m;
  localColSize_ = n;

  Clear();

  diagMatrixGraphPtr_->clear();
  diagMatrixGraphPtr_->resize(localRowSize_);
  offDiagMatrixGraphPtr_->clear();
  offDiagMatrixGraphPtr_->resize(localRowSize_);

  unsigned long localRowSize = m;
  unsigned long localColSize = n;
  unsigned long globalRowSize, globalColSize;
  MPI_Allreduce(&localRowSize, &globalRowSize, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&localColSize, &globalColSize, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);

  globalRowSize_ = globalRowSize;
  globalColSize_ = globalColSize;

  rankColSize_.resize(mpiSize_);
  MPI_Allgather(&localColSize, 1, MPI_UNSIGNED_LONG, rankColSize_.data(), 1,
                MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
  colRangeLow_ = 0;
  for (int i = 0; i < mpiRank_; i++)
    colRangeLow_ += rankColSize_[i];
  colRangeHigh_ = colRangeLow_ + rankColSize_[mpiRank_];

  rankRowSize_.resize(mpiSize_);
  MPI_Allgather(&localRowSize, 1, MPI_UNSIGNED_LONG, rankRowSize_.data(), 1,
                MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
  rowRangeLow_ = 0;
  for (int i = 0; i < mpiRank_; i++)
    rowRangeLow_ += rankRowSize_[i];
  rowRangeHigh_ = rowRangeLow_ + rankRowSize_[mpiRank_];

  rankColOffset_.resize(mpiSize_ + 1);
  rankColOffset_[0] = 0;
  for (int i = 0; i < mpiSize_; i++)
    rankColOffset_[i + 1] = rankColOffset_[i] + rankColSize_[i];

  rankRowOffset_.resize(mpiSize_ + 1);
  rankRowOffset_[0] = 0;
  for (int i = 0; i < mpiSize_; i++)
    rankRowOffset_[i + 1] = rankRowOffset_[i] + rankRowSize_[i];
}

Void LinearAlgebra::Impl::DefaultMatrix::Transpose(DefaultMatrix &mat) {}

Void LinearAlgebra::Impl::DefaultMatrix::Clear() {}

LocalIndex LinearAlgebra::Impl::DefaultMatrix::GetLocalColSize() const {
  return localColSize_;
}

LocalIndex LinearAlgebra::Impl::DefaultMatrix::GetLocalRowSize() const {
  return localRowSize_;
}

GlobalIndex LinearAlgebra::Impl::DefaultMatrix::GetGlobalColSize() const {
  return globalColSize_;
}

GlobalIndex LinearAlgebra::Impl::DefaultMatrix::GetGlobalRowSize() const {
  return globalRowSize_;
}

Void LinearAlgebra::Impl::DefaultMatrix::SetColIndex(
    const GlobalIndex row, const std::vector<GlobalIndex> &index) {
  auto &diagMatrixGraph = *diagMatrixGraphPtr_;
  auto &offDiagMatrixGraph = *offDiagMatrixGraphPtr_;

  std::vector<GlobalIndex> &diagIndex = diagMatrixGraph[row];
  std::vector<GlobalIndex> &offDiagIndex = offDiagMatrixGraph[row];
  auto colRangeLowFlag =
      std::lower_bound(index.begin(), index.end(), colRangeLow_);
  auto colRangeHighFlag =
      std::lower_bound(index.begin(), index.end(), colRangeHigh_);
  std::size_t diagIndexCount = colRangeHighFlag - colRangeLowFlag;
  std::size_t offDiagIndexCount = index.size() - diagIndexCount;
  diagIndex.resize(diagIndexCount);
  offDiagIndex.resize(offDiagIndexCount);
  auto diagIt = diagIndex.begin();
  auto offDiagIt = offDiagIndex.begin();
  for (auto it = index.begin(); it != colRangeLowFlag; it++, offDiagIt++)
    *offDiagIt = *it;
  for (auto it = colRangeLowFlag; it != colRangeHighFlag; it++, diagIt++)
    *diagIt = *it - colRangeLow_;
  for (auto it = colRangeHighFlag; it != index.end(); it++, offDiagIt++)
    *offDiagIt = *it;
}

Void LinearAlgebra::Impl::DefaultMatrix::Increment(const GlobalIndex row,
                                                   const GlobalIndex col,
                                                   const Scalar value) {
  auto &hostDiagMatrixRowIndex = *hostDiagMatrixRowIndexPtr_;
  auto &hostOffDiagMatrixRowIndex = *hostOffDiagMatrixRowIndexPtr_;

  auto &hostDiagMatrixColIndex = *hostDiagMatrixColIndexPtr_;
  auto &hostOffDiagMatrixColIndex = *hostOffDiagMatrixColIndexPtr_;

  auto &hostDiagMatrixValue = *hostDiagMatrixValuePtr_;
  auto &hostOffDiagMatrixValue = *hostOffDiagMatrixValuePtr_;

  if (col >= colRangeLow_ && col < colRangeHigh_) {
    std::size_t low = hostDiagMatrixRowIndex(row);
    std::size_t high = hostDiagMatrixRowIndex(row + 1);
    std::size_t offset = (low + high) / 2;
    GlobalIndex target = col - colRangeLow_;
    while (true) {
      if (hostDiagMatrixColIndex(offset) == target || offset == low ||
          offset == high)
        break;
      if (hostDiagMatrixColIndex(offset) < target) {
        low = offset;
        offset = (offset + high) / 2;
      }
      if (hostDiagMatrixColIndex(offset) > target) {
        high = offset;
        offset = (low + offset) / 2;
      }
    }
    hostDiagMatrixValue(offset) = value;
  } else {
    std::size_t low = hostOffDiagMatrixRowIndex(row);
    std::size_t high = hostOffDiagMatrixRowIndex(row + 1);
    std::size_t offset = (low + high) / 2;
    GlobalIndex target = col;
    while (true) {
      if (hostOffDiagMatrixColIndex(offset) == target || offset == low ||
          offset == high)
        break;
      if (hostOffDiagMatrixColIndex(offset) < target) {
        low = offset;
        offset = (offset + high) / 2;
      }
      if (hostOffDiagMatrixColIndex(offset) > target) {
        high = offset;
        offset = (low + offset) / 2;
      }
    }
    hostOffDiagMatrixValue(offset) = value;
  }
}

Void LinearAlgebra::Impl::DefaultMatrix::GraphAssemble() {
  unsigned long diagNumNonzero = 0;
  unsigned long offDiagNumNonzero = 0;
  unsigned long maxDiagNonzero = 0;
  unsigned long maxOffDiagNonzero = 0;

  auto &diagMatrixGraph = *diagMatrixGraphPtr_;
  auto &offDiagMatrixGraph = *offDiagMatrixGraphPtr_;

  for (int i = 0; i < localRowSize_; i++) {
    diagNumNonzero += diagMatrixGraph[i].size();
    if (diagMatrixGraph[i].size() > maxDiagNonzero)
      maxDiagNonzero = diagMatrixGraph[i].size();
    offDiagNumNonzero += offDiagMatrixGraph[i].size();
    if (offDiagMatrixGraph[i].size() > maxOffDiagNonzero)
      maxOffDiagNonzero = offDiagMatrixGraph[i].size();
  }

  std::vector<GlobalIndex> diagNonzero(localRowSize_);
  std::vector<GlobalIndex> offDiagNonzero(localRowSize_);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, localRowSize_),
      [&](const unsigned int i) {
        diagNonzero[i] = diagMatrixGraph[i].size();
        offDiagNonzero[i] = offDiagMatrixGraph[i].size();
      });
  Kokkos::fence();

  auto &hostDiagMatrixRowIndex = *hostDiagMatrixRowIndexPtr_;
  auto &hostOffDiagMatrixRowIndex = *hostOffDiagMatrixRowIndexPtr_;

  auto &hostDiagMatrixColIndex = *hostDiagMatrixColIndexPtr_;
  auto &hostOffDiagMatrixColIndex = *hostOffDiagMatrixColIndexPtr_;

  auto &hostDiagMatrixValue = *hostDiagMatrixValuePtr_;
  auto &hostOffDiagMatrixValue = *hostOffDiagMatrixValuePtr_;

  auto &deviceDiagMatrixRowIndex = *deviceDiagMatrixRowIndexPtr_;
  auto &deviceOffDiagMatrixRowIndex = *deviceOffDiagMatrixRowIndexPtr_;

  auto &deviceDiagMatrixColIndex = *deviceDiagMatrixColIndexPtr_;
  auto &deviceOffDiagMatrixColIndex = *deviceOffDiagMatrixColIndexPtr_;

  auto &deviceDiagMatrixValue = *deviceDiagMatrixValuePtr_;
  auto &deviceOffDiagMatrixValue = *deviceOffDiagMatrixValuePtr_;

  Kokkos::resize(deviceDiagMatrixRowIndex, localRowSize_ + 1);
  Kokkos::resize(deviceOffDiagMatrixRowIndex, localRowSize_ + 1);

  Kokkos::resize(deviceDiagMatrixColIndex, diagNumNonzero);
  Kokkos::resize(deviceOffDiagMatrixColIndex, offDiagNumNonzero);

  Kokkos::resize(deviceDiagMatrixValue, diagNumNonzero);
  Kokkos::resize(deviceOffDiagMatrixValue, offDiagNumNonzero);

  hostDiagMatrixRowIndex = Kokkos::create_mirror_view(deviceDiagMatrixRowIndex);
  hostOffDiagMatrixRowIndex =
      Kokkos::create_mirror_view(deviceOffDiagMatrixRowIndex);

  hostDiagMatrixColIndex = Kokkos::create_mirror_view(deviceDiagMatrixColIndex);
  hostOffDiagMatrixColIndex =
      Kokkos::create_mirror_view(deviceOffDiagMatrixColIndex);

  hostDiagMatrixValue = Kokkos::create_mirror_view(deviceDiagMatrixValue);
  hostOffDiagMatrixValue = Kokkos::create_mirror_view(deviceOffDiagMatrixValue);

  // build ia for diag and off-diag mat
  for (unsigned int i = 0; i < localRowSize_; i++)
    hostDiagMatrixRowIndex(i + 1) = hostDiagMatrixRowIndex(i) + diagNonzero[i];

  for (unsigned int i = 0; i < localRowSize_; i++)
    hostOffDiagMatrixRowIndex(i + 1) =
        hostOffDiagMatrixRowIndex(i) + offDiagNonzero[i];

  // build ja for diag and off-diag mat
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>(localRowSize_,
                                                            Kokkos::AUTO()),
      KOKKOS_LAMBDA(
          const Kokkos::TeamPolicy<
              Kokkos::DefaultHostExecutionSpace>::member_type &teamMember) {
        const LocalIndex i = teamMember.league_rank();
        const LocalIndex colSize =
            hostDiagMatrixRowIndex(i + 1) - hostDiagMatrixRowIndex(i);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, colSize),
                             [&](const GlobalIndex j) {
                               hostDiagMatrixColIndex(
                                   j + hostDiagMatrixRowIndex(i)) =
                                   diagMatrixGraph[i][j];
                             });
      });
  Kokkos::fence();

  Kokkos::parallel_for(
      Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>(localRowSize_,
                                                            Kokkos::AUTO()),
      KOKKOS_LAMBDA(
          const Kokkos::TeamPolicy<
              Kokkos::DefaultHostExecutionSpace>::member_type &teamMember) {
        const LocalIndex i = teamMember.league_rank();
        const LocalIndex colSize =
            hostOffDiagMatrixRowIndex(i + 1) - hostOffDiagMatrixRowIndex(i);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, colSize),
                             [&](const GlobalIndex j) {
                               hostOffDiagMatrixColIndex(
                                   j + hostOffDiagMatrixRowIndex(i)) =
                                   offDiagMatrixGraph[i][j];
                             });
      });
  Kokkos::fence();
}

Void LinearAlgebra::Impl::DefaultMatrix::Assemble() {
  // build communication graph

  auto &hostDiagMatrixRowIndex = *hostDiagMatrixRowIndexPtr_;
  auto &hostOffDiagMatrixRowIndex = *hostOffDiagMatrixRowIndexPtr_;

  auto &hostDiagMatrixColIndex = *hostDiagMatrixColIndexPtr_;
  auto &hostOffDiagMatrixColIndex = *hostOffDiagMatrixColIndexPtr_;

  auto &hostDiagMatrixValue = *hostDiagMatrixValuePtr_;
  auto &hostOffDiagMatrixValue = *hostOffDiagMatrixValuePtr_;

  auto &deviceDiagMatrixRowIndex = *deviceDiagMatrixRowIndexPtr_;
  auto &deviceOffDiagMatrixRowIndex = *deviceOffDiagMatrixRowIndexPtr_;

  auto &deviceDiagMatrixColIndex = *deviceDiagMatrixColIndexPtr_;
  auto &deviceOffDiagMatrixColIndex = *deviceOffDiagMatrixColIndexPtr_;

  auto &deviceDiagMatrixValue = *deviceDiagMatrixValuePtr_;
  auto &deviceOffDiagMatrixValue = *deviceOffDiagMatrixValuePtr_;

  /* TODO: The functionality of communication graph is very similar to
   * Geometry::Ghost.  These two parts should be merged together one day.
   */

  /*!
   *! \note Brief design idea behind the communication graph
   *! \date Dec 21, 2022
   *! \author Zisheng Ye <zisheng_ye@outlook.com>
   * Communication graph has two stages.  In the first stage, vector from
   * different MPI process is gathered.  In the second stage, off-diag mat is
   * multiplied with the gathered vector.  Therefore, in the assemble stage, the
   * gather assignment should be scheduled and the reordered off-diag mat for a
   * faster multiplication should be formed.
   */

  // first stage
  std::vector<GlobalIndex> offDiagCol;
  offDiagCol.resize(hostOffDiagMatrixColIndex.extent(0));

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,
                                                             offDiagCol.size()),
      [&](const int i) { offDiagCol[i] = hostOffDiagMatrixColIndex(i); });
  Kokkos::fence();
  std::sort(std::execution::par_unseq, offDiagCol.begin(), offDiagCol.end());
  offDiagCol.erase(std::unique(std::execution::par_unseq, offDiagCol.begin(),
                               offDiagCol.end()),
                   offDiagCol.end());

  // align to corresponding external ranks
  /*!
   *! \note 'int' used here is required by MPI MPI_Gatherv
   *! \author Zisheng Ye <zisheng_ye@outlook.com>
   *! \date Dec 22, 2022
   */
  std::vector<int> sendVectorSizeByRank(mpiSize_);
  std::vector<int> sendVectorOffsetByRank(mpiSize_ + 1);

  sendRank_.clear();
  recvRank_.clear();
  sendCountByRank_.clear();
  recvCountByRank_.clear();
  sendOffsetByRank_.clear();
  recvOffsetByRank_.clear();

  recvOffsetByRank_.push_back(0);

  for (int i = 0; i < mpiSize_; i++) {
    if (mpiRank_ == i) {
      int gatherVectorSize = 0;

      MPI_Gather(&gatherVectorSize, 1, MPI_INT, sendVectorSizeByRank.data(), 1,
                 MPI_INT, i, MPI_COMM_WORLD);

      for (int i = 0; i < mpiSize_; i++) {
        sendVectorOffsetByRank[i + 1] =
            sendVectorOffsetByRank[i] + sendVectorSizeByRank[i];
      }

      sendVectorColIndexPtr_ = std::make_shared<std::vector<GlobalIndex>>(
          sendVectorOffsetByRank[mpiSize_]);
      auto &sendVectorColIndex = *sendVectorColIndexPtr_;

      MPI_Gatherv(nullptr, 0, MPI_UNSIGNED_LONG, sendVectorColIndex.data(),
                  sendVectorSizeByRank.data(), sendVectorOffsetByRank.data(),
                  MPI_UNSIGNED_LONG, i, MPI_COMM_WORLD);

      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
              0, sendVectorColIndex.size()),
          [&](const int i) { sendVectorColIndex[i] -= colRangeLow_; });
      Kokkos::fence();

      for (int i = 0; i < mpiSize_; i++)
        if (sendVectorSizeByRank[i] != 0) {
          sendRank_.push_back(i);
          sendCountByRank_.push_back(sendVectorSizeByRank[i]);
        }

      sendOffsetByRank_.resize(sendCountByRank_.size() + 1);
      sendOffsetByRank_[0] = 0;
      for (int i = 0; i < sendCountByRank_.size(); i++)
        sendOffsetByRank_[i + 1] = sendOffsetByRank_[i] + sendCountByRank_[i];
    } else {
      auto startPosition = std::lower_bound(
          offDiagCol.begin(), offDiagCol.end(), rankColOffset_[i]);
      auto endPosition = std::lower_bound(offDiagCol.begin(), offDiagCol.end(),
                                          rankColOffset_[i + 1]);

      int gatherVectorSize = endPosition - startPosition;

      MPI_Gather(&gatherVectorSize, 1, MPI_INT, nullptr, 1, MPI_INT, i,
                 MPI_COMM_WORLD);

      MPI_Gatherv(&(*startPosition), gatherVectorSize, MPI_UNSIGNED_LONG,
                  nullptr, nullptr, nullptr, MPI_UNSIGNED_LONG, i,
                  MPI_COMM_WORLD);

      if (gatherVectorSize != 0) {
        recvRank_.push_back(i);
        recvCountByRank_.push_back(gatherVectorSize);
        recvOffsetByRank_.push_back(recvOffsetByRank_.back() +
                                    gatherVectorSize);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // second stage
  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
                           0, hostOffDiagMatrixColIndex.extent(0)),
                       [&](const int i) {
                         std::size_t newColIndex =
                             std::lower_bound(offDiagCol.begin(),
                                              offDiagCol.end(),
                                              hostOffDiagMatrixColIndex[i]) -
                             offDiagCol.begin();
                         hostOffDiagMatrixColIndex(i) = newColIndex;
                       });
  Kokkos::fence();

  sendRequestPtr_ = std::make_shared<std::vector<MPI_Request>>();
  recvRequestPtr_ = std::make_shared<std::vector<MPI_Request>>();

  sendStatusPtr_ = std::make_shared<std::vector<MPI_Status>>();
  recvStatusPtr_ = std::make_shared<std::vector<MPI_Status>>();

  sendRequestPtr_->resize(sendRank_.size());
  sendStatusPtr_->resize(sendRank_.size());

  recvRequestPtr_->resize(recvRank_.size());
  recvStatusPtr_->resize(recvRank_.size());

  Kokkos::resize(*deviceOffDiagVectorPtr_, offDiagCol.size());
  *hostOffDiagVectorPtr_ = Kokkos::create_mirror_view(*deviceOffDiagVectorPtr_);

  // update device memory
  Kokkos::deep_copy(deviceDiagMatrixRowIndex, hostDiagMatrixRowIndex);
  Kokkos::deep_copy(deviceOffDiagMatrixRowIndex, hostOffDiagMatrixRowIndex);

  Kokkos::deep_copy(deviceDiagMatrixColIndex, hostDiagMatrixColIndex);
  Kokkos::deep_copy(deviceOffDiagMatrixColIndex, hostOffDiagMatrixColIndex);

  Kokkos::deep_copy(deviceDiagMatrixValue, hostDiagMatrixValue);
  Kokkos::deep_copy(deviceOffDiagMatrixValue, hostOffDiagMatrixValue);

  isAssembled_ = true;
}

Void LinearAlgebra::Impl::DefaultMatrix::MatrixVectorMultiplication(
    DefaultVector &vec1, DefaultVector &vec2) const {
  if (!isAssembled_) {
    printf("Matrix is not assembled\n");
    while (true) {
    }
  }

  auto &deviceDiagMatrixRowIndex = *deviceDiagMatrixRowIndexPtr_;
  auto &deviceDiagMatrixColIndex = *deviceDiagMatrixColIndexPtr_;
  auto &deviceDiagMatrixValue = *deviceDiagMatrixValuePtr_;

  CommunicateOffDiagVectorBegin(vec1.GetDeviceVector());

  SequentialMatrixMultiplication(
      deviceDiagMatrixRowIndex, deviceDiagMatrixColIndex, deviceDiagMatrixValue,
      vec1.GetDeviceVector(), vec2.GetDeviceVector());

  auto &deviceOffDiagMatrixRowIndex = *deviceOffDiagMatrixRowIndexPtr_;
  auto &deviceOffDiagMatrixColIndex = *deviceOffDiagMatrixColIndexPtr_;
  auto &deviceOffDiagMatrixValue = *deviceOffDiagMatrixValuePtr_;

  CommunicateOffDiagVectorEnd();

  SequentialMatrixMultiplicationAddition(
      deviceOffDiagMatrixRowIndex, deviceOffDiagMatrixColIndex,
      deviceOffDiagMatrixValue, *deviceOffDiagVectorPtr_,
      vec2.GetDeviceVector());
}

Void LinearAlgebra::Impl::DefaultMatrix::MatrixVectorMultiplicationAddition(
    DefaultVector &vec1, DefaultVector &vec2) const {
  if (!isAssembled_) {
    printf("Matrix is not assembled\n");
    while (true) {
    }
  }

  auto &deviceDiagMatrixRowIndex = *deviceDiagMatrixRowIndexPtr_;
  auto &deviceDiagMatrixColIndex = *deviceDiagMatrixColIndexPtr_;
  auto &deviceDiagMatrixValue = *deviceDiagMatrixValuePtr_;

  CommunicateOffDiagVectorBegin(vec1.GetDeviceVector());

  SequentialMatrixMultiplicationAddition(
      deviceDiagMatrixRowIndex, deviceDiagMatrixColIndex, deviceDiagMatrixValue,
      vec1.GetDeviceVector(), vec2.GetDeviceVector());

  auto &deviceOffDiagMatrixRowIndex = *deviceOffDiagMatrixRowIndexPtr_;
  auto &deviceOffDiagMatrixColIndex = *deviceOffDiagMatrixColIndexPtr_;
  auto &deviceOffDiagMatrixValue = *deviceOffDiagMatrixValuePtr_;

  CommunicateOffDiagVectorEnd();

  SequentialMatrixMultiplicationAddition(
      deviceOffDiagMatrixRowIndex, deviceOffDiagMatrixColIndex,
      deviceOffDiagMatrixValue, *deviceOffDiagVectorPtr_,
      vec2.GetDeviceVector());
}

Void LinearAlgebra::Impl::DefaultMatrix::JacobiPreconditioning(
    DefaultVector &b, DefaultVector &x) {
  if (!isAssembled_) {
    printf("Matrix is not assembled\n");
    while (true) {
    }
  }

  if (!isJacobiPreconditioningPrepared_)
    PrepareJacobiPreconditioning();

  auto &xVector = x.GetDeviceVector();
  auto &bVector = b.GetDeviceVector();

  auto &deviceDiagMatrixValue = *deviceDiagMatrixValuePtr_;
  auto &deviceSeqDiagOffset = *deviceSeqDiagOffsetPtr_;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localRowSize_),
      KOKKOS_LAMBDA(const GlobalIndex row) {
        xVector(row) =
            bVector(row) / deviceDiagMatrixValue(deviceSeqDiagOffset(row));
      });
  Kokkos::fence();
}

Void LinearAlgebra::Impl::DefaultMatrix::SorPreconditioning(DefaultVector &b,
                                                            DefaultVector &x) {
  if (!isAssembled_) {
    printf("Matrix is not assembled\n");
    while (true) {
    }
  }

  if (!isSorPreconditioningPrepared_)
    PrepareSorPreconditioning();

  auto &xVector = x.GetDeviceVector();
  auto &bVector = b.GetDeviceVector();

  DeviceRealVector yVector;
  Kokkos::resize(yVector, localRowSize_);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localRowSize_),
      KOKKOS_LAMBDA(const GlobalIndex row) {
        yVector(row) = bVector(row);
        xVector(row) = 0.0;
      });
  Kokkos::fence();

  auto &hostMultiColorReordering = *hostMultiColorReorderingPtr_;
  auto &deviceMultiColorReordering = *deviceMultiColorReorderingPtr_;

  auto &deviceMultiColorReorderingRow = *deviceMultiColorReorderingRowPtr_;

  auto &deviceDiagMatrixRowIndex = *deviceDiagMatrixRowIndexPtr_;
  auto &deviceDiagMatrixColIndex = *deviceDiagMatrixColIndexPtr_;
  auto &deviceDiagMatrixValue = *deviceDiagMatrixValuePtr_;
  auto &deviceSeqDiagOffset = *deviceSeqDiagOffsetPtr_;

  for (int i = 0; i < hostMultiColorReordering.extent(0) - 1; i++) {
    if (i != 0) {
      LocalIndex rowSize =
          hostMultiColorReordering(i + 1) - hostMultiColorReordering(i);
      Kokkos::parallel_for(
          Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(rowSize,
                                                            Kokkos::AUTO()),
          KOKKOS_LAMBDA(
              const Kokkos::TeamPolicy<
                  Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
            const LocalIndex row =
                teamMember.league_rank() + hostMultiColorReordering(i);
            const LocalIndex rowIndex = deviceMultiColorReorderingRow(row);
            const GlobalIndex colSize = deviceDiagMatrixRowIndex(rowIndex + 1) -
                                        deviceDiagMatrixRowIndex(rowIndex);

            Scalar sum;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(teamMember, colSize),
                [&](const GlobalIndex j, Scalar &tSum) {
                  GlobalIndex offset = deviceDiagMatrixRowIndex(rowIndex) + j;
                  tSum += deviceDiagMatrixValue(offset) *
                          xVector(deviceDiagMatrixColIndex(offset));
                },
                Kokkos::Sum<Scalar>(sum));
            teamMember.team_barrier();
            yVector(rowIndex) -= sum;
          });
      Kokkos::fence();
    }

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            hostMultiColorReordering(i), hostMultiColorReordering(i + 1)),
        KOKKOS_LAMBDA(const GlobalIndex row) {
          const LocalIndex rowIndex = deviceMultiColorReorderingRow(row);
          xVector(rowIndex) =
              yVector(rowIndex) /
              deviceDiagMatrixValue(deviceSeqDiagOffset(rowIndex));
        });
    Kokkos::fence();
  }
}