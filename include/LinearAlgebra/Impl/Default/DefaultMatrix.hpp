#ifndef _LinearAlgebra_Impl_Default_DefaultMatrix_Hpp_
#define _LinearAlgebra_Impl_Default_DefaultMatrix_Hpp_

#include <memory>
#include <vector>

#include <mpi.h>

#include "Core/Typedef.hpp"
#include "LinearAlgebra/Impl/Default/Default.hpp"
#include "LinearAlgebra/Impl/Default/DefaultVector.hpp"

namespace LinearAlgebra {
namespace Impl {
Void SequentialMatrixMultiplication(DeviceIndexVector &ia,
                                    DeviceIndexVector &ja, DeviceRealVector &a,
                                    DeviceRealVector &x, DeviceRealVector &y);
Void SequentialMatrixMultiplicationAddition(DeviceIndexVector &ia,
                                            DeviceIndexVector &ja,
                                            DeviceRealVector &a,
                                            DeviceRealVector &x,
                                            DeviceRealVector &y);

Void SequentialOffDiagMatrixMultiplication(DeviceIndexVector &ia,
                                           DeviceIndexVector &ja,
                                           DeviceRealVector &a,
                                           DeviceRealVector &x,
                                           DeviceRealVector &y);

class DefaultMatrix {
protected:
  int mpiRank_, mpiSize_;

  Boolean isHostDeviceSynchronized_; // is host data updated but has not been
                                     // synchronized on device
  Boolean isDeviceHostSynchronized_; // is device data update but has not been
                                     // synchronized on host

  Boolean isAssembled_;

  LocalIndex localColSize_, localRowSize_;
  GlobalIndex globalColSize_, globalRowSize_;
  LocalIndex colRangeLow_, colRangeHigh_;
  LocalIndex rowRangeLow_, rowRangeHigh_;

  std::vector<std::size_t> rankColSize_, rankRowSize_;
  std::vector<std::size_t> rankColOffset_, rankRowOffset_;
  std::shared_ptr<std::vector<std::vector<GlobalIndex>>> diagMatrixGraphPtr_,
      offDiagMatrixGraphPtr_;

  // crs matrix
  std::shared_ptr<DeviceIndexVector::HostMirror> hostDiagMatrixRowIndexPtr_,
      hostOffDiagMatrixRowIndexPtr_;
  std::shared_ptr<DeviceIndexVector::HostMirror> hostDiagMatrixColIndexPtr_,
      hostOffDiagMatrixColIndexPtr_;
  std::shared_ptr<DeviceRealVector::HostMirror> hostDiagMatrixValuePtr_,
      hostOffDiagMatrixValuePtr_;

  std::shared_ptr<DeviceIndexVector> deviceDiagMatrixRowIndexPtr_,
      deviceOffDiagMatrixRowIndexPtr_;
  std::shared_ptr<DeviceIndexVector> deviceDiagMatrixColIndexPtr_,
      deviceOffDiagMatrixColIndexPtr_;
  std::shared_ptr<DeviceRealVector> deviceDiagMatrixValuePtr_,
      deviceOffDiagMatrixValuePtr_;

  // auxiliary vector
  std::shared_ptr<DeviceRealVector::HostMirror> hostOffDiagVectorPtr_;
  std::shared_ptr<DeviceRealVector> deviceOffDiagVectorPtr_;
  std::shared_ptr<std::vector<GlobalIndex>> sendVectorColIndexPtr_;

  std::vector<int> sendRank_, recvRank_;
  std::vector<int> sendCountByRank_, recvCountByRank_;
  std::vector<int> sendOffsetByRank_, recvOffsetByRank_;

  std::shared_ptr<std::vector<MPI_Request>> sendRequestPtr_, recvRequestPtr_;
  std::shared_ptr<std::vector<MPI_Status>> sendStatusPtr_, recvStatusPtr_;

  // To split the communication into two stages is to take advantage of
  // asynchronous communication
  Void CommunicateOffDiagVectorBegin(const DeviceRealVector &vec) const;
  Void CommunicateOffDiagVectorEnd() const;

  // Preconditioning
  std::shared_ptr<DeviceIndexVector> deviceSeqDiagOffsetPtr_;
  std::shared_ptr<DeviceIndexVector::HostMirror> hostSeqDiagOffsetPtr_;
  Void FindSeqDiagOffset();

  Boolean isJacobiPreconditioningPrepared_;

  Void PrepareJacobiPreconditioning();

  Boolean isSorPreconditioningPrepared_;

  // multi-coloring reordering
  std::shared_ptr<DeviceIntVector> deviceMultiColorReorderingPtr_,
      deviceMultiColorReorderingRowPtr_;
  std::shared_ptr<DeviceIntVector::HostMirror> hostMultiColorReorderingPtr_,
      hostMultiColorReorderingRowPtr_;

  Void PrepareSorPreconditioning();

public:
  DefaultMatrix();

  ~DefaultMatrix();

  virtual Void Resize(const GlobalIndex m, const GlobalIndex n,
                      const LocalIndex blockSize = 1);

  Void Transpose(DefaultMatrix &mat);

  Void Clear();

  LocalIndex GetLocalColSize() const;
  LocalIndex GetLocalRowSize() const;

  GlobalIndex GetGlobalColSize() const;
  GlobalIndex GetGlobalRowSize() const;

  inline virtual Void SetColIndex(const GlobalIndex row,
                                  const std::vector<GlobalIndex> &index);
  inline virtual Void Increment(const GlobalIndex row, const GlobalIndex col,
                                const Scalar value);

  virtual Void GraphAssemble();
  virtual Void Assemble();

  virtual Void MatrixVectorMultiplication(DefaultVector &vec1,
                                          DefaultVector &vec2) const;
  virtual Void MatrixVectorMultiplicationAddition(DefaultVector &vec1,
                                                  DefaultVector &vec2) const;

  virtual Void JacobiPreconditioning(DefaultVector &b, DefaultVector &x);
  virtual Void SorPreconditioning(DefaultVector &b, DefaultVector &x);
};
} // namespace Impl
} // namespace LinearAlgebra

#endif