#include "LinearAlgebra/Impl/Default/DefaultVector.hpp"

#include "Core/Typedef.hpp"
#include "LinearAlgebra/Impl/Default/Default.hpp"
#include "LinearAlgebra/Impl/Default/DefaultMatrix.hpp"

#include <Kokkos_Core.hpp>
#include <memory>
#include <mpi.h>
#include <vector>

Void LinearAlgebra::Impl::DefaultVector::ResizeInternal(
    const LocalIndex localSize) {
  auto &deviceData = *deviceDataPtr_;
  auto &hostData = *hostDataPtr_;

  Kokkos::resize(deviceData, localSize);
  hostData = Kokkos::create_mirror_view(deviceData);

  localSize_ = localSize;
  globalSize_ = localSize;
  MPI_Allreduce(MPI_IN_PLACE, &globalSize_, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);

  isHostDeviceSynchronized_ = false;
}

Void LinearAlgebra::Impl::DefaultVector::HostDeviceSynchronization() {
  auto &deviceData = *deviceDataPtr_;
  auto &hostData = *hostDataPtr_;

  if (isDeviceAvailable_ && !isHostDeviceSynchronized_)
    Kokkos::deep_copy(deviceData, hostData);

  isHostDeviceSynchronized_ = true;
}

Void LinearAlgebra::Impl::DefaultVector::DeviceHostSynchronization() {
  auto &deviceData = *deviceDataPtr_;
  auto &hostData = *hostDataPtr_;

  if (isDeviceAvailable_ && !isDeviceHostSynchronized_)
    Kokkos::deep_copy(hostData, deviceData);

  isDeviceHostSynchronized_ = true;
}

LinearAlgebra::Impl::DefaultVector::DefaultVector()
    : isDeviceAvailable_(true), isHostDeviceSynchronized_(false),
      isDeviceHostSynchronized_(false), localSize_(0), globalSize_(0) {
  deviceDataPtr_ = std::make_shared<DeviceRealVector>();
  hostDataPtr_ = std::make_shared<DeviceRealVector::HostMirror>();

  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);
}

LinearAlgebra::Impl::DefaultVector::DefaultVector(const LocalIndex localSize)
    : isDeviceAvailable_(true), isHostDeviceSynchronized_(false),
      isDeviceHostSynchronized_(false), localSize_(0), globalSize_(0) {
  deviceDataPtr_ = std::make_shared<DeviceRealVector>();
  hostDataPtr_ = std::make_shared<DeviceRealVector::HostMirror>();

  ResizeInternal(localSize);

  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);
}

LinearAlgebra::Impl::DefaultVector::~DefaultVector() {}

LocalIndex LinearAlgebra::Impl::DefaultVector::GetLocalSize() const {
  return localSize_;
}

GlobalIndex LinearAlgebra::Impl::DefaultVector::GetGlobalSize() const {
  return globalSize_;
}

Void LinearAlgebra::Impl::DefaultVector::Resize(const LocalIndex localSize) {
  ResizeInternal(localSize);
}

Void LinearAlgebra::Impl::DefaultVector::Create(
    const std::vector<Scalar> &vec) {
  ResizeInternal(vec.size());

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, localSize_),
      [&](const LocalIndex i) { (*hostDataPtr_)(i) = vec[i]; });

  isHostDeviceSynchronized_ = false;
}

Void LinearAlgebra::Impl::DefaultVector::Create(const HostRealVector &vec) {
  ResizeInternal(vec.extent(0));

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, localSize_),
      [&](const LocalIndex i) { (*hostDataPtr_)(i) = vec(i); });

  isHostDeviceSynchronized_ = false;
}

Void LinearAlgebra::Impl::DefaultVector::Create(const DefaultVector &vec) {
  ResizeInternal(vec.GetLocalSize());

  auto &targetHostData = *hostDataPtr_;
  auto &sourceHostData = *vec.hostDataPtr_;

  Kokkos::deep_copy(targetHostData, sourceHostData);

  isHostDeviceSynchronized_ = false;
}

Void LinearAlgebra::Impl::DefaultVector::Copy(std::vector<Scalar> &vec) {
  vec.resize(localSize_);

  DeviceHostSynchronization();

  auto &sourceHostData = *hostDataPtr_;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, localSize_),
      [&](const LocalIndex i) { vec[i] = sourceHostData(i); });
}

Void LinearAlgebra::Impl::DefaultVector::Copy(HostRealVector &vec) {
  Kokkos::resize(vec, localSize_);

  DeviceHostSynchronization();

  auto &sourceHostData = *hostDataPtr_;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, localSize_),
      [&](const LocalIndex i) { vec(i) = sourceHostData(i); });
}

Void LinearAlgebra::Impl::DefaultVector::Copy(DefaultVector &vec) {
  vec.ResizeInternal(localSize_);

  auto &targetHostData = *vec.deviceDataPtr_;
  auto &sourceHostData = *hostDataPtr_;

  Kokkos::deep_copy(targetHostData, sourceHostData);

  vec.isDeviceHostSynchronized_ = false;
}

Void LinearAlgebra::Impl::DefaultVector::Clear() {}

Scalar LinearAlgebra::Impl::DefaultVector::Norm1() {
  HostDeviceSynchronization();

  auto &targetDeviceData = *deviceDataPtr_;

  Scalar normResult = 0.0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
      KOKKOS_LAMBDA(const LocalIndex i, Scalar &tNormResult) {
        tNormResult += std::abs(targetDeviceData(i));
      },
      Kokkos::Sum<Scalar>(normResult));
  Kokkos::fence();

  MPI_Allreduce(MPI_IN_PLACE, &normResult, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  return normResult;
}

Scalar LinearAlgebra::Impl::DefaultVector::Norm2() {
  HostDeviceSynchronization();

  auto &targetDeviceData = *deviceDataPtr_;

  Scalar normResult = 0.0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
      KOKKOS_LAMBDA(const LocalIndex i, Scalar &tNormResult) {
        tNormResult += targetDeviceData(i) * targetDeviceData(i);
      },
      Kokkos::Sum<Scalar>(normResult));
  Kokkos::fence();

  MPI_Allreduce(MPI_IN_PLACE, &normResult, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  normResult = sqrt(normResult);

  return normResult;
}

Scalar LinearAlgebra::Impl::DefaultVector::NormInf() {
  HostDeviceSynchronization();

  auto &targetDeviceData = *deviceDataPtr_;

  Scalar normResult = 0.0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
      KOKKOS_LAMBDA(const LocalIndex i, Scalar &tNormResult) {
        tNormResult = std::max(std::abs(targetDeviceData(i)), tNormResult);
      },
      Kokkos::Sum<Scalar>(normResult));
  Kokkos::fence();

  MPI_Allreduce(MPI_IN_PLACE, &normResult, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);

  return normResult;
}

Void LinearAlgebra::Impl::DefaultVector::Normalize() {
  const Scalar norm = Norm2();

  auto &targetDeviceData = *deviceDataPtr_;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
      KOKKOS_LAMBDA(const LocalIndex i) { targetDeviceData(i) /= norm; });
  Kokkos::fence();
}

Void LinearAlgebra::Impl::DefaultVector::Scale(const Scalar factor) {
  HostDeviceSynchronization();

  auto &leftDeviceVector = *deviceDataPtr_;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
      KOKKOS_LAMBDA(const LocalIndex i) { leftDeviceVector(i) *= factor; });
  Kokkos::fence();
}

Void LinearAlgebra::Impl::DefaultVector::Set(const Scalar value) {
  auto &leftDeviceVector = *deviceDataPtr_;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
      KOKKOS_LAMBDA(const LocalIndex i) { leftDeviceVector(i) = value; });
  Kokkos::fence();
}

Void LinearAlgebra::Impl::DefaultVector::Zeros() {
  auto &leftDeviceVector = *deviceDataPtr_;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
      KOKKOS_LAMBDA(const LocalIndex i) { leftDeviceVector(i) = 0.0; });
  Kokkos::fence();
}

Void LinearAlgebra::Impl::DefaultVector::Ones() {
  auto &leftDeviceVector = *deviceDataPtr_;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
      KOKKOS_LAMBDA(const LocalIndex i) { leftDeviceVector(i) = 1.0; });
  Kokkos::fence();
}

Scalar &LinearAlgebra::Impl::DefaultVector::operator()(const LocalIndex index) {
  isHostDeviceSynchronized_ = false;
  return (*hostDataPtr_)(index);
}

const Scalar &
LinearAlgebra::Impl::DefaultVector::operator()(const LocalIndex index) const {
  return (*hostDataPtr_)(index);
}

Void LinearAlgebra::Impl::DefaultVector::operator=(const DefaultVector &vec) {
  ResizeInternal(vec.GetLocalSize());

  auto &leftDeviceVector = *deviceDataPtr_;
  auto &rightDeviceVector = *vec.deviceDataPtr_;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
      KOKKOS_LAMBDA(const LocalIndex i) {
        leftDeviceVector(i) = rightDeviceVector(i);
      });
  Kokkos::fence();
}

Void LinearAlgebra::Impl::DefaultVector::operator=(
    const DefaultVectorEntry &vecEntry) {
  LocalIndex localSize;

  if (vecEntry.ExistLeftMatrixOperand())
    localSize = vecEntry.GetLeftMatrixOperand()->GetLocalRowSize();
  else
    localSize = vecEntry.GetLeftVectorOperand()->GetLocalSize();

  ResizeInternal(localSize);

  if (vecEntry.ExistLeftMatrixOperand())
    vecEntry.GetLeftMatrixOperand()->MatrixVectorMultiplication(
        *vecEntry.GetRightVectorOperand(), *this);
  else {
    if (vecEntry.ExistRightScalarOperand()) {
      Scalar scalar = vecEntry.GetRightScalarOperand();

      auto &leftDeviceVector = *deviceDataPtr_;
      auto &rightDeviceVector =
          *(vecEntry.GetLeftVectorOperand()->deviceDataPtr_);

      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
          KOKKOS_LAMBDA(const LocalIndex i) {
            leftDeviceVector(i) = scalar * rightDeviceVector(i);
          });
      Kokkos::fence();
    } else {
      auto &leftDeviceVector = *deviceDataPtr_;
      auto &rightDeviceVector1 =
          *(vecEntry.GetLeftVectorOperand()->deviceDataPtr_);
      auto &rightDeviceVector2 =
          *(vecEntry.GetRightVectorOperand()->deviceDataPtr_);

      char operatorName = vecEntry.GetOperatorName();

      if (operatorName == '+') {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
            KOKKOS_LAMBDA(const LocalIndex i) {
              leftDeviceVector(i) =
                  rightDeviceVector1(i) + rightDeviceVector2(i);
            });
        Kokkos::fence();
      } else if (operatorName == '-') {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
            KOKKOS_LAMBDA(const LocalIndex i) {
              leftDeviceVector(i) =
                  rightDeviceVector1(i) - rightDeviceVector2(i);
            });
        Kokkos::fence();
      }
    }
  }
}

Void LinearAlgebra::Impl::DefaultVector::operator+=(const DefaultVector &vec) {
  HostDeviceSynchronization();

  auto &leftDeviceVector = *deviceDataPtr_;
  auto &rightDeviceVector = *vec.deviceDataPtr_;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
      KOKKOS_LAMBDA(const LocalIndex i) {
        leftDeviceVector(i) += rightDeviceVector(i);
      });
  Kokkos::fence();
}

Void LinearAlgebra::Impl::DefaultVector::operator+=(
    const DefaultVectorEntry &vecEntry) {
  HostDeviceSynchronization();

  if (vecEntry.ExistLeftMatrixOperand())
    vecEntry.GetLeftMatrixOperand()->MatrixVectorMultiplicationAddition(
        *vecEntry.GetRightVectorOperand(), *this);
  else {
    if (vecEntry.ExistRightScalarOperand()) {
      Scalar scalar = vecEntry.GetRightScalarOperand();

      auto &leftDeviceVector = *deviceDataPtr_;
      auto &rightDeviceVector =
          *(vecEntry.GetLeftVectorOperand()->deviceDataPtr_);

      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
          KOKKOS_LAMBDA(const LocalIndex i) {
            leftDeviceVector(i) += scalar * rightDeviceVector(i);
          });
      Kokkos::fence();
    } else {
      auto &leftDeviceVector = *deviceDataPtr_;
      auto &rightDeviceVector1 =
          *(vecEntry.GetLeftVectorOperand()->deviceDataPtr_);
      auto &rightDeviceVector2 =
          *(vecEntry.GetRightVectorOperand()->deviceDataPtr_);

      char operatorName = vecEntry.GetOperatorName();

      if (operatorName == '+') {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
            KOKKOS_LAMBDA(const LocalIndex i) {
              leftDeviceVector(i) +=
                  rightDeviceVector1(i) + rightDeviceVector2(i);
            });
        Kokkos::fence();
      } else if (operatorName == '-') {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
            KOKKOS_LAMBDA(const LocalIndex i) {
              leftDeviceVector(i) +=
                  rightDeviceVector1(i) - rightDeviceVector2(i);
            });
        Kokkos::fence();
      }
    }
  }
}

Void LinearAlgebra::Impl::DefaultVector::operator-=(const DefaultVector &vec) {
  HostDeviceSynchronization();

  auto &leftDeviceVector = *deviceDataPtr_;
  auto &rightDeviceVector = *vec.deviceDataPtr_;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
      KOKKOS_LAMBDA(const LocalIndex i) {
        leftDeviceVector(i) -= rightDeviceVector(i);
      });
  Kokkos::fence();
}

Void LinearAlgebra::Impl::DefaultVector::operator-=(
    const DefaultVectorEntry &vecEntry) {
  HostDeviceSynchronization();

  if (vecEntry.ExistLeftMatrixOperand()) {
    DefaultVector vec;
    vec.Resize(localSize_);
    vecEntry.GetLeftMatrixOperand()->MatrixVectorMultiplication(
        *vecEntry.GetRightVectorOperand(), vec);

    auto &leftDeviceVector = *deviceDataPtr_;
    auto &rightDeviceVector = *vec.deviceDataPtr_;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
        KOKKOS_LAMBDA(const LocalIndex i) {
          leftDeviceVector(i) -= rightDeviceVector(i);
        });
    Kokkos::fence();
  } else {
    if (vecEntry.ExistRightScalarOperand()) {
      Scalar scalar = vecEntry.GetRightScalarOperand();

      auto &leftDeviceVector = *deviceDataPtr_;
      auto &rightDeviceVector =
          *(vecEntry.GetLeftVectorOperand()->deviceDataPtr_);

      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
          KOKKOS_LAMBDA(const LocalIndex i) {
            leftDeviceVector(i) -= scalar * rightDeviceVector(i);
          });
      Kokkos::fence();
    } else {
      auto &leftDeviceVector = *deviceDataPtr_;
      auto &rightDeviceVector1 =
          *(vecEntry.GetLeftVectorOperand()->deviceDataPtr_);
      auto &rightDeviceVector2 =
          *(vecEntry.GetRightVectorOperand()->deviceDataPtr_);

      char operatorName = vecEntry.GetOperatorName();

      if (operatorName == '+') {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
            KOKKOS_LAMBDA(const LocalIndex i) {
              leftDeviceVector(i) -=
                  rightDeviceVector1(i) + rightDeviceVector2(i);
            });
        Kokkos::fence();
      } else if (operatorName == '-') {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
            KOKKOS_LAMBDA(const LocalIndex i) {
              leftDeviceVector(i) -=
                  rightDeviceVector1(i) - rightDeviceVector2(i);
            });
        Kokkos::fence();
      }
    }
  }
}

Void LinearAlgebra::Impl::DefaultVector::operator*=(const Scalar scalar) {
  HostDeviceSynchronization();

  auto &leftDeviceVector = *deviceDataPtr_;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
      KOKKOS_LAMBDA(const LocalIndex i) { leftDeviceVector(i) *= scalar; });
  Kokkos::fence();
}

LinearAlgebra::Impl::DefaultVectorEntry
LinearAlgebra::Impl::DefaultVector::operator*(const Scalar scalar) {
  DefaultVectorEntry vecEntry;

  vecEntry.InsertLeftVectorOperand(std::make_shared<DefaultVector>(*this));
  vecEntry.InsertRightScalarOperand(scalar);

  return vecEntry;
}

LinearAlgebra::Impl::DefaultVectorEntry
LinearAlgebra::Impl::DefaultVector::operator/(const Scalar scalar) {
  DefaultVectorEntry vecEntry;

  vecEntry.InsertLeftVectorOperand(std::make_shared<DefaultVector>(*this));
  vecEntry.InsertRightScalarOperand(1.0 / scalar);

  return vecEntry;
}

Scalar LinearAlgebra::Impl::DefaultVector::Dot(const DefaultVector &vec) {
  HostDeviceSynchronization();

  auto &leftDeviceVector = *deviceDataPtr_;
  auto &rightDeviceVector = *vec.deviceDataPtr_;

  Scalar sum = 0.0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
      KOKKOS_LAMBDA(const LocalIndex i, Scalar &tSum) {
        tSum += leftDeviceVector(i) * rightDeviceVector(i);
      },
      Kokkos::Sum<Scalar>(sum));
  Kokkos::fence();

  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return sum;
}

DeviceRealVector &LinearAlgebra::Impl::DefaultVector::GetDeviceVector() {
  return *deviceDataPtr_;
}

Void LinearAlgebra::Impl::DefaultVector::OrthogonalizeToConstant(
    const LocalIndex start, const LocalIndex end) {
  Scalar sum = 0.0;

  auto &targetDeviceData = *deviceDataPtr_;

  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
      KOKKOS_LAMBDA(const LocalIndex i, Scalar &tSum) {
        tSum += targetDeviceData(i);
      },
      Kokkos::Sum<Scalar>(sum));
  Kokkos::fence();

  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  GlobalIndex length = end - start;
  MPI_Allreduce(MPI_IN_PLACE, &length, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);

  Scalar average = sum / (Scalar)(length);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localSize_),
      KOKKOS_LAMBDA(const LocalIndex i) { targetDeviceData(i) -= average; });
  Kokkos::fence();
}

LinearAlgebra::Impl::DefaultVectorEntry
LinearAlgebra::Impl::operator*(const Scalar scalar,
                               LinearAlgebra::Impl::DefaultVector &vec) {
  LinearAlgebra::Impl::DefaultVectorEntry vecEntry;
  vecEntry.InsertLeftVectorOperand(
      std::make_shared<LinearAlgebra::Impl::DefaultVector>(vec));
  vecEntry.InsertRightScalarOperand(scalar);

  return vecEntry;
}