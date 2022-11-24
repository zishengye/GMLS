#include "LinearAlgebra/Impl/Petsc/PetscVector.hpp"
#include "petsclog.h"
#include "petscsys.h"
#include "petscvec.h"
#include <memory>
#include <mpi.h>

LinearAlgebra::Impl::PetscVector::PetscVector() : ptr_(nullptr) {
  vecPtr_ = std::make_shared<Vec>();
}

LinearAlgebra::Impl::PetscVector::~PetscVector() {
  if (vecPtr_.use_count() == 1 && *vecPtr_ != PETSC_NULL)
    VecDestroy(vecPtr_.get());
}

LinearAlgebra::Impl::PetscVector::PetscVector(const int localSize) {
  localSize_ = localSize;
  vecPtr_ = std::make_shared<Vec>();

  VecCreateMPI(MPI_COMM_WORLD, localSize, PETSC_DECIDE, vecPtr_.get());
  VecGetArray(*vecPtr_, &ptr_);
}

PetscInt LinearAlgebra::Impl::PetscVector::GetLocalSize() { return localSize_; }

Void LinearAlgebra::Impl::PetscVector::Create(const std::vector<Scalar> &vec) {
  if (*vecPtr_ != PETSC_NULL)
    VecDestroy(&*vecPtr_);
  VecCreateMPI(MPI_COMM_WORLD, vec.size(), PETSC_DECIDE, &*vecPtr_);

  localSize_ = vec.size();

  VecGetArray(*vecPtr_, &ptr_);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, localSize_),
      [&](const std::size_t i) { ptr_[i] = vec[i]; });
}

Void LinearAlgebra::Impl::PetscVector::Create(const Vec &vec) {
  if (*vecPtr_ != PETSC_NULL)
    VecDestroy(&*vecPtr_);

  VecDuplicate(vec, &*vecPtr_);
  VecCopy(vec, *vecPtr_);
  VecGetArray(*vecPtr_, &ptr_);

  VecGetLocalSize(*vecPtr_, &localSize_);
}

Void LinearAlgebra::Impl::PetscVector::Create(const PetscVector &vec) {
  if (*vecPtr_ != PETSC_NULL)
    VecDestroy(&*vecPtr_);

  VecDuplicate(*(vec.vecPtr_), vecPtr_.get());
  VecCopy(*(vec.vecPtr_), *vecPtr_);
  VecGetArray(*vecPtr_, &ptr_);

  VecGetLocalSize(*vecPtr_, &localSize_);
}

Void LinearAlgebra::Impl::PetscVector::Create(const HostRealVector &vec) {
  if (*vecPtr_ != PETSC_NULL)
    VecDestroy(&*vecPtr_);
  VecCreateMPI(MPI_COMM_WORLD, vec.extent(0), PETSC_DECIDE, &*vecPtr_);

  VecGetArray(*vecPtr_, &ptr_);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, vec.extent(0)),
      [&](const std::size_t i) { ptr_[i] = vec(i); });

  VecGetLocalSize(*vecPtr_, &localSize_);
}

Void LinearAlgebra::Impl::PetscVector::Copy(std::vector<Scalar> &vec) {
  PetscInt size;
  VecGetLocalSize(*vecPtr_, &size);
  vec.resize(size);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, size),
      [&](const std::size_t i) { vec[i] = ptr_[i]; });
}

Void LinearAlgebra::Impl::PetscVector::Copy(Vec &vec) {
  VecCopy(*vecPtr_, vec);
}

Void LinearAlgebra::Impl::PetscVector::Copy(PetscVector &vec) {
  VecCopy(*vecPtr_, *(vec.vecPtr_));
}

Void LinearAlgebra::Impl::PetscVector::Copy(HostRealVector &vec) {
  PetscInt size;
  VecGetLocalSize(*vecPtr_, &size);
  Kokkos::resize(vec, size);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, size),
      [&](const std::size_t i) { vec(i) = ptr_[i]; });
}

Void LinearAlgebra::Impl::PetscVector::Clear() {
  if (vecPtr_.use_count() == 1 && *vecPtr_ == PETSC_NULL) {
    VecDestroy(vecPtr_.get());
  }

  *vecPtr_ = PETSC_NULL;
}

Void LinearAlgebra::Impl::PetscVector::Scale(const Scalar scalar) {
  VecScale(*vecPtr_, scalar);
}

Scalar &LinearAlgebra::Impl::PetscVector::operator()(const LocalIndex index) {
  return ptr_[index];
}

Void LinearAlgebra::Impl::PetscVector::operator=(const PetscVector &vec) {
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, localSize_),
      [&](const std::size_t i) { ptr_[i] = vec.ptr_[i]; });
}

Void LinearAlgebra::Impl::PetscVector::operator+=(const PetscVector &vec) {
  VecAXPY(*vecPtr_, 1.0, *(vec.vecPtr_));
}

Void LinearAlgebra::Impl::PetscVector::operator-=(const PetscVector &vec) {
  VecAXPY(*vecPtr_, -1.0, *(vec.vecPtr_));
}

Void LinearAlgebra::Impl::PetscVector::operator*=(const PetscReal scalar) {
  VecScale(*vecPtr_, scalar);
}

Void LinearAlgebra::Impl::PetscVector::OrthogonalizeToConstant(
    const PetscInt start, const PetscInt end) {
  PetscReal sum, average;
  PetscInt length = end - start;

  PetscReal *a;
  VecGetArray(*vecPtr_, &a);
  sum = 0.0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(start, end),
      [&](const std::size_t i, double &tSum) { tSum += a[i]; },
      Kokkos::Sum<double>(sum));
  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &length, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  average = sum / (double)length;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(start, end),
      [&](const std::size_t i) { a[i] -= average; });
  VecRestoreArray(*vecPtr_, &a);
}