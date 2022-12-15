#include "LinearAlgebra/Impl/Petsc/PetscMatrix.hpp"
#include "LinearAlgebra/Impl/Petsc/PetscVector.hpp"

#include <algorithm>
#include <memory>

#include <mpi.h>
#include <petscmat.h>

LinearAlgebra::Impl::PetscMatrix::PetscMatrix() {
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

  matSharedPtr_ = std::make_shared<Mat>();
  *matSharedPtr_ = PETSC_NULL;
}

LinearAlgebra::Impl::PetscMatrix::~PetscMatrix() {
  if (matSharedPtr_.use_count() == 1)
    if (*matSharedPtr_ != PETSC_NULL) {
      MatDestroy(matSharedPtr_.get());
    }
}

Void LinearAlgebra::Impl::PetscMatrix::Resize(const PetscInt m,
                                              const PetscInt n,
                                              const PetscInt blockSize) {
  localRowSize_ = m;
  localColSize_ = n;

  Clear();

  diagMatrixCol_.clear();
  diagMatrixCol_.resize(localRowSize_);
  offDiagMatrixCol_.clear();
  offDiagMatrixCol_.resize(localRowSize_);

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
}

Void LinearAlgebra::Impl::PetscMatrix::Transpose(PetscMatrix &mat) {
  MatTranspose(*mat.matSharedPtr_, MAT_INITIAL_MATRIX, matSharedPtr_.get());
}

Void LinearAlgebra::Impl::PetscMatrix::Clear() {
  if (matSharedPtr_.use_count() == 1 && matSharedPtr_ == PETSC_NULL) {
    MatDestroy(matSharedPtr_.get());
  }

  *matSharedPtr_ = PETSC_NULL;
}

PetscInt LinearAlgebra::Impl::PetscMatrix::GetLocalColSize() const {
  return localColSize_;
}

PetscInt LinearAlgebra::Impl::PetscMatrix::GetLocalRowSize() const {
  return localRowSize_;
}

PetscInt LinearAlgebra::Impl::PetscMatrix::GetGlobalColSize() const {
  return globalColSize_;
}

PetscInt LinearAlgebra::Impl::PetscMatrix::GetGlobalRowSize() const {
  return globalRowSize_;
}

void LinearAlgebra::Impl::PetscMatrix::SetColIndex(
    const PetscInt row, const std::vector<PetscInt> &index) {
  std::vector<PetscInt> &diagIndex = diagMatrixCol_[row];
  std::vector<PetscInt> &offDiagIndex = offDiagMatrixCol_[row];
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

Void LinearAlgebra::Impl::PetscMatrix::Increment(const PetscInt row,
                                                 const PetscInt col,
                                                 const PetscReal value) {
  if (col >= colRangeLow_ && col < colRangeHigh_) {
    std::size_t offset =
        std::lower_bound(
            diagMatrixColIndex_.begin() + diagMatrixRowOffset_[row],
            diagMatrixColIndex_.begin() + diagMatrixRowOffset_[row + 1],
            col - colRangeLow_) -
        diagMatrixColIndex_.begin();
    diagMatrixVal_[offset] = value;
  } else {
    std::size_t offset =
        std::lower_bound(
            offDiagMatrixColIndex_.begin() + offDiagMatrixRowOffset_[row],
            offDiagMatrixColIndex_.begin() + offDiagMatrixRowOffset_[row + 1],
            col) -
        offDiagMatrixColIndex_.begin();
    offDiagMatrixVal_[offset] = value;
  }
}

Void LinearAlgebra::Impl::PetscMatrix::Increment(
    const PetscInt row, const std::vector<PetscInt> &index,
    const std::vector<PetscReal> &value) {}

Void LinearAlgebra::Impl::PetscMatrix::GraphAssemble() {
  unsigned long diagNumNonzero = 0;
  unsigned long offDiagNumNonzero = 0;
  unsigned long maxDiagNonzero = 0;
  unsigned long maxOffDiagNonzero = 0;
  for (int i = 0; i < localRowSize_; i++) {
    diagNumNonzero += diagMatrixCol_[i].size();
    if (diagMatrixCol_[i].size() > maxDiagNonzero)
      maxDiagNonzero = diagMatrixCol_[i].size();
    offDiagNumNonzero += offDiagMatrixCol_[i].size();
    if (offDiagMatrixCol_[i].size() > maxOffDiagNonzero)
      maxOffDiagNonzero = offDiagMatrixCol_[i].size();
  }

  std::vector<PetscInt> diagNonzero(localRowSize_);
  std::vector<PetscInt> offDiagNonzero(localRowSize_);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, localRowSize_),
      [&](const unsigned int i) {
        diagNonzero[i] = diagMatrixCol_[i].size();
        offDiagNonzero[i] = offDiagMatrixCol_[i].size();
      });
  Kokkos::fence();

  diagMatrixColIndex_.resize(diagNumNonzero);
  diagMatrixVal_.resize(diagNumNonzero);
  auto it = diagMatrixColIndex_.begin();
  for (int i = 0; i < localRowSize_; i++) {
    for (int j = 0; j < diagMatrixCol_[i].size(); j++) {
      *it = diagMatrixCol_[i][j];
      it++;
    }
  }

  diagMatrixRowOffset_.resize(localRowSize_ + 1);
  diagMatrixRowOffset_[0] = 0;
  for (int i = 0; i < localRowSize_; i++)
    diagMatrixRowOffset_[i + 1] =
        diagMatrixRowOffset_[i] + diagMatrixCol_[i].size();

  offDiagMatrixColIndex_.resize(offDiagNumNonzero);
  offDiagMatrixVal_.resize(offDiagNumNonzero);
  it = offDiagMatrixColIndex_.begin();
  for (int i = 0; i < localRowSize_; i++) {
    for (int j = 0; j < offDiagMatrixCol_[i].size(); j++) {
      *it = offDiagMatrixCol_[i][j];
      it++;
    }
  }

  offDiagMatrixRowOffset_.resize(localRowSize_ + 1);
  offDiagMatrixRowOffset_[0] = 0;
  for (int i = 0; i < localRowSize_; i++)
    offDiagMatrixRowOffset_[i + 1] =
        offDiagMatrixRowOffset_[i] + offDiagMatrixCol_[i].size();
}

Void LinearAlgebra::Impl::PetscMatrix::Assemble() {
  MatCreateMPIAIJWithSplitArrays(
      MPI_COMM_WORLD, localRowSize_, localColSize_, PETSC_DECIDE, PETSC_DECIDE,
      diagMatrixRowOffset_.data(), diagMatrixColIndex_.data(),
      diagMatrixVal_.data(), offDiagMatrixRowOffset_.data(),
      offDiagMatrixColIndex_.data(), offDiagMatrixVal_.data(),
      matSharedPtr_.get());
}

Void LinearAlgebra::Impl::PetscMatrix::MatrixVectorMultiplication(
    PetscVector &vec1, PetscVector &vec2) {
  MatMult(*matSharedPtr_, *(vec1.vecPtr_), *(vec2.vecPtr_));
}