#include "PetscMatrix.hpp"

PetscMatrix::PetscMatrix() : PetscMatrixBase() {}

PetscMatrix::PetscMatrix(const PetscInt m, const PetscInt n,
                         const PetscInt blockSize)
    : PetscMatrixBase(), localRowSize_(m), localColSize_(n),
      blockSize_(blockSize) {
  diagMatrixCol_.resize(localRowSize_);
  offDiagMatrixCol_.resize(localRowSize_);

  blockStorage_ = blockSize_ * blockSize_;
}

PetscMatrix::~PetscMatrix() {}

void PetscMatrix::Resize(const PetscInt m, const PetscInt n,
                         const PetscInt blockSize) {
  localRowSize_ = m;
  localColSize_ = n;
  blockSize_ = blockSize;
  blockStorage_ = blockSize_ * blockSize_;

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

PetscInt PetscMatrix::GetRowSize() { return localRowSize_ * blockSize_; }

PetscInt PetscMatrix::GetColSize() { return localColSize_ * blockSize_; }

void PetscMatrix::SetColIndex(const PetscInt row,
                              const std::vector<PetscInt> &index) {
  if (row < 0 || row > localRowSize_) {
    std::cout << "wrong row index " << row << std::endl;
    return;
  }
  std::vector<PetscInt> &diagIndex = diagMatrixCol_[row];
  std::vector<PetscInt> &offDiagIndex = offDiagMatrixCol_[row];
  diagIndex.clear();
  offDiagIndex.clear();
  for (auto it = index.begin(); it != index.end(); it++) {
    if (*it >= colRangeLow_ && *it < colRangeHigh_)
      diagIndex.push_back(*it - colRangeLow_);
    else
      offDiagIndex.push_back(*it);
  }
}

void PetscMatrix::Increment(const PetscInt row, const PetscInt col,
                            const PetscReal value) {
  if (row > localRowSize_) {
    std::cout << "increment wrong row setup" << std::endl;
    return;
  }
  MatSetValue(mat_, row + rowRangeLow_, col, value, INSERT_VALUES);
}

void PetscMatrix::IncrementGlobalIndex(const PetscInt row, const PetscInt col,
                                       const PetscReal value) {
  MatSetValue(mat_, row, col, value, INSERT_VALUES);
}

void PetscMatrix::Increment(const PetscInt row,
                            const std::vector<PetscInt> &index,
                            const std::vector<PetscReal> &value) {
  if (row > localRowSize_) {
    std::cout << "increment wrong row setup" << std::endl;
    return;
  }

  std::vector<PetscInt> rowIndex(1);
  rowIndex[0] = row + rowRangeLow_;

  if (blockSize_ == 1) {
    MatSetValues(mat_, 1, rowIndex.data(), index.size(), index.data(),
                 value.data(), INSERT_VALUES);
  } else
    MatSetValuesBlocked(mat_, 1, rowIndex.data(), index.size(), index.data(),
                        value.data(), INSERT_VALUES);
}

unsigned long PetscMatrix::GraphAssemble() {
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
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localRowSize_),
      [&](const unsigned int i) {
        diagNonzero[i] = diagMatrixCol_[i].size();
        offDiagNonzero[i] = offDiagMatrixCol_[i].size();
      });
  Kokkos::fence();

  if (blockSize_ == 1)
    MatCreateAIJ(MPI_COMM_WORLD, localRowSize_, localColSize_, PETSC_DECIDE,
                 PETSC_DECIDE, 0, diagNonzero.data(), 0, offDiagNonzero.data(),
                 &mat_);
  else
    MatCreateBAIJ(MPI_COMM_WORLD, blockSize_, localRowSize_ * blockSize_,
                  localColSize_ * blockSize_, PETSC_DECIDE, PETSC_DECIDE, 0,
                  diagNonzero.data(), 0, offDiagNonzero.data(), &mat_);

  MatSetUp(mat_);

  return diagNumNonzero + offDiagNumNonzero;
}

unsigned long PetscMatrix::Assemble() {
  MatAssemblyBegin(mat_, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(mat_, MAT_FINAL_ASSEMBLY);

  return diagCol_.size() + offDiagCol_.size();
}