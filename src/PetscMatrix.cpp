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

  std::vector<unsigned long> rankColSize(mpiSize_);
  MPI_Allgather(&localColSize, 1, MPI_UNSIGNED_LONG, rankColSize.data(), 1,
                MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
  colRangeLow_ = 0;
  for (int i = 0; i < mpiRank_; i++)
    colRangeLow_ += rankColSize[i];
  colRangeHigh_ = colRangeLow_ + rankColSize[mpiRank_];
}

PetscInt PetscMatrix::GetRowSize() { return localRowSize_ * blockSize_; }

PetscInt PetscMatrix::GetColSize() { return localColSize_ * blockSize_; }

void PetscMatrix::SetColIndex(const PetscInt row,
                              const std::vector<PetscInt> &index) {
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
  PetscInt blockRow = row / blockSize_;
  PetscInt blockRowReminder = row % blockSize_;
  PetscInt blockCol = col / blockSize_;
  PetscInt blockColReminder = col % blockSize_;
  if (row > localRowSize_) {
    std::cout << "Row index larger than local row size" << std::endl;
    return;
  }
  if (blockCol >= colRangeLow_ && blockCol < colRangeHigh_) {
    auto it = lower_bound(diagCol_.begin() + diagRow_[blockRow],
                          diagCol_.begin() + diagRow_[blockRow + 1],
                          blockCol - colRangeLow_);
    if (it != diagCol_.begin() + diagRow_[blockRow + 1] &&
        *it == blockCol - colRangeLow_)
      diagVal_[(it - diagCol_.begin()) * blockStorage_ +
               blockRowReminder * blockSize_ + blockColReminder] += value;
    else
      std::cout << blockRow << ' ' << blockCol
                << " diagonal increment misplacement" << std::endl;
  } else {
    auto it =
        lower_bound(offDiagCol_.begin() + offDiagRow_[blockRow],
                    offDiagCol_.begin() + offDiagRow_[blockRow + 1], blockCol);
    if (it != offDiagCol_.begin() + offDiagRow_[blockRow + 1] &&
        *it == blockCol)
      offDiagVal_[(it - offDiagCol_.begin()) * blockStorage_ +
                  blockRowReminder * blockSize_ + blockColReminder] += value;
    else
      std::cout << blockRow << ' ' << blockCol
                << " off-diagonal increment misplacement" << std::endl;
  }
}

void PetscMatrix::Increment(const PetscInt row,
                            const std::vector<PetscInt> &index,
                            const std::vector<PetscReal> &value) {
  PetscInt blockRow = row / blockSize_;
  PetscInt blockRowReminder = row % blockSize_;
  if (blockRow > localRowSize_) {
    std::cout << "Row index larger than local row size" << std::endl;
    return;
  }
  if (index.size() != value.size()) {
    std::cout << "Wrong increment setup in row: " << blockRow << std::endl;
    return;
  }
  for (std::size_t i = 0; i < index.size(); i++) {
    PetscInt col = index[i];
    PetscInt blockCol = col / blockSize_;
    PetscInt blockColReminder = col % blockSize_;
    if (blockCol >= colRangeLow_ && blockCol < colRangeHigh_) {
      auto it = lower_bound(diagCol_.begin() + diagRow_[blockRow],
                            diagCol_.begin() + diagRow_[blockRow + 1],
                            blockCol - colRangeLow_);
      if (it != diagCol_.begin() + diagRow_[blockRow + 1] &&
          *it == blockCol - colRangeLow_)
        diagVal_[(it - diagCol_.begin()) * blockStorage_ +
                 blockRowReminder * blockSize_ + blockColReminder] += value[i];
      else
        printf("MPI rank: %d, (%d, %d), (%d, %d), diagonal increment "
               "misplacement, %d-%d\n",
               mpiRank_, blockRow, row, blockCol - colRangeLow_,
               col - colRangeLow_ * blockSize_, diagCol_[diagRow_[blockRow]],
               diagCol_[diagRow_[blockRow + 1] - 1]);
    } else {
      auto it = lower_bound(offDiagCol_.begin() + offDiagRow_[blockRow],
                            offDiagCol_.begin() + offDiagRow_[blockRow + 1],
                            blockCol);
      if (it != offDiagCol_.begin() + offDiagRow_[blockRow + 1] &&
          *it == blockCol)
        offDiagVal_[(it - offDiagCol_.begin()) * blockStorage_ +
                    blockRowReminder * blockSize_ + blockColReminder] +=
            value[i];
      else
        printf("MPI rank: %d, (%d, %d), (%d, %d), off-diagonal increment "
               "misplacement, %d-%d\n",
               mpiRank_, blockRow, row, blockCol, col,
               offDiagCol_[offDiagRow_[blockRow]],
               offDiagCol_[offDiagRow_[blockRow + 1] - 1]);
    }
  }
}

unsigned long PetscMatrix::GraphAssemble() {
  unsigned long diagNumNonzero = 0;
  unsigned long offDiagNumNonzero = 0;
  for (int i = 0; i < localRowSize_; i++) {
    diagNumNonzero += diagMatrixCol_[i].size();
    offDiagNumNonzero += offDiagMatrixCol_[i].size() * blockStorage_;
  }

  diagCol_.resize(diagNumNonzero);
  diagVal_.resize(diagNumNonzero * blockStorage_);
  offDiagCol_.resize(offDiagNumNonzero);
  offDiagVal_.resize(offDiagNumNonzero * blockStorage_);

  diagRow_.resize(localRowSize_ + 1);
  offDiagRow_.resize(localRowSize_ + 1);

  diagRow_[0] = 0;
  offDiagRow_[0] = 0;
  for (PetscInt i = 0; i < localRowSize_; i++) {
    diagRow_[i + 1] = diagRow_[i] + diagMatrixCol_[i].size();
    offDiagRow_[i + 1] = offDiagRow_[i] + offDiagMatrixCol_[i].size();
  }

  for (PetscInt i = 0; i < localRowSize_; i++) {
    for (PetscInt j = diagRow_[i], k = 0; j < diagRow_[i + 1]; j++, k++) {
      diagCol_[j] = diagMatrixCol_[i][k];
    }
    for (PetscInt j = offDiagRow_[i], k = 0; j < offDiagRow_[i + 1]; j++, k++) {
      offDiagCol_[j] = offDiagMatrixCol_[i][k];
    }
  }

  decltype(diagMatrixCol_)().swap(diagMatrixCol_);
  decltype(offDiagMatrixCol_)().swap(offDiagMatrixCol_);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, diagVal_.size()),
      KOKKOS_LAMBDA(const unsigned int i) { diagVal_[i] = 0.0; });
  Kokkos::fence();

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, offDiagVal_.size()),
      KOKKOS_LAMBDA(const unsigned int i) { offDiagVal_[i] = 0.0; });
  Kokkos::fence();

  return diagNumNonzero + offDiagNumNonzero;
}

unsigned long GraphAssemble(const PetscInt blockSize) { return 0; }

unsigned long PetscMatrix::Assemble() {
  MatCreateMPIAIJWithSplitArrays(
      PETSC_COMM_WORLD, localRowSize_, localColSize_, globalRowSize_,
      globalColSize_, diagRow_.data(), diagCol_.data(), diagVal_.data(),
      offDiagRow_.data(), offDiagCol_.data(), offDiagVal_.data(), &mat_);

  return diagCol_.size() + offDiagCol_.size();
}