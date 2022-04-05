#include "PetscMatrix.hpp"

PetscMatrix::PetscMatrix() : mat_(PETSC_NULL) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);
}

PetscMatrix::PetscMatrix(const PetscInt m, const PetscInt n)
    : localRowSize_(m), localColSize_(n), mat_(PETSC_NULL) {
  diagMatrixCol_.resize(localRowSize_);
  offDiagMatrixCol_.resize(localRowSize_);

  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);
}

PetscMatrix::~PetscMatrix() {
  if (mat_ != PETSC_NULL) {
    MatDestroy(&mat_);
  }
}

void PetscMatrix::Resize(const PetscInt m, const PetscInt n) {
  localRowSize_ = m;
  localColSize_ = n;
  if (mat_ != PETSC_NULL)
    MatDestroy(&mat_);
  mat_ = PETSC_NULL;

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
  colRangeLow = 0;
  for (int i = 0; i < mpiRank_; i++)
    colRangeLow += rankColSize[i];
  colRangeHigh = colRangeLow + rankColSize[mpiRank_];
}

const int PetscMatrix::GetRowSize() { return diagMatrixCol_.size(); }

void PetscMatrix::SetColIndex(const PetscInt row,
                              const std::vector<PetscInt> &index) {
  std::vector<PetscInt> &diagIndex = diagMatrixCol_[row];
  std::vector<PetscInt> &offDiagIndex = offDiagMatrixCol_[row];
  diagIndex.clear();
  offDiagIndex.clear();
  for (auto it = index.begin(); it != index.end(); it++) {
    if (*it >= colRangeLow && *it < colRangeHigh)
      diagIndex.push_back(*it - colRangeLow);
    else
      offDiagIndex.push_back(*it);
  }
}

void PetscMatrix::Increment(const PetscInt row, const PetscInt col,
                            const PetscReal value) {
  if (row > localRowSize_) {
    std::cout << "Row index larger than local row size" << std::endl;
    return;
  }
  if (col >= colRangeLow && col < colRangeHigh) {
    auto it =
        lower_bound(diagCol_.begin() + diagRow_[row],
                    diagCol_.begin() + diagRow_[row + 1], col - colRangeLow);
    if (it != diagCol_.begin() + diagRow_[row + 1] && *it == col - colRangeLow)
      diagVal_[it - diagCol_.begin()] += value;
    else
      std::cout << row << ' ' << col << " diagonal increment misplacement"
                << std::endl;
  } else {
    auto it = lower_bound(offDiagCol_.begin() + offDiagRow_[row],
                          offDiagCol_.begin() + offDiagRow_[row + 1], col);
    if (it != offDiagCol_.begin() + offDiagRow_[row + 1] && *it == col)
      offDiagVal_[it - offDiagCol_.begin()] += value;
    else
      std::cout << row << ' ' << col << " off-diagonal increment misplacement"
                << std::endl;
  }
}

void PetscMatrix::Increment(const PetscInt row,
                            const std::vector<PetscInt> &index,
                            const std::vector<PetscReal> &value) {
  if (row > localRowSize_) {
    std::cout << "Row index larger than local row size" << std::endl;
    return;
  }
  if (index.size() != value.size()) {
    std::cout << "Wrong increment setup in row: " << row << std::endl;
    return;
  }
  for (int i = 0; i < index.size(); i++) {
    PetscInt col = index[i];
    if (col >= colRangeLow && col < colRangeHigh) {
      auto it =
          lower_bound(diagCol_.begin() + diagRow_[row],
                      diagCol_.begin() + diagRow_[row + 1], col - colRangeLow);
      if (it != diagCol_.begin() + diagRow_[row + 1] &&
          *it == col - colRangeLow)
        diagVal_[it - diagCol_.begin()] += value[i];
      else
        std::cout << row << ' ' << col << " diagonal increment misplacement"
                  << std::endl;
    } else {
      auto it = lower_bound(offDiagCol_.begin() + offDiagRow_[row],
                            offDiagCol_.begin() + offDiagRow_[row + 1], col);
      if (it != offDiagCol_.begin() + offDiagRow_[row + 1] && *it == col)
        offDiagVal_[it - offDiagCol_.begin()] += value[i];
      else
        std::cout << row << ' ' << col
                  << " off-diagonal increment misplacement " << std::endl;
    }
  }
}

const unsigned long PetscMatrix::GraphAssemble() {
  unsigned long diagNumNonzero = 0;
  unsigned long offDiagNumNonzero = 0;
  for (int i = 0; i < localRowSize_; i++) {
    diagNumNonzero += diagMatrixCol_[i].size();
    offDiagNumNonzero += offDiagMatrixCol_[i].size();
  }

  diagCol_.resize(diagNumNonzero);
  diagVal_.resize(diagNumNonzero);
  offDiagCol_.resize(offDiagNumNonzero);
  offDiagVal_.resize(offDiagNumNonzero);

  diagRow_.resize(localRowSize_ + 1);
  offDiagRow_.resize(localRowSize_ + 1);

  diagRow_[0] = 0;
  offDiagRow_[0] = 0;
  for (int i = 0; i < localRowSize_; i++) {
    diagRow_[i + 1] = diagRow_[i] + diagMatrixCol_[i].size();
    offDiagRow_[i + 1] = offDiagRow_[i] + offDiagMatrixCol_[i].size();
  }

  for (int i = 0; i < localRowSize_; i++) {
    for (int j = diagRow_[i], k = 0; j < diagRow_[i + 1]; j++, k++) {
      diagCol_[j] = diagMatrixCol_[i][k];
    }
    for (int j = offDiagRow_[i], k = 0; j < offDiagRow_[i + 1]; j++, k++) {
      offDiagCol_[j] = offDiagMatrixCol_[i][k];
    }
  }

  decltype(diagMatrixCol_)().swap(diagMatrixCol_);
  decltype(offDiagMatrixCol_)().swap(offDiagMatrixCol_);

  return diagNumNonzero + offDiagNumNonzero;
}

const unsigned long PetscMatrix::Assemble() {
  MatCreateMPIAIJWithSplitArrays(
      PETSC_COMM_WORLD, localRowSize_, localColSize_, globalRowSize_,
      globalColSize_, diagRow_.data(), diagCol_.data(), diagVal_.data(),
      offDiagRow_.data(), offDiagCol_.data(), offDiagVal_.data(), &mat_);

  return diagCol_.size() + offDiagCol_.size();
}

Mat &PetscMatrix::GetReference() { return mat_; }