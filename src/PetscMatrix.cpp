#include "PetscMatrix.hpp"

PetscMatrix::PetscMatrix() : mat_(PETSC_NULL) {}

PetscMatrix::PetscMatrix(const PetscInt m, const PetscInt n)
    : localRowSize_(m), localColSize_(n), mat_(PETSC_NULL) {
  matrix_.resize(localRowSize_);
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

  matrix_.clear();
  matrix_.resize(localRowSize_);
}

const int PetscMatrix::GetRowSize() { return matrix_.size(); }

void PetscMatrix::SetColIndex(const PetscInt row,
                              const std::vector<PetscInt> &index) {
  matrix_[row].resize(index.size());
  std::size_t counter = 0;
  for (std::vector<Entry>::iterator it = matrix_[row].begin();
       it != matrix_[row].end(); it++) {
    it->first = index[counter++];
    it->second = 0.0;
  }
}

void PetscMatrix::Increment(const PetscInt row, const PetscInt col,
                            const PetscReal value) {
  if (std::abs(value) > 1e-15) {
    auto it = lower_bound(matrix_[row].begin(), matrix_[row].end(),
                          Entry(col, value), CompareIndex);
    if (it->first == col)
      it->second += value;
    else
      std::cout << row << ' ' << col << " increment misplacement" << std::endl;
  }
}

void PetscMatrix::Increment(const PetscInt row,
                            const std::vector<PetscInt> &index,
                            const std::vector<PetscReal> &value) {
  if (row > matrix_.size()) {
    std::cout << "Row index larger than local row size" << std::endl;
    return;
  }
  if (index.size() != value.size()) {
    std::cout << "Wrong increment setup in row: " << row << std::endl;
    return;
  }
  for (int i = 0; i < index.size(); i++) {
    auto it = lower_bound(matrix_[row].begin(), matrix_[row].end(),
                          Entry(index[i], value[i]), CompareIndex);
    if (it == matrix_[row].end())
      std::cout << row << ' ' << index[i] << ' ' << matrix_[row].size()
                << " out of column range" << std::endl;
    else if (it->first == index[i])
      it->second += value[i];
    else
      std::cout << row << ' ' << index[i] << " increment misplacement"
                << std::endl;
  }
}

const unsigned long PetscMatrix::Assemble() {
  const unsigned long nnz = 0;

  row_.resize(localRowSize_ + 1);
  for (int i = 0; i < localRowSize_; i++) {
    row_[i] = 0;
    nnz += matrix_[i].size();
  }

  col_.resize(nnz);
  val_.resize(nnz);

  for (int i = 1; i <= localRowSize_; i++) {
    if (row_[i - 1] == 0) {
      row_[i] = 0;
      for (int j = i - 1; j >= 0; j--) {
        if (row_[j] == 0) {
          row_[i] += matrix_[j].size();
        } else {
          row_[i] += row_[j] + matrix_[j].size();
          break;
        }
      }
    } else {
      row_[i] = row_[i - 1] + matrix_[i - 1].size();
    }
  }

  for (int i = 0; i < localRowSize_; i++) {
    for (auto n = 0; n < matrix_[i].size(); n++) {
      col_[row_[i] + n] = matrix_[i][n].first;
      val_[row_[i] + n] = matrix_[i][n].second;
    }
  }

  MatCreateMPIAIJWithSplitArrays(PETSC_COMM_WORLD, localRowSize_, localColSize_,
                                 PETSC_DECIDE, PETSC_DECIDE, row_.data(),
                                 col_.data(), val_.data(), &mat_);

  return nnz;
}