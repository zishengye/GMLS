#include "StokesMatrix.hpp"
#include "petscmat.h"
#include "petscsys.h"
#include "petscvec.h"

#include <algorithm>
#include <mpi.h>

PetscErrorCode FieldMatrixMultWrapper(Mat mat, Vec x, Vec y) {
  StokesMatrix *ctx;
  MatShellGetContext(mat, &ctx);

  return ctx->FieldMatrixMult(x, y);
}

PetscErrorCode FieldMatrixSORWrapper(Mat mat, Vec b, PetscReal omega,
                                     MatSORType flag, PetscReal shift,
                                     PetscInt its, PetscInt lits, Vec x) {
  StokesMatrix *ctx;
  MatShellGetContext(mat, &ctx);

  return ctx->FieldMatrixSOR(b, omega, flag, shift, its, lits, x);
}

void StokesMatrix::InitInternal() {
  MPI_Allreduce(&numLocalParticle_, &numGlobalParticle_, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  numLocalRigidBody_ =
      numRigidBody_ / (unsigned int)mpiSize_ +
      ((numRigidBody_ % (unsigned int)mpiSize_ > mpiRank_) ? 1 : 0);

  rigidBodyStartIndex_ = 0;
  for (int i = 0; i < mpiRank_; i++) {
    rigidBodyStartIndex_ +=
        numRigidBody_ / (unsigned int)mpiSize_ +
        ((numRigidBody_ % (unsigned int)mpiSize_ > i) ? 1 : 0);
  }
  rigidBodyEndIndex_ = rigidBodyStartIndex_ + numLocalRigidBody_;

  translationDof_ = dimension_;
  rotationDof_ = (dimension_ == 3) ? 3 : 1;

  auto a00 = PetscNestedMatrix::GetMatrix(0, 0);
  auto a01 = PetscNestedMatrix::GetMatrix(0, 1);
  auto a10 = PetscNestedMatrix::GetMatrix(1, 0);
  auto a11 = PetscNestedMatrix::GetMatrix(1, 1);

  a00->Resize(numLocalParticle_, numLocalParticle_, fieldDof_);
  a01->Resize(numLocalParticle_ * fieldDof_,
              numLocalRigidBody_ * rigidBodyDof_);
  a10->Resize(numLocalRigidBody_ * rigidBodyDof_,
              numLocalParticle_ * fieldDof_);
  a11->Resize(numLocalRigidBody_ * rigidBodyDof_,
              numLocalRigidBody_ * rigidBodyDof_);

  x_ = std::make_shared<PetscNestedVec>(2);
  b_ = std::make_shared<PetscNestedVec>(2);
}

StokesMatrix::StokesMatrix() : PetscNestedMatrix(2, 2) {}

StokesMatrix::StokesMatrix(const unsigned int dimension)
    : dimension_(dimension), numRigidBody_(0), numLocalParticle_(0),
      fieldDof_(dimension + 1), velocityDof_(dimension),
      PetscNestedMatrix(2, 2) {}

StokesMatrix::StokesMatrix(const unsigned long numLocalParticle,
                           const unsigned int dimension)
    : dimension_(dimension), numRigidBody_(0),
      numLocalParticle_(numLocalParticle), fieldDof_(dimension + 1),
      velocityDof_(dimension), PetscNestedMatrix(2, 2) {
  rigidBodyDof_ = (dimension_ == 3) ? 6 : 3;

  InitInternal();
}

StokesMatrix::StokesMatrix(const unsigned long numLocalParticle,
                           const unsigned int numRigidBody,
                           const unsigned int dimension)
    : dimension_(dimension), numRigidBody_(numRigidBody),
      numLocalParticle_(numLocalParticle), fieldDof_(dimension + 1),
      velocityDof_(dimension), PetscNestedMatrix(2, 2) {
  rigidBodyDof_ = (dimension_ == 3) ? 6 : 3;

  InitInternal();
}

StokesMatrix::~StokesMatrix() {
  MatDestroy(&neighborSchurMat_);
  MatDestroy(&neighborMat_);
  MatDestroy(&neighborWholeMat_);
  MatDestroy(&nestedMat_[0]);

  ISDestroy(&isgNeighbor_);

  for (unsigned int i = 0; i < 3; i++) {
    if (nestedNeighborMat_[i] != PETSC_NULL)
      MatDestroy(&nestedNeighborMat_[i]);
  }
  nestedNeighborMat_[3] = PETSC_NULL;

  for (unsigned int i = 0; i < 3; i++) {
    if (nestedNeighborWholeMat_[i] != PETSC_NULL)
      MatDestroy(&nestedNeighborWholeMat_[i]);
  }
  nestedNeighborWholeMat_[3] = PETSC_NULL;
}

void StokesMatrix::SetSize(const unsigned long numLocalParticle,
                           const unsigned int numRigidBody) {
  rigidBodyDof_ = (dimension_ == 3) ? 6 : 3;
  numLocalParticle_ = numLocalParticle;
  numRigidBody_ = numRigidBody;

  InitInternal();
}

void StokesMatrix::SetGraph(
    const std::vector<int> &localIndex, const std::vector<int> &globalIndex,
    const std::vector<int> &particleType,
    const std::vector<int> &attachedRigidBody,
    const Kokkos::View<int **, Kokkos::HostSpace> &neighborLists) {
  auto a00 = PetscNestedMatrix::GetMatrix(0, 0);
  auto a01 = PetscNestedMatrix::GetMatrix(0, 1);
  auto a10 = PetscNestedMatrix::GetMatrix(1, 0);
  auto a11 = PetscNestedMatrix::GetMatrix(1, 1);

  for (unsigned int i = 0; i < numLocalParticle_; i++) {
    const int currentParticleLocalIndex = localIndex[i];

    std::vector<PetscInt> index;
    index.resize(neighborLists(i, 0));
    for (unsigned int j = 0; j < neighborLists(i, 0); j++) {
      index[j] = globalIndex[neighborLists(i, j + 1)];
    }

    a00->SetColIndex(currentParticleLocalIndex, index);

    if (particleType[i] >= 4) {
      index.clear();
      index.resize(1 + rotationDof_);

      for (unsigned int axes = 0; axes < rotationDof_; axes++) {
        index[1 + axes] =
            attachedRigidBody[i] * rigidBodyDof_ + translationDof_ + axes;
      }

      for (unsigned int axes = 0; axes < velocityDof_; axes++) {
        index[0] = attachedRigidBody[i] * rigidBodyDof_ + axes;
        a01->SetColIndex(currentParticleLocalIndex * fieldDof_ + axes, index);
      }
    }
  }

  rigidBodyFieldIndexMap_.resize(numRigidBody_);
  for (unsigned int i = 0; i < numLocalParticle_; i++) {
    if (particleType[i] >= 4) {
      for (unsigned int j = 0; j < neighborLists(i, 0); j++) {
        const unsigned int neighborParticleIndex =
            globalIndex[neighborLists(i, j + 1)];
        for (unsigned int axes = 0; axes < fieldDof_; axes++) {
          rigidBodyFieldIndexMap_[attachedRigidBody[i]].push_back(
              neighborParticleIndex * fieldDof_ + axes);
        }
      }
    }
  }

  for (unsigned int i = 0; i < numRigidBody_; i++) {
    std::sort(rigidBodyFieldIndexMap_[i].begin(),
              rigidBodyFieldIndexMap_[i].end());
    rigidBodyFieldIndexMap_[i].erase(
        std::unique(rigidBodyFieldIndexMap_[i].begin(),
                    rigidBodyFieldIndexMap_[i].end()),
        rigidBodyFieldIndexMap_[i].end());
    MPI_Barrier(MPI_COMM_WORLD);

    int targetRank = 0;
    if (i >= rigidBodyStartIndex_ && i < rigidBodyEndIndex_)
      targetRank = mpiRank_;
    MPI_Allreduce(MPI_IN_PLACE, &targetRank, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    int sendCount = rigidBodyFieldIndexMap_[i].size();
    std::vector<int> index;
    std::vector<int> recvCount(mpiSize_);
    std::vector<int> recvOffset(mpiSize_ + 1);
    MPI_Gather(&sendCount, 1, MPI_INT, recvCount.data(), 1, MPI_INT, targetRank,
               MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (targetRank == mpiRank_) {
      recvOffset[0] = 0;
      for (unsigned int j = 0; j < mpiSize_; j++) {
        recvOffset[j + 1] = recvCount[j] + recvOffset[j];
      }
      index.resize(recvOffset[mpiSize_]);
    }
    MPI_Gatherv(rigidBodyFieldIndexMap_[i].data(), sendCount, MPI_INT,
                index.data(), recvCount.data(), recvOffset.data(), MPI_INT,
                targetRank, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (targetRank == mpiRank_) {
      std::sort(index.begin(), index.end());
      index.erase(std::unique(index.begin(), index.end()), index.end());
      for (unsigned int j = 0; j < rigidBodyDof_; j++) {
        a10->SetColIndex((i - rigidBodyStartIndex_) * rigidBodyDof_ + j, index);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  rigidBodyFieldValueMap_.resize(rigidBodyDof_ * numRigidBody_);
  for (unsigned int i = 0; i < numRigidBody_; i++) {
    for (unsigned int j = 0; j < rigidBodyDof_; j++) {
      rigidBodyFieldValueMap_[i * rigidBodyDof_ + j].resize(
          rigidBodyFieldIndexMap_[i].size());
      for (auto &v : rigidBodyFieldValueMap_[i * rigidBodyDof_ + j])
        v = 0.0;
    }
  }

  PetscNestedMatrix::GraphAssemble();

  std::vector<unsigned int> flattenedRigidBodyFieldIndexMap_;
  for (unsigned int i = 0; i < numRigidBody_; i++) {
    for (unsigned int j = 0; j < rigidBodyFieldIndexMap_[i].size(); j++) {
      flattenedRigidBodyFieldIndexMap_.push_back(rigidBodyFieldIndexMap_[i][j] /
                                                 fieldDof_);
    }
  }
  std::sort(flattenedRigidBodyFieldIndexMap_.begin(),
            flattenedRigidBodyFieldIndexMap_.end());
  flattenedRigidBodyFieldIndexMap_.erase(
      std::unique(flattenedRigidBodyFieldIndexMap_.begin(),
                  flattenedRigidBodyFieldIndexMap_.end()),
      flattenedRigidBodyFieldIndexMap_.end());
  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<int> gatheredField;
  std::vector<int> localField;
  std::vector<PetscInt> idxNeighbor;
  unsigned int gatheredParticleNum;
  unsigned int startIndex = 0;
  unsigned int endIndex = 0;

  for (int i = 0; i < mpiSize_; i++) {
    if (mpiRank_ == i) {
      gatheredParticleNum = numLocalParticle_;
      localField.resize(numLocalParticle_);
    }
    MPI_Bcast(&gatheredParticleNum, 1, MPI_UNSIGNED, i, MPI_COMM_WORLD);
    gatheredField.resize(gatheredParticleNum);
    for (unsigned int j = 0; j < gatheredParticleNum; j++) {
      gatheredField[j] = 0;
    }

    startIndex = endIndex;
    endIndex = startIndex + gatheredParticleNum;

    auto start = std::lower_bound(flattenedRigidBodyFieldIndexMap_.begin(),
                                  flattenedRigidBodyFieldIndexMap_.end(),
                                  (PetscInt)startIndex);
    auto end = std::lower_bound(start, flattenedRigidBodyFieldIndexMap_.end(),
                                (PetscInt)endIndex);
    auto it = start;
    while (it != end) {
      gatheredField[*it - startIndex] = 1;

      it++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(gatheredField.data(), localField.data(), gatheredParticleNum,
               MPI_INT, MPI_SUM, i, MPI_COMM_WORLD);

    if (i == mpiRank_) {
      for (unsigned int j = 0; j < localField.size(); j++)
        if (localField[j] != 0) {
          for (unsigned int k = 0; k < fieldDof_; k++)
            idxNeighbor.push_back((j + startIndex) * fieldDof_ + k);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  std::sort(idxNeighbor.begin(), idxNeighbor.end());
  idxNeighbor.erase(std::unique(idxNeighbor.begin(), idxNeighbor.end()),
                    idxNeighbor.end());

  ISCreateGeneral(MPI_COMM_WORLD, idxNeighbor.size(), idxNeighbor.data(),
                  PETSC_COPY_VALUES, &isgNeighbor_);

  x_->Create(0, idxNeighbor.size());
  b_->Create(0, idxNeighbor.size());
}

unsigned long StokesMatrix::Assemble() {
  // move data
  auto a10 = PetscNestedMatrix::GetMatrix(1, 0);

  for (unsigned int i = 0; i < numRigidBody_; i++) {
    int targetRank = 0;
    if (i >= rigidBodyStartIndex_ && i < rigidBodyEndIndex_)
      targetRank = mpiRank_;
    MPI_Allreduce(MPI_IN_PLACE, &targetRank, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    int sendCount = rigidBodyFieldIndexMap_[i].size();
    std::vector<int> index;
    std::vector<double> value;
    std::vector<int> recvCount(mpiSize_);
    std::vector<int> recvOffset(mpiSize_ + 1);
    MPI_Gather(&sendCount, 1, MPI_INT, recvCount.data(), 1, MPI_INT, targetRank,
               MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (targetRank == mpiRank_) {
      recvOffset[0] = 0;
      for (unsigned int j = 0; j < mpiSize_; j++) {
        recvOffset[j + 1] = recvCount[j] + recvOffset[j];
      }
      index.resize(recvOffset[mpiSize_]);
      value.resize(recvOffset[mpiSize_]);
    }
    MPI_Gatherv(rigidBodyFieldIndexMap_[i].data(), sendCount, MPI_INT,
                index.data(), recvCount.data(), recvOffset.data(), MPI_INT,
                targetRank, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    for (unsigned int j = 0; j < rigidBodyDof_; j++) {
      MPI_Gatherv(rigidBodyFieldValueMap_[i * rigidBodyDof_ + j].data(),
                  sendCount, MPI_DOUBLE, value.data(), recvCount.data(),
                  recvOffset.data(), MPI_DOUBLE, targetRank, MPI_COMM_WORLD);

      if (targetRank == mpiRank_) {
        a10->Increment((i - rigidBodyStartIndex_) * rigidBodyDof_ + j, index,
                       value);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // replace the matrix with a shell matrix
  for (auto &it : nestedWrappedMat_)
    it->Assemble();

  nestedMat_[1] = nestedWrappedMat_[1]->GetReference();
  nestedMat_[2] = nestedWrappedMat_[2]->GetReference();
  nestedMat_[3] = nestedWrappedMat_[3]->GetReference();

  PetscInt rowSize, colSize;
  MatGetLocalSize(nestedWrappedMat_[0]->GetReference(), &rowSize, &colSize);
  MatCreateShell(MPI_COMM_WORLD, rowSize, colSize, PETSC_DECIDE, PETSC_DECIDE,
                 this, &nestedMat_[0]);

  MatShellSetOperation(nestedMat_[0], MATOP_MULT,
                       (void (*)(void))FieldMatrixMultWrapper);
  MatShellSetOperation(nestedMat_[0], MATOP_SOR,
                       (void (*)(void))FieldMatrixSORWrapper);

  MatCreateNest(MPI_COMM_WORLD, 2, PETSC_NULL, 2, PETSC_NULL, nestedMat_.data(),
                &mat_);

  nestedNeighborMat_.resize(4);
  nestedNeighborWholeMat_.resize(4);

  std::vector<PetscInt> idxColloid;
  PetscInt lowRange, highRange;
  MatGetOwnershipRange(nestedMat_[3], &lowRange, &highRange);
  idxColloid.resize(highRange - lowRange);
  for (unsigned int i = 0; i < idxColloid.size(); i++)
    idxColloid[i] = lowRange + i;
  IS isgColloid;
  ISCreateGeneral(PETSC_COMM_WORLD, idxColloid.size(), idxColloid.data(),
                  PETSC_COPY_VALUES, &isgColloid);

  x_->Create(1, idxColloid.size());
  b_->Create(1, idxColloid.size());

  x_->Create();
  b_->Create();

  MatCreateSubMatrix(nestedWrappedMat_[0]->GetReference(), isgNeighbor_,
                     isgNeighbor_, MAT_INITIAL_MATRIX, &nestedNeighborMat_[0]);
  MatCreateSubMatrix(nestedMat_[1], isgNeighbor_, PETSC_NULL,
                     MAT_INITIAL_MATRIX, &nestedNeighborMat_[1]);
  MatCreateSubMatrix(nestedMat_[2], isgColloid, isgNeighbor_,
                     MAT_INITIAL_MATRIX, &nestedNeighborMat_[2]);
  nestedNeighborMat_[3] = nestedMat_[3];
  MatCreateNest(MPI_COMM_WORLD, 2, PETSC_NULL, 2, PETSC_NULL,
                nestedNeighborMat_.data(), &neighborMat_);

  MatCreateSubMatrix(nestedWrappedMat_[0]->GetReference(), isgNeighbor_,
                     PETSC_NULL, MAT_INITIAL_MATRIX,
                     &nestedNeighborWholeMat_[0]);
  MatCreateSubMatrix(nestedMat_[1], isgNeighbor_, PETSC_NULL,
                     MAT_INITIAL_MATRIX, &nestedNeighborWholeMat_[1]);
  MatCreateSubMatrix(nestedMat_[2], isgColloid, PETSC_NULL, MAT_INITIAL_MATRIX,
                     &nestedNeighborWholeMat_[2]);
  nestedNeighborWholeMat_[3] = nestedMat_[3];
  MatCreateNest(MPI_COMM_WORLD, 2, PETSC_NULL, 2, PETSC_NULL,
                nestedNeighborWholeMat_.data(), &neighborWholeMat_);

  ISDestroy(&isgColloid);

  Mat B, C;
  MatCreate(MPI_COMM_WORLD, &B);
  MatSetType(B, MATMPIAIJ);
  MatInvertBlockDiagonalMat(nestedNeighborMat_[0], B);

  MatMatMult(B, nestedNeighborMat_[1], MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);
  MatMatMult(nestedNeighborMat_[2], C, MAT_INITIAL_MATRIX, PETSC_DEFAULT,
             &neighborSchurMat_);
  MatScale(neighborSchurMat_, -1.0);
  MatAXPY(neighborSchurMat_, 1.0, nestedNeighborMat_[3],
          DIFFERENT_NONZERO_PATTERN);

  MatDestroy(&B);
  MatDestroy(&C);

  return 0;
}

void StokesMatrix::IncrementFieldField(const PetscInt row,
                                       const std::vector<PetscInt> &index,
                                       const std::vector<PetscReal> &value) {
  PetscNestedMatrix::GetMatrix(0, 0)->Increment(row, index, value);
}

void StokesMatrix::IncrementFieldRigidBody(const PetscInt row,
                                           const PetscInt index,
                                           const PetscReal value) {
  PetscNestedMatrix::GetMatrix(0, 1)->Increment(row, index, value);
}

void StokesMatrix::IncrementRigidBodyField(const PetscInt row,
                                           const PetscInt index,
                                           const PetscReal value) {
  unsigned int rigidBodyIndex = row / rigidBodyDof_;
  auto result =
      std::lower_bound(rigidBodyFieldIndexMap_[rigidBodyIndex].begin(),
                       rigidBodyFieldIndexMap_[rigidBodyIndex].end(), index);
  std::size_t mapIndex =
      result - rigidBodyFieldIndexMap_[rigidBodyIndex].begin();
  if (mapIndex < rigidBodyFieldIndexMap_[rigidBodyIndex].size())
    rigidBodyFieldValueMap_[row][mapIndex] += value;
  else
    std::cout << "rigid body-field wrong increment" << std::endl;
}

void StokesMatrix::IncrementRigidBodyRigidBody(const PetscInt row,
                                               const PetscInt index,
                                               const PetscReal value) {
  PetscNestedMatrix::GetMatrix(1, 1)->Increment(row, index, value);
}

PetscErrorCode StokesMatrix::FieldMatrixMult(Vec x, Vec y) {
  PetscReal *a;

  PetscReal pressureSum = 0.0;
  PetscReal averagePressure;

  VecGetArray(x, &a);

  pressureSum = 0.0;
  for (int i = 0; i < numLocalParticle_; i++) {
    pressureSum += a[i * fieldDof_ + velocityDof_];
  }
  MPI_Allreduce(MPI_IN_PLACE, &pressureSum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  averagePressure = pressureSum / (double)numGlobalParticle_;
  for (int i = 0; i < numLocalParticle_; i++) {
    a[i * fieldDof_ + velocityDof_] -= averagePressure;
  }

  VecRestoreArray(x, &a);

  MatMult(nestedWrappedMat_[0]->GetReference(), x, y);

  VecGetArray(y, &a);

  pressureSum = 0.0;
  for (int i = 0; i < numLocalParticle_; i++) {
    pressureSum += a[i * fieldDof_ + velocityDof_];
  }
  MPI_Allreduce(MPI_IN_PLACE, &pressureSum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  averagePressure = pressureSum / (double)numGlobalParticle_;
  for (int i = 0; i < numLocalParticle_; i++) {
    a[i * fieldDof_ + velocityDof_] -= averagePressure;
  }

  VecRestoreArray(y, &a);

  return 0;
}

PetscErrorCode StokesMatrix::FieldMatrixSOR(Vec b, PetscReal omega,
                                            MatSORType flag, PetscReal shift,
                                            PetscInt its, PetscInt lits,
                                            Vec x) {
  PetscReal *a;

  PetscReal pressureSum = 0.0;
  PetscReal averagePressure;

  VecGetArray(b, &a);

  pressureSum = 0.0;
  for (int i = 0; i < numLocalParticle_; i++) {
    pressureSum += a[i * fieldDof_ + velocityDof_];
  }
  MPI_Allreduce(MPI_IN_PLACE, &pressureSum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  averagePressure = pressureSum / (double)numGlobalParticle_;
  for (int i = 0; i < numLocalParticle_; i++) {
    a[i * fieldDof_ + velocityDof_] -= averagePressure;
  }

  VecRestoreArray(b, &a);

  MatSOR(nestedWrappedMat_[0]->GetReference(), b, omega, flag, shift, its, lits,
         x);

  VecGetArray(x, &a);

  pressureSum = 0.0;
  for (int i = 0; i < numLocalParticle_; i++) {
    pressureSum += a[i * fieldDof_ + velocityDof_];
  }
  MPI_Allreduce(MPI_IN_PLACE, &pressureSum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  averagePressure = pressureSum / (double)numGlobalParticle_;
  for (int i = 0; i < numLocalParticle_; i++) {
    a[i * fieldDof_ + velocityDof_] -= averagePressure;
  }

  VecRestoreArray(x, &a);

  return 0;
}