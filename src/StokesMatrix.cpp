#include "StokesMatrix.hpp"
#include "PetscVector.hpp"
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

  auto uu = PetscNestedMatrix::GetMatrix(0, 0);
  auto up = PetscNestedMatrix::GetMatrix(0, 1);
  auto uc = PetscNestedMatrix::GetMatrix(0, 2);
  auto pu = PetscNestedMatrix::GetMatrix(1, 0);
  auto pp = PetscNestedMatrix::GetMatrix(1, 1);
  auto pc = PetscNestedMatrix::GetMatrix(1, 2);
  auto cu = PetscNestedMatrix::GetMatrix(2, 0);
  auto cp = PetscNestedMatrix::GetMatrix(2, 1);
  auto cc = PetscNestedMatrix::GetMatrix(2, 2);

  uu->Resize(numLocalParticle_, numLocalParticle_, velocityDof_);
  up->Resize(numLocalParticle_ * velocityDof_, numLocalRigidBody_);
  uc->Resize(numLocalParticle_ * velocityDof_,
             numLocalRigidBody_ * rigidBodyDof_);
  pu->Resize(numLocalParticle_, numLocalParticle_ * velocityDof_);
  pp->Resize(numLocalParticle_, numLocalParticle_);
  pc->Resize(numLocalParticle_, numLocalRigidBody_ * rigidBodyDof_);
  cu->Resize(numLocalRigidBody_ * rigidBodyDof_,
             numLocalParticle_ * velocityDof_);
  cp->Resize(numLocalRigidBody_ * rigidBodyDof_, numLocalParticle_);
  cc->Resize(numLocalRigidBody_ * rigidBodyDof_,
             numLocalRigidBody_ * rigidBodyDof_);

  x_ = std::make_shared<PetscVector>();
  b_ = std::make_shared<PetscVector>();
}

StokesMatrix::StokesMatrix() : PetscNestedMatrix(3, 3) {}

StokesMatrix::StokesMatrix(const unsigned int dimension)
    : dimension_(dimension), numRigidBody_(0), numLocalParticle_(0),
      fieldDof_(dimension + 1), velocityDof_(dimension),
      PetscNestedMatrix(3, 3) {}

StokesMatrix::StokesMatrix(const unsigned long numLocalParticle,
                           const unsigned int dimension)
    : dimension_(dimension), numRigidBody_(0),
      numLocalParticle_(numLocalParticle), fieldDof_(dimension + 1),
      velocityDof_(dimension), PetscNestedMatrix(3, 3) {
  rigidBodyDof_ = (dimension_ == 3) ? 6 : 3;

  InitInternal();
}

StokesMatrix::StokesMatrix(const unsigned long numLocalParticle,
                           const unsigned int numRigidBody,
                           const unsigned int dimension)
    : dimension_(dimension), numRigidBody_(numRigidBody),
      numLocalParticle_(numLocalParticle), fieldDof_(dimension + 1),
      velocityDof_(dimension), PetscNestedMatrix(3, 3) {
  rigidBodyDof_ = (dimension_ == 3) ? 6 : 3;

  InitInternal();
}

StokesMatrix::~StokesMatrix() {
  MatDestroy(&neighborSchurMat_);
  MatDestroy(&neighborMat_);
  MatDestroy(&nestedMat_[0]);
  MatDestroy(&schurMat_);

  ISDestroy(&isgNeighbor_);
  ISDestroy(&isgColloid_);
  ISDestroy(&isgField_);

  for (unsigned int i = 0; i < 3; i++) {
    if (nestedNeighborMat_[i] != PETSC_NULL)
      MatDestroy(&nestedNeighborMat_[i]);
  }
  nestedNeighborMat_[3] = PETSC_NULL;
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

    idxNeighborPressure_.clear();
    idxNonNeighborPressure_.clear();

    if (i == mpiRank_) {
      for (unsigned int j = 0; j < localField.size(); j++) {
        if (localField[j] != 0) {
          for (unsigned int k = 0; k < fieldDof_; k++) {
            idxNeighbor.push_back((j + startIndex) * fieldDof_ + k);
            idxNeighbor_.push_back(j * fieldDof_ + k);
          }

          idxNeighborPressure_.push_back(j * fieldDof_ + velocityDof_);
        } else {
          idxNonNeighborPressure_.push_back(j * fieldDof_ + velocityDof_);
        }
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  std::sort(idxNeighbor.begin(), idxNeighbor.end());
  idxNeighbor.erase(std::unique(idxNeighbor.begin(), idxNeighbor.end()),
                    idxNeighbor.end());

  std::sort(idxNeighbor_.begin(), idxNeighbor_.end());
  idxNeighbor_.erase(std::unique(idxNeighbor_.begin(), idxNeighbor_.end()),
                     idxNeighbor_.end());

  std::sort(idxNeighborPressure_.begin(), idxNeighborPressure_.end());
  idxNeighborPressure_.erase(
      std::unique(idxNeighborPressure_.begin(), idxNeighborPressure_.end()),
      idxNeighborPressure_.end());

  std::sort(idxNonNeighborPressure_.begin(), idxNonNeighborPressure_.end());
  idxNonNeighborPressure_.erase(std::unique(idxNonNeighborPressure_.begin(),
                                            idxNonNeighborPressure_.end()),
                                idxNonNeighborPressure_.end());

  ISCreateGeneral(MPI_COMM_WORLD, idxNeighbor.size(), idxNeighbor.data(),
                  PETSC_COPY_VALUES, &isgNeighbor_);

  x_->Create(idxNeighbor_.size() + numLocalRigidBody_ * rigidBodyDof_);
  b_->Create(idxNeighbor_.size() + numLocalRigidBody_ * rigidBodyDof_);
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

  PetscInt rowSize0, rowSize1;
  MatGetLocalSize(nestedWrappedMat_[0]->GetReference(), &rowSize0, PETSC_NULL);
  MatGetLocalSize(nestedWrappedMat_[3]->GetReference(), &rowSize1, PETSC_NULL);

  MatCreateShell(MPI_COMM_WORLD, rowSize0 + rowSize1, rowSize0 + rowSize1,
                 PETSC_DECIDE, PETSC_DECIDE, this, &mat_);
  MatShellSetOperation(mat_, MATOP_MULT, (void (*)(void))NestedMatMultWrapper);

  nestedNeighborMat_.resize(4);

  std::vector<PetscInt> idxColloid;
  PetscInt lowRange, highRange;
  MatGetOwnershipRange(nestedMat_[3], &lowRange, &highRange);
  idxColloid.resize(highRange - lowRange);
  for (unsigned int i = 0; i < idxColloid.size(); i++)
    idxColloid[i] = lowRange + i;
  ISCreateGeneral(PETSC_COMM_WORLD, idxColloid.size(), idxColloid.data(),
                  PETSC_COPY_VALUES, &isgColloid_);

  std::vector<PetscInt> idxField;
  idxField.resize(numLocalParticle_ * fieldDof_);
  for (unsigned int i = 0; i < idxField.size(); i++)
    idxField[i] = lowRange + i;
  ISCreateGeneral(PETSC_COMM_WORLD, idxField.size(), idxField.data(),
                  PETSC_COPY_VALUES, &isgField_);

  MatCreateSubMatrix(nestedWrappedMat_[0]->GetReference(), isgNeighbor_,
                     isgNeighbor_, MAT_INITIAL_MATRIX, &nestedNeighborMat_[0]);
  MatCreateSubMatrix(nestedMat_[1], isgNeighbor_, PETSC_NULL,
                     MAT_INITIAL_MATRIX, &nestedNeighborMat_[1]);
  MatCreateSubMatrix(nestedMat_[2], isgColloid_, isgNeighbor_,
                     MAT_INITIAL_MATRIX, &nestedNeighborMat_[2]);
  nestedNeighborMat_[3] = nestedMat_[3];
  MatCreateNest(MPI_COMM_WORLD, 2, PETSC_NULL, 2, PETSC_NULL,
                nestedNeighborMat_.data(), &neighborMat_);

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

  MatCreate(MPI_COMM_WORLD, &B);
  MatSetType(B, MATMPIAIJ);
  MatInvertBlockDiagonalMat(nestedWrappedMat_[0]->GetReference(), B);

  MatMatMult(B, nestedWrappedMat_[1]->GetReference(), MAT_INITIAL_MATRIX,
             PETSC_DEFAULT, &C);
  MatMatMult(nestedWrappedMat_[2]->GetReference(), C, MAT_INITIAL_MATRIX,
             PETSC_DEFAULT, &schurMat_);
  MatScale(schurMat_, -1.0);
  MatAXPY(schurMat_, 1.0, nestedWrappedMat_[3]->GetReference(),
          DIFFERENT_NONZERO_PATTERN);

  MatDestroy(&B);
  MatDestroy(&C);

  return 0;
}

void StokesMatrix::IncrementVelocityVelocity(
    const PetscInt row, const std::vector<PetscInt> &index,
    const std::vector<PetscReal> &value) {
  PetscNestedMatrix::GetMatrix(0, 0)->Increment(row, index, value);
}

void StokesMatrix::IncrementVelocityPressure(
    const PetscInt row, const std::vector<PetscInt> &index,
    const std::vector<PetscReal> &value) {
  PetscNestedMatrix::GetMatrix(0, 1)->Increment(row, index, value);
}

void StokesMatrix::IncrementVelocityRigidBody(const PetscInt row,
                                              const PetscInt index,
                                              const PetscReal value) {
  PetscNestedMatrix::GetMatrix(0, 2)->Increment(row, index, value);
}

void StokesMatrix::IncrementPressureVelocity(
    const PetscInt row, const std::vector<PetscInt> &index,
    const std::vector<PetscReal> &value) {
  PetscNestedMatrix::GetMatrix(1, 0)->Increment(row, index, value);
}

void StokesMatrix::IncrementPressurePressure(
    const PetscInt row, const std::vector<PetscInt> &index,
    const std::vector<PetscReal> &value) {
  PetscNestedMatrix::GetMatrix(1, 1)->Increment(row, index, value);
}

void StokesMatrix::IncrementPressureRigidBody(const PetscInt row,
                                              const PetscInt index,
                                              const PetscReal value) {
  PetscNestedMatrix::GetMatrix(1, 2)->Increment(row, index, value);
}

void StokesMatrix::IncrementRigidBodyVelocity(const PetscInt row,
                                              const PetscInt index,
                                              const PetscReal value) {
  PetscNestedMatrix::GetMatrix(2, 0)->Increment(row, index, value);
}

void StokesMatrix::IncrementRigidBodyPressure(const PetscInt row,
                                              const PetscInt index,
                                              const PetscReal value) {
  PetscNestedMatrix::GetMatrix(2, 0)->Increment(row, index, value);
}

void StokesMatrix::IncrementRigidBodyRigidBody(const PetscInt row,
                                               const PetscInt index,
                                               const PetscReal value) {
  PetscNestedMatrix::GetMatrix(2, 0)->Increment(row, index, value);
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

void StokesMatrix::ConstantVec(Vec v) {
  PetscReal *a;
  VecGetArray(v, &a);
  PetscReal sum = 0.0, average;
  for (unsigned int i = 0; i < numLocalParticle_; i++) {
    sum += a[fieldDof_ * i + velocityDof_];
  }
  MPI_Allreduce(&sum, &average, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  average /= numGlobalParticle_;
  for (unsigned int i = 0; i < numLocalParticle_; i++) {
    a[fieldDof_ * i + velocityDof_] -= average;
  }
  VecRestoreArray(v, &a);
}

void StokesMatrix::ConstantVecNonNeighborPressure(Vec v) {
  PetscReal *a;
  VecGetArray(v, &a);
  PetscReal sum = 0.0;
  PetscReal sum1, sum2;
  PetscReal average;
  for (unsigned int i = 0; i < numLocalParticle_; i++) {
    sum += a[fieldDof_ * i + velocityDof_];
  }
  MPI_Allreduce(&sum, &sum1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  sum = 0.0;
  for (unsigned int i = 0; i < idxNeighborPressure_.size(); i++) {
    sum += a[idxNeighborPressure_[i]];
  }
  MPI_Allreduce(&sum, &sum2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  int numNonNeighborPressure = idxNonNeighborPressure_.size();
  MPI_Allreduce(MPI_IN_PLACE, &numNonNeighborPressure, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  average = (sum1 - sum2) / (double)numNonNeighborPressure;
  for (unsigned int i = 0; i < idxNonNeighborPressure_.size(); i++) {
    a[idxNonNeighborPressure_[i]] -= average;
  }
  sum = 0.0;
  for (unsigned int i = 0; i < numLocalParticle_; i++) {
    sum += a[fieldDof_ * i + velocityDof_];
  }
  MPI_Allreduce(&sum, &average, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

void StokesMatrix::ForwardField(Vec x, Vec y) {
  PetscReal *a1, *a2;
  VecGetArray(x, &a1);
  VecGetArray(y, &a2);
  for (unsigned int i = 0; i < numLocalParticle_ * fieldDof_; i++) {
    a2[i] = a1[i];
  }
  VecRestoreArray(x, &a1);
  VecRestoreArray(y, &a2);
}

void StokesMatrix::BackwardField(Vec x, Vec y) {
  PetscReal *a1, *a2;
  VecGetArray(x, &a1);
  VecGetArray(y, &a2);
  for (unsigned int i = 0; i < numLocalParticle_ * fieldDof_; i++) {
    a1[i] = a2[i];
  }
  VecRestoreArray(x, &a1);
  VecRestoreArray(y, &a2);
}

void StokesMatrix::ForwardNeighbor(Vec x, Vec y) {
  PetscReal *a1, *a2;
  VecGetArray(x, &a1);
  VecGetArray(y, &a2);
  for (unsigned int i = 0; i < idxNeighbor_.size(); i++) {
    a2[i] = a1[idxNeighbor_[i]];
  }
  for (unsigned int i = 0; i < numLocalRigidBody_ * rigidBodyDof_; i++) {
    a2[i + idxNeighbor_.size()] = a1[numLocalParticle_ * fieldDof_ + i];
  }
  VecRestoreArray(x, &a1);
  VecRestoreArray(y, &a2);
}

void StokesMatrix::BackwardNeighbor(Vec x, Vec y) {
  PetscReal *a1, *a2;
  VecGetArray(x, &a1);
  VecGetArray(y, &a2);
  for (unsigned int i = 0; i < idxNeighbor_.size(); i++) {
    a1[idxNeighbor_[i]] = a2[i];
  }
  for (unsigned int i = 0; i < numLocalRigidBody_ * rigidBodyDof_; i++) {
    a1[numLocalParticle_ * fieldDof_ + i] = a2[i + idxNeighbor_.size()];
  }
  VecRestoreArray(x, &a1);
  VecRestoreArray(y, &a2);
}