#include "StokesMatrix.hpp"

#include <algorithm>

void StokesMatrix::InitInternal() {
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

void StokesMatrix::Resize(const unsigned long numLocalParticle,
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

  std::vector<std::vector<PetscInt>> rigidBodyFieldIndexMap(numRigidBody_);
  for (unsigned int i = 0; i < numLocalParticle_; i++) {
    if (particleType[i] >= 4) {
      for (unsigned int j = 0; j < neighborLists(i, 0); j++) {
        const unsigned int neighborParticleIndex =
            globalIndex[neighborLists(i, j + 1)];
        for (unsigned int axes = 0; axes < fieldDof_; axes++) {
          rigidBodyFieldIndexMap[attachedRigidBody[i]].push_back(
              neighborParticleIndex * fieldDof_ + axes);
        }
      }
    }
  }

  for (unsigned int i = 0; i < numRigidBody_; i++) {
    std::sort(rigidBodyFieldIndexMap[i].begin(),
              rigidBodyFieldIndexMap[i].end());
    rigidBodyFieldIndexMap[i].erase(
        std::unique(rigidBodyFieldIndexMap[i].begin(),
                    rigidBodyFieldIndexMap[i].end()),
        rigidBodyFieldIndexMap[i].end());
    MPI_Barrier(MPI_COMM_WORLD);

    int targetRank = 0;
    if (i >= rigidBodyStartIndex_ && i < rigidBodyEndIndex_)
      targetRank = mpiRank_;
    MPI_Allreduce(MPI_IN_PLACE, &targetRank, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    int sendCount = rigidBodyFieldIndexMap[i].size();
    std::vector<int> index;
    std::vector<int> recvCount(mpiSize_);
    std::vector<int> recvOffset(mpiSize_ + 1);
    MPI_Gather(&sendCount, 1, MPI_INT, recvCount.data(), 1, MPI_INT, targetRank,
               MPI_COMM_WORLD);
    if (targetRank == mpiRank_) {
      recvOffset[0] = 0;
      for (unsigned int j = 0; j < mpiSize_; j++) {
        recvOffset[j + 1] = recvCount[j] + recvOffset[j];
      }
      index.resize(recvOffset[mpiSize_]);
    }
    MPI_Gatherv(rigidBodyFieldIndexMap[i].data(), sendCount, MPI_INT,
                index.data(), recvCount.data(), recvOffset.data(), MPI_INT,
                targetRank, MPI_COMM_WORLD);

    if (targetRank == mpiRank_) {
      std::sort(index.begin(), index.end());
      index.erase(std::unique(index.begin(), index.end()), index.end());
      for (unsigned int j = 0; j < rigidBodyDof_; j++) {
        a10->SetColIndex((i - rigidBodyStartIndex_) * rigidBodyDof_ + j, index);
      }
    }
  }

  a00->GraphAssemble();
  a01->GraphAssemble();
  a10->GraphAssemble();
  a11->GraphAssemble();

  std::vector<unsigned int> flattenedRigidBodyFieldIndexMap;
  for (unsigned int i = 0; i < numRigidBody_; i++) {
    for (unsigned int j = 0; j < rigidBodyFieldIndexMap[i].size(); j++) {
      flattenedRigidBodyFieldIndexMap.push_back(rigidBodyFieldIndexMap[i][j] /
                                                fieldDof_);
    }
  }
  std::sort(flattenedRigidBodyFieldIndexMap.begin(),
            flattenedRigidBodyFieldIndexMap.end());
  flattenedRigidBodyFieldIndexMap.erase(
      std::unique(flattenedRigidBodyFieldIndexMap.begin(),
                  flattenedRigidBodyFieldIndexMap.end()),
      flattenedRigidBodyFieldIndexMap.end());
  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<int> gatheredField;
  std::vector<int> localField;
  std::vector<PetscInt> idxNeighbor, idxNeighborBlock;
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

    auto start = std::lower_bound(flattenedRigidBodyFieldIndexMap.begin(),
                                  flattenedRigidBodyFieldIndexMap.end(),
                                  (PetscInt)startIndex);
    auto end = std::lower_bound(start, flattenedRigidBodyFieldIndexMap.end(),
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
          idxNeighborBlock.push_back(j + startIndex);
          for (unsigned int k = 0; k < fieldDof_; k++)
            idxNeighbor.push_back((j + startIndex) * fieldDof_ + k);
        }
    }
  }

  idxNeighborBlock.clear();
  if (mpiRank_ == 0)
    idxNeighborBlock.push_back(0);

  ISCreateGeneral(MPI_COMM_WORLD, idxNeighbor.size(), idxNeighbor.data(),
                  PETSC_COPY_VALUES, &isgNeighbor_);
  ISCreateGeneral(MPI_COMM_WORLD, idxNeighborBlock.size(),
                  idxNeighborBlock.data(), PETSC_COPY_VALUES,
                  &isgNeighborBlock_);
}

unsigned long StokesMatrix::Assemble() {
  PetscNestedMatrix::Assemble();

  nestedMat_[0] = PetscNestedMatrix::GetMatrix(0, 0)->GetReference();
  nestedMat_[1] = PetscNestedMatrix::GetMatrix(0, 1)->GetReference();
  nestedMat_[2] = PetscNestedMatrix::GetMatrix(1, 0)->GetReference();
  nestedMat_[3] = PetscNestedMatrix::GetMatrix(1, 1)->GetReference();

  MatCreateNest(MPI_COMM_WORLD, 2, PETSC_NULL, 2, PETSC_NULL, nestedMat_.data(),
                &mat_);

  Mat B, C;
  MatCreate(MPI_COMM_WORLD, &B);
  MatSetType(B, MATMPIAIJ);
  MatInvertBlockDiagonalMat(nestedMat_[0], B);

  MatMatMult(B, nestedMat_[1], MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);
  MatMatMult(nestedMat_[2], C, MAT_INITIAL_MATRIX, PETSC_DEFAULT,
             &neighborSchurMat_);
  MatScale(neighborSchurMat_, -1.0);
  MatAXPY(neighborSchurMat_, 1.0, nestedMat_[3], DIFFERENT_NONZERO_PATTERN);

  nestedNeighborMat_.resize(4);
  nestedNeighborWholeMat_.resize(4);

  std::vector<PetscInt> idxColloid;
  PetscInt lowRange, highRange;
  MatGetOwnershipRange(nestedMat_[2], &lowRange, &highRange);
  idxColloid.resize(highRange - lowRange);
  for (unsigned int i = 0; i < idxColloid.size(); i++)
    idxColloid[i] = lowRange + i;
  IS isgColloid;
  ISCreateGeneral(PETSC_COMM_WORLD, idxColloid.size(), idxColloid.data(),
                  PETSC_COPY_VALUES, &isgColloid);

  MatCreateSubMatrix(nestedMat_[0], isgNeighbor_, isgNeighbor_,
                     MAT_INITIAL_MATRIX, &nestedNeighborMat_[0]);
  // MatCreateSubMatrix(nestedMat_[1], isgNeighbor_, PETSC_NULL,
  //                    MAT_INITIAL_MATRIX, &nestedNeighborMat_[1]);
  // MatCreateSubMatrix(nestedMat_[2], isgColloid, isgNeighbor_,
  //                    MAT_INITIAL_MATRIX, &nestedNeighborMat_[2]);
  // nestedNeighborMat_[3] = nestedMat_[3];
  // MatCreateNest(MPI_COMM_WORLD, 2, PETSC_NULL, 2, PETSC_NULL,
  //               nestedNeighborMat_.data(), &neighborMat_);

  MatCreateSubMatrix(nestedMat_[0], isgNeighbor_, PETSC_NULL,
                     MAT_INITIAL_MATRIX, &nestedNeighborWholeMat_[0]);
  // MatCreateSubMatrix(nestedMat_[1], isgNeighbor_, PETSC_NULL,
  //                    MAT_INITIAL_MATRIX, &nestedNeighborWholeMat_[1]);
  // MatCreateSubMatrix(nestedMat_[2], isgColloid, isgNeighbor_,
  //                    MAT_INITIAL_MATRIX, &nestedNeighborWholeMat_[2]);
  // nestedNeighborWholeMat_[3] = nestedMat_[3];
  // MatCreateNest(MPI_COMM_WORLD, 2, PETSC_NULL, 2, PETSC_NULL,
  //               nestedNeighborWholeMat_.data(), &neighborWholeMat_);

  ISDestroy(&isgColloid);

  MatDestroy(&B);
  MatDestroy(&C);
}

void StokesMatrix::IncrementFieldField(const PetscInt row,
                                       const std::vector<PetscInt> &index,
                                       const std::vector<PetscReal> &value) {
  PetscNestedMatrix::GetMatrix(0, 0)->Increment(row, index, value);
}

void StokesMatrix::IncrementFieldRigidBody(const PetscInt row,
                                           const PetscInt index,
                                           const PetscInt value) {
  PetscNestedMatrix::GetMatrix(0, 1)->Increment(row, index, value);
}

void StokesMatrix::IncrementRigidBodyField(const PetscInt row,
                                           const PetscInt index,
                                           const PetscInt value) {
  PetscNestedMatrix::GetMatrix(1, 0)->IncrementGlobalIndex(row, index, value);
}

void StokesMatrix::IncrementRigidBodyRigidBody(const PetscInt row,
                                               const PetscInt index,
                                               const PetscInt value) {
  PetscNestedMatrix::GetMatrix(0, 0)->Increment(row, index, value);
}