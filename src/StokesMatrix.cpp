#include "StokesMatrix.hpp"

#include <algorithm>

void StokesMatrix::InitInternal() {
  numLocalRigidBody_ =
      numRigidBody_ / mpiSize_ + (numRigidBody_ % mpiSize_ > mpiRank_) ? 1 : 0;

  rigidBodyStartIndex_ = 0;
  for (int i = 0; i < mpiRank_; i++) {
    rigidBodyStartIndex_ +=
        numRigidBody_ / mpiSize_ + (numRigidBody_ % mpiSize_ > i) ? 1 : 0;
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
}

unsigned long StokesMatrix::Assemble() { PetscNestedMatrix::Assemble(); }

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