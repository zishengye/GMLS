#include "StokesMatrix.hpp"

void StokesMatrix::InitInternal() {
  numLocalRigidBody_ =
      numRigidBody_ / mpiSize_ + (numRigidBody_ % mpiSize_ < mpiRank_) ? 1 : 0;

  auto a00 = PetscNestedMatrix::GetMatrix(0, 0);
  auto a01 = PetscNestedMatrix::GetMatrix(0, 1);
  auto a10 = PetscNestedMatrix::GetMatrix(1, 0);
  auto a11 = PetscNestedMatrix::GetMatrix(1, 1);

  a00->Resize(numLocalParticle_ * fieldDof_, numLocalParticle_ * fieldDof_,
              fieldDof_);
  a01->Resize(numLocalParticle_ * fieldDof_,
              numLocalRigidBody_ * rigidBodyDof_);
  a10->Resize(numLocalRigidBody_ * rigidBodyDof_,
              numLocalParticle_ * fieldDof_);
  a11->Resize(numLocalRigidBody_ * rigidBodyDof_,
              numLocalRigidBody_ * rigidBodyDof_, rigidBodyDof_);
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
  rigidBodyDof_ = (dimension == 3) ? 6 : 3;

  InitInternal();
}

StokesMatrix::StokesMatrix(const unsigned long numLocalParticle,
                           const unsigned int numRigidBody,
                           const unsigned int dimension)
    : dimension_(dimension), numRigidBody_(numRigidBody),
      numLocalParticle_(numLocalParticle), fieldDof_(dimension + 1),
      velocityDof_(dimension), PetscNestedMatrix(2, 2) {
  rigidBodyDof_ = (dimension == 3) ? 6 : 3;

  InitInternal();
}