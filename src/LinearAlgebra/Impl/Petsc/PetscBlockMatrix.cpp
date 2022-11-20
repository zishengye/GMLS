#include "LinearAlgebra/Impl/Petsc/PetscBlockMatrix.hpp"
#include "LinearAlgebra/Impl/Petsc/PetscBackend.hpp"
#include "LinearAlgebra/Impl/Petsc/PetscMatrix.hpp"
#include "LinearAlgebra/Impl/Petsc/PetscVector.hpp"
#include "petscksp.h"
#include "petscpc.h"
#include "petscsys.h"
#include "petscvec.h"

#include <memory>

#include <mpi.h>
#include <petscmat.h>

LinearAlgebra::Impl::PetscBlockMatrix::PetscBlockMatrix()
    : PetscMatrix(), schur_(PETSC_NULL), a00Ksp_(PETSC_NULL),
      a11Ksp_(PETSC_NULL) {}

LinearAlgebra::Impl::PetscBlockMatrix::~PetscBlockMatrix() {
  if (matPtr_.use_count() == 1)
    if (*matPtr_ != PETSC_NULL)
      MatDestroy(matPtr_.get());

  if (schur_ != PETSC_NULL)
    MatDestroy(&schur_);

  if (a00Ksp_ != PETSC_NULL)
    KSPDestroy(&a00Ksp_);
  if (a11Ksp_ != PETSC_NULL)
    KSPDestroy(&a11Ksp_);

  for (unsigned int i = 0; i < lhsVector_.size(); i++) {
    if (lhsVector_[i] != PETSC_NULL)
      VecDestroy(&lhsVector_[i]);
  }

  for (unsigned int i = 0; i < rhsVector_.size(); i++) {
    if (rhsVector_[i] != PETSC_NULL)
      VecDestroy(&rhsVector_[i]);
  }
}

Void LinearAlgebra::Impl::PetscBlockMatrix::Resize(const PetscInt blockM,
                                                   const PetscInt blockN,
                                                   const PetscInt blockSize) {
  blockM_ = blockM;
  blockN_ = blockN;

  subMat_.resize(blockM * blockN);
}

std::shared_ptr<LinearAlgebra::Impl::PetscMatrix>
LinearAlgebra::Impl::PetscBlockMatrix::GetSubMat(const PetscInt blockI,
                                                 const PetscInt blockJ) {
  return subMat_[blockI * blockN_ + blockJ];
}

Void LinearAlgebra::Impl::PetscBlockMatrix::SetSubMat(
    const PetscInt blockI, const PetscInt blockJ,
    std::shared_ptr<PetscMatrix> mat) {
  subMat_[blockI * blockN_ + blockJ] = mat;
}

Void LinearAlgebra::Impl::PetscBlockMatrix::SetCallbackPointer(
    LinearAlgebra::BlockMatrix<LinearAlgebra::Impl::PetscBackend> *ptr) {
  callbackPtr_ = ptr;
}

Void LinearAlgebra::Impl::PetscBlockMatrix::Assemble() {
  for (unsigned int i = 0; i < subMat_.size(); i++) {
    subMat_[i]->Assemble();
  }

  PetscInt localRow = 0, localCol = 0;
  lhsVector_.resize(blockM_);
  rhsVector_.resize(blockN_);
  localLhsVectorOffset_.resize(blockM_ + 1);
  localRhsVectorOffset_.resize(blockN_ + 1);
  localLhsVectorOffset_[0] = 0;
  localRhsVectorOffset_[0] = 0;
  for (PetscInt i = 0; i < blockM_; i++) {
    PetscInt tLocalRow = subMat_[i * blockN_]->GetLocalRowSize();
    localRow += tLocalRow;
    localLhsVectorOffset_[i + 1] = localRow;
    VecCreateMPI(MPI_COMM_WORLD, tLocalRow, PETSC_DECIDE, &lhsVector_[i]);
  }

  for (PetscInt i = 0; i < blockN_; i++) {
    PetscInt tLocalCol = subMat_[i]->GetLocalColSize();
    localCol += tLocalCol;
    localRhsVectorOffset_[i + 1] = localCol;
    VecCreateMPI(MPI_COMM_WORLD, tLocalCol, PETSC_DECIDE, &rhsVector_[i]);
  }

  MatCreateShell(PETSC_COMM_WORLD, localRow, localCol, PETSC_DECIDE,
                 PETSC_DECIDE, (void *)callbackPtr_, matPtr_.get());
  MatShellSetOperation(*matPtr_, MATOP_MULT,
                       (Void(*)(Void))PetscBlockMatrixMatMultWrapper);
}

Void LinearAlgebra::Impl::PetscBlockMatrix::MatrixVectorMultiplication(
    PetscVector &vec1, PetscVector &vec2) {
  PetscReal *a, *b;
  VecGetArray(*(vec1.vecPtr_), &a);
  for (PetscInt i = 0; i < blockN_; i++) {
    VecGetArray(rhsVector_[i], &b);
    for (unsigned int j = localRhsVectorOffset_[i];
         j < localRhsVectorOffset_[i + 1]; j++) {
      b[j - localRhsVectorOffset_[i]] = a[j];
    }
    VecRestoreArray(rhsVector_[i], &b);
  }
  VecRestoreArray(*(vec1.vecPtr_), &a);

  for (PetscInt i = 0; i < blockM_; i++) {
    VecSet(lhsVector_[i], 0.0);
  }

  // PetscReal sum, average;
  // PetscInt length;
  // VecSum(rhsVector_[1], &sum);
  // VecGetArray(rhsVector_[1], &a);
  // VecGetSize(rhsVector_[1], &length);
  // average = sum / (double)length;
  // for (unsigned int i = 0;
  //      i < localRhsVectorOffset_[2] - localRhsVectorOffset_[1]; i++) {
  //   a[i] -= average;
  // }
  // VecRestoreArray(rhsVector_[1], &a);

  for (PetscInt i = 0; i < blockM_; i++) {
    for (PetscInt j = 0; j < blockN_; j++) {
      MatMultAdd(*(subMat_[i * blockN_ + j]->matPtr_), rhsVector_[j],
                 lhsVector_[i], lhsVector_[i]);
    }
  }

  // VecSum(lhsVector_[1], &sum);
  // VecGetArray(lhsVector_[1], &a);
  // VecGetSize(lhsVector_[1], &length);
  // average = sum / (double)length;
  // for (unsigned int i = 0;
  //      i < localLhsVectorOffset_[2] - localLhsVectorOffset_[1]; i++) {
  //   a[i] -= average;
  // }
  // VecRestoreArray(lhsVector_[1], &a);

  VecGetArray(*(vec2.vecPtr_), &a);
  for (PetscInt i = 0; i < blockM_; i++) {
    VecGetArray(lhsVector_[i], &b);
    for (unsigned int j = localLhsVectorOffset_[i];
         j < localLhsVectorOffset_[i + 1]; j++) {
      a[j] = b[j - localLhsVectorOffset_[i]];
    }
    VecRestoreArray(lhsVector_[i], &b);
  }
  VecRestoreArray(*(vec2.vecPtr_), &a);
}

Void LinearAlgebra::Impl::PetscBlockMatrix::
    PrepareSchurComplementPreconditioner() {
  // prepare schur complement matrix
  Mat B, C;

  auto &a00 = *(subMat_[0]->matPtr_);
  auto &a01 = *(subMat_[1]->matPtr_);
  auto &a10 = *(subMat_[2]->matPtr_);
  auto &a11 = *(subMat_[3]->matPtr_);
  MatCreate(MPI_COMM_WORLD, &B);
  MatSetType(B, MATAIJMKL);
  MatInvertBlockDiagonalMat(a00, B);

  MatMatMult(B, a01, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);

  MPI_Barrier(MPI_COMM_WORLD);
  MatMatMult(a10, C, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &schur_);
  MatScale(schur_, -1.0);
  MatAXPY(schur_, 1.0, a11, DIFFERENT_NONZERO_PATTERN);

  MatDestroy(&B);
  MatDestroy(&C);

  // setup preconditioner for diagonal blocks
  KSPCreate(MPI_COMM_WORLD, &a00Ksp_);
  KSPSetType(a00Ksp_, KSPPREONLY);
  KSPSetOperators(a00Ksp_, a00, a00);
  PC a00Pc;
  KSPGetPC(a00Ksp_, &a00Pc);
  PCSetType(a00Pc, PCSOR);
  PCSetUp(a00Pc);
  KSPSetUp(a00Ksp_);

  KSPCreate(MPI_COMM_WORLD, &a11Ksp_);
  KSPSetType(a11Ksp_, KSPPREONLY);
  KSPSetOperators(a11Ksp_, a11, a11);
  PC a11Pc;
  KSPGetPC(a11Ksp_, &a11Pc);
  PCSetType(a11Pc, PCSOR);
  PCSetUp(a11Pc);
  KSPSetUp(a11Ksp_);
}

Void LinearAlgebra::Impl::PetscBlockMatrix::
    ApplySchurComplementPreconditioningIteration(PetscVector &x,
                                                 PetscVector &y) {
  PetscReal *a, *b;
  VecGetArray(*(x.vecPtr_), &a);
  for (PetscInt i = 0; i < blockN_; i++) {
    VecGetArray(rhsVector_[i], &b);
    for (unsigned int j = localRhsVectorOffset_[i];
         j < localRhsVectorOffset_[i + 1]; j++) {
      b[j - localRhsVectorOffset_[i]] = a[j];
    }
    VecRestoreArray(rhsVector_[i], &b);
  }
  VecRestoreArray(*(x.vecPtr_), &a);

  for (PetscInt i = 0; i < blockM_; i++) {
    VecSet(lhsVector_[i], 0.0);
  }

  KSPSolve(a00Ksp_, rhsVector_[0], lhsVector_[0]);
  KSPSolve(a11Ksp_, rhsVector_[1], lhsVector_[1]);

  VecGetArray(*(y.vecPtr_), &a);
  for (PetscInt i = 0; i < blockM_; i++) {
    VecGetArray(lhsVector_[i], &b);
    for (unsigned int j = localLhsVectorOffset_[i];
         j < localLhsVectorOffset_[i + 1]; j++) {
      a[j] = b[j - localLhsVectorOffset_[i]];
    }
    VecRestoreArray(lhsVector_[i], &b);
  }
  VecRestoreArray(*(y.vecPtr_), &a);
}

PetscErrorCode PetscBlockMatrixMatMultWrapper(Mat mat, Vec x, Vec y) {
  LinearAlgebra::BlockMatrix<LinearAlgebra::Impl::PetscBackend> *ctx;
  MatShellGetContext(mat, ((void **)&ctx));

  LinearAlgebra::Vector<LinearAlgebra::Impl::PetscBackend> vecX;
  LinearAlgebra::Vector<LinearAlgebra::Impl::PetscBackend> vecY;

  LinearAlgebra::Impl::PetscVector vecPetscX;

  vecPetscX.Create(x);

  vecX.Create(vecPetscX);
  vecY.Create(vecPetscX);

  ctx->MatrixVectorMultiplication(vecX, vecY);

  vecY.Copy(vecPetscX);

  vecPetscX.Copy(y);

  return 0;
}