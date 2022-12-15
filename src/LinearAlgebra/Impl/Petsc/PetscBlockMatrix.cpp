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
  if (matSharedPtr_.use_count() == 1)
    if (*matSharedPtr_ != PETSC_NULL)
      MatDestroy(matSharedPtr_.get());

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

Void LinearAlgebra::Impl::PetscBlockMatrix::ClearTimer() {
  a00Timer_ = 0;
  a11Timer_ = 0;
}

PetscReal LinearAlgebra::Impl::PetscBlockMatrix::GetA00Timer() {
  return a00Timer_;
}

PetscReal LinearAlgebra::Impl::PetscBlockMatrix::GetA11Timer() {
  return a11Timer_;
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
                 PETSC_DECIDE, (void *)callbackPtr_, matSharedPtr_.get());
  MatShellSetOperation(*matSharedPtr_, MATOP_MULT,
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

  for (PetscInt i = 0; i < blockM_; i++) {
    for (PetscInt j = 0; j < blockN_; j++) {
      MatMultAdd(*(subMat_[i * blockN_ + j]->matSharedPtr_), rhsVector_[j],
                 lhsVector_[i], lhsVector_[i]);
    }
  }

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

  auto &a00 = *(subMat_[0]->matSharedPtr_);
  auto &a01 = *(subMat_[1]->matSharedPtr_);
  auto &a10 = *(subMat_[2]->matSharedPtr_);
  auto &a11 = *(subMat_[3]->matSharedPtr_);

  // MatDuplicate(a10, MAT_COPY_VALUES, &B);

  // Vec diag;
  // MatCreateVecs(a00, &diag, NULL);
  // MatGetDiagonal(a00, diag);
  // VecReciprocal(diag);

  // MatDiagonalScale(B, NULL, diag);
  // PetscPrintf(MPI_COMM_WORLD, "After MatDiagonalScale\n");
  // MPI_Barrier(MPI_COMM_WORLD);
  // MatMatMult(B, a01, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &schur_);
  // PetscPrintf(MPI_COMM_WORLD, "After MatMatMult\n");
  // MPI_Barrier(MPI_COMM_WORLD);
  // MatScale(schur_, -1.0);
  // MPI_Barrier(MPI_COMM_WORLD);
  // PetscPrintf(MPI_COMM_WORLD, "After MatScale\n");
  // MatAXPY(schur_, 1.0, a11, DIFFERENT_NONZERO_PATTERN);
  // MPI_Barrier(MPI_COMM_WORLD);
  // PetscPrintf(MPI_COMM_WORLD, "After MatAXPY\n");

  // MatDestroy(&B);
  // VecDestroy(&diag);

  // MPI_Barrier(MPI_COMM_WORLD);

  // setup preconditioner for diagonal blocks
  KSPCreate(MPI_COMM_WORLD, &a00Ksp_);
  KSPSetType(a00Ksp_, KSPRICHARDSON);
  KSPSetTolerances(a00Ksp_, 1e-3, 1e-50, 1e20, 1);
  KSPSetOperators(a00Ksp_, a00, a00);
  PC a00Pc;
  KSPGetPC(a00Ksp_, &a00Pc);
  PCSetType(a00Pc, PCSOR);
  PCSetUp(a00Pc);
  KSPSetUp(a00Ksp_);

  KSPCreate(MPI_COMM_WORLD, &a11Ksp_);
  KSPSetType(a11Ksp_, KSPRICHARDSON);
  KSPSetTolerances(a11Ksp_, 1e-3, 1e-50, 1e20, 1);
  KSPSetOperators(a11Ksp_, a11, a11);
  PC a11Pc;
  KSPGetPC(a11Ksp_, &a11Pc);
  PCSetType(a11Pc, PCSOR);
  PCSetFromOptions(a11Pc);
  PCSetUp(a11Pc);
  KSPSetUp(a11Ksp_);
}

Void LinearAlgebra::Impl::PetscBlockMatrix::
    ApplySchurComplementPreconditioningIteration(PetscVector &x,
                                                 PetscVector &y) {
  double timer1, timer2;

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

  Vec a0, a1;
  VecDuplicate(rhsVector_[0], &a0);
  VecDuplicate(rhsVector_[1], &a1);

  auto &a01 = *(subMat_[1]->matSharedPtr_);
  auto &a10 = *(subMat_[2]->matSharedPtr_);

  MPI_Barrier(MPI_COMM_WORLD);
  timer1 = MPI_Wtime();
  KSPSolve(a00Ksp_, rhsVector_[0], lhsVector_[0]);
  timer2 = MPI_Wtime();
  a00Timer_ += timer2 - timer1;

  MatMult(a10, lhsVector_[0], a1);
  VecAXPY(rhsVector_[1], -1.0, a1);

  MPI_Barrier(MPI_COMM_WORLD);
  timer1 = MPI_Wtime();
  KSPSolve(a11Ksp_, rhsVector_[1], lhsVector_[1]);
  timer2 = MPI_Wtime();
  a11Timer_ += timer2 - timer1;

  MatMult(a01, lhsVector_[1], a0);

  MPI_Barrier(MPI_COMM_WORLD);
  timer1 = MPI_Wtime();
  KSPSolve(a00Ksp_, a0, a0);
  timer2 = MPI_Wtime();
  a00Timer_ += timer2 - timer1;

  VecAXPY(lhsVector_[0], -1.0, a0);

  VecDestroy(&a0);
  VecDestroy(&a1);

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