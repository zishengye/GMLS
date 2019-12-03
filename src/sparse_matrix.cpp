#include "sparse_matrix.h"

#include <fstream>
#include <string>

using namespace std;

int PetscSparseMatrix::FinalAssemble() {
  // move data from outProcessIncrement
  int myid, MPIsize;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

  int sendCount = __unsorted_val.size();
  vector<int> recvCount(MPIsize);
  vector<int> displs(MPIsize + 1);

  MPI_Gather(&sendCount, 1, MPI_INT, recvCount.data(), 1, MPI_INT, MPIsize - 1,
             MPI_COMM_WORLD);

  if (myid == MPIsize - 1) {
    displs[0] = 0;
    for (int i = 1; i <= MPIsize; i++) {
      displs[i] = displs[i - 1] + recvCount[i - 1];
    }
  }

  vector<PetscInt> iRecv;
  vector<PetscInt> jRecv;
  vector<PetscReal> valRecv;

  iRecv.resize(displs[MPIsize]);
  jRecv.resize(displs[MPIsize]);
  valRecv.resize(displs[MPIsize]);

  MPI_Gatherv(__unsorted_i.data(), sendCount, MPI_UNSIGNED, iRecv.data(),
              recvCount.data(), displs.data(), MPI_UNSIGNED, MPIsize - 1,
              MPI_COMM_WORLD);
  MPI_Gatherv(__unsorted_j.data(), sendCount, MPI_UNSIGNED, jRecv.data(),
              recvCount.data(), displs.data(), MPI_UNSIGNED, MPIsize - 1,
              MPI_COMM_WORLD);
  MPI_Gatherv(__unsorted_val.data(), sendCount, MPI_DOUBLE, valRecv.data(),
              recvCount.data(), displs.data(), MPI_DOUBLE, MPIsize - 1,
              MPI_COMM_WORLD);

  // merge data
  if (myid == MPIsize - 1) {
    for (int i = 0; i < iRecv.size(); i++) {
      this->increment(iRecv[i], jRecv[i], valRecv[i]);
    }
  }

  // prepare data for Petsc construction
  sortbyj();

  __i.resize(__row + 1);

  __nnz = 0;
  for (int i = 0; i < __row; i++) {
    __nnz += __matrix[i].size();
  }

  __j.resize(__nnz);
  __val.resize(__nnz);

  int count = 0;
  for (int i = 0; i < __row; i++) {
    __i[i] = count;
    for (list<entry>::iterator it = __matrix[i].begin();
         it != __matrix[i].end(); it++) {
      __j[count] = (it->first);
      __val[count] = it->second;
      count++;
    }
  }
  __i[__row] = count;

  MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, __row, __col, PETSC_DECIDE, __Col,
                            __i.data(), __j.data(), __val.data(), &__mat);

  __isAssembled = true;

  return __nnz;
}

void PetscSparseMatrix::Solve(vector<double> &rhs, vector<double> &x) {
  if (__isAssembled) {
    Vec _rhs, _x;
    VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, rhs.size(), PETSC_DECIDE,
                          rhs.data(), &_rhs);
    VecDuplicate(_rhs, &_x);

    KSP _ksp;
    KSPCreate(PETSC_COMM_WORLD, &_ksp);
    KSPSetOperators(_ksp, __mat, __mat);
    KSPSetFromOptions(_ksp);

    PetscPrintf(PETSC_COMM_WORLD, "final solving of linear system\n");
    KSPSolve(_ksp, _rhs, _x);
    PetscPrintf(PETSC_COMM_WORLD, "ksp solving finished\n");

    KSPDestroy(&_ksp);

    PetscScalar *a;
    VecGetArray(_x, &a);
    for (size_t i = 0; i < rhs.size(); i++) {
      x[i] = a[i];
    }

    VecDestroy(&_rhs);
    VecDestroy(&_x);
  }
}

void Solve(PetscSparseMatrix &A, PetscSparseMatrix &Bt, PetscSparseMatrix &B,
           PetscSparseMatrix &C, std::vector<double> &f, std::vector<double> &g,
           std::vector<double> &x, std::vector<double> &y) {
  if (!A.__isAssembled || !Bt.__isAssembled || !B.__isAssembled ||
      !C.__isAssembled)
    return;

  // setup rhs
  Vec _bpSub[2], _xpSub[2], _bp, _xp;
  VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, f.size(), PETSC_DECIDE, f.data(),
                        &_bpSub[0]);
  VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, g.size(), PETSC_DECIDE, g.data(),
                        &_bpSub[1]);

  // setup matrix
  Mat _ASub[5], _A;

  VecCreateNest(PETSC_COMM_WORLD, 2, NULL, _bpSub, &_bp);
  MatDuplicate(A.__mat, MAT_COPY_VALUES, &_ASub[0]);
  MatDuplicate(Bt.__mat, MAT_COPY_VALUES, &_ASub[1]);
  MatDuplicate(B.__mat, MAT_COPY_VALUES, &_ASub[2]);
  MatDuplicate(C.__mat, MAT_COPY_VALUES, &_ASub[3]);
  MatCreateNest(MPI_COMM_WORLD, 2, NULL, 2, NULL, _ASub, &_A);

  // setup schur complement approximation
  Vec _diag;
  VecCreate(PETSC_COMM_WORLD, &_diag);
  VecSetSizes(_diag, f.size(), PETSC_DECIDE);
  VecSetType(_diag, VECMPI);
  MatGetDiagonal(_ASub[0], _diag);
  VecReciprocal(_diag);
  MatDiagonalScale(_ASub[1], _diag, NULL);

  MatMatMult(_ASub[2], _ASub[1], MAT_INITIAL_MATRIX, PETSC_DEFAULT, _ASub + 4);

  MatScale(_ASub[4], -1.0);
  MatAXPY(_ASub[4], 1.0, _ASub[3], DIFFERENT_NONZERO_PATTERN);

  MatGetDiagonal(_ASub[0], _diag);
  MatDiagonalScale(_ASub[1], _diag, NULL);
  VecDestroy(&_diag);

  PetscPrintf(PETSC_COMM_WORLD, "Setup Schur Complement Preconditioner\n");

  // setup ksp
  KSP _ksp;
  PC _pc;

  KSPCreate(PETSC_COMM_WORLD, &_ksp);
  KSPSetOperators(_ksp, _A, _A);
  KSPSetFromOptions(_ksp);

  IS _isg[2];

  PetscFunctionBeginUser;

  // setup index sets
  MatNestGetISs(_A, _isg, NULL);

  PetscFunctionBeginUser;
  KSPGetPC(_ksp, &_pc);
  PCFieldSplitSetIS(_pc, "0", _isg[0]);
  PCFieldSplitSetIS(_pc, "1", _isg[1]);

  PCFieldSplitSetSchurPre(_pc, PC_FIELDSPLIT_SCHUR_PRE_USER, _ASub[4]);

  // setup sub solver
  KSP *_subKsp;
  PetscInt n = 1;
  PCSetUp(_pc);
  PCFieldSplitGetSubKSP(_pc, &n, &_subKsp);
  KSPSetOperators(_subKsp[1], _ASub[4], _ASub[4]);
  KSPSetFromOptions(_subKsp[0]);
  PetscFree(_subKsp);

  VecDuplicate(_bp, &_xp);

  // final solve
  PetscPrintf(PETSC_COMM_WORLD, "final solving of linear system\n");
  KSPSolve(_ksp, _bp, _xp);
  PetscPrintf(PETSC_COMM_WORLD, "finish ksp solving\n");

  KSPDestroy(&_ksp);

  // copy result
  VecGetSubVector(_xp, _isg[0], &_xpSub[0]);
  VecGetSubVector(_xp, _isg[1], &_xpSub[1]);

  double *p;
  VecGetArray(_xpSub[0], &p);
  for (size_t i = 0; i < f.size(); i++) {
    x[i] = p[i];
  }
  VecRestoreArray(_xpSub[0], &p);
  VecGetArray(_xpSub[1], &p);
  for (size_t i = 0; i < g.size(); i++) {
    y[i] = p[i];
  }
  VecRestoreArray(_xpSub[1], &p);

  VecRestoreSubVector(_xp, _isg[0], &_xpSub[0]);
  VecRestoreSubVector(_xp, _isg[1], &_xpSub[1]);

  // destroy objects
  VecDestroy(_bpSub);
  VecDestroy(_bpSub + 1);
  VecDestroy(&_bp);
  VecDestroy(&_xp);

  for (int i = 0; i < 5; i++) {
    MatDestroy(_ASub + i);
  }

  MatDestroy(&_A);
}