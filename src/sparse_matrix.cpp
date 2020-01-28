#include "sparse_matrix.h"
#include "composite_preconditioner.h"

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

void PetscSparseMatrix::Solve(vector<double> &rhs, vector<double> &x,
                              int dimension) {
  int fieldDof = dimension + 1;
  int velocityDof = dimension;
  int pressureDof = 1;

  vector<int> idx_velocity, idx_pressure;

  int MPIsize, myId;
  MPI_Comm_rank(MPI_COMM_WORLD, &myId);
  MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

  PetscInt localN1, localN2;
  MatGetOwnershipRange(__mat, &localN1, &localN2);

  if (myId != MPIsize - 1) {
    int localParticleNum = (localN2 - localN1) / fieldDof;
    idx_velocity.resize(velocityDof * localParticleNum);
    idx_pressure.resize(localParticleNum);

    for (int i = 0; i < localParticleNum; i++) {
      for (int j = 0; j < dimension; j++) {
        idx_velocity[velocityDof * i + j] = localN1 + fieldDof * i + j;
      }
      idx_pressure[i] = localN1 + fieldDof * i + velocityDof;
    }
  } else {
    int localParticleNum = (localN2 - localN1 - 1) / fieldDof;
    idx_velocity.resize(velocityDof * localParticleNum);
    idx_pressure.resize(localParticleNum + 1);

    for (int i = 0; i < localParticleNum; i++) {
      for (int j = 0; j < dimension; j++) {
        idx_velocity[velocityDof * i + j] = localN1 + fieldDof * i + j;
      }
      idx_pressure[i] = localN1 + fieldDof * i + velocityDof;
    }
    idx_pressure[localParticleNum] = localN1 + fieldDof * localParticleNum;
  }

  IS isg_velocity, isg_pressure;

  ISCreateGeneral(MPI_COMM_WORLD, idx_velocity.size(), idx_velocity.data(),
                  PETSC_COPY_VALUES, &isg_velocity);
  ISCreateGeneral(MPI_COMM_WORLD, idx_pressure.size(), idx_pressure.data(),
                  PETSC_COPY_VALUES, &isg_pressure);

  Mat uu, up, pu, pp, up_s;

  MatCreateSubMatrix(__mat, isg_velocity, isg_velocity, MAT_INITIAL_MATRIX,
                     &uu);
  MatCreateSubMatrix(__mat, isg_pressure, isg_velocity, MAT_INITIAL_MATRIX,
                     &pu);
  MatCreateSubMatrix(__mat, isg_velocity, isg_pressure, MAT_INITIAL_MATRIX,
                     &up);
  MatCreateSubMatrix(__mat, isg_pressure, isg_pressure, MAT_INITIAL_MATRIX,
                     &pp);

  Vec diag;

  MatCreateVecs(uu, &diag, NULL);
  MatGetDiagonal(uu, diag);
  VecReciprocal(diag);
  MatDiagonalScale(up, diag, NULL);
  MatMatMult(pu, up, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &up_s);
  MatScale(up_s, -1.0);
  MatAXPY(up_s, 1.0, pp, DIFFERENT_NONZERO_PATTERN);

  Vec _rhs, _x;
  VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, rhs.size(), PETSC_DECIDE,
                        rhs.data(), &_rhs);
  VecDuplicate(_rhs, &_x);

  KSP _ksp;
  KSPCreate(PETSC_COMM_WORLD, &_ksp);
  KSPSetOperators(_ksp, __mat, __mat);
  KSPSetFromOptions(_ksp);

  PC _pc;

  KSPGetPC(_ksp, &_pc);
  PCFieldSplitSetIS(_pc, "0", isg_velocity);
  PCFieldSplitSetIS(_pc, "1", isg_pressure);

  PCFieldSplitSetSchurPre(_pc, PC_FIELDSPLIT_SCHUR_PRE_USER, up_s);

  KSP *_subKsp;
  PetscInt n = 1;
  PCSetUp(_pc);
  PCFieldSplitGetSubKSP(_pc, &n, &_subKsp);
  KSPSetOperators(_subKsp[1], up_s, up_s);
  KSPSetFromOptions(_subKsp[0]);
  KSPSetOperators(_subKsp[0], uu, uu);

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

  MatDestroy(&uu);
  MatDestroy(&up);
  MatDestroy(&pu);
  MatDestroy(&pp);
  MatDestroy(&up_s);

  ISDestroy(&isg_velocity);
  ISDestroy(&isg_pressure);
}

void PetscSparseMatrix::Solve(vector<double> &rhs, vector<double> &x,
                              int dimension, int numRigidBody) {
  int fieldDof = dimension + 1;
  int velocityDof = dimension;
  int pressureDof = 1;
  int rigidBodyDof = (dimension == 3) ? 6 : 3;

  vector<int> idx_field, idx_rigid;
  vector<int> idx_velocity, idx_pressure;

  int MPIsize, myId;
  MPI_Comm_rank(MPI_COMM_WORLD, &myId);
  MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

  PetscInt localN1, localN2;
  MatGetOwnershipRange(__mat, &localN1, &localN2);

  if (myId != MPIsize - 1) {
    int localParticleNum = (localN2 - localN1) / fieldDof;
    idx_field.resize(fieldDof * localParticleNum);
    idx_velocity.resize(velocityDof * localParticleNum);
    idx_pressure.resize(localParticleNum);

    for (int i = 0; i < localParticleNum; i++) {
      for (int j = 0; j < dimension; j++) {
        idx_field[fieldDof * i + j] = localN1 + fieldDof * i + j;
        idx_velocity[velocityDof * i + j] = localN1 + fieldDof * i + j;
      }
      idx_field[fieldDof * i + velocityDof] =
          localN1 + fieldDof * i + velocityDof;
      idx_pressure[i] = localN1 + fieldDof * i + velocityDof;
    }
  } else {
    int localParticleNum =
        (localN2 - localN1 - 1 - numRigidBody * rigidBodyDof) / fieldDof;
    idx_field.resize(fieldDof * localParticleNum + 1);
    idx_velocity.resize(velocityDof * localParticleNum);
    idx_pressure.resize(localParticleNum + 1);
    idx_rigid.resize(rigidBodyDof * numRigidBody);

    for (int i = 0; i < localParticleNum; i++) {
      for (int j = 0; j < dimension; j++) {
        idx_field[fieldDof * i + j] = localN1 + fieldDof * i + j;
        idx_velocity[velocityDof * i + j] = localN1 + fieldDof * i + j;
      }
      idx_field[fieldDof * i + velocityDof] =
          localN1 + fieldDof * i + velocityDof;
      idx_pressure[i] = localN1 + fieldDof * i + velocityDof;
    }

    idx_field[fieldDof * localParticleNum] =
        localN1 + fieldDof * localParticleNum;
    idx_pressure[localParticleNum] = localN1 + fieldDof * localParticleNum;

    for (int i = 0; i < numRigidBody; i++) {
      for (int j = 0; j < rigidBodyDof; j++) {
        idx_rigid[rigidBodyDof * i + j] =
            localN1 + fieldDof * localParticleNum + 1 + i * rigidBodyDof + j;
      }
    }
  }

  IS isg_field, isg_rigid;
  IS isg_velocity, isg_pressure;

  ISCreateGeneral(MPI_COMM_WORLD, idx_velocity.size(), idx_velocity.data(),
                  PETSC_COPY_VALUES, &isg_velocity);
  ISCreateGeneral(MPI_COMM_WORLD, idx_pressure.size(), idx_pressure.data(),
                  PETSC_COPY_VALUES, &isg_pressure);
  ISCreateGeneral(MPI_COMM_WORLD, idx_field.size(), idx_field.data(),
                  PETSC_COPY_VALUES, &isg_field);
  ISCreateGeneral(MPI_COMM_WORLD, idx_rigid.size(), idx_rigid.data(),
                  PETSC_COPY_VALUES, &isg_rigid);

  Mat ff, fr, rf, fr_s;

  Mat uu, up, pu, pp, up_s;

  MatCreateSubMatrix(__mat, isg_field, isg_field, MAT_INITIAL_MATRIX, &ff);
  MatCreateSubMatrix(__mat, isg_field, isg_rigid, MAT_INITIAL_MATRIX, &fr);
  MatCreateSubMatrix(__mat, isg_rigid, isg_field, MAT_INITIAL_MATRIX, &rf);

  MatCreateSubMatrix(__mat, isg_velocity, isg_velocity, MAT_INITIAL_MATRIX,
                     &uu);
  MatCreateSubMatrix(__mat, isg_pressure, isg_velocity, MAT_INITIAL_MATRIX,
                     &pu);
  MatCreateSubMatrix(__mat, isg_velocity, isg_pressure, MAT_INITIAL_MATRIX,
                     &up);
  MatCreateSubMatrix(__mat, isg_pressure, isg_pressure, MAT_INITIAL_MATRIX,
                     &pp);

  Vec diag;

  MatCreateVecs(ff, &diag, NULL);
  MatGetDiagonal(ff, diag);
  VecReciprocal(diag);
  MatDiagonalScale(fr, diag, NULL);
  MatMatMult(rf, fr, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &fr_s);
  MatScale(fr_s, -1.0);

  MatCreateVecs(uu, &diag, NULL);
  MatGetDiagonal(uu, diag);
  VecReciprocal(diag);
  MatDiagonalScale(up, diag, NULL);
  MatMatMult(pu, up, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &up_s);
  MatScale(up_s, -1.0);
  MatAXPY(up_s, 1.0, pp, DIFFERENT_NONZERO_PATTERN);

  Vec _rhs, _x;
  VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, rhs.size(), PETSC_DECIDE,
                        rhs.data(), &_rhs);
  VecDuplicate(_rhs, &_x);

  KSP _ksp;
  KSPCreate(PETSC_COMM_WORLD, &_ksp);
  KSPSetOperators(_ksp, __mat, __mat);
  KSPSetFromOptions(_ksp);

  PC _pc;

  KSPGetPC(_ksp, &_pc);
  PCFieldSplitSetIS(_pc, "0", isg_field);
  PCFieldSplitSetIS(_pc, "1", isg_rigid);

  PCFieldSplitSetSchurPre(_pc, PC_FIELDSPLIT_SCHUR_PRE_USER, fr_s);

  KSP *_subKsp;
  PetscInt n = 1;
  PCSetUp(_pc);
  PCFieldSplitGetSubKSP(_pc, &n, &_subKsp);
  KSPSetOperators(_subKsp[1], fr_s, fr_s);
  KSPSetFromOptions(_subKsp[0]);
  KSPSetOperators(_subKsp[0], ff, ff);

  PC subpc;
  KSPGetPC(_subKsp[0], &subpc);

  PCFieldSplitSetIS(subpc, "0", isg_velocity);
  PCFieldSplitSetIS(subpc, "1", isg_pressure);

  PCFieldSplitSetSchurPre(subpc, PC_FIELDSPLIT_SCHUR_PRE_USER, up_s);

  // setup sub solver
  KSP *_subsubKsp;
  PCSetFromOptions(subpc);
  PCSetUp(subpc);
  PCFieldSplitGetSubKSP(subpc, &n, &_subsubKsp);
  KSPSetOperators(_subsubKsp[1], up_s, up_s);
  KSPSetFromOptions(_subsubKsp[0]);
  PetscFree(_subsubKsp);

  MPI_Barrier(MPI_COMM_WORLD);
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

  MatDestroy(&ff);
  MatDestroy(&fr);
  MatDestroy(&rf);
  MatDestroy(&fr_s);

  MatDestroy(&uu);
  MatDestroy(&up);
  MatDestroy(&pu);
  MatDestroy(&pp);
  MatDestroy(&up_s);

  ISDestroy(&isg_field);
  ISDestroy(&isg_rigid);
  ISDestroy(&isg_velocity);
  ISDestroy(&isg_pressure);
}

void Solve(PetscSparseMatrix &A, PetscSparseMatrix &Bt, PetscSparseMatrix &B,
           PetscSparseMatrix &C, vector<double> &f, vector<double> &g,
           vector<double> &x, vector<double> &y, int numRigid,
           int rigidBodyDof) {
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
  MatCreateVecs(_ASub[0], &_diag, NULL);

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
  KSPSetOperators(_subKsp[0], _ASub[0], _ASub[0]);

  int MPIsize, myId;
  MPI_Comm_rank(MPI_COMM_WORLD, &myId);
  MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

  vector<PetscInt> idx1, idx2;
  PetscInt localN1, localN2;
  MatGetOwnershipRange(_ASub[0], &localN1, &localN2);
  if (myId != MPIsize - 1) {
    idx1.resize(localN2 - localN1);
    idx2.resize(0);
    for (int i = 0; i < localN2 - localN1; i++) {
      idx1[i] = localN1 + i;
    }
  } else {
    idx1.resize(localN2 - localN1 - rigidBodyDof * numRigid);
    idx2.resize(rigidBodyDof * numRigid);

    for (int i = 0; i < localN2 - localN1 - rigidBodyDof * numRigid; i++) {
      idx1[i] = localN1 + i;
    }
    for (int i = 0; i < rigidBodyDof * numRigid; i++) {
      idx2[i] = localN2 - rigidBodyDof * numRigid + i;
    }
  }

  IS isg1, isg2;
  ISCreateGeneral(MPI_COMM_WORLD, idx1.size(), idx1.data(), PETSC_COPY_VALUES,
                  &isg1);
  ISCreateGeneral(MPI_COMM_WORLD, idx2.size(), idx2.data(), PETSC_COPY_VALUES,
                  &isg2);

  Mat sub_A, sub_Bt, sub_B, sub_S;

  MatCreateSubMatrix(_ASub[0], isg1, isg1, MAT_INITIAL_MATRIX, &sub_A);
  MatCreateSubMatrix(_ASub[0], isg1, isg2, MAT_INITIAL_MATRIX, &sub_Bt);
  MatCreateSubMatrix(_ASub[0], isg2, isg1, MAT_INITIAL_MATRIX, &sub_B);

  MatCreateVecs(sub_A, &_diag, NULL);

  MatGetDiagonal(sub_A, _diag);
  VecReciprocal(_diag);
  MatDiagonalScale(sub_Bt, _diag, NULL);

  MatMatMult(sub_B, sub_Bt, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &sub_S);

  MatScale(sub_S, -1.0);

  PC subpc;
  KSPGetPC(_subKsp[0], &subpc);

  PCFieldSplitSetIS(subpc, "0", isg1);
  PCFieldSplitSetIS(subpc, "1", isg2);

  PCFieldSplitSetSchurPre(subpc, PC_FIELDSPLIT_SCHUR_PRE_USER, sub_S);

  // setup sub solver
  KSP *_subsubKsp;
  PCSetFromOptions(subpc);
  PCSetUp(subpc);
  PCFieldSplitGetSubKSP(subpc, &n, &_subsubKsp);
  KSPSetOperators(_subsubKsp[1], sub_S, sub_S);
  KSPSetFromOptions(_subsubKsp[0]);
  PetscFree(_subsubKsp);

  // vector<PetscInt> idx3, idx4;
  // MatGetOwnershipRange(_ASub[4], &localN1, &localN2);
  // if (myId != MPIsize - 1) {
  //   idx3.resize(localN2 - localN1);
  //   idx4.resize(0);
  //   for (int i = 0; i < localN2 - localN1; i++) {
  //     idx3[i] = localN1 + i;
  //   }
  // } else {
  //   idx3.resize(localN2 - localN1 - 1);
  //   idx4.resize(1);

  //   for (int i = 0; i < localN2 - localN1 - 1; i++) {
  //     idx3[i] = localN1 + i;
  //   }
  //   idx4[0] = localN2 - 1;
  // }

  // IS isg3, isg4;
  // ISCreateGeneral(MPI_COMM_WORLD, idx3.size(), idx3.data(),
  // PETSC_COPY_VALUES,
  //                 &isg3);
  // ISCreateGeneral(MPI_COMM_WORLD, idx4.size(), idx4.data(),
  // PETSC_COPY_VALUES,
  //                 &isg4);

  // Mat S_A, S_Bt, S_B, S_S;

  // MatCreateSubMatrix(_ASub[4], isg3, isg3, MAT_INITIAL_MATRIX, &S_A);
  // MatCreateSubMatrix(_ASub[4], isg3, isg4, MAT_INITIAL_MATRIX, &S_Bt);
  // MatCreateSubMatrix(_ASub[4], isg4, isg3, MAT_INITIAL_MATRIX, &S_B);

  // MatCreateVecs(S_A, &_diag, NULL);

  // MatGetDiagonal(S_A, _diag);
  // VecReciprocal(_diag);
  // MatDiagonalScale(S_Bt, _diag, NULL);

  // MatMatMult(S_B, S_Bt, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &S_S);

  // MatScale(S_S, -1.0);

  // PC subpc_s;
  // KSPGetPC(_subKsp[1], &subpc_s);

  // PCFieldSplitSetIS(subpc_s, "0", isg3);
  // PCFieldSplitSetIS(subpc_s, "1", isg4);

  // PCFieldSplitSetSchurPre(subpc_s, PC_FIELDSPLIT_SCHUR_PRE_USER, S_S);

  // // setup sub solver
  // KSP *_subsubKsp_s;
  // PCSetFromOptions(subpc_s);
  // PCSetUp(subpc_s);
  // PCFieldSplitGetSubKSP(subpc_s, &n, &_subsubKsp_s);
  // KSPSetOperators(_subsubKsp_s[1], S_S, S_S);
  // KSPSetFromOptions(_subsubKsp_s[0]);
  // PetscFree(_subsubKsp_s);
  PetscFree(_subKsp);

  MPI_Barrier(MPI_COMM_WORLD);

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

  MatDestroy(&sub_A);
  MatDestroy(&sub_Bt);
  MatDestroy(&sub_B);
  MatDestroy(&sub_S);

  ISDestroy(&isg1);
  ISDestroy(&isg2);

  VecDestroy(&_diag);
}

void PetscSparseMatrix::Solve(vector<double> &rhs, vector<double> &x,
                              vector<int> &neighborInclusion, int dimension,
                              int numRigidBody) {
  int fieldDof = dimension + 1;
  int velocityDof = dimension;
  int pressureDof = 1;
  int rigidBodyDof = (dimension == 3) ? 6 : 3;

  vector<int> idx_field, idx_rigid;
  vector<int> idx_velocity, idx_pressure;

  int MPIsize, myId;
  MPI_Comm_rank(MPI_COMM_WORLD, &myId);
  MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

  PetscInt localN1, localN2;
  MatGetOwnershipRange(__mat, &localN1, &localN2);

  if (myId != MPIsize - 1) {
    int localParticleNum = (localN2 - localN1) / fieldDof;
    idx_field.resize(fieldDof * localParticleNum);
    idx_velocity.resize(velocityDof * localParticleNum);
    idx_pressure.resize(localParticleNum);

    for (int i = 0; i < localParticleNum; i++) {
      for (int j = 0; j < dimension; j++) {
        idx_field[fieldDof * i + j] = localN1 + fieldDof * i + j;
        idx_velocity[velocityDof * i + j] = localN1 + fieldDof * i + j;
      }
      idx_field[fieldDof * i + velocityDof] =
          localN1 + fieldDof * i + velocityDof;
      idx_pressure[i] = localN1 + fieldDof * i + velocityDof;
    }
  } else {
    int localParticleNum =
        (localN2 - localN1 - 1 - numRigidBody * rigidBodyDof) / fieldDof;
    idx_field.resize(fieldDof * localParticleNum + 1);
    idx_velocity.resize(velocityDof * localParticleNum);
    idx_pressure.resize(localParticleNum + 1);
    idx_rigid.resize(rigidBodyDof * numRigidBody);

    for (int i = 0; i < localParticleNum; i++) {
      for (int j = 0; j < dimension; j++) {
        idx_field[fieldDof * i + j] = localN1 + fieldDof * i + j;
        idx_velocity[velocityDof * i + j] = localN1 + fieldDof * i + j;
      }
      idx_field[fieldDof * i + velocityDof] =
          localN1 + fieldDof * i + velocityDof;
      idx_pressure[i] = localN1 + fieldDof * i + velocityDof;
    }

    idx_field[fieldDof * localParticleNum] =
        localN1 + fieldDof * localParticleNum;
    idx_pressure[localParticleNum] = localN1 + fieldDof * localParticleNum;

    for (int i = 0; i < numRigidBody; i++) {
      for (int j = 0; j < rigidBodyDof; j++) {
        idx_rigid[rigidBodyDof * i + j] =
            localN1 + fieldDof * localParticleNum + 1 + i * rigidBodyDof + j;
      }
    }
  }

  vector<PetscInt> neighbor_idx;
  for (int i = 0; i < neighborInclusion.size(); i++) {
    if (neighborInclusion[i] >= localN1 && neighborInclusion[i] < localN2) {
      neighbor_idx.push_back(neighborInclusion[i]);
    }
  }

  IS isg_field, isg_neighbor;
  IS isg_velocity, isg_pressure;

  ISCreateGeneral(MPI_COMM_WORLD, idx_velocity.size(), idx_velocity.data(),
                  PETSC_COPY_VALUES, &isg_velocity);
  ISCreateGeneral(MPI_COMM_WORLD, idx_pressure.size(), idx_pressure.data(),
                  PETSC_COPY_VALUES, &isg_pressure);
  ISCreateGeneral(MPI_COMM_WORLD, idx_field.size(), idx_field.data(),
                  PETSC_COPY_VALUES, &isg_field);
  ISCreateGeneral(MPI_COMM_WORLD, neighbor_idx.size(), neighbor_idx.data(),
                  PETSC_COPY_VALUES, &isg_neighbor);

  Mat ff, nn;

  Mat uu, up, pu, pp, up_s;

  MatCreateSubMatrix(__mat, isg_field, isg_field, MAT_INITIAL_MATRIX, &ff);
  MatCreateSubMatrix(__mat, isg_neighbor, isg_neighbor, MAT_INITIAL_MATRIX,
                     &nn);

  MatCreateSubMatrix(__mat, isg_velocity, isg_velocity, MAT_INITIAL_MATRIX,
                     &uu);
  MatCreateSubMatrix(__mat, isg_pressure, isg_velocity, MAT_INITIAL_MATRIX,
                     &pu);
  MatCreateSubMatrix(__mat, isg_velocity, isg_pressure, MAT_INITIAL_MATRIX,
                     &up);
  MatCreateSubMatrix(__mat, isg_pressure, isg_pressure, MAT_INITIAL_MATRIX,
                     &pp);

  Vec diag;

  MatCreateVecs(uu, &diag, NULL);
  MatGetDiagonal(uu, diag);
  VecReciprocal(diag);
  MatDiagonalScale(up, diag, NULL);
  MatMatMult(pu, up, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &up_s);
  MatScale(up_s, -1.0);
  MatAXPY(up_s, 1.0, pp, DIFFERENT_NONZERO_PATTERN);

  Vec _rhs, _x;
  VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, rhs.size(), PETSC_DECIDE,
                        rhs.data(), &_rhs);
  VecDuplicate(_rhs, &_x);

  KSP _ksp;
  KSPCreate(PETSC_COMM_WORLD, &_ksp);
  KSPSetOperators(_ksp, __mat, __mat);
  KSPSetFromOptions(_ksp);

  PC _pc;

  KSPGetPC(_ksp, &_pc);
  PCSetType(_pc, PCSHELL);

  HypreLUShellPC *shell;

  HypreLUShellPCCreate(&shell);

  PCShellSetApply(_pc, HypreLUShellPCApply);
  PCShellSetContext(_pc, shell);
  PCShellSetDestroy(_pc, HypreLUShellPCDestroy);

  HypreLUShellPCSetUp(_pc, &__mat, &ff, &nn, &isg_field, &isg_neighbor,
                      &isg_velocity, &isg_pressure, &up_s, _rhs);

  MPI_Barrier(MPI_COMM_WORLD);
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
  VecDestroy(&diag);

  MatDestroy(&ff);
  MatDestroy(&nn);

  MatDestroy(&uu);
  MatDestroy(&up);
  MatDestroy(&pu);
  MatDestroy(&pp);
  MatDestroy(&up_s);

  ISDestroy(&isg_field);
  ISDestroy(&isg_velocity);
  ISDestroy(&isg_pressure);
  ISDestroy(&isg_neighbor);
}