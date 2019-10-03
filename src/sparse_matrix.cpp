#include "sparse_matrix.h"

#include <fstream>
#include <string>

using namespace std;

int PetscSparseMatrix::FinalAssemble() {
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

  ofstream file("output.mat", ios::trunc);
  for (int i = 0; i < __row; i++) {
    for (list<entry>::iterator it = __matrix[i].begin();
         it != __matrix[i].end(); it++) {
      file << i << '\t' << it->first << '\t' << it->second << endl;
    }
  }
  file.close();

  MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, __row, __row, PETSC_DECIDE,
                            PETSC_DECIDE, __i.data(), __j.data(), __val.data(),
                            &__mat);

  __isAssembled = true;

  return __nnz;
}

void PetscSparseMatrix::Solve(vector<double> &rhs, vector<double> &x) {
  if (__isAssembled) {
    Vec _rhs, _x;
    VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, rhs.size(), PETSC_DECIDE,
                          rhs.data(), &_rhs);
    VecDuplicate(_rhs, &_x);

    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, __mat, __mat);
    KSPSetFromOptions(ksp);

    PetscPrintf(PETSC_COMM_WORLD, "final solving of linear system\n");
    KSPSolve(ksp, _rhs, _x);
    PetscPrintf(PETSC_COMM_WORLD, "ksp solving finished\n");

    KSPDestroy(&ksp);

    PetscScalar *a;
    VecGetArray(_x, &a);
    for (size_t i = 0; i < rhs.size(); i++) {
      x[i] = a[i];
    }

    VecDestroy(&_rhs);
    VecDestroy(&_x);
  }
}