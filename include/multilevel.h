#ifndef _MULTILEVEL_H_
#define _MULTILEVEL_H_

#include <vector>

#include "sparse_matrix.h"

class multilevel {
private:
  std::vector<PetscSparseMatrix> A_list; // coefficient matrix list
  std::vector<PetscSparseMatrix> I_list; // interpolation matrix list

public:
  multilevel() {}

  ~multilevel() { clear(); }

  PetscSparseMatrix &getA(int num_level) { return A_list[num_level]; }
  PetscSparseMatrix &getI(int num_level) { return I_list[num_level]; }

  void add_new_level() {
    A_list.push_back(std::move(PetscSparseMatrix()));
    I_list.push_back(std::move(PetscSparseMatrix()));
  }

  void clear() {
    MPI_Barrier(MPI_COMM_WORLD);
    A_list.clear();
    I_list.clear();
  }
};

#endif