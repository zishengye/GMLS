#ifndef _MULTILEVEL_H_
#define _MULTILEVEL_H_

#include <vector>

#include "sparse_matrix.h"

class multilevel {
private:
  std::vector<PetscSparseMatrix> A_list; // coefficient matrix list
  std::vector<PetscSparseMatrix> I_list; // interpolation matrix list

  int myid, mpi_size;

  int dimension, num_rigid_body;

  int field_dof, velocity_dof, pressure_dof;

public:
  multilevel() {}

  ~multilevel() {}

  void init(int _dimension) {
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    set_dimension(_dimension);
  }

  inline void set_dimension(int _dimension) { dimension = _dimension; }

  inline void set_num_rigid_body(int _num_rigid_body) {
    num_rigid_body = _num_rigid_body;
  }

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

  void InitialGuessFromPreviousAdaptiveStep(std::vector<double> &initial_guess) {}

  void Solve(std::vector<double> &rhs, std::vector<double> &x,
             std::vector<int> &idx_neighbor);
};

#endif