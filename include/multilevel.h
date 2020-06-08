#ifndef _MULTILEVEL_H_
#define _MULTILEVEL_H_

#include <vector>

#include "sparse_matrix.h"

class multilevel {
private:
  std::vector<PetscSparseMatrix *> A_list; // coefficient matrix list
  std::vector<PetscSparseMatrix *> I_list; // interpolation matrix list
  std::vector<PetscSparseMatrix *> R_list; // restriction matrix list
  std::vector<Mat *>
      ff_lag_list; // field with lagrange multiplier sub-matrix list
  std::vector<Mat *>
      ff_list; // field without lagrange multiplier sub-matrix list
  std::vector<Mat *> nn_list;                 // nearfield sub-matrix list
  std::vector<KSP *> ksp_list;                // main ksp list
  std::vector<KSP *> field_smoother_ksp_list; // field value smoother ksp list
  std::vector<IS *> isg_field_lag_list;
  std::vector<IS *> isg_field_list;
  std::vector<IS *> isg_neighbor_list;

  // vector list
  std::vector<Vec *> x_list;
  std::vector<Vec *> y_list;
  std::vector<Vec *> b_list;
  std::vector<Vec *> r_list;
  std::vector<Vec *> t_list;

  // relaxation list
  std::vector<KSP *> relaxation_list;

  KSP ksp_field_base;

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

  PetscSparseMatrix &getA(int num_level) { return *A_list[num_level]; }
  PetscSparseMatrix &getI(int num_level) { return *I_list[num_level]; }
  PetscSparseMatrix &getR(int num_level) { return *R_list[num_level]; }
  KSP &getKsp(int num_level) { return *ksp_list[num_level]; }
  KSP &getSmootherKsp(int num_level) {
    return *field_smoother_ksp_list[num_level];
  }
  KSP &getRelaxation(int num_level) { return *relaxation_list[num_level]; }

  Mat &getFieldMat(int num_level) { return *ff_lag_list[num_level]; }

  void add_new_level() {
    A_list.push_back(new PetscSparseMatrix());
    I_list.push_back(new PetscSparseMatrix());
    R_list.push_back(new PetscSparseMatrix());

    ksp_list.push_back(new KSP);
    field_smoother_ksp_list.push_back(new KSP);

    isg_field_lag_list.push_back(new IS);
    isg_field_list.push_back(new IS);
    isg_neighbor_list.push_back(new IS);

    ff_lag_list.push_back(new Mat);
    ff_list.push_back(new Mat);
    nn_list.push_back(new Mat);
  }

  void clear() {
    MPI_Barrier(MPI_COMM_WORLD);
    A_list.clear();
    I_list.clear();
    R_list.clear();

    for (int i = 1; i < ksp_list.size(); i++) {
      KSPDestroy(ksp_list[i]);
    }

    ksp_list.clear();
  }

  void InitialGuessFromPreviousAdaptiveStep(std::vector<double> &initial_guess);

  std::vector<PetscSparseMatrix *> *GetInterpolationList() { return &I_list; }
  std::vector<PetscSparseMatrix *> *GetRestrictionList() { return &R_list; }

  std::vector<Vec *> *GetXList() { return &x_list; }
  std::vector<Vec *> *GetYList() { return &y_list; }
  std::vector<Vec *> *GetBList() { return &b_list; }
  std::vector<Vec *> *GetRList() { return &r_list; }
  std::vector<Vec *> *GetTList() { return &t_list; }

  void Solve(std::vector<double> &rhs, std::vector<double> &x,
             std::vector<int> &idx_neighbor);
};

#endif