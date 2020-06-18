#ifndef _MULTILEVEL_H_
#define _MULTILEVEL_H_

#include <vector>

#include "sparse_matrix.h"

class multilevel {
private:
  std::vector<PetscSparseMatrix *> A_list;    // coefficient matrix list
  std::vector<PetscSparseMatrix *> I_list;    // interpolation matrix list
  std::vector<PetscSparseMatrix *> R_list;    // restriction matrix list
  std::vector<Mat *> ff_list;                 // field sub-matrix list
  std::vector<Mat *> nn_list;                 // nearfield sub-matrix list
  std::vector<Mat *> nw_list;                 // nearfield-whole sub-matrix list
  std::vector<KSP *> ksp_list;                // main ksp list
  std::vector<KSP *> field_smoother_ksp_list; // field value smoother ksp list
  std::vector<IS *> isg_field_list;
  std::vector<IS *> isg_neighbor_list;
  std::vector<IS *> isg_pressure_list;

  // vector list
  std::vector<Vec *> x_list;
  std::vector<Vec *> y_list;
  std::vector<Vec *> b_list;
  std::vector<Vec *> r_list;
  std::vector<Vec *> t_list;

  std::vector<Vec *> x_field_list;
  std::vector<Vec *> y_field_list;
  std::vector<Vec *> b_field_list;
  std::vector<Vec *> r_field_list;
  std::vector<Vec *> t_field_list;

  std::vector<Vec *> x_neighbor_list;
  std::vector<Vec *> y_neighbor_list;
  std::vector<Vec *> b_neighbor_list;
  std::vector<Vec *> r_neighbor_list;
  std::vector<Vec *> t_neighbor_list;

  std::vector<Vec *> x_pressure_list;

  std::vector<VecScatter *> field_scatter_list;
  std::vector<VecScatter *> neighbor_scatter_list;
  std::vector<VecScatter *> pressure_scatter_list;

  Vec x_neighbor, y_neighbor;

  // relaxation list
  std::vector<KSP *> field_relaxation_list;
  std::vector<KSP *> neighbor_relaxation_list;

  KSP ksp_field_base, ksp_neighbor_base;

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

  inline int get_dimension() { return dimension; }

  inline int get_num_rigid_body() { return num_rigid_body; }

  PetscSparseMatrix &getA(int num_level) { return *A_list[num_level]; }
  PetscSparseMatrix &getI(int num_level) { return *I_list[num_level]; }
  PetscSparseMatrix &getR(int num_level) { return *R_list[num_level]; }
  KSP &getKsp(int num_level) { return *ksp_list[num_level]; }
  KSP &getSmootherKsp(int num_level) {
    return *field_smoother_ksp_list[num_level];
  }
  KSP &getFieldRelaxation(int num_level) {
    return *field_relaxation_list[num_level];
  }
  KSP &getNeighborRelaxation(int num_level) {
    return *neighbor_relaxation_list[num_level];
  }
  KSP &getFieldBase() { return ksp_field_base; }
  KSP &getNeighborBase() { return ksp_neighbor_base; }

  Mat &getFieldMat(int num_level) { return *ff_list[num_level]; }
  Mat &getNeighborWholeMat(int num_level) { return *nw_list[num_level]; }

  Vec *getXNeighbor() { return &x_neighbor; }
  Vec *getYNeighbor() { return &y_neighbor; }

  void add_new_level() {
    A_list.push_back(new PetscSparseMatrix());
    I_list.push_back(new PetscSparseMatrix());
    R_list.push_back(new PetscSparseMatrix());

    ksp_list.push_back(new KSP);
    field_smoother_ksp_list.push_back(new KSP);

    isg_field_list.push_back(new IS);
    isg_neighbor_list.push_back(new IS);
    isg_pressure_list.push_back(new IS);

    ff_list.push_back(new Mat);
    nn_list.push_back(new Mat);
    nw_list.push_back(new Mat);
  }

  void clear() {
    MPI_Barrier(MPI_COMM_WORLD);
    A_list.clear();
    I_list.clear();
    R_list.clear();
  }

  void InitialGuessFromPreviousAdaptiveStep(std::vector<double> &initial_guess);

  std::vector<PetscSparseMatrix *> *GetInterpolationList() { return &I_list; }
  std::vector<PetscSparseMatrix *> *GetRestrictionList() { return &R_list; }

  std::vector<Vec *> *GetXList() { return &x_list; }
  std::vector<Vec *> *GetYList() { return &y_list; }
  std::vector<Vec *> *GetBList() { return &b_list; }
  std::vector<Vec *> *GetRList() { return &r_list; }
  std::vector<Vec *> *GetTList() { return &t_list; }

  std::vector<Vec *> *GetXFieldList() { return &x_field_list; }
  std::vector<Vec *> *GetYFieldList() { return &y_field_list; }
  std::vector<Vec *> *GetBFieldList() { return &b_field_list; }
  std::vector<Vec *> *GetRFieldList() { return &r_field_list; }
  std::vector<Vec *> *GetTFieldList() { return &t_field_list; }

  std::vector<Vec *> *GetXNeighborList() { return &x_neighbor_list; }
  std::vector<Vec *> *GetYNeighborList() { return &y_neighbor_list; }
  std::vector<Vec *> *GetBNeighborList() { return &b_neighbor_list; }
  std::vector<Vec *> *GetRNeighborList() { return &r_neighbor_list; }
  std::vector<Vec *> *GetTNeighborList() { return &t_neighbor_list; }

  std::vector<Vec *> *GetXPressureList() { return &x_pressure_list; }

  std::vector<VecScatter *> *GetFieldScatterList() {
    return &field_scatter_list;
  }

  std::vector<VecScatter *> *GetNeighborScatterList() {
    return &neighbor_scatter_list;
  }

  std::vector<VecScatter *> *GetPressureScatterList() {
    return &pressure_scatter_list;
  }

  void Solve(std::vector<double> &rhs, std::vector<double> &x,
             std::vector<int> &idx_neighbor);
};

#endif