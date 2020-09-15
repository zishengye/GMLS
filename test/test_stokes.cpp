#include "stokes_equation.hpp"

using namespace std;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  PetscInitialize(&argc, &argv, NULL, NULL);

  Kokkos::initialize(argc, argv);

  shared_ptr<geometry> geo_2d(new geometry);

  stokes_equation stokes;
  stokes.set_dimension(2);
  stokes.add_geometry(geo_2d);

  geo_2d->set_dimension(2);
  geo_2d->set_global_x_particle_num(400);
  geo_2d->set_global_y_particle_num(400);
  geo_2d->set_local_x_particle_num_min(25);
  geo_2d->set_local_y_particle_num_min(25);
  geo_2d->init();
  geo_2d->update();

  stokes.build_coarse_level_matrix();
  stokes.build_interpolation_restriction_operators();

  // Vec x, y;
  // MatCreateVecs(stokes.get_interpolation(0)->__mat, &x, &y);

  // auto coarse = geo_2d->get_particle_set(0);
  // auto fine = geo_2d->get_particle_set(1);

  // int field_dof = 3;
  // PetscReal *a;
  // VecGetArray(x, &a);
  // for (int i = 0; i < coarse->size(); i++) {
  //   double x = (*coarse)[i].coord[0];
  //   double y = (*coarse)[i].coord[1];
  //   a[field_dof * i] = cos(M_PI * x) * sin(M_PI * y);
  //   a[field_dof * i + 1] = -sin(M_PI * x) * cos(M_PI * y);
  //   a[field_dof * i + 2] = -cos(2 * M_PI * x) - cos(2 * M_PI * y);
  // }
  // VecRestoreArray(x, &a);

  // MatMult(stokes.get_interpolation(0)->__mat, x, y);

  // double error_velocity = 0.0;
  // VecGetArray(y, &a);
  // for (int i = 0; i < fine->size(); i++) {
  //   double xx = (*fine)[i].coord[0];
  //   double yy = (*fine)[i].coord[1];
  //   error_velocity +=
  //       pow(cos(M_PI * xx) * sin(M_PI * yy) - a[field_dof * i], 2);
  //   error_velocity +=
  //       pow(-sin(M_PI * xx) * cos(M_PI * yy) - a[field_dof * i + 1], 2);
  // }
  // VecRestoreArray(y, &a);

  // cout << error_velocity << endl;

  Vec x, y;
  MatCreateVecs(stokes.get_restriction(0)->__mat, &x, &y);

  auto coarse = geo_2d->get_particle_set(0);
  auto fine = geo_2d->get_particle_set(1);

  int field_dof = 3;
  PetscReal *a;
  VecGetArray(x, &a);
  for (int i = 0; i < fine->size(); i++) {
    double x = (*fine)[i].coord[0];
    double y = (*fine)[i].coord[1];
    a[field_dof * i] = cos(M_PI * x) * sin(M_PI * y);
    a[field_dof * i + 1] = -sin(M_PI * x) * cos(M_PI * y);
    a[field_dof * i + 2] = -cos(2 * M_PI * x) - cos(2 * M_PI * y);
  }
  VecRestoreArray(x, &a);

  MatMult(stokes.get_restriction(0)->__mat, x, y);

  double error_velocity = 0.0;
  VecGetArray(y, &a);
  for (int i = 0; i < coarse->size(); i++) {
    double xx = (*coarse)[i].coord[0];
    double yy = (*coarse)[i].coord[1];
    error_velocity +=
        pow(cos(M_PI * xx) * sin(M_PI * yy) - a[field_dof * i], 2);
    error_velocity +=
        pow(-sin(M_PI * xx) * cos(M_PI * yy) - a[field_dof * i + 1], 2);
  }
  VecRestoreArray(y, &a);

  cout << error_velocity << endl;

  stokes.clear();

  Kokkos::finalize();

  PetscFinalize();

  MPI_Finalize();

  return 0;
}