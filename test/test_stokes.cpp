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
  geo_2d->set_global_x_particle_num(100);
  geo_2d->set_global_y_particle_num(100);
  geo_2d->set_local_x_particle_num_min(25);
  geo_2d->set_local_y_particle_num_min(25);
  geo_2d->init();
  geo_2d->update();

  stokes.build_coarse_level_matrix();
  stokes.build_interpolation_restriction_operators();
  stokes.clear();

  Kokkos::finalize();

  PetscFinalize();

  MPI_Finalize();

  return 0;
}