#include "geometry.hpp"

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  geometry geo_2d;

  geo_2d.set_dimension(2);
  geo_2d.set_global_x_particle_num(100);
  geo_2d.set_global_y_particle_num(100);
  geo_2d.set_local_x_particle_num_min(25);
  geo_2d.set_local_y_particle_num_min(25);
  geo_2d.init();
  geo_2d.update();
  geo_2d.write_all_init_level("2d_test_result");

  geometry geo_3d;
  geo_3d.set_dimension(3);

  MPI_Finalize();

  return 0;
}