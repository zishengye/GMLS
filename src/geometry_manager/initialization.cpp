#include "geometry_manager.hpp"

#include <cmath>
#include <iostream>

using namespace std;

void geometry_manager::initialization() {
  // init domain decompsition
  int proc_i = 0, proc_j = 0, proc_k = 0;
  int proc_x = 0, proc_y = 0, proc_z = 0;

  int proc_num = _net->get_proc_num();
  int id = _net->get_id();

  if (_dimension == 1) {
  }

  if (_dimension == 2) {
    proc_x = sqrt(proc_num);
    while (proc_x > 0) {
      proc_y = proc_num / proc_x;
      if (proc_num == proc_x * proc_y) {
        break;
      } else {
        proc_x--;
      }
    }

    proc_i = id % proc_x;
    proc_j = id % proc_x;
  }

  if (_dimension == 3) {
  }

  // get domain range
  std::vector<int> count_x;
  std::vector<int> count_y;
  std::vector<int> count_z;

  count_x.resize(proc_x);
  for (int i = 0; i < proc_x; i++) {
    count_x[i] = (_nx % proc_x > i) ? (_nx / proc_x + 1) : (_nx / proc_x);
  }

  if (_dimension > 1) {
    count_y.resize(proc_y);
    for (int i = 0; i < proc_y; i++) {
      count_y[i] = (_ny % proc_y > i) ? (_ny / proc_y + 1) : (_ny / proc_y);
    }
  }

  if (_dimension > 2) {
    count_z.resize(proc_z);
    for (int i = 0; i < proc_z; i++) {
      count_z[i] = (_nz % proc_z > i) ? (_nz / proc_z + 1) : (_nz / proc_z);
    }
  }

  vec3 domain_count;
  domain_count[0] = count_x[proc_i];
  if (_dimension > 1)
    domain_count[1] = count_y[proc_j];
  if (_dimension > 2)
    domain_count[2] = count_z[proc_k];

  double particle_size = 2.0 / _nx;

  vec3 bounding_box[2];

  bounding_box[0][0] = -1.0;
  bounding_box[0][1] = (_dimension > 1) ? -1.0 : 0.0;
  bounding_box[0][2] = (_dimension > 2) ? -1.0 : 0.0;
  bounding_box[1][0] = 1.0;
  bounding_box[1][1] = (_dimension > 1) ? 1.0 : 0.0;
  bounding_box[1][2] = (_dimension > 2) ? 1.0 : 0.0;

  double x_start = bounding_box[0][0];
  double y_start = bounding_box[0][1];
  double z_start = bounding_box[0][2];
  for (int i = 0; i < proc_i; i++) {
    x_start += count_x[i] * particle_size;
  }
  for (int i = 0; i < proc_j; i++) {
    y_start += count_y[i] * particle_size;
  }
  for (int i = 0; i < proc_k; i++) {
    z_start += count_z[i] * particle_size;
  }

  double x_end = x_start + count_x[proc_i] * particle_size;
  double y_end =
      (_dimension > 1) ? (y_start + count_y[proc_j] * particle_size) : (0.0);
  double z_end =
      (_dimension > 2) ? (z_start + count_z[proc_k] * particle_size) : (0.0);

  _domain[0][0] = x_start;
  _domain[0][1] = y_start;
  _domain[0][2] = z_start;
  _domain[1][0] = x_end;
  _domain[1][1] = y_end;
  _domain[1][2] = z_end;

  _domain_boundary_type.resize(2 * _dimension);
  _domain_boundary_type[0] = (proc_i == 0) ? 1 : 0;
  _domain_boundary_type[1] = (proc_i == proc_x - 1) ? 1 : 0;
  if (_dimension > 1) {
    _domain_boundary_type[2] = (proc_j == 0) ? 1 : 0;
    _domain_boundary_type[3] = (proc_j == proc_y - 1) ? 1 : 0;
  }
  if (_dimension > 2) {
    _domain_boundary_type[4] = (proc_k == 0) ? 1 : 0;
    _domain_boundary_type[5] = (proc_k == proc_z - 1) ? 1 : 0;
  }

  // init base
  vector<double> x_pos, y_pos, z_pos;
  vector<int> x_flag, y_flag, z_flag;

  if (_domain_boundary_type[0] == 1) {
    x_pos.push_back(_domain[0][0]);
    x_flag.push_back(1);
  }
  for (int i = 0; i < count_x[proc_i]; i++) {
    x_pos.push_back(_domain[0][0] + (i + 0.5) * particle_size);
    x_flag.push_back(0);
  }
  if (_domain_boundary_type[1] == 1) {
    x_pos.push_back(_domain[1][0]);
    x_flag.push_back(-1);
  }

  if (_dimension > 1) {
    if (_domain_boundary_type[2] == 1) {
      y_pos.push_back(_domain[0][1]);
      y_flag.push_back(1);
    }
    for (int i = 0; i < count_y[proc_j]; i++) {
      y_pos.push_back(_domain[0][1] + (i + 0.5) * particle_size);
      y_flag.push_back(0);
    }
    if (_domain_boundary_type[3] == 1) {
      y_pos.push_back(_domain[1][1]);
      y_flag.push_back(-1);
    }
  } else {
    y_pos.push_back(0.0);
    y_flag.push_back(0);
  }

  if (_dimension > 2) {
    if (_domain_boundary_type[4] == 1) {
      z_pos.push_back(_domain[0][2]);
      z_flag.push_back(1);
    }
    for (int i = 0; i < count_z[proc_k]; i++) {
      z_pos.push_back(_domain[0][2] + (i + 0.5) * particle_size);
      z_flag.push_back(0);
    }
    if (_domain_boundary_type[5] == 1) {
      z_pos.push_back(_domain[1][2]);
      z_flag.push_back(-1);
    }
  } else {
    z_pos.push_back(0.0);
    z_flag.push_back(-1);
  }

  for (size_t i = 0; i < x_pos.size(); i++) {
    for (size_t j = 0; j < y_pos.size(); j++) {
      for (size_t k = 0; k < z_pos.size(); k++) {
        double x = x_pos[i];
        double y = y_pos[j];
        double z = z_pos[k];

        vec3 normal;

        int boundary_counter = 0;
        if (x_flag[i] != 0) {
          boundary_counter++;
          normal[0] = x_flag[i];
        }
        if (y_flag[j] != 0) {
          boundary_counter++;
          normal[1] = y_flag[j];
        }
        if (z_flag[k] != 0) {
          boundary_counter++;
          normal[2] = z_flag[k];
        }

        normal *= (1.0 / sqrt((double)boundary_counter));

        insert_base_particle(vec3(x, y, z), normal, boundary_counter,
                             particle_size);
      }
    }
  }

  int particle_count = _base.size();
  MPI_Allreduce(MPI_IN_PLACE, &particle_count, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
}