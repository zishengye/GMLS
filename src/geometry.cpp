#include "geometry.hpp"
#include "domain_decomposition.hpp"

#include <iostream>

using namespace std;

void geometry::set_local_bounding_box_boundary() {
  if (_dimension == 2) {
    // four eages as boundaries
    // 0 down edge
    // 1 right edge
    // 2 up edge
    // 3 left edge
    _local_bounding_box_boundary_type.resize(4);
    if (abs(_local_bounding_box[0][0] - _global_bounding_box[0][0]) < 1e-6) {
      _local_bounding_box_boundary_type[3] = 1;
    } else {
      _local_bounding_box_boundary_type[3] = 0;
    }

    if (abs(_local_bounding_box[0][1] - _global_bounding_box[0][1]) < 1e-6) {
      _local_bounding_box_boundary_type[0] = 1;
    } else {
      _local_bounding_box_boundary_type[0] = 0;
    }

    if (abs(_local_bounding_box[1][0] - _global_bounding_box[1][0]) < 1e-6) {
      _local_bounding_box_boundary_type[1] = 1;
    } else {
      _local_bounding_box_boundary_type[1] = 0;
    }

    if (abs(_local_bounding_box[1][1] - _global_bounding_box[1][1]) < 1e-6) {
      _local_bounding_box_boundary_type[2] = 1;
    } else {
      _local_bounding_box_boundary_type[2] = 0;
    }
  }
  if (_dimension == 3) {
  }
}

void geometry::init_nonmanifold() {
  process_split(_global_block_size, _local_block_coord, _mpi_size, _id,
                _dimension);
}

void geometry::init_manifold() {}

void geometry::update_nonmanifold() {
  for (int i = 0; i < _particle_set.size(); i++) {
    delete _particle_set[i];
  }
  for (int i = 0; i < _background_particle_set.size(); i++) {
    delete _background_particle_set[i];
  }

  _particle_set.clear();
  _background_particle_set.clear();
  _gap_particle_set.clear();

  vec3 current_particle_size;
  triple<int> current_global_particle_num = _global_particle_num;
  triple<int> current_local_particle_num = _local_particle_num;

  int current_particle_level = 0;

  do {
    bounding_box_split(_global_bounding_box, _local_bounding_box,
                       current_global_particle_num, current_local_particle_num,
                       current_particle_size, _global_block_size,
                       _local_block_coord, _dimension);
    set_local_bounding_box_boundary();

    std::vector<particle> *current_particle_set;
    current_particle_set = new std::vector<particle>();

    update_uniform_nonmanifold(*current_particle_set, current_particle_size,
                               current_particle_level);

    size_t global_particle_count = global_indexing(*current_particle_set);

    _particle_set.insert(_particle_set.begin(), current_particle_set);
    _global_particle_count.insert(_global_particle_count.begin(),
                                  global_particle_count);

    for (int i = 0; i < 3; i++) {
      current_global_particle_num[i] /= 2;
    }

    current_particle_level--;
  } while (current_local_particle_num[0] > _local_particle_num_min[0]);
}

void geometry::update_manifold() {}

size_t geometry::global_indexing(std::vector<particle> &current_particle_set) {
  size_t local_particle_count = current_particle_set.size();
  size_t global_particle_count;
  MPI_Allreduce(&local_particle_count, &global_particle_count, 1,
                MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

  vector<size_t> particle_offset;
  particle_offset.resize(_mpi_size + 1);

  MPI_Allgather(&local_particle_count, 1, MPI_UNSIGNED_LONG,
                particle_offset.data() + 1, 1, MPI_UNSIGNED_LONG,
                MPI_COMM_WORLD);

  particle_offset[0];
  for (int i = 0; i < _mpi_size; i++) {
    particle_offset[i + 1] += particle_offset[i];
  }

  for (size_t i = 0; i < current_particle_set.size(); i++) {
    current_particle_set[i].global_index = particle_offset[_id] + i;
  }

  return global_particle_count;
}

void geometry::update_uniform_nonmanifold(
    std::vector<particle> &current_particle_set, vec3 &current_particle_size,
    int particle_level) {
  particle current_particle;
  current_particle.particle_adaptive_level = particle_level;
  current_particle.volume = current_particle_size[0] * current_particle_size[1];
  current_particle.particle_size = current_particle_size;

  bool add_gap_particle = (particle_level < 0) ? false : true;

  const double tolerance = 1e-5;

  double x_pos, y_pos, z_pos;
  if (_dimension == 2) {
    z_pos = 0.0;

    if (_local_bounding_box_boundary_type[0] != 0) {
      x_pos = _local_bounding_box[0][0];
      y_pos = _local_bounding_box[0][1];
      if (_local_bounding_box_boundary_type[3] != 0) {
        current_particle.coord = vec3(x_pos, y_pos, z_pos);
        current_particle.normal = vec3(sqrt(2) / 2.0, sqrt(2) / 2.0, 0.0);
        current_particle.particle_type = 1;

        current_particle_set.push_back(current_particle);
      }
      x_pos = _local_bounding_box[0][0] + 0.5 * current_particle_size[0];

      while (x_pos < _local_bounding_box[1][0] - tolerance) {
        current_particle.coord = vec3(x_pos, y_pos, z_pos);
        current_particle.normal = vec3(0.0, 1.0, 0.0);
        current_particle.particle_type = 2;

        current_particle_set.push_back(current_particle);

        x_pos += current_particle_size[0];
      }

      if (_local_bounding_box_boundary_type[1] != 0) {
        x_pos = _local_bounding_box[1][0];
        current_particle.coord = vec3(x_pos, y_pos, z_pos);
        current_particle.normal = vec3(-sqrt(2) / 2.0, sqrt(2) / 2.0, 0.0);
        current_particle.particle_type = 1;

        current_particle_set.push_back(current_particle);
      }
    }

    y_pos = _local_bounding_box[0][1] + 0.5 * current_particle_size[1];
    while (y_pos < _local_bounding_box[1][1] - tolerance) {
      if (_local_bounding_box_boundary_type[3] != 0) {
        x_pos = _local_bounding_box[0][0];
        current_particle.coord = vec3(x_pos, y_pos, z_pos);
        current_particle.normal = vec3(1.0, 0.0, 0.0);
        current_particle.particle_type = 2;

        current_particle_set.push_back(current_particle);
      }

      x_pos = _local_bounding_box[0][0] + 0.5 * current_particle_size[0];
      while (x_pos < _local_bounding_box[1][0] - tolerance) {
        current_particle.coord = vec3(x_pos, y_pos, z_pos);
        current_particle.normal = vec3(1.0, 0.0, 0.0);
        current_particle.particle_type = 0;

        current_particle_set.push_back(current_particle);

        x_pos += current_particle_size[0];
      }

      if (_local_bounding_box_boundary_type[1] != 0) {
        x_pos = _local_bounding_box[1][0];
        current_particle.coord = vec3(x_pos, y_pos, z_pos);
        current_particle.normal = vec3(-1.0, 0.0, 0.0);
        current_particle.particle_type = 2;

        current_particle_set.push_back(current_particle);
      }

      y_pos += current_particle_size[1];
    }

    if (_local_bounding_box_boundary_type[2] != 0) {
      x_pos = _local_bounding_box[0][0];
      y_pos = _local_bounding_box[1][1];
      if (_local_bounding_box_boundary_type[3] != 0) {
        current_particle.coord = vec3(x_pos, y_pos, z_pos);
        current_particle.normal = vec3(sqrt(2) / 2.0, -sqrt(2) / 2.0, 0.0);
        current_particle.particle_type = 1;

        current_particle_set.push_back(current_particle);
      }
      x_pos = _local_bounding_box[0][0] + 0.5 * current_particle_size[0];

      while (x_pos < _local_bounding_box[1][0] - tolerance) {
        current_particle.coord = vec3(x_pos, y_pos, z_pos);
        current_particle.normal = vec3(0.0, -1.0, 0.0);
        current_particle.particle_type = 2;

        current_particle_set.push_back(current_particle);

        x_pos += current_particle_size[0];
      }

      x_pos = _local_bounding_box[1][0];
      if (_local_bounding_box_boundary_type[1] != 0) {
        current_particle.coord = vec3(x_pos, y_pos, z_pos);
        current_particle.normal = vec3(-sqrt(2) / 2.0, -sqrt(2) / 2.0, 0.0);
        current_particle.particle_type = 1;

        current_particle_set.push_back(current_particle);
      }
    }
  }
  if (_dimension == 3) {
  }
}

#include <fstream>
#include <string>

using namespace std;

void geometry::write_all_init_level(std::string output_filename_prefix) {
  if (_manifold_order != 0) {
  } else {
    for (int layer = 0; layer < _particle_set.size(); layer++) {
      if (_id == 0) {
        ofstream file;
        file.open("./vtk/" + output_filename_prefix + to_string(layer) + ".vtk",
                  ios::trunc);
        file << "# vtk DataFile Version 2.0" << endl;
        file << "particlePositions" << endl;
        file << "ASCII" << endl;
        file << "DATASET POLYDATA " << endl;
        file << " POINTS " << _global_particle_count[layer] << " float" << endl;
        file.close();
      }

      for (int id = 0; id < _mpi_size; id++) {
        if (id == _id) {
          ofstream file;
          file.open("./vtk/" + output_filename_prefix + to_string(layer) +
                        ".vtk",
                    ios::app);
          std::vector<particle> &current_particle_set = *_particle_set[layer];
          for (size_t i = 0; i < current_particle_set.size(); i++) {
            file << current_particle_set[i].coord[0] << ' '
                 << current_particle_set[i].coord[1] << ' '
                 << current_particle_set[i].coord[2] << endl;
          }
          file.close();
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }

      if (_id == 0) {
        ofstream file;
        file.open("./vtk/" + output_filename_prefix + to_string(layer) + ".vtk",
                  ios::app);
        file << "POINT_DATA " << _global_particle_count[layer] << endl;
        file.close();
      }

      // particle type
      if (_id == 0) {
        ofstream file;
        file.open("./vtk/" + output_filename_prefix + to_string(layer) + ".vtk",
                  ios::app);
        file << "SCALARS ID int 1" << endl;
        file << "LOOKUP_TABLE default" << endl;
        file.close();
      }

      for (int id = 0; id < _mpi_size; id++) {
        if (id == _id) {
          ofstream file;
          file.open("./vtk/" + output_filename_prefix + to_string(layer) +
                        ".vtk",
                    ios::app);
          std::vector<particle> &current_particle_set = *_particle_set[layer];
          for (size_t i = 0; i < current_particle_set.size(); i++) {
            file << current_particle_set[i].particle_type << endl;
          }
          file.close();
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }

      // particle size
      if (_id == 0) {
        ofstream file;
        file.open("./vtk/" + output_filename_prefix + to_string(layer) + ".vtk",
                  ios::app);
        file << "SCALARS d float 1" << endl;
        file << "LOOKUP_TABLE default" << endl;
        file.close();
      }

      for (int id = 0; id < _mpi_size; id++) {
        if (id == _id) {
          ofstream file;
          file.open("./vtk/" + output_filename_prefix + to_string(layer) +
                        ".vtk",
                    ios::app);
          std::vector<particle> &current_particle_set = *_particle_set[layer];
          for (size_t i = 0; i < current_particle_set.size(); i++) {
            file << current_particle_set[i].particle_size[0] << endl;
          }
          file.close();
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }

      // normal
      if (_id == 0) {
        ofstream file;
        file.open("./vtk/" + output_filename_prefix + to_string(layer) + ".vtk",
                  ios::app);
        file << "SCALARS n float 3" << endl;
        file << "LOOKUP_TABLE default" << endl;
        file.close();
      }

      for (int id = 0; id < _mpi_size; id++) {
        if (id == _id) {
          ofstream file;
          file.open("./vtk/" + output_filename_prefix + to_string(layer) +
                        ".vtk",
                    ios::app);
          std::vector<particle> &current_particle_set = *_particle_set[layer];
          for (size_t i = 0; i < current_particle_set.size(); i++) {
            file << current_particle_set[i].normal[0] << ' '
                 << current_particle_set[i].normal[1] << ' '
                 << current_particle_set[i].normal[2] << endl;
          }
          file.close();
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
}