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

  setup_block_neighbor_nonmanifold();
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

  // build background
  for (size_t i = 0; i < _particle_set.size(); i++) {
    _background_particle_set.push_back(new vector<particle>());
    setup_background_nonmanifold(*_particle_set[i],
                                 *_background_particle_set[i]);
  }

  // build hierarchy
  int layer_num = _particle_set.size() - 1;
  _hierarchy.resize(layer_num);
  for (int i = 0; i < layer_num; i++) {
    setup_hierarchy_nonmanifold(*_particle_set[i], *_particle_set[i + 1],
                                _hierarchy[i]);
  }
}

void geometry::update_manifold() {}

void geometry::refine_nonmanifold(std::vector<int> &local_refinement_idnex) {}

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

void geometry::setup_background_nonmanifold(
    std::vector<particle> &current_particle_set,
    std::vector<particle> &background_particle_set) {
  background_particle_set = current_particle_set;
  double cutoff_distance = 1.5 * current_particle_set[0].particle_size[0];

  vector<bool> is_neighbor;
  int neighbor_num = pow(3, _dimension);
  is_neighbor.resize(neighbor_num);

  vector<vector<size_t>> neighbor_send_index;
  neighbor_send_index.resize(neighbor_num);

  for (size_t i = 0; i < current_particle_set.size(); i++) {
    vec3 pos = current_particle_set[i].coord;
    for (auto it = is_neighbor.begin(); it != is_neighbor.end(); it++)
      *it = false;

    if (_dimension == 2) {
      is_neighbor[0] = (pos[0] < _local_bounding_box[0][0] + cutoff_distance &&
                        pos[1] < _local_bounding_box[0][1] + cutoff_distance);

      is_neighbor[1] = (pos[1] < _local_bounding_box[0][1] + cutoff_distance);

      is_neighbor[2] = (pos[0] > _local_bounding_box[1][0] - cutoff_distance &&
                        pos[1] < _local_bounding_box[0][1] + cutoff_distance);

      is_neighbor[3] = (pos[0] < _local_bounding_box[0][0] + cutoff_distance);

      is_neighbor[5] = (pos[0] > _local_bounding_box[1][0] - cutoff_distance);

      is_neighbor[6] = (pos[0] < _local_bounding_box[0][0] + cutoff_distance &&
                        pos[1] > _local_bounding_box[1][1] - cutoff_distance);

      is_neighbor[7] = (pos[1] > _local_bounding_box[1][1] - cutoff_distance);

      is_neighbor[8] = (pos[0] > _local_bounding_box[1][0] - cutoff_distance &&
                        pos[1] > _local_bounding_box[1][1] - cutoff_distance);
    }

    for (size_t j = 0; j < is_neighbor.size(); j++) {
      if (is_neighbor[j] == true) {
        neighbor_send_index[j].push_back(i);
      }
    }
  }

  vector<int> destination_index;
  vector<int> send_count;
  vector<int> recv_offset;
  vector<int> recv_count;

  destination_index.resize(neighbor_num);
  send_count.resize(neighbor_num);
  recv_count.resize(neighbor_num);
  recv_offset.resize(neighbor_num + 1);

  MPI_Barrier(MPI_COMM_WORLD);

  int count;
  vector<MPI_Request> send_request;
  vector<MPI_Request> recv_request;
  vector<MPI_Status> status;

  send_request.resize(neighbor_num);
  recv_request.resize(neighbor_num);
  status.resize(neighbor_num);

  if (_dimension == 2) {
    int offset[3] = {-1, 0, 1};

    count = 0;

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        int index = i + j * 3;
        if (_neighbor_flag[index] == true) {
          int destination =
              (_local_block_coord[0] + offset[i]) +
              (_local_block_coord[1] + offset[j]) * _global_block_size[0];

          send_count[index] = neighbor_send_index[index].size();
          destination_index[index] = destination;

          MPI_Isend(send_count.data() + index, 1, MPI_INT,
                    destination_index[index], 0, MPI_COMM_WORLD,
                    send_request.data() + count);
          MPI_Irecv(recv_count.data() + index, 1, MPI_INT,
                    destination_index[index], 0, MPI_COMM_WORLD,
                    recv_request.data() + count);

          count++;
        }
      }
    }

    MPI_Waitall(count, send_request.data(), status.data());
    MPI_Waitall(count, recv_request.data(), status.data());
  }
  if (_dimension == 3) {
  }

  recv_offset[0] = 0;
  for (int i = 0; i < neighbor_num; i++) {
    recv_offset[i + 1] = recv_offset[i] + recv_count[i];
  }

  MPI_Barrier(MPI_COMM_WORLD);
  int total_neighbor_particle_num = recv_offset[neighbor_num];

  int local_particle_num = current_particle_set.size();

  background_particle_set.resize(local_particle_num +
                                 total_neighbor_particle_num);

  // swap background coordinate
  vector<vector<double>> send_coord_block(neighbor_num);
  vector<double> recv_coord(total_neighbor_particle_num * 3);

  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < neighbor_num; i++) {
    if (_neighbor_flag[i] == true) {
      send_coord_block[i].resize(send_count[i] * 3);

      for (int j = 0; j < send_count[i]; j++) {
        for (int k = 0; k < 3; k++) {
          send_coord_block[i][j * 3 + k] =
              current_particle_set[neighbor_send_index[i][j]].coord[k];
        }
      }
    }
  }

  count = 0;

  for (int i = 0; i < neighbor_num; i++) {
    if (_neighbor_flag[i] == true) {
      MPI_Isend(send_coord_block[i].data(), send_count[i] * 3, MPI_DOUBLE,
                destination_index[i], 0, MPI_COMM_WORLD,
                send_request.data() + count);
      MPI_Irecv(recv_coord.data() + recv_offset[i] * 3, recv_count[i] * 3,
                MPI_DOUBLE, destination_index[i], 0, MPI_COMM_WORLD,
                recv_request.data() + count);

      count++;
    }
  }

  MPI_Waitall(count, send_request.data(), status.data());
  MPI_Waitall(count, recv_request.data(), status.data());

  for (int i = 0; i < total_neighbor_particle_num; i++) {
    for (int j = 0; j < 3; j++)
      background_particle_set[local_particle_num + i].coord[j] =
          recv_coord[i * 3 + j];
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // swap background index
  vector<vector<size_t>> send_index_block(neighbor_num);
  vector<size_t> recv_index(total_neighbor_particle_num);

  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < neighbor_num; i++) {
    if (_neighbor_flag[i] == true) {
      send_index_block[i].resize(send_count[i]);

      for (int j = 0; j < send_count[i]; j++) {
        send_index_block[i][j] =
            current_particle_set[neighbor_send_index[i][j]].global_index;
      }
    }
  }

  count = 0;

  for (int i = 0; i < neighbor_num; i++) {
    if (_neighbor_flag[i] == true) {
      MPI_Isend(send_index_block[i].data(), send_count[i], MPI_UNSIGNED_LONG,
                destination_index[i], 0, MPI_COMM_WORLD,
                send_request.data() + count);
      MPI_Irecv(recv_index.data() + recv_offset[i], recv_count[i],
                MPI_UNSIGNED_LONG, destination_index[i], 0, MPI_COMM_WORLD,
                recv_request.data() + count);

      count++;
    }
  }

  MPI_Waitall(count, send_request.data(), status.data());
  MPI_Waitall(count, recv_request.data(), status.data());

  for (int i = 0; i < total_neighbor_particle_num; i++) {
    background_particle_set[local_particle_num + i].global_index =
        recv_index[i];
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // swap background particle type
  vector<vector<int>> send_particle_type_block(neighbor_num);
  vector<int> recv_particle_type(total_neighbor_particle_num);

  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < neighbor_num; i++) {
    if (_neighbor_flag[i] == true) {
      send_particle_type_block[i].resize(send_count[i]);

      for (int j = 0; j < send_count[i]; j++) {
        send_particle_type_block[i][j] =
            current_particle_set[neighbor_send_index[i][j]].global_index;
      }
    }
  }

  count = 0;

  for (int i = 0; i < neighbor_num; i++) {
    if (_neighbor_flag[i] == true) {
      MPI_Isend(send_particle_type_block[i].data(), send_count[i], MPI_INT,
                destination_index[i], 0, MPI_COMM_WORLD,
                send_request.data() + count);
      MPI_Irecv(recv_particle_type.data() + recv_offset[i], recv_count[i],
                MPI_INT, destination_index[i], 0, MPI_COMM_WORLD,
                recv_request.data() + count);

      count++;
    }
  }

  MPI_Waitall(count, send_request.data(), status.data());
  MPI_Waitall(count, recv_request.data(), status.data());

  for (int i = 0; i < total_neighbor_particle_num; i++) {
    background_particle_set[local_particle_num + i].particle_type =
        recv_particle_type[i];
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

void geometry::setup_background_manifold() {}

void geometry::setup_block_neighbor_nonmanifold() {
  int neighbor_num = pow(3, _dimension);

  _neighbor_flag.resize(neighbor_num);
  for (int i = 0; i < neighbor_num; i++) {
    _neighbor_flag[i] = false;
  }

  if (_dimension == 2) {
    int offset[3] = {-1, 0, 1};

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        triple<int> new_coord =
            triple<int>(_local_block_coord[0] + offset[i],
                        _local_block_coord[1] + offset[j], 0);

        if (new_coord[0] >= 0 && new_coord[0] < _global_block_size[0]) {
          if (new_coord[1] >= 0 && new_coord[1] < _global_block_size[1]) {
            int destination =
                new_coord[0] + new_coord[1] * _global_block_size[0];
            if (destination >= 0 && destination != _id &&
                destination < _mpi_size) {
              _neighbor_flag[i + j * 3] = false;
            }
          }
        }
      }
    }
  }

  if (_dimension == 3) {
    int offset[3] = {-1, 0, 1};
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          triple<int> new_coord =
              triple<int>(_local_block_coord[0] + offset[i],
                          _local_block_coord[1] + offset[j],
                          _local_block_coord[2] + offset[k]);

          if (new_coord[0] >= 0 && new_coord[0] < _global_block_size[0]) {
            if (new_coord[1] >= 0 && new_coord[1] < _global_block_size[1]) {
              if (new_coord[2] >= 0 && new_coord[2] < _global_block_size[2]) {
                int destination = new_coord[0] +
                                  new_coord[1] * _global_block_size[0] +
                                  new_coord[2] * _global_block_size[0] *
                                      _global_block_size[1];
                if (destination >= 0 && destination != _id &&
                    destination < _mpi_size) {
                  _neighbor_flag[i + j * 3 + k * 9] = true;
                }
              }
            }
          }
        }
      }
    }
  }
}

void geometry::setup_hierarchy_nonmanifold(
    std::vector<particle> &coarse_level_particle_set,
    std::vector<particle> &fine_level_particle_set,
    std::vector<std::vector<size_t>> &hierarchy) {
  hierarchy.resize(coarse_level_particle_set.size());

  space_tree tree(_dimension);
  double domain_size = _local_bounding_box[1][0] - _local_bounding_box[0][0];
  int level = 1 + log2((domain_size / 2.0) /
                       coarse_level_particle_set[0].particle_size[0]);
  tree.make_tree(level, _local_bounding_box[0], _local_bounding_box[1]);
  tree.add_leaf(coarse_level_particle_set);

  for (size_t i = 0; i < fine_level_particle_set.size(); i++) {
    size_t nearest_neighbor_index =
        tree.find_nearest_neighbor(fine_level_particle_set[i].coord);
    if (coarse_level_particle_set[nearest_neighbor_index].particle_type ==
        fine_level_particle_set[i].particle_type) {
      hierarchy[nearest_neighbor_index].push_back(i);
    }
  }
}

#include <fstream>
#include <string>

using namespace std;

void geometry::write_all_level(std::string output_filename_prefix) {
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

#include <queue>

void space_tree::make_tree(int level, vec3 &domain_low, vec3 &domain_high) {
  _max_level = level;

  std::queue<std::shared_ptr<space_tree_node>> working_queue;
  working_queue.push(_root);

  _root->center_point[0] = 0.5 * (domain_low[0] + domain_high[0]);
  _root->center_point[1] = 0.5 * (domain_low[1] + domain_high[1]);
  _root->center_point[2] = 0.5 * (domain_low[2] + domain_high[2]);
  _root->domain_low = domain_low;
  _root->domain_high = domain_high;

  int current_level = 0;
  while (current_level < level) {
    size_t current_queue_size = working_queue.size();
    for (int i = 0; i < current_queue_size; i++) {
      size_t children_size = (_dimension == 2) ? 4 : 8;

      auto working_item = working_queue.front();
      working_item->level = current_level;
      working_item->children.resize(children_size);

      for (size_t j = 0; j < children_size; j++) {
        working_item->children[j] = std::make_shared<space_tree_node>();
        working_queue.push(working_item->children[j]);
      }

      if (_dimension == 2) {
        working_item->children[0]->center_point[0] =
            0.5 * (working_item->domain_low[0] + working_item->center_point[0]);
        working_item->children[0]->center_point[1] =
            0.5 * (working_item->domain_low[1] + working_item->center_point[1]);
        working_item->children[0]->level = current_level + 1;
        working_item->children[0]->domain_low = working_item->domain_low;
        working_item->children[0]->domain_high = working_item->center_point;

        working_item->children[1]->center_point[0] =
            0.5 * (working_item->domain_low[0] + working_item->center_point[0]);
        working_item->children[1]->center_point[1] =
            0.5 *
            (working_item->domain_high[1] + working_item->center_point[1]);
        working_item->children[1]->level = current_level + 1;
        working_item->children[1]->domain_low[0] = working_item->domain_low[0];
        working_item->children[1]->domain_low[1] =
            working_item->center_point[1];
        working_item->children[1]->domain_high[0] =
            working_item->center_point[0];
        working_item->children[1]->domain_high[1] =
            working_item->domain_high[1];

        working_item->children[2]->center_point[0] =
            0.5 *
            (working_item->domain_high[0] + working_item->center_point[0]);
        working_item->children[2]->center_point[1] =
            0.5 * (working_item->domain_low[1] + working_item->center_point[1]);
        working_item->children[2]->level = current_level + 1;
        working_item->children[2]->domain_low[0] =
            working_item->center_point[0];
        working_item->children[2]->domain_low[1] = working_item->domain_low[1];
        working_item->children[2]->domain_high[0] =
            working_item->domain_high[0];
        working_item->children[2]->domain_high[1] =
            working_item->center_point[1];

        working_item->children[3]->center_point[0] =
            0.5 *
            (working_item->domain_high[0] + working_item->center_point[0]);
        working_item->children[3]->center_point[1] =
            0.5 *
            (working_item->domain_high[1] + working_item->center_point[1]);
        working_item->children[3]->level = current_level + 1;
        working_item->children[3]->domain_low = working_item->center_point;
        working_item->children[3]->domain_high = working_item->domain_high;
      }
      if (_dimension == 3) {
      }

      working_queue.pop();
    }

    current_level++;
  }
}

void space_tree::add_leaf(std::vector<particle> &source) {
  for (size_t i = 0; i < source.size(); i++) {
    auto ptr = _root;
    while (ptr->children.size() != 0) {
      if (_dimension == 2) {
        if (source[i].coord[0] >= ptr->center_point[0] &&
            source[i].coord[1] >= ptr->center_point[1]) {
          ptr = ptr->children[3];
          continue;
        }
        if (source[i].coord[0] >= ptr->center_point[0] &&
            source[i].coord[1] < ptr->center_point[1]) {
          ptr = ptr->children[2];
          continue;
        }
        if (source[i].coord[0] < ptr->center_point[0] &&
            source[i].coord[1] >= ptr->center_point[1]) {
          ptr = ptr->children[1];
          continue;
        }
        if (source[i].coord[0] < ptr->center_point[0] &&
            source[i].coord[1] < ptr->center_point[1]) {
          ptr = ptr->children[0];
          continue;
        }
      }
    }
    space_tree_leaf new_leaf;
    new_leaf.index = i;
    new_leaf.position = source[i].coord;
    ptr->leaf.push_back(new_leaf);
  }
}

double dis(vec3 &pos1, vec3 &pos2) {
  return sqrt(pow(pos1[0] - pos2[0], 2) + pow(pos1[1] - pos2[1], 2) +
              pow(pos1[2] - pos2[2], 2));
}

size_t space_tree::find_nearest_neighbor(vec3 &position) {
  auto ptr = _root;
  while (ptr->children.size() != 0) {
    if (_dimension == 2) {
      if (position[0] >= ptr->center_point[0] &&
          position[1] >= ptr->center_point[1]) {
        ptr = ptr->children[3];
        continue;
      }
      if (position[0] >= ptr->center_point[0] &&
          position[1] < ptr->center_point[1]) {
        ptr = ptr->children[2];
        continue;
      }
      if (position[0] < ptr->center_point[0] &&
          position[1] >= ptr->center_point[1]) {
        ptr = ptr->children[1];
        continue;
      }
      if (position[0] < ptr->center_point[0] &&
          position[1] < ptr->center_point[1]) {
        ptr = ptr->children[1];
        continue;
      }
    }
  }

  size_t index = 0;
  double distance = 10;
  for (size_t i = 0; i < ptr->leaf.size(); i++) {
    double new_distance = dis(position, ptr->leaf[i].position);
    if (new_distance < distance) {
      distance = new_distance;
      index = i;
    }
  }

  return ptr->leaf[index].index;
}

void space_tree::write_tree() {
  auto ptr = _root;
  std::queue<std::shared_ptr<space_tree_node>> working_queue;
  working_queue.push(_root);
  while (working_queue.size() != 0) {
    auto working_item = working_queue.front();
    cout << working_item->level << ": (" << working_item->center_point[0]
         << ", " << working_item->center_point[1] << ", "
         << working_item->center_point[2] << ")" << endl;
    if (working_item->children.size() != 0) {
      for (int i = 0; i < working_item->children.size(); i++) {
        working_queue.push(working_item->children[i]);
      }
    } else {
      for (int i = 0; i < working_item->leaf.size(); i++) {
        cout << "\t";
        cout << working_item->leaf[i].index << ": ("
             << working_item->leaf[i].position[0] << ", "
             << working_item->leaf[i].position[1] << ", "
             << working_item->leaf[i].position[2] << ")" << endl;
      }
      cout << endl;
    }
    working_queue.pop();
  }
}