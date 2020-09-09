#ifndef _GEOMETRY_HPP_
#define _GEOMETRY_HPP_

#include <mpi.h>

#include <memory>
#include <vector>

#include "vec3.h"

struct particle {
  vec3 coord;
  vec3 normal;
  vec3 particle_size;
  vec3 p_coord;

  double volume;

  std::size_t global_index;

  int particle_type;
  int attached_rigid_body_index;
  int particle_adaptive_level;
};

struct gap_particle {
  vec3 coord;
  vec3 normal;
  vec3 particle_size;

  int particle_type;
  int particle_adaptive_level;
};

class geometry {
private:
  int _dimension, _manifold_dimension, _manifold_order;

  int _id, _mpi_size;

  triple<int> _global_block_size, _local_block_coord;

  vec3 _global_bounding_box[2];
  vec3 _local_bounding_box[2];

  std::vector<std::vector<particle> *> _particle_set;
  std::vector<std::vector<particle> *> _background_particle_set;
  std::vector<gap_particle> _gap_particle_set;

  std::vector<std::size_t> _global_particle_count;

  triple<int> _global_particle_num;
  triple<int> _local_particle_num;
  triple<int> _local_particle_num_min;

protected:
  std::vector<int> _local_bounding_box_boundary_type;

  void set_local_bounding_box_boundary();

  void init_nonmanifold();
  void init_manifold();

  void update_nonmanifold();
  void update_manifold();

  size_t global_indexing(std::vector<particle> &current_particle_set);

  void update_uniform_nonmanifold(std::vector<particle> &current_particle_set,
                                  vec3 &current_particle_size,
                                  int particle_level);

public:
  geometry() : _manifold_dimension(0), _manifold_order(0) {
    _global_bounding_box[0][0] = -1;
    _global_bounding_box[0][1] = -1;
    _global_bounding_box[0][2] = -1;
    _global_bounding_box[1][0] = 1;
    _global_bounding_box[1][1] = 1;
    _global_bounding_box[1][2] = 1;
  }

  // paramemter setup
  void set_dimension(int dimension) { _dimension = dimension; }

  void set_manifold_order(int manifold_order) {
    _manifold_order = manifold_order;
  }

  void set_manifold_dimension(int manifold_dimension) {
    _manifold_dimension = manifold_dimension;
  }

  void set_global_x_particle_num(int particle_num) {
    _global_particle_num[0] = particle_num;
  }

  void set_global_y_particle_num(int particle_num) {
    _global_particle_num[1] = particle_num;
  }

  void set_global_z_particle_num(int particle_num) {
    _global_particle_num[2] = particle_num;
  }

  void set_local_x_particle_num_min(int particle_num) {
    _local_particle_num_min[0] = particle_num;
  }

  void set_local_y_particle_num_min(int particle_num) {
    _local_particle_num[1] = particle_num;
  }

  void set_local_z_particle_num_min(int particle_num) {
    _local_particle_num[2] = particle_num;
  }

  // ouput
  void write_all_init_level(std::string output_filename_prefix = "");

  void init() {
    MPI_Comm_size(MPI_COMM_WORLD, &_mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &_id);

    if (_manifold_dimension != 0) {
      init_manifold();
    } else {
      init_nonmanifold();
    }
  }

  void update() {
    if (_manifold_dimension != 0) {
      update_manifold();
    } else {
      update_nonmanifold();
    }
  }
};

#endif