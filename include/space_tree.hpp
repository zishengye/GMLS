#ifndef SPACE_TREE_HPP
#define SPACE_TREE_HPP

#include <mpi.h>
#include <vector>

#include "vec3.h"

struct particle {
  vec3 coord;
  int particle_type;
  double particle_size;
  vec3 normal;
  int global_index;
};

class space_tree {
private:
  int _dim, _manifold_dim, _manifold_order;

  std::vector<std::vector<particle> *> _particle_set;

  int _id, _mpi_size;

protected:
  void init_nonmanifold(int particle_num, int particle_num_min);

  void init_manifold() {}

public:
  space_tree() : _manifold_dim(0), _manifold_order(0) {}

  void set_dimension(int dim) { _dim = dim; }

  void set_manifold_order(int manifold_order) {
    _manifold_order = manifold_order;
  }

  void set_manifold_dimension(int manifold_dim) {
    _manifold_dim = manifold_dim;
  }

  void init(int particle_num, int particle_num_min) {
    MPI_Comm_size(MPI_COMM_WORLD, &_mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &_id);

    if (_manifold_dim != 0)
      init_manifold();
    else
      init_nonmanifold(particle_num, particle_num_min);
  }
};

#endif