#ifndef _GEOMETRY_MANAGER_HPP_
#define _GEOMETRY_MANAGER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "network.hpp"
#include "vec3.h"

enum discretization_type { particle_discretization, mesh_discretization };
enum discretization_update_type {
  eulerian,
  lagrangian,
  arbitrary_eulerian_lagrangian
};

class geometry_manager {
private:
  int _dimension;
  int _nx, _ny, _nz;
  int _output_level;

  vec3 _domain[2];

  std::vector<int> _domain_boundary_type;

  std::shared_ptr<network> _net;

  discretization_type _discretization;
  discretization_update_type _update;

  size_t _base_num, _source_num, _background_num;

  std::vector<vec3> _base;
  std::vector<vec3> _source;
  std::vector<vec3> _background;

  std::vector<int> _base2source_index;
  std::vector<int> _source2base_index;

  // variables for particle discretization
  std::vector<double> _base_particle_size;
  std::vector<double> _source_particle_size;
  std::vector<double> _background_particle_size;

  std::vector<int> _base_particle_type;
  std::vector<int> _source_particle_type;

  std::vector<vec3> _base_normal;
  std::vector<vec3> _source_normal;

  std::vector<int> _source_global_index;
  std::vector<int> _background_global_index;

  std::vector<std::vector<int>> _source_neighbor_list_on_background;

public:
  geometry_manager(std::shared_ptr<network> net) : _net(net) {}

  void set_dimension(int dimension) { _dimension = dimension; }

  void set_discretization_type(discretization_type discretization) {
    _discretization = discretization;
  }

  void set_discretization_size(int nx, int ny, int nz) {
    _nx = nx;
    _ny = ny;
    _nz = nz;
  }

  void set_discretization_update_scheme(discretization_update_type update) {
    _update = update;
  }

  void set_output_level(int output_level) { _output_level = output_level; }

  void initialization();

  void write_data(std::string output_file_name) {}

  // particle discretization management
  void insert_base_particle(vec3 pos, vec3 normal, int particle_type,
                            double particle_size);
};

#endif