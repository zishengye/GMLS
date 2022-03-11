#ifndef _PARTICLE_MANAGER_PARTICLE_MANAGER_HPP_
#define _PARTICLE_MANAGER_PARTICLE_MANAGER_HPP_

#include <memory>
#include <vector>

#include <Kokkos_Core.hpp>

#include "vec3.hpp"

class particle_manager {
public:
  typedef std::vector<vec3> vec_type;
  typedef std::vector<long long> idx_type;
  typedef std::vector<int> int_type;
  typedef std::vector<double> real_type;
};

class hierarchical_particle_manager {
public:
  hierarchical_particle_manager();

  void init();
  void clear();
};

#endif