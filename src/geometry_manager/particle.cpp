#include "geometry_manager.hpp"

using namespace std;

void geometry_manager::insert_base_particle(vec3 pos, vec3 normal,
                                            int particle_type,
                                            double particle_size) {
  _base.push_back(pos);
  _base_normal.push_back(normal);
  _base_particle_type.push_back(particle_type);
  _base_particle_size.push_back(particle_size);
}