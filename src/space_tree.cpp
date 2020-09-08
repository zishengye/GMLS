#include "space_tree.hpp"

void space_tree::init_nonmanifold(int particle_num, int particle_num_min) {
  int current_level_particle_num = particle_num;
  int current_level = 0;
  while (current_level_particle_num >= particle_num_min) {
    _particle_set.push_back(new std::vector<particle>());

    std::vector<particle> &current_level_particle =
        *_particle_set[current_level];

    particle part;

    part.coord = vec3(0, 0, 0);

    current_level_particle_num /= 2;
  }
}