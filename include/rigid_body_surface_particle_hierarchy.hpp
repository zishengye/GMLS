#ifndef _RIGID_BODY_SURFACE_PARTICLE_HIERARCHY_HPP_
#define _RIGID_BODY_SURFACE_PARTICLE_HIERARCHY_HPP_

#include <memory>
#include <vector>

#include <Compadre_PointCloudSearch.hpp>
#include <petscksp.h>

#include "geometry.hpp"
#include "rigid_body_manager.hpp"
#include "vec3.hpp"

class rigid_body_surface_particle_hierarchy {
private:
  std::vector<std::vector<vec3>> hierarchy_coord;
  std::vector<std::vector<vec3>> hierarchy_normal;
  std::vector<std::vector<vec3>> hierarchy_spacing;
  std::vector<std::vector<triple<int>>> hierarchy_element;
  std::vector<int> hierarchy_adaptive_level;

  // the refinement relation is seen as a CRS matrix
  std::vector<std::vector<int>> hierarchy;
  std::vector<std::vector<int>> hierarchy_index;

  std::vector<int> rigid_body_type_list;
  std::vector<std::vector<double>> rigid_body_size_list;

  std::vector<std::vector<int>> mapping;
  std::vector<std::vector<int>> hierarchy_mapping;

  std::shared_ptr<rigid_body_manager> rb_mgr;

  std::vector<int> rb_idx;

  double coarse_level_resolution;

  int dimension;

protected:
  int find_rigid_body(const int rigid_body_index, const int refinement_level);

  void extend_hierarchy(const int compressed_rigid_body_index);

  void add_sphere(const double radius, const double h);
  void add_rounded_square(const double half_side_length, const double h);
  void add_customized_shape(const double size, const double h);

  void build_hierarchy_mapping(const int coarse_level_idx,
                               const int fine_level_idx);

public:
  rigid_body_surface_particle_hierarchy() {}

  ~rigid_body_surface_particle_hierarchy() {}

  void init(std::shared_ptr<rigid_body_manager> mgr, const int dim);

  void set_coarse_level_resolution(const double h0) {
    coarse_level_resolution = h0;
  }

  void find_refined_particle(int rigid_body_index, int refinement_level,
                             int particle_index,
                             std::vector<int> &refined_particle_index);

  vec3 get_coordinate(int rigid_body_index, int refinement_level,
                      int particle_index);

  vec3 get_normal(int rigid_body_index, int refinement_level,
                  int particle_index);

  vec3 get_spacing(int rigid_body_index, int refinement_level,
                   int particle_idnex);

  void
  get_coarse_level_coordinate(const int rigid_body_index,
                              std::shared_ptr<std::vector<vec3>> &coord_ptr);

  void get_coarse_level_normal(const int rigid_body_index,
                               std::shared_ptr<std::vector<vec3>> &normal_ptr);

  void
  get_coarse_level_spacing(const int rigid_body_index,
                           std::shared_ptr<std::vector<vec3>> &spacing_ptr);

  int get_coarse_level_adaptive_level(const int rigid_body_index) {
    return hierarchy_adaptive_level[find_rigid_body(rigid_body_index, 0)];
  }

  void get_coarse_level_element(
      const int rigid_body_index,
      std::shared_ptr<std::vector<triple<int>>> &element_ptr);

  void write_log();

  void move_to_boundary(int rigid_body_index, vec3 &pos);

  void get_normal(int rigid_body_index, vec3 pos, vec3 &norm);
};

#endif