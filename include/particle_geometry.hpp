#ifndef _PARTICLE_GEOMETRY_HPP_
#define _PARTICLE_GEOMETRY_HPP_

#define UNIFORM_REFINE 1
#define ADAPTIVE_REFINE 2

#define STANDARD 1
#define MANIFOLD 2

#include <memory>
#include <vector>

#include "rigid_body_manager.hpp"
#include "trilinos_wrapper.hpp"
#include "vec3.hpp"

class particle_geometry {
private:
  int refinement_type;
  int problem_type;
  int dim;

  double spacing;
  double cutoff_multiplier;
  double cutoff_distance;

  std::shared_ptr<rigid_body_manager> rb_mgr;

  std::shared_ptr<std::vector<vec3>> current_local_work_particle_coord;
  std::shared_ptr<std::vector<vec3>> current_local_work_ghost_particle_coord;
  std::shared_ptr<std::vector<vec3>> current_local_gmls_particle_coord;
  std::shared_ptr<std::vector<vec3>> current_local_gmls_ghost_particle_coord;
  std::shared_ptr<std::vector<long long>> current_local_work_particle_index;
  std::shared_ptr<std::vector<int>> current_local_work_ghost_particle_index;
  std::shared_ptr<std::vector<int>> current_local_gmls_particle_index;
  std::shared_ptr<std::vector<int>> current_local_gmls_ghost_particle_index;

  std::shared_ptr<std::vector<vec3>> last_local_work_particle_coord;
  std::shared_ptr<std::vector<vec3>> last_local_work_ghost_particle_coord;
  std::shared_ptr<std::vector<vec3>> last_local_gmls_particle_coord;
  std::shared_ptr<std::vector<vec3>> last_local_gmls_ghost_particle_coord;
  std::shared_ptr<std::vector<vec3>> last_local_managing_particle_coord;
  std::shared_ptr<std::vector<long long>> last_local_work_particle_index;
  std::shared_ptr<std::vector<int>> last_local_work_ghost_particle_index;
  std::shared_ptr<std::vector<int>> last_local_gmls_particle_index;
  std::shared_ptr<std::vector<int>> last_local_gmls_ghost_particle_index;
  std::shared_ptr<std::vector<int>> last_local_managing_particle_index;

  std::shared_ptr<std::vector<vec3>> current_local_managing_particle_coord;
  std::shared_ptr<std::vector<vec3>> current_local_managing_particle_normal;
  std::shared_ptr<std::vector<vec3>> current_local_managing_particle_p_coord;
  std::shared_ptr<std::vector<double>> current_local_managing_particle_spacing;
  std::shared_ptr<std::vector<double>> current_local_managing_particle_volume;
  std::shared_ptr<std::vector<long long>> current_local_managing_particle_index;
  std::shared_ptr<std::vector<int>> current_local_managing_particle_type;
  std::shared_ptr<std::vector<int>>
      current_local_managing_particle_adaptive_level;
  std::shared_ptr<std::vector<int>> current_local_managing_particle_new_added;
  std::shared_ptr<std::vector<int>>
      current_local_managing_particle_attached_rigid_body;

  std::shared_ptr<std::vector<vec3>> local_managing_gap_particle_coord;
  std::shared_ptr<std::vector<vec3>> local_managing_gap_particle_normal;
  std::shared_ptr<std::vector<vec3>> local_managing_gap_particle_p_coord;
  std::shared_ptr<std::vector<double>> local_managing_gap_particle_volume;
  std::shared_ptr<std::vector<double>> local_managing_gap_particle_spacing;
  std::shared_ptr<std::vector<int>> local_managing_gap_particle_particle_type;
  std::shared_ptr<std::vector<int>> local_managing_gap_particle_adaptive_level;

  std::vector<vec3> rigid_body_surface_particle;

  vec3 bounding_box[2];
  vec3 bounding_box_size;
  triple<int> bounding_box_count;

  vec3 domain_bounding_box[2];
  vec3 domain[2];
  triple<int> domain_count;

  std::vector<int> domain_boundary_type;

  int process_x, process_y, process_z, process_i, process_j, process_k;

  trilinos_rcp_partitioner partitioner;

public:
  particle_geometry()
      : dim(3), refinement_type(ADAPTIVE_REFINE), problem_type(STANDARD) {}

  ~particle_geometry() {}

  void init(const int _dim, const int _problem_type = STANDARD,
            const int _refinement_type = ADAPTIVE_REFINE, double _spacing = 0.1,
            double _cutoff_multiplier = 3.0,
            std::string geometry_input_file_name = "");
  void init_rigid_body(rigid_body_manager &mgr);

  void generate_uniform_particle();

  void clear_particle();

  void mitigate_forward();
  void mitigate_backward();

  void split_particle();

  void refine();

protected:
  void init_domain_boundary();

  void generate_field_particle();
  void generate_rigid_body_surface_particle();

  void uniform_refine();
  void adaptive_refine();

  void insert_particle(const vec3 &_pos, int _particle_type,
                       const double _spacing, const vec3 &_normal,
                       int _adaptive_level, double _volume,
                       bool _rigid_body_particle = false,
                       int _rigid_body_index = -1,
                       vec3 _p_coord = vec3(0.0, 0.0, 0.0));

  void split_field_particle();
  void split_rigid_body_surface_particle();
  void split_gap_particle();

  int is_gap_particle(const vec3 &_pos, double _spacing,
                      int _attached_rigid_body_index);

  void index_particle();

  void balance_workload();
};

#endif