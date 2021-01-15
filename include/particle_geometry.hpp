#ifndef _PARTICLE_GEOMETRY_HPP_
#define _PARTICLE_GEOMETRY_HPP_

#define UNIFORM_REFINE 1
#define ADAPTIVE_REFINE 2

#define STANDARD_PROBLEM 1
#define MANIFOLD_PROBLEM 2

#include <memory>
#include <vector>

#include "rigid_body_manager.hpp"
#include "trilinos_wrapper.hpp"
#include "vec3.hpp"

class particle_geometry {
public:
  typedef std::shared_ptr<std::vector<vec3>> vec_type;
  typedef std::shared_ptr<std::vector<long long>> idx_type;
  typedef std::shared_ptr<std::vector<int>> int_type;
  typedef std::shared_ptr<std::vector<double>> real_type;

private:
  int refinement_type;
  int problem_type;
  int dim;

  double spacing;
  double cutoff_multiplier;
  double cutoff_distance, old_cutoff_distance;

  std::shared_ptr<rigid_body_manager> rb_mgr;

  // work domain
  vec_type current_local_work_particle_coord;
  vec_type current_local_work_particle_normal;
  vec_type current_local_work_particle_p_spacing;
  real_type current_local_work_particle_spacing;
  real_type current_local_work_particle_volume;
  int_type current_local_work_particle_index;
  int_type current_local_work_particle_type;
  int_type current_local_work_particle_adaptive_level;
  int_type current_local_work_particle_new_added;
  int_type current_local_work_particle_attached_rigid_body;
  int_type current_local_work_particle_num_neighbor;

  vec_type last_local_work_particle_coord;
  vec_type last_local_work_particle_normal;
  real_type last_local_work_particle_spacing;
  real_type last_local_work_particle_volume;
  int_type last_local_work_particle_index;
  int_type last_local_work_particle_type;
  int_type last_local_work_particle_adaptive_level;

  // work ghost domain
  vec_type current_local_work_ghost_particle_coord;
  real_type current_local_work_ghost_particle_volume;
  int_type current_local_work_ghost_particle_index;

  vec_type last_local_work_ghost_particle_coord;
  real_type last_local_work_ghost_particle_volume;
  int_type last_local_work_ghost_particle_index;

  // ghost for current level from last level [abbreviated by "clll"]
  vec_type clll_particle_coord;
  int_type clll_particle_index;
  int_type clll_particle_type;

  // ghost for last level from current level [abbreviated by "llcl"]
  vec_type llcl_particle_coord;
  int_type llcl_particle_index;
  int_type llcl_particle_type;

  // managing domain
  vec_type current_local_managing_particle_coord;
  vec_type current_local_managing_particle_normal;
  vec_type current_local_managing_particle_p_coord;
  vec_type current_local_managing_particle_p_spacing;
  real_type current_local_managing_particle_spacing;
  real_type current_local_managing_particle_volume;
  idx_type current_local_managing_particle_index;
  int_type current_local_managing_particle_type;
  int_type current_local_managing_particle_adaptive_level;
  int_type current_local_managing_particle_new_added;
  int_type current_local_managing_particle_attached_rigid_body;

  // managing domain gap particles
  vec_type local_managing_gap_particle_coord;
  vec_type local_managing_gap_particle_normal;
  vec_type local_managing_gap_particle_p_coord;
  real_type local_managing_gap_particle_volume;
  real_type local_managing_gap_particle_spacing;
  int_type local_managing_gap_particle_particle_type;
  int_type local_managing_gap_particle_adaptive_level;

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

  // mitigation
  std::vector<int> mitigation_in_graph, mitigation_out_graph;
  std::vector<int> mitigation_in_num, mitigation_out_num;
  std::vector<int> mitigation_in_offset, mitigation_out_offset;
  std::vector<int> local_mitigation_map;
  std::vector<int> local_reserve_map;

  // ghost
  std::vector<int> ghost_in_graph, ghost_out_graph;
  std::vector<int> ghost_in_num, ghost_out_num;
  std::vector<int> ghost_in_offset, ghost_out_offset;
  std::vector<int> ghost_map;

  // ghost for current level from last level [abbreviated by "clll"]
  std::vector<int> ghost_clll_in_graph, ghost_clll_out_graph;
  std::vector<int> ghost_clll_in_num, ghost_clll_out_num;
  std::vector<int> ghost_clll_in_offset, ghost_clll_out_offset;
  std::vector<int> ghost_clll_map;
  std::vector<int> reserve_clll_map;

  // ghost for last level from current level [abbreviated by "llcl"]
  std::vector<int> ghost_llcl_in_graph, ghost_llcl_out_graph;
  std::vector<int> ghost_llcl_in_num, ghost_llcl_out_num;
  std::vector<int> ghost_llcl_in_offset, ghost_llcl_out_offset;
  std::vector<int> ghost_llcl_map;
  std::vector<int> reserve_llcl_map;

  // mpi
  int rank, size;

  int min_count, max_count, current_count, stride;

public:
  particle_geometry()
      : dim(3), refinement_type(ADAPTIVE_REFINE),
        problem_type(STANDARD_PROBLEM) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  }

  ~particle_geometry() {}

  void init(const int _dim, const int _problem_type = STANDARD_PROBLEM,
            const int _refinement_type = ADAPTIVE_REFINE, double _spacing = 0.1,
            double _cutoff_multiplier = 3.0, const int _min_count = 0,
            const int _max_count = 0, const int _stride = 0,
            std::string geometry_input_file_name = "");
  void init_rigid_body(std::shared_ptr<rigid_body_manager> mgr);

  void generate_uniform_particle();

  void clear_particle();

  void mitigate_forward(int_type source, int_type target);
  void mitigate_forward(real_type source, real_type target);
  void mitigate_forward(vec_type source, vec_type target);
  void mitigate_backward(std::vector<int> &source, std::vector<int> &target);

  void ghost_forward(int_type source, int_type target);
  void ghost_forward(real_type source, real_type target);
  void ghost_forward(vec_type source, vec_type target);
  void ghost_forward(std::vector<int> &source_vec,
                     std::vector<int> &target_vec);
  void ghost_forward(std::vector<double> &source_vec,
                     std::vector<double> &target_vec);
  void ghost_forward(std::vector<vec3> &source_vec,
                     std::vector<vec3> &target_vec);
  void ghost_forward(std::vector<std::vector<double>> &source_chunk,
                     std::vector<std::vector<double>> &target_chunk,
                     const size_t unit_length);

  void ghost_clll_forward(int_type source, int_type target);
  void ghost_clll_forward(real_type source, real_type target);
  void ghost_clll_forward(vec_type source, vec_type target);
  void ghost_llcl_forward(int_type source, int_type target);
  void ghost_llcl_forward(real_type source, real_type target);
  void ghost_llcl_forward(vec_type source, vec_type target);

  void refine(std::vector<int> &split_tag);

  // get work domain data
  vec_type get_current_work_particle_coord() {
    return current_local_work_particle_coord;
  }

  vec_type get_current_work_particle_normal() {
    return current_local_work_particle_normal;
  }

  vec_type get_current_work_particle_p_spacing() {
    return current_local_work_particle_p_spacing;
  }

  real_type get_current_work_particle_spacing() {
    return current_local_work_particle_spacing;
  }

  real_type get_current_work_particle_volume() {
    return current_local_work_particle_volume;
  }

  int_type get_current_work_particle_index() {
    return current_local_work_particle_index;
  }

  int_type get_current_work_particle_type() {
    return current_local_work_particle_type;
  }

  int_type get_current_work_particle_adaptive_level() {
    return current_local_work_particle_adaptive_level;
  }

  int_type get_current_work_particle_new_added() {
    return current_local_work_particle_new_added;
  }

  int_type get_current_work_particle_attached_rigid_body() {
    return current_local_work_particle_attached_rigid_body;
  }

  int_type get_current_work_particle_num_neighbor() {
    return current_local_work_particle_num_neighbor;
  }

  vec_type get_current_work_ghost_particle_coord() {
    return current_local_work_ghost_particle_coord;
  }

  real_type get_current_work_ghost_particle_volume() {
    return current_local_work_ghost_particle_volume;
  }

  int_type get_current_work_ghost_particle_index() {
    return current_local_work_ghost_particle_index;
  }

  vec_type get_last_work_particle_coord() {
    return last_local_work_particle_coord;
  }

  vec_type get_last_work_particle_normal() {
    return last_local_work_particle_normal;
  }

  real_type get_last_work_particle_spacing() {
    return last_local_work_particle_spacing;
  }

  real_type get_last_work_particle_volume() {
    return last_local_work_particle_volume;
  }

  int_type get_last_work_particle_index() {
    return last_local_work_particle_index;
  }

  int_type get_last_work_particle_type() {
    return last_local_work_particle_type;
  }

  int_type get_last_work_particle_adaptive_level() {
    return last_local_work_particle_adaptive_level;
  }

  vec_type get_last_work_ghost_particle_coord() {
    return last_local_work_ghost_particle_coord;
  }

  int_type get_last_work_ghost_particle_index() {
    return last_local_work_ghost_particle_index;
  }

  vec_type get_clll_particle_coord() { return clll_particle_coord; }

  int_type get_clll_particle_index() { return clll_particle_index; }

  int_type get_clll_particle_type() { return clll_particle_type; }

  vec_type get_llcl_particle_coord() { return llcl_particle_coord; }

  int_type get_llcl_particle_index() { return llcl_particle_index; }

  int_type get_llcl_particle_type() { return llcl_particle_type; }

  double get_cutoff_distance() { return cutoff_distance; }

  double get_old_cutoff_distance() { return old_cutoff_distance; }

protected:
  void init_domain_boundary();

  void generate_field_particle();
  void generate_rigid_body_surface_particle();

  void uniform_refine();
  void adaptive_refine(std::vector<int> &split_tag);

  void insert_particle(const vec3 &_pos, int _particle_type,
                       const double _spacing, const vec3 &_normal,
                       int _adaptive_level, double _volume,
                       bool _rigid_body_particle = false,
                       int _rigid_body_index = -1,
                       vec3 _p_coord = vec3(0.0, 0.0, 0.0),
                       vec3 _p_spacing = vec3(0.0, 0.0, 0.0));

  void split_field_particle(std::vector<int> &split_tag);
  void split_rigid_body_surface_particle(std::vector<int> &split_tag);
  void split_gap_particle(std::vector<int> &split_tag);

  int is_gap_particle(const vec3 &_pos, double _spacing,
                      int _attached_rigid_body_index);

  void index_particle();
  void index_work_particle();

  void balance_workload();

  void build_ghost();

  void build_ghost_from_last_level();

  void build_ghost_for_last_level();
};

#endif