#include "rigid_body_surface_particle_hierarchy.hpp"

#include <iostream>

using namespace std;
using namespace Compadre;

int rigid_body_surface_particle_hierarchy::find_rigid_body(
    const int rigid_body_index, const int refinement_level) {
  while (refinement_level >= mapping[rb_idx[rigid_body_index]].size())
    extend_hierarchy(rb_idx[rigid_body_index]);
  return mapping[rb_idx[rigid_body_index]][refinement_level];
}

void rigid_body_surface_particle_hierarchy::extend_hierarchy(
    const int compressed_rigid_body_index) {
  int new_refinement_level = mapping[compressed_rigid_body_index].size();

  double resolution = pow(0.5, new_refinement_level) * coarse_level_resolution;

  switch (rigid_body_type_list[compressed_rigid_body_index]) {
  case 1:
    if (dimension == 3) {
      add_sphere(rigid_body_size_list[compressed_rigid_body_index], resolution);
    }
    break;
  case 2:
    if (dimension == 2) {
      add_rounded_square(rigid_body_size_list[compressed_rigid_body_index],
                         resolution);
    }
  }

  mapping[compressed_rigid_body_index].push_back(hierarchy_coord.size() - 1);
  hierarchy_index.push_back(vector<int>());
  hierarchy.push_back(vector<int>());
}

void rigid_body_surface_particle_hierarchy::add_sphere(const double radius,
                                                       const double h) {
  hierarchy_coord.push_back(vector<vec3>());
  hierarchy_normal.push_back(vector<vec3>());
  hierarchy_spacing.push_back(vector<vec3>());

  vector<vec3> &coord = hierarchy_coord[hierarchy_coord.size() - 1];
  vector<vec3> &normal = hierarchy_normal[hierarchy_normal.size() - 1];
  vector<vec3> &spacing = hierarchy_spacing[hierarchy_spacing.size() - 1];

  const double r = radius;
  const double a = pow(h, 2);

  int N = round(4 * M_PI * r * r) / a;
  double area = 4 * M_PI * r * r / N;

  double phi = M_PI * (3 - sqrt(5));

  for (int i = 0; i < N; i++) {
    double y = (1.0 - ((double)i / (N - 1)) * 2);
    double r0 = sqrt(1.0 - y * y);

    double theta = phi * i;

    double x = cos(theta) * r0;
    double z = sin(theta) * r0;

    vec3 norm = vec3(x, y, z);
    vec3 ps = vec3(area, 1.0, 0.0);

    coord.push_back(norm * r);
    normal.push_back(norm);
    spacing.push_back(ps);
  }

  // int M_theta = round(r * M_PI / h) + 1;
  // double d_theta = r * M_PI / M_theta;
  // double d_phi = a / d_theta;

  // for (int i = 0; i < M_theta; ++i) {
  //   double theta = M_PI * (i + 0.5) / M_theta;
  //   int M_phi = round(2 * M_PI * r * sin(theta) / d_phi);
  //   for (int j = 0; j < M_phi; ++j) {
  //     double phi = 2 * M_PI * (j + 0.5) / M_phi;

  //     double theta0 = M_PI * i / M_theta;
  //     double theta1 = M_PI * (i + 1) / M_theta;

  //     double phi0 = 2 * M_PI * j / M_phi;
  //     double phi1 = 2 * M_PI * (j + 1) / M_phi;

  //     vec3 norm =
  //         vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
  //     vec3 ps = vec3(r * (cos(theta0) - cos(theta1)), r * (phi1 - phi0),
  //     0.0);

  //     coord.push_back(norm * r);
  //     normal.push_back(norm);
  //     spacing.push_back(ps);
  //   }
  // }
}

void rigid_body_surface_particle_hierarchy::add_rounded_square(
    const double half_side_length, const double h) {
  hierarchy_coord.push_back(vector<vec3>());
  hierarchy_normal.push_back(vector<vec3>());
  hierarchy_spacing.push_back(vector<vec3>());

  vector<vec3> &coord = hierarchy_coord[hierarchy_coord.size() - 1];
  vector<vec3> &normal = hierarchy_normal[hierarchy_normal.size() - 1];
  vector<vec3> &spacing = hierarchy_spacing[hierarchy_spacing.size() - 1];

  const double rounded_ratio = 0.2;

  const double hs = half_side_length;
  const double r = rounded_ratio * hs;

  // place particles on the straight lines
  const double ratio = 1.0 - rounded_ratio;
  int N = round(2.0 * ratio * half_side_length / h);
  double dist = 2.0 * ratio * half_side_length / N;

  double start_point = -ratio * half_side_length + 0.5 * dist;
  double end_point = ratio * half_side_length;
  double xPos, yPos;
  xPos = start_point;
  yPos = half_side_length;
  while (xPos < end_point) {
    coord.push_back(vec3(xPos, yPos, 0.0));
    normal.push_back(vec3(0.0, 1.0, 0.0));
    spacing.push_back(vec3(dist, 0.0, 0.0));
    xPos += dist;
  }

  xPos = start_point;
  yPos = -half_side_length;
  while (xPos < end_point) {
    coord.push_back(vec3(xPos, yPos, 0.0));
    normal.push_back(vec3(0.0, -1.0, 0.0));
    spacing.push_back(vec3(dist, 0.0, 0.0));
    xPos += dist;
  }

  xPos = half_side_length;
  yPos = start_point;
  while (yPos < end_point) {
    coord.push_back(vec3(xPos, yPos, 0.0));
    normal.push_back(vec3(1.0, 0.0, 0.0));
    spacing.push_back(vec3(dist, 0.0, 0.0));
    yPos += dist;
  }

  xPos = -half_side_length;
  yPos = start_point;
  while (yPos < end_point) {
    coord.push_back(vec3(xPos, yPos, 0.0));
    normal.push_back(vec3(-1.0, 0.0, 0.0));
    spacing.push_back(vec3(dist, 0.0, 0.0));
    yPos += dist;
  }

  // place particles on rounded corners
  int M_theta = round(0.5 * M_PI * r / h);
  double d_theta = 0.5 * M_PI * r / M_theta;

  vec3 p_spacing = vec3(d_theta, 0, 0);

  for (int i = 0; i < M_theta; ++i) {
    double theta = 0.5 * M_PI / M_theta * (i + 0.5);
    vec3 norm = vec3(cos(theta), sin(theta), 0.0);

    coord.push_back(norm * r + vec3(end_point, end_point, 0.0));
    normal.push_back(norm);
    spacing.push_back(p_spacing);
  }

  for (int i = 0; i < M_theta; ++i) {
    double theta = 0.5 * M_PI / M_theta * (i + 0.5) + M_PI * 0.5;
    vec3 norm = vec3(cos(theta), sin(theta), 0.0);

    coord.push_back(norm * r + vec3(-end_point, end_point, 0.0));
    normal.push_back(norm);
    spacing.push_back(p_spacing);
  }

  for (int i = 0; i < M_theta; ++i) {
    double theta = 0.5 * M_PI / M_theta * (i + 0.5) + M_PI;
    vec3 norm = vec3(cos(theta), sin(theta), 0.0);

    coord.push_back(norm * r + vec3(-end_point, -end_point, 0.0));
    normal.push_back(norm);
    spacing.push_back(p_spacing);
  }

  for (int i = 0; i < M_theta; ++i) {
    double theta = 0.5 * M_PI / M_theta * (i + 0.5) + M_PI * 1.5;
    vec3 norm = vec3(cos(theta), sin(theta), 0.0);

    coord.push_back(norm * r + vec3(end_point, -end_point, 0.0));
    normal.push_back(norm);
    spacing.push_back(p_spacing);
  }
}

void rigid_body_surface_particle_hierarchy::build_hierarchy_mapping(
    const int coarse_level_idx, const int fine_level_idx) {
  auto &source_coord = hierarchy_coord[coarse_level_idx];
  auto &target_coord = hierarchy_coord[fine_level_idx];

  int num_source_coord = source_coord.size();
  int num_target_coord = target_coord.size();

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> source_coord_device(
      "source coordinates", num_source_coord, 3);
  Kokkos::View<double **>::HostMirror source_coord_host =
      Kokkos::create_mirror_view(source_coord_device);

  for (size_t i = 0; i < num_source_coord; i++) {
    for (int j = 0; j < 3; j++) {
      source_coord_host(i, j) = source_coord[i][j];
    }
  }

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> target_coord_device(
      "target coordinates", num_target_coord, 3);
  Kokkos::View<double **>::HostMirror target_coord_host =
      Kokkos::create_mirror_view(target_coord_device);

  for (int i = 0; i < num_target_coord; i++) {
    for (int j = 0; j < 3; j++) {
      target_coord_host(i, j) = target_coord[i][j];
    }
  }

  auto point_cloud_search(CreatePointCloudSearch(source_coord_host, dimension));

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> neighbor_list_device(
      "neighbor lists", num_target_coord, 10);
  Kokkos::View<int **>::HostMirror neighbor_list_host =
      Kokkos::create_mirror_view(neighbor_list_device);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> epsilon_device(
      "h supports", num_target_coord);
  Kokkos::View<double *>::HostMirror epsilon_host =
      Kokkos::create_mirror_view(epsilon_device);

  int num_neighbor = point_cloud_search.generate2DNeighborListsFromKNNSearch(
      true, target_coord_host, neighbor_list_host, epsilon_host, 2, 1.0001,
      0.0);
  if (num_neighbor > 2) {
    neighbor_list_device = Kokkos::View<int **, Kokkos::DefaultExecutionSpace>(
        "neighbor lists", num_target_coord, num_neighbor + 1);
    neighbor_list_host = Kokkos::create_mirror_view(neighbor_list_device);
  }

  point_cloud_search.generate2DNeighborListsFromKNNSearch(
      false, target_coord_host, neighbor_list_host, epsilon_host, 2, 1.0001,
      0.0);

  vector<int> child_num;
  child_num.resize(num_source_coord);
  for (int i = 0; i < num_source_coord; i++) {
    child_num[i] = 0;
  }

  for (int i = 0; i < num_target_coord; i++) {
    child_num[neighbor_list_host(i, 1)]++;
  }

  hierarchy_index[coarse_level_idx].resize(num_source_coord + 1);
  hierarchy_index[coarse_level_idx][0] = 0;
  for (int i = 0; i < num_source_coord; i++) {
    hierarchy_index[coarse_level_idx][i + 1] =
        hierarchy_index[coarse_level_idx][i] + child_num[i];
  }

  vector<int> child_idx;
  child_idx.resize(num_source_coord);
  for (int i = 0; i < num_source_coord; i++) {
    child_idx[i] = 0;
  }

  hierarchy[coarse_level_idx].resize(
      hierarchy_index[coarse_level_idx][num_source_coord]);
  for (int i = 0; i < num_target_coord; i++) {
    int coarse_level_particle_idx = neighbor_list_host(i, 1);
    hierarchy[coarse_level_idx]
             [hierarchy_index[coarse_level_idx][coarse_level_particle_idx] +
              child_idx[coarse_level_particle_idx]] = i;
    child_idx[coarse_level_particle_idx]++;
  }
}

void rigid_body_surface_particle_hierarchy::init(
    shared_ptr<rigid_body_manager> mgr, const int dim) {
  rb_mgr = mgr;
  dimension = dim;

  const int rigid_body_num = rb_mgr->get_rigid_body_num();

  rb_idx.resize(rigid_body_num);

  // won't have to build a hierarchy for each rigid body, merge the size and
  // type first.
  for (int i = 0; i < rigid_body_num; i++) {
    rb_idx[i] = -1;
    bool flag = false;

    const int current_rigid_body_type = rb_mgr->get_rigid_body_type(i);
    const double current_rigid_body_size = rb_mgr->get_rigid_body_size(i);

    for (int j = 0; j < rigid_body_type_list.size(); j++) {
      if (rigid_body_type_list[j] == current_rigid_body_type &&
          abs(rigid_body_size_list[j] - current_rigid_body_size) <
              1e-5 * current_rigid_body_size) {
        flag = true;
        rb_idx[i] = j;
      }
    }

    if (!flag) {
      rb_idx[i] = rigid_body_type_list.size();
      rigid_body_type_list.push_back(current_rigid_body_type);
      rigid_body_size_list.push_back(current_rigid_body_size);
    }
  }

  int num_rigid_body_hierarchy_needed = rigid_body_type_list.size();
  mapping.resize(num_rigid_body_hierarchy_needed);
}

void rigid_body_surface_particle_hierarchy::find_refined_particle(
    int rigid_body_index, int refinement_level, int particle_index,
    std::vector<int> &refined_particle_index) {
  int refinement_level_needed = refinement_level + 1;
  bool hierarchy_available =
      (refinement_level_needed < mapping[rb_idx[rigid_body_index]].size());

  if (!hierarchy_available) {
    for (int i = mapping[rb_idx[rigid_body_index]].size();
         i <= refinement_level_needed; i++) {
      extend_hierarchy(rb_idx[rigid_body_index]);
    }

    build_hierarchy_mapping(
        mapping[rb_idx[rigid_body_index]][refinement_level],
        mapping[rb_idx[rigid_body_index]][refinement_level + 1]);
  }

  int hierarchy_idx = mapping[rb_idx[rigid_body_index]][refinement_level];
  refined_particle_index.clear();

  for (int i = hierarchy_index[hierarchy_idx][particle_index];
       i < hierarchy_index[hierarchy_idx][particle_index + 1]; i++) {
    refined_particle_index.push_back(hierarchy[hierarchy_idx][i]);
  }
}

vec3 rigid_body_surface_particle_hierarchy::get_coordinate(int rigid_body_index,
                                                           int refinement_level,
                                                           int particle_index) {
  return hierarchy_coord[find_rigid_body(rigid_body_index, refinement_level)]
                        [particle_index];
}

vec3 rigid_body_surface_particle_hierarchy::get_normal(int rigid_body_index,
                                                       int refinement_level,
                                                       int particle_index) {
  return hierarchy_normal[find_rigid_body(rigid_body_index, refinement_level)]
                         [particle_index];
}

vec3 rigid_body_surface_particle_hierarchy::get_spacing(int rigid_body_index,
                                                        int refinement_level,
                                                        int particle_index) {
  return hierarchy_spacing[find_rigid_body(rigid_body_index, refinement_level)]
                          [particle_index];
}

void rigid_body_surface_particle_hierarchy::get_coarse_level_coordinate(
    const int rigid_body_index, shared_ptr<vector<vec3>> &coord_ptr) {
  coord_ptr = make_shared<vector<vec3>>(
      hierarchy_coord[find_rigid_body(rigid_body_index, 0)]);
}

void rigid_body_surface_particle_hierarchy::get_coarse_level_normal(
    const int rigid_body_index, shared_ptr<vector<vec3>> &normal_ptr) {
  normal_ptr = make_shared<vector<vec3>>(
      hierarchy_normal[find_rigid_body(rigid_body_index, 0)]);
}

void rigid_body_surface_particle_hierarchy::get_coarse_level_spacing(
    const int rigid_body_index, shared_ptr<vector<vec3>> &spacing_ptr) {
  spacing_ptr = make_shared<vector<vec3>>(
      hierarchy_spacing[find_rigid_body(rigid_body_index, 0)]);
}

void rigid_body_surface_particle_hierarchy::write_log() {
  for (int i = 0; i < rb_idx.size(); i++) {
    cout << rb_idx[i] << endl;
  }
}