#include "rigid_body_surface_particle_hierarchy.hpp"

#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace Compadre;

void rigid_body_surface_particle_hierarchy::custom_shape_normal(
    int rigid_body_index, const vec3 &pos, vec3 &norm) {
  auto &rigid_body_size = rb_mgr->get_rigid_body_size(rigid_body_index);
  double r1 = rigid_body_size[0];
  double r2 = rigid_body_size[1];
  double d = rigid_body_size[2];

  double theta1 = 0.5 * M_PI + asin((d - r2) / (r1 - r2));
  double s = sqrt((pow(r1 - r2, 2.0) - pow(d - r2, 2.0)));
  double theta2 = M_PI - atan(s / d);

  double r = pos.mag();
  double theta = acos(pos[2] / r);
  double phi = atan2(pos[1], pos[0]);

  if (theta < theta1) {
    norm = vec3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
  } else if (theta < theta2) {
    vec3 new_center =
        vec3((r1 - r2) * cos(phi) * sin(theta1),
             (r1 - r2) * sin(phi) * sin(theta1), (r1 - r2) * cos(theta1));
    vec3 new_pos = pos - new_center;
    double new_r = new_pos.mag();
    double new_theta = acos(new_pos[2] / new_r);
    double new_phi = atan2(new_pos[1], new_pos[0]);

    norm = vec3(cos(new_phi) * sin(new_theta), sin(new_phi) * sin(new_theta),
                cos(new_theta));
  } else {
    norm = vec3(0.0, 0.0, -1.0);
  }
}

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
      add_sphere(rigid_body_size_list[compressed_rigid_body_index][0],
                 resolution);
    }
    if (dimension == 2) {
      add_circle(rigid_body_size_list[compressed_rigid_body_index][0],
                 resolution);
    }
    break;
  case 2:
    if (dimension == 2) {
      add_rounded_square(rigid_body_size_list[compressed_rigid_body_index][0],
                         resolution);
    }
    break;
  case 4:
    if (dimension == 3) {
      add_ellipsoid(rigid_body_size_list[compressed_rigid_body_index][0],
                    rigid_body_size_list[compressed_rigid_body_index][1],
                    rigid_body_size_list[compressed_rigid_body_index][2],
                    resolution);
    }
    break;
  case 5:
    if (dimension == 3) {
      add_customized_shape(rigid_body_size_list[compressed_rigid_body_index][0],
                           resolution);
    }
    break;
  }

  mapping[compressed_rigid_body_index].push_back(hierarchy_coord.size() - 1);
  hierarchy_index.push_back(vector<int>());
  hierarchy.push_back(vector<int>());
}

void rigid_body_surface_particle_hierarchy::add_circle(const double radius,
                                                       const double h) {
  hierarchy_coord.push_back(vector<vec3>());
  hierarchy_normal.push_back(vector<vec3>());
  hierarchy_spacing.push_back(vector<vec3>());
  hierarchy_element.push_back(vector<triple<int>>());

  vector<vec3> &coord = hierarchy_coord[hierarchy_coord.size() - 1];
  vector<vec3> &normal = hierarchy_normal[hierarchy_normal.size() - 1];
  vector<vec3> &spacing = hierarchy_spacing[hierarchy_spacing.size() - 1];

  int M_theta = round(2.0 * M_PI * radius / h);
  double d_theta = 2.0 * M_PI * radius / M_theta;

  vec3 p_spacing = vec3(d_theta, 0, 0);

  for (int i = 0; i < M_theta; ++i) {
    double theta = 2.0 * M_PI / M_theta * (i + 0.5);
    vec3 norm = vec3(cos(theta), sin(theta), 0.0);

    coord.push_back(norm * radius);
    normal.push_back(norm);
    spacing.push_back(p_spacing);
  }
}

void rigid_body_surface_particle_hierarchy::add_sphere(const double radius,
                                                       const double h) {
  hierarchy_coord.push_back(vector<vec3>());
  hierarchy_normal.push_back(vector<vec3>());
  hierarchy_spacing.push_back(vector<vec3>());
  hierarchy_element.push_back(vector<triple<int>>());

  vector<vec3> &coord = hierarchy_coord[hierarchy_coord.size() - 1];
  vector<vec3> &normal = hierarchy_normal[hierarchy_normal.size() - 1];
  vector<vec3> &spacing = hierarchy_spacing[hierarchy_spacing.size() - 1];
  vector<triple<int>> &element =
      hierarchy_element[hierarchy_element.size() - 1];

  ifstream input("shape/sphere.txt", ios::in);
  if (!input.is_open()) {
    PetscPrintf(PETSC_COMM_WORLD, "surface point input file does not exist\n");
    return;
  }

  int adaptive_level;
  input >> adaptive_level;
  hierarchy_adaptive_level.push_back(adaptive_level);

  while (!input.eof()) {
    vec3 xyz;
    for (int i = 0; i < 3; i++) {
      input >> xyz[i];
    }

    coord.push_back(xyz);
    double mag = 1.0 / xyz.mag();
    normal.push_back(xyz * mag);
    spacing.push_back(vec3(1.0, 0.0, 0.0));
  }

  input.close();

  input.open("shape/sphere_element.txt", ios::in);

  if (!input.is_open()) {
    PetscPrintf(PETSC_COMM_WORLD,
                "surface element input file does not exist\n");
    return;
  }

  while (!input.eof()) {
    triple<int> idx;
    for (int i = 0; i < 3; i++) {
      input >> idx[i];
      idx[i]--;
    }

    element.push_back(idx);
  }

  input.close();
}

void rigid_body_surface_particle_hierarchy::add_rounded_square(
    const double half_side_length, const double h) {
  hierarchy_coord.push_back(vector<vec3>());
  hierarchy_normal.push_back(vector<vec3>());
  hierarchy_spacing.push_back(vector<vec3>());
  hierarchy_element.push_back(vector<triple<int>>());

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

void rigid_body_surface_particle_hierarchy::add_ellipsoid(const double x,
                                                          const double y,
                                                          const double z,
                                                          const double h) {
  hierarchy_coord.push_back(vector<vec3>());
  hierarchy_normal.push_back(vector<vec3>());
  hierarchy_spacing.push_back(vector<vec3>());
  hierarchy_element.push_back(vector<triple<int>>());

  vector<vec3> &coord = hierarchy_coord[hierarchy_coord.size() - 1];
  vector<vec3> &normal = hierarchy_normal[hierarchy_normal.size() - 1];
  vector<vec3> &spacing = hierarchy_spacing[hierarchy_spacing.size() - 1];
  vector<triple<int>> &element =
      hierarchy_element[hierarchy_element.size() - 1];

  ifstream input("shape/ellipsoid.txt", ios::in);
  if (!input.is_open()) {
    PetscPrintf(PETSC_COMM_WORLD, "surface point input file does not exist\n");
    return;
  }

  int adaptive_level;
  input >> adaptive_level;
  hierarchy_adaptive_level.push_back(adaptive_level);

  while (!input.eof()) {
    vec3 xyz;
    for (int i = 0; i < 3; i++) {
      input >> xyz[i];
    }

    coord.push_back(xyz);
    vec3 norm =
        vec3(xyz[0] / pow(x, 2.0), xyz[1] / pow(y, 2.0), xyz[2] / pow(z, 2.0));
    double mag = 1.0 / norm.mag();
    normal.push_back(norm * mag);
    spacing.push_back(vec3(1.0, 0.0, 0.0));
  }

  input.close();

  input.open("shape/ellipsoid_element.txt", ios::in);

  if (!input.is_open()) {
    PetscPrintf(PETSC_COMM_WORLD,
                "surface element input file does not exist\n");
    return;
  }

  while (!input.eof()) {
    triple<int> idx;
    for (int i = 0; i < 3; i++) {
      input >> idx[i];
      idx[i]--;
    }

    element.push_back(idx);
  }

  input.close();
}

void rigid_body_surface_particle_hierarchy::add_customized_shape(
    const double size, const double h) {
  hierarchy_coord.push_back(vector<vec3>());
  hierarchy_normal.push_back(vector<vec3>());
  hierarchy_spacing.push_back(vector<vec3>());
  hierarchy_element.push_back(vector<triple<int>>());

  vector<vec3> &coord = hierarchy_coord[hierarchy_coord.size() - 1];
  vector<vec3> &normal = hierarchy_normal[hierarchy_normal.size() - 1];
  vector<vec3> &spacing = hierarchy_spacing[hierarchy_spacing.size() - 1];
  vector<triple<int>> &element =
      hierarchy_element[hierarchy_element.size() - 1];

  ifstream input("shape/microcolony.txt", ios::in);
  if (!input.is_open()) {
    PetscPrintf(PETSC_COMM_WORLD, "surface point input file does not exist\n");
    return;
  }

  int adaptive_level;
  input >> adaptive_level;
  hierarchy_adaptive_level.push_back(adaptive_level);

  while (!input.eof()) {
    vec3 xyz;
    for (int i = 0; i < 3; i++) {
      input >> xyz[i];
    }

    vec3 norm;
    custom_shape_normal(0, xyz, norm);

    coord.push_back(xyz);
    normal.push_back(norm);
    spacing.push_back(vec3(1.0, 0.0, 0.0));
  }

  input.close();

  input.open("shape/microcolony_element.txt", ios::in);

  if (!input.is_open()) {
    PetscPrintf(PETSC_COMM_WORLD,
                "surface element input file does not exist\n");
    return;
  }

  while (!input.eof()) {
    triple<int> idx;
    for (int i = 0; i < 3; i++) {
      input >> idx[i];
      idx[i]--;
    }

    element.push_back(idx);
  }

  input.close();
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
    auto &current_rigid_body_size = rb_mgr->get_rigid_body_size(i);

    for (int j = 0; j < rigid_body_type_list.size(); j++) {
      if (rigid_body_type_list[j] == current_rigid_body_type &&
          abs(rigid_body_size_list[j][0] - current_rigid_body_size[0]) <
              1e-5 * current_rigid_body_size[0]) {
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

void rigid_body_surface_particle_hierarchy::get_coarse_level_element(
    const int rigid_body_index, shared_ptr<vector<triple<int>>> &element_ptr) {
  element_ptr = make_shared<vector<triple<int>>>(
      hierarchy_element[find_rigid_body(rigid_body_index, 0)]);
}

void rigid_body_surface_particle_hierarchy::write_log() {
  for (int i = 0; i < rb_idx.size(); i++) {
    cout << rb_idx[i] << endl;
  }
}

void rigid_body_surface_particle_hierarchy::move_to_boundary(
    int rigid_body_index, vec3 &pos) {
  switch (rigid_body_type_list[rb_idx[rigid_body_index]]) {
  case 1: {
    double mag = pos.mag();
    double r = rigid_body_size_list[rb_idx[rigid_body_index]][0];
    pos = pos * (r / mag);
  } break;
  case 2:
    break;
  case 4: {
    double a = rigid_body_size_list[rb_idx[rigid_body_index]][0];
    double b = rigid_body_size_list[rb_idx[rigid_body_index]][1];
    double c = rigid_body_size_list[rb_idx[rigid_body_index]][2];

    pos[0] /= a;
    pos[1] /= b;
    pos[2] /= c;

    double mag = pos.mag();
    pos *= (1.0 / mag);

    pos[0] *= a;
    pos[1] *= b;
    pos[2] *= c;
  } break;
  case 5: {
    auto &rigid_body_size = rb_mgr->get_rigid_body_size(rigid_body_index);
    double r1 = rigid_body_size[0];
    double r2 = rigid_body_size[1];
    double d = rigid_body_size[2];

    double theta1 = 0.5 * M_PI + asin((d - r2) / (r1 - r2));
    double s = sqrt((pow(r1 - r2, 2.0) - pow(d - r2, 2.0)));
    double theta2 = M_PI - atan(s / d);

    double r = pos.mag();
    double theta = acos(pos[2] / r);
    double phi = atan2(pos[1], pos[0]);

    if (theta < theta1) {
      r = r1;
    } else if (theta < theta2) {
      double theta_prime = theta - theta1;
      r = (r1 - r2) * cos(theta_prime) +
          sqrt(pow(r1 - r2, 2.0) * pow(cos(theta_prime), 2.0) -
               (pow(r1 - r2, 2.0) - pow(r2, 2.0)));
    } else {
      r = d / cos(M_PI - theta);
    }

    pos[0] = r * cos(phi) * sin(theta);
    pos[1] = r * sin(phi) * sin(theta);
    pos[2] = r * cos(theta);

    if (isnan(pos.mag()))
      cout << r << ' ' << phi << ' ' << theta << endl;
  } break;
  }
}

void rigid_body_surface_particle_hierarchy::get_normal(int rigid_body_index,
                                                       vec3 pos, vec3 &norm) {

  switch (rigid_body_type_list[rb_idx[rigid_body_index]]) {
  case 1: {
    double mag = pos.mag();
    norm = pos * (1.0 / mag);
  } break;
  case 2:
    break;
  case 4: {
    double x = rigid_body_size_list[rb_idx[rigid_body_index]][0];
    double y = rigid_body_size_list[rb_idx[rigid_body_index]][1];
    double z = rigid_body_size_list[rb_idx[rigid_body_index]][2];

    norm =
        vec3(pos[0] / pow(x, 2.0), pos[1] / pow(y, 2.0), pos[2] / pow(z, 2.0));
    double mag = 1.0 / norm.mag();
    norm *= mag;
  } break;
  case 5: {
    auto &rigid_body_size = rb_mgr->get_rigid_body_size(rigid_body_index);
    double r1 = rigid_body_size[0];
    double r2 = rigid_body_size[1];
    double d = rigid_body_size[2];

    double theta1 = 0.5 * M_PI + asin((d - r2) / (r1 - r2));
    double s = sqrt((pow(r1 - r2, 2.0) - pow(d - r2, 2.0)));
    double theta2 = M_PI - atan(s / d);

    double r = pos.mag();
    double theta = acos(pos[2] / r);
    double phi = atan2(pos[1], pos[0]);

    if (theta < theta1) {
      norm = vec3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
    } else if (theta < theta2) {
      vec3 new_center =
          vec3((r1 - r2) * cos(phi) * sin(theta1),
               (r1 - r2) * sin(phi) * sin(theta1), (r1 - r2) * cos(theta1));
      vec3 new_pos = pos - new_center;
      double new_r = new_pos.mag();
      double new_theta = acos(new_pos[2] / new_r);
      double new_phi = atan2(new_pos[1], new_pos[0]);

      norm = vec3(cos(new_phi) * sin(new_theta), sin(new_phi) * sin(new_theta),
                  cos(new_theta));
    } else {
      norm = vec3(0.0, 0.0, -1.0);
    }
  } break;
  }
}