#include "particle_geometry.hpp"

#include <Compadre_PointCloudSearch.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <tgmath.h>
#include <vector>

#include <mpi.h>

using namespace std;
using namespace Compadre;

static void process_split(int &x, int &y, int &i, int &j, const int size,
                          const int rank) {
  x = sqrt(size);
  while (x > 0) {
    y = size / x;
    if (size == x * y) {
      break;
    } else {
      x--;
    }
  }

  i = rank % x;
  j = rank / y;
}

static void process_split(int &x, int &y, int &z, int &i, int &j, int &k,
                          const int size, const int rank) {
  x = cbrt(size);
  bool splitFound = false;
  while (x > 0 && splitFound == false) {
    y = x;
    while (y > 0 && splitFound == false) {
      z = size / (x * y);
      if (size == (x * y * z)) {
        splitFound = true;
        break;
      } else {
        y--;
      }
    }
    if (splitFound == false) {
      x--;
    }
  }

  i = (rank % (x * y)) % x;
  j = (rank % (x * y)) / x;
  k = rank / (x * y);
}

static int bounding_box_split(vec3 &bounding_box_size,
                              triple<int> &bounding_box_count,
                              vec3 &bounding_box_low, double _spacing,
                              vec3 &domain_bounding_box_low,
                              vec3 &domain_bounding_box_high, vec3 &domain_low,
                              vec3 &domain_high, triple<int> &domain_count,
                              int x, int y, int i, int j) {
  for (int ite = 0; ite < 2; ite++) {
    bounding_box_count[ite] = bounding_box_size[ite] / _spacing;
  }

  vector<int> count_x, count_y;
  for (int ite = 0; ite < x; ite++) {
    if (bounding_box_count[0] % x > ite) {
      count_x.push_back(bounding_box_count[0] / x + 1);
    } else {
      count_x.push_back(bounding_box_count[0] / x);
    }
  }

  for (int ite = 0; ite < y; ite++) {
    if (bounding_box_count[1] % y > ite) {
      count_y.push_back(bounding_box_count[1] / y + 1);
    } else {
      count_y.push_back(bounding_box_count[1] / y);
    }
  }

  domain_count[0] = count_x[i];
  domain_count[1] = count_y[j];

  double x_start = bounding_box_low[0];
  double y_start = bounding_box_low[1];

  for (int ite = 0; ite < i; ite++) {
    x_start += count_x[ite] * _spacing;
  }
  for (int ite = 0; ite < j; ite++) {
    y_start += count_y[ite] * _spacing;
  }

  double x_end = x_start + count_x[i] * _spacing;
  double y_end = y_start + count_y[j] * _spacing;

  domain_low[0] = x_start;
  domain_low[1] = y_start;
  domain_high[0] = x_end;
  domain_high[1] = y_end;

  domain_bounding_box_low[0] =
      bounding_box_size[0] / x * i + bounding_box_low[0];
  domain_bounding_box_low[1] =
      bounding_box_size[1] / y * j + bounding_box_low[1];
  domain_bounding_box_high[0] =
      bounding_box_size[0] / x * (i + 1) + bounding_box_low[0];
  domain_bounding_box_high[1] =
      bounding_box_size[1] / y * (j + 1) + bounding_box_low[1];

  return 0;
}

static int bounding_box_split(vec3 &bounding_box_size,
                              triple<int> &bounding_box_count,
                              vec3 &bounding_box_low, double _spacing,
                              vec3 &domain_bounding_box_low,
                              vec3 &domain_bounding_box_high, vec3 &domain_low,
                              vec3 &domain_high, triple<int> &domain_count,
                              const int x, const int y, const int z,
                              const int i, const int j, const int k) {
  for (int ite = 0; ite < 3; ite++) {
    bounding_box_count[ite] = bounding_box_size[ite] / _spacing;
  }

  std::vector<int> count_x;
  std::vector<int> count_y;
  std::vector<int> count_z;

  for (int ite = 0; ite < x; ite++) {
    if (bounding_box_count[0] % x > ite) {
      count_x.push_back(bounding_box_count[0] / x + 1);
    } else {
      count_x.push_back(bounding_box_count[0] / x);
    }
  }

  for (int ite = 0; ite < y; ite++) {
    if (bounding_box_count[1] % y > ite) {
      count_y.push_back(bounding_box_count[1] / y + 1);
    } else {
      count_y.push_back(bounding_box_count[1] / y);
    }
  }

  for (int ite = 0; ite < z; ite++) {
    if (bounding_box_count[2] % z > ite) {
      count_z.push_back(bounding_box_count[2] / z + 1);
    } else {
      count_z.push_back(bounding_box_count[2] / z);
    }
  }

  domain_count[0] = count_x[i];
  domain_count[1] = count_y[j];
  domain_count[2] = count_z[k];

  double x_start = bounding_box_low[0];
  double y_start = bounding_box_low[1];
  double z_start = bounding_box_low[2];
  for (int ite = 0; ite < i; ite++) {
    x_start += count_x[ite] * _spacing;
  }
  for (int ite = 0; ite < j; ite++) {
    y_start += count_y[ite] * _spacing;
  }
  for (int ite = 0; ite < k; ite++) {
    z_start += count_z[ite] * _spacing;
  }

  double x_end = x_start + count_x[i] * _spacing;
  double y_end = y_start + count_y[j] * _spacing;
  double z_end = z_start + count_z[k] * _spacing;

  domain_low[0] = x_start;
  domain_low[1] = y_start;
  domain_low[2] = z_start;
  domain_high[0] = x_end;
  domain_high[1] = y_end;
  domain_high[2] = z_end;

  domain_bounding_box_low[0] =
      bounding_box_size[0] / x * i + bounding_box_low[0];
  domain_bounding_box_low[1] =
      bounding_box_size[1] / y * j + bounding_box_low[1];
  domain_bounding_box_low[2] =
      bounding_box_size[2] / z * k + bounding_box_low[2];
  domain_bounding_box_high[0] =
      bounding_box_size[0] / x * (i + 1) + bounding_box_low[0];
  domain_bounding_box_high[1] =
      bounding_box_size[1] / y * (j + 1) + bounding_box_low[1];
  domain_bounding_box_high[2] =
      bounding_box_size[2] / z * (k + 1) + bounding_box_low[2];

  return 0;
}

void particle_geometry::init(const int _dim, const int _problem_type,
                             const int _refinement_type, double _spacing,
                             double _cutoff_multiplier,
                             string geometry_input_file_name) {
  dim = _dim;
  problem_type = _problem_type;
  refinement_type = _refinement_type;
  spacing = _spacing;
  cutoff_multiplier = _cutoff_multiplier;
  cutoff_distance = _spacing * (cutoff_multiplier + 0.5);

  if (geometry_input_file_name != "") {
  } else {
    // default setup
    bounding_box_size[0] = 2.0;
    bounding_box_size[1] = 2.0;
    bounding_box_size[2] = 2.0;

    bounding_box[0][0] = -1.0;
    bounding_box[1][0] = 1.0;
    bounding_box[0][1] = -1.0;
    bounding_box[1][1] = 1.0;
    bounding_box[0][2] = -1.0;
    bounding_box[1][2] = 1.0;

    if (dim == 2) {
      process_split(process_x, process_y, process_i, process_j, size, rank);
    } else if (dim == 3) {
      process_split(process_x, process_y, process_z, process_i, process_j,
                    process_k, size, rank);
    }
  }

  if (dim == 2) {
    bounding_box_split(bounding_box_size, bounding_box_count, bounding_box[0],
                       _spacing, domain_bounding_box[0], domain_bounding_box[1],
                       domain[0], domain[1], domain_count, process_x, process_y,
                       process_i, process_j);
  }
  if (dim == 3) {
    bounding_box_split(bounding_box_size, bounding_box_count, bounding_box[0],
                       _spacing, domain_bounding_box[0], domain_bounding_box[1],
                       domain[0], domain[1], domain_count, process_x, process_y,
                       process_z, process_i, process_j, process_k);
  }

  init_domain_boundary();
}

void particle_geometry::init_rigid_body(shared_ptr<rigid_body_manager> mgr) {
  rb_mgr = mgr;
}

void particle_geometry::generate_uniform_particle() {
  // prepare data storage
  current_local_work_particle_coord = make_shared<vector<vec3>>();
  current_local_work_particle_normal = make_shared<vector<vec3>>();
  current_local_work_particle_p_spacing = make_shared<vector<vec3>>();
  current_local_work_particle_spacing = make_shared<vector<double>>();
  current_local_work_particle_volume = make_shared<vector<double>>();
  current_local_work_particle_index = make_shared<vector<int>>();
  current_local_work_particle_type = make_shared<vector<int>>();
  current_local_work_particle_adaptive_level = make_shared<vector<int>>();
  current_local_work_particle_new_added = make_shared<vector<int>>();
  current_local_work_particle_attached_rigid_body = make_shared<vector<int>>();
  current_local_work_particle_num_neighbor = make_shared<vector<int>>();

  current_local_work_ghost_particle_coord = make_shared<vector<vec3>>();
  current_local_work_ghost_particle_volume = make_shared<vector<double>>();
  current_local_work_ghost_particle_index = make_shared<vector<int>>();

  current_local_managing_particle_coord = make_shared<vector<vec3>>();
  current_local_managing_particle_normal = make_shared<vector<vec3>>();
  current_local_managing_particle_p_spacing = make_shared<vector<vec3>>();
  current_local_managing_particle_p_coord = make_shared<vector<vec3>>();
  current_local_managing_particle_spacing = make_shared<vector<double>>();
  current_local_managing_particle_volume = make_shared<vector<double>>();
  current_local_managing_particle_index = make_shared<vector<long long>>();
  current_local_managing_particle_type = make_shared<vector<int>>();
  current_local_managing_particle_adaptive_level = make_shared<vector<int>>();
  current_local_managing_particle_new_added = make_shared<vector<int>>();
  current_local_managing_particle_attached_rigid_body =
      make_shared<vector<int>>();

  local_managing_gap_particle_coord = make_shared<vector<vec3>>();
  local_managing_gap_particle_normal = make_shared<vector<vec3>>();
  local_managing_gap_particle_p_coord = make_shared<vector<vec3>>();
  local_managing_gap_particle_volume = make_shared<vector<double>>();
  local_managing_gap_particle_spacing = make_shared<vector<double>>();
  local_managing_gap_particle_particle_type = make_shared<vector<int>>();
  local_managing_gap_particle_adaptive_level = make_shared<vector<int>>();

  generate_rigid_body_surface_particle();
  generate_field_particle();

  index_particle();

  balance_workload();

  mitigate_forward(current_local_managing_particle_coord,
                   current_local_work_particle_coord);
  mitigate_forward(current_local_managing_particle_normal,
                   current_local_work_particle_normal);
  mitigate_forward(current_local_managing_particle_p_spacing,
                   current_local_work_particle_p_spacing);
  mitigate_forward(current_local_managing_particle_spacing,
                   current_local_work_particle_spacing);
  mitigate_forward(current_local_managing_particle_volume,
                   current_local_work_particle_volume);
  mitigate_forward(current_local_managing_particle_type,
                   current_local_work_particle_type);
  mitigate_forward(current_local_managing_particle_adaptive_level,
                   current_local_work_particle_adaptive_level);
  mitigate_forward(current_local_managing_particle_new_added,
                   current_local_work_particle_new_added);
  mitigate_forward(current_local_managing_particle_attached_rigid_body,
                   current_local_work_particle_attached_rigid_body);

  index_work_particle();

  build_ghost();

  ghost_forward(current_local_work_particle_coord,
                current_local_work_ghost_particle_coord);
  ghost_forward(current_local_work_particle_volume,
                current_local_work_ghost_particle_volume);
  ghost_forward(current_local_work_particle_index,
                current_local_work_ghost_particle_index);
}

void particle_geometry::clear_particle() {}

void particle_geometry::mitigate_forward(int_type source, int_type target) {
  int num_target_num = source->size();
  for (int i = 0; i < mitigation_in_num.size(); i++) {
    num_target_num += mitigation_in_num[i];
  }
  for (int i = 0; i < mitigation_out_num.size(); i++) {
    num_target_num -= mitigation_out_num[i];
  }

  target->resize(num_target_num);

  auto &source_vec = *source;
  auto &target_vec = *target;

  vector<MPI_Request> send_request;
  vector<MPI_Request> recv_request;

  vector<MPI_Status> send_status;
  vector<MPI_Status> recv_status;

  send_request.resize(mitigation_out_graph.size());
  recv_request.resize(mitigation_in_graph.size());
  send_status.resize(mitigation_out_graph.size());
  recv_status.resize(mitigation_in_graph.size());

  vector<int> send_buffer, recv_buffer;
  send_buffer.resize(mitigation_out_offset[mitigation_out_graph.size()]);
  recv_buffer.resize(mitigation_in_offset[mitigation_in_graph.size()]);

  // prepare send buffer
  for (int i = 0; i < local_mitigation_map.size(); i++) {
    send_buffer[i] = source_vec[local_mitigation_map[i]];
  }

  // send and recv data buffer
  for (int i = 0; i < mitigation_out_graph.size(); i++) {
    MPI_Isend(send_buffer.data() + mitigation_out_offset[i],
              mitigation_out_num[i], MPI_INT, mitigation_out_graph[i], 0,
              MPI_COMM_WORLD, send_request.data() + i);
  }

  for (int i = 0; i < mitigation_in_graph.size(); i++) {
    MPI_Irecv(recv_buffer.data() + mitigation_in_offset[i],
              mitigation_in_num[i], MPI_INT, mitigation_in_graph[i], 0,
              MPI_COMM_WORLD, recv_request.data() + i);
  }

  MPI_Waitall(send_request.size(), send_request.data(), send_status.data());
  MPI_Waitall(recv_request.size(), recv_request.data(), recv_status.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store data
  for (int i = 0; i < local_reserve_map.size(); i++) {
    target_vec[i] = source_vec[local_reserve_map[i]];
  }
  const int local_reserve_size = local_reserve_map.size();
  for (int i = 0; i < recv_buffer.size(); i++) {
    target_vec[i + local_reserve_size] = recv_buffer[i];
  }
}

void particle_geometry::mitigate_forward(real_type source, real_type target) {
  int num_target_num = source->size();
  for (int i = 0; i < mitigation_in_num.size(); i++) {
    num_target_num += mitigation_in_num[i];
  }
  for (int i = 0; i < mitigation_out_num.size(); i++) {
    num_target_num -= mitigation_out_num[i];
  }

  target->resize(num_target_num);

  auto &source_vec = *source;
  auto &target_vec = *target;

  vector<MPI_Request> send_request;
  vector<MPI_Request> recv_request;

  vector<MPI_Status> send_status;
  vector<MPI_Status> recv_status;

  send_request.resize(mitigation_out_graph.size());
  recv_request.resize(mitigation_in_graph.size());
  send_status.resize(mitigation_out_graph.size());
  recv_status.resize(mitigation_in_graph.size());

  vector<double> send_buffer, recv_buffer;
  send_buffer.resize(mitigation_out_offset[mitigation_out_graph.size()]);
  recv_buffer.resize(mitigation_in_offset[mitigation_in_graph.size()]);

  // prepare send buffer
  for (int i = 0; i < local_mitigation_map.size(); i++) {
    send_buffer[i] = source_vec[local_mitigation_map[i]];
  }

  // send and recv data buffer
  for (int i = 0; i < mitigation_out_graph.size(); i++) {
    MPI_Isend(send_buffer.data() + mitigation_out_offset[i],
              mitigation_out_num[i], MPI_DOUBLE, mitigation_out_graph[i], 0,
              MPI_COMM_WORLD, send_request.data() + i);
  }

  for (int i = 0; i < mitigation_in_graph.size(); i++) {
    MPI_Irecv(recv_buffer.data() + mitigation_in_offset[i],
              mitigation_in_num[i], MPI_DOUBLE, mitigation_in_graph[i], 0,
              MPI_COMM_WORLD, recv_request.data() + i);
  }

  MPI_Waitall(send_request.size(), send_request.data(), send_status.data());
  MPI_Waitall(recv_request.size(), recv_request.data(), recv_status.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store data
  for (int i = 0; i < local_reserve_map.size(); i++) {
    target_vec[i] = source_vec[local_reserve_map[i]];
  }
  const int local_reserve_size = local_reserve_map.size();
  for (int i = 0; i < recv_buffer.size(); i++) {
    target_vec[i + local_reserve_size] = recv_buffer[i];
  }
}

void particle_geometry::mitigate_forward(vec_type source, vec_type target) {
  const int unit_length = 3;
  int num_target_num = source->size();
  for (int i = 0; i < mitigation_in_num.size(); i++) {
    num_target_num += mitigation_in_num[i];
  }
  for (int i = 0; i < mitigation_out_num.size(); i++) {
    num_target_num -= mitigation_out_num[i];
  }

  target->resize(num_target_num);

  auto &source_vec = *source;
  auto &target_vec = *target;

  vector<MPI_Request> send_request;
  vector<MPI_Request> recv_request;

  vector<MPI_Status> send_status;
  vector<MPI_Status> recv_status;

  send_request.resize(mitigation_out_graph.size());
  recv_request.resize(mitigation_in_graph.size());
  send_status.resize(mitigation_out_graph.size());
  recv_status.resize(mitigation_in_graph.size());

  vector<double> send_buffer, recv_buffer;
  send_buffer.resize(mitigation_out_offset[mitigation_out_graph.size()] *
                     unit_length);
  recv_buffer.resize(mitigation_in_offset[mitigation_in_graph.size()] *
                     unit_length);

  // prepare send buffer
  for (int i = 0; i < local_mitigation_map.size(); i++) {
    for (int j = 0; j < unit_length; j++)
      send_buffer[i * unit_length + j] = source_vec[local_mitigation_map[i]][j];
  }

  // send and recv data buffer
  for (int i = 0; i < mitigation_out_graph.size(); i++) {
    MPI_Isend(send_buffer.data() + mitigation_out_offset[i] * unit_length,
              mitigation_out_num[i] * unit_length, MPI_DOUBLE,
              mitigation_out_graph[i], 0, MPI_COMM_WORLD,
              send_request.data() + i);
  }

  for (int i = 0; i < mitigation_in_graph.size(); i++) {
    MPI_Irecv(recv_buffer.data() + mitigation_in_offset[i] * unit_length,
              mitigation_in_num[i] * unit_length, MPI_DOUBLE,
              mitigation_in_graph[i], 0, MPI_COMM_WORLD,
              recv_request.data() + i);
  }

  MPI_Waitall(send_request.size(), send_request.data(), send_status.data());
  MPI_Waitall(recv_request.size(), recv_request.data(), recv_status.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store data
  for (int i = 0; i < local_reserve_map.size(); i++) {
    target_vec[i] = source_vec[local_reserve_map[i]];
  }
  const int local_reserve_size = local_reserve_map.size();
  for (int i = 0; i < mitigation_in_offset[mitigation_in_graph.size()]; i++) {
    for (int j = 0; j < unit_length; j++)
      target_vec[i + local_reserve_size][j] = recv_buffer[i * unit_length + j];
  }
}

void particle_geometry::mitigate_backward(vector<int> &source,
                                          vector<int> &target) {
  int num_target_num = source.size();
  for (int i = 0; i < mitigation_in_num.size(); i++) {
    num_target_num -= mitigation_in_num[i];
  }
  for (int i = 0; i < mitigation_out_num.size(); i++) {
    num_target_num += mitigation_out_num[i];
  }

  target.resize(num_target_num);

  vector<MPI_Request> send_request;
  vector<MPI_Request> recv_request;

  vector<MPI_Status> send_status;
  vector<MPI_Status> recv_status;

  send_request.resize(mitigation_in_graph.size());
  recv_request.resize(mitigation_out_graph.size());
  send_status.resize(mitigation_in_graph.size());
  recv_status.resize(mitigation_out_graph.size());

  vector<int> send_buffer, recv_buffer;
  send_buffer.resize(mitigation_in_offset[mitigation_in_graph.size()]);
  recv_buffer.resize(mitigation_out_offset[mitigation_out_graph.size()]);

  // prepare send buffer
  const int local_reserve_size = local_reserve_map.size();
  for (int i = 0; i < mitigation_in_offset[mitigation_in_graph.size()]; i++) {
    send_buffer[i] = source[i + local_reserve_size];
  }

  // send and recv data buffer
  for (int i = 0; i < mitigation_in_graph.size(); i++) {
    MPI_Isend(send_buffer.data() + mitigation_in_offset[i],
              mitigation_in_num[i], MPI_INT, mitigation_in_graph[i], 0,
              MPI_COMM_WORLD, send_request.data() + i);
  }

  for (int i = 0; i < mitigation_out_graph.size(); i++) {
    MPI_Irecv(recv_buffer.data() + mitigation_out_offset[i],
              mitigation_out_num[i], MPI_INT, mitigation_out_graph[i], 0,
              MPI_COMM_WORLD, recv_request.data() + i);
  }

  MPI_Waitall(send_request.size(), send_request.data(), send_status.data());
  MPI_Waitall(recv_request.size(), recv_request.data(), recv_status.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store data
  for (int i = 0; i < local_mitigation_map.size(); i++) {
    target[local_mitigation_map[i]] = recv_buffer[i];
  }
  for (int i = 0; i < local_reserve_map.size(); i++) {
    target[local_reserve_map[i]] = source[i];
  }
}

void particle_geometry::ghost_forward(int_type source, int_type target) {
  int num_target_num = source->size();
  for (int i = 0; i < ghost_in_num.size(); i++) {
    num_target_num += ghost_in_num[i];
  }

  target->resize(num_target_num);

  auto &source_vec = *source;
  auto &target_vec = *target;

  vector<MPI_Request> send_request;
  vector<MPI_Request> recv_request;

  vector<MPI_Status> send_status;
  vector<MPI_Status> recv_status;

  send_request.resize(ghost_out_graph.size());
  recv_request.resize(ghost_in_graph.size());
  send_status.resize(ghost_out_graph.size());
  recv_status.resize(ghost_in_graph.size());

  vector<int> send_buffer, recv_buffer;
  send_buffer.resize(ghost_out_offset[ghost_out_graph.size()]);
  recv_buffer.resize(ghost_in_offset[ghost_in_graph.size()]);

  // prepare send buffer
  for (int i = 0; i < ghost_map.size(); i++) {
    send_buffer[i] = source_vec[ghost_map[i]];
  }

  // send and recv data buffer
  for (int i = 0; i < ghost_out_graph.size(); i++) {
    MPI_Isend(send_buffer.data() + ghost_out_offset[i], ghost_out_num[i],
              MPI_INT, ghost_out_graph[i], 0, MPI_COMM_WORLD,
              send_request.data() + i);
  }

  for (int i = 0; i < ghost_in_graph.size(); i++) {
    MPI_Irecv(recv_buffer.data() + ghost_in_offset[i], ghost_in_num[i], MPI_INT,
              ghost_in_graph[i], 0, MPI_COMM_WORLD, recv_request.data() + i);
  }

  MPI_Waitall(send_request.size(), send_request.data(), send_status.data());
  MPI_Waitall(recv_request.size(), recv_request.data(), recv_status.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store data
  for (int i = 0; i < source->size(); i++) {
    target_vec[i] = source_vec[i];
  }
  const int recv_offset = source->size();
  for (int i = 0; i < recv_buffer.size(); i++) {
    target_vec[i + recv_offset] = recv_buffer[i];
  }
}

void particle_geometry::ghost_forward(real_type source, real_type target) {
  int num_target_num = source->size();
  for (int i = 0; i < ghost_in_num.size(); i++) {
    num_target_num += ghost_in_num[i];
  }

  target->resize(num_target_num);

  auto &source_vec = *source;
  auto &target_vec = *target;

  vector<MPI_Request> send_request;
  vector<MPI_Request> recv_request;

  vector<MPI_Status> send_status;
  vector<MPI_Status> recv_status;

  send_request.resize(ghost_out_graph.size());
  recv_request.resize(ghost_in_graph.size());
  send_status.resize(ghost_out_graph.size());
  recv_status.resize(ghost_in_graph.size());

  vector<double> send_buffer, recv_buffer;
  send_buffer.resize(ghost_out_offset[ghost_out_graph.size()]);
  recv_buffer.resize(ghost_in_offset[ghost_in_graph.size()]);

  // prepare send buffer
  for (int i = 0; i < ghost_map.size(); i++) {
    send_buffer[i] = source_vec[ghost_map[i]];
  }

  // send and recv data buffer
  for (int i = 0; i < ghost_out_graph.size(); i++) {
    MPI_Isend(send_buffer.data() + ghost_out_offset[i], ghost_out_num[i],
              MPI_DOUBLE, ghost_out_graph[i], 0, MPI_COMM_WORLD,
              send_request.data() + i);
  }

  for (int i = 0; i < ghost_in_graph.size(); i++) {
    MPI_Irecv(recv_buffer.data() + ghost_in_offset[i], ghost_in_num[i],
              MPI_DOUBLE, ghost_in_graph[i], 0, MPI_COMM_WORLD,
              recv_request.data() + i);
  }

  MPI_Waitall(send_request.size(), send_request.data(), send_status.data());
  MPI_Waitall(recv_request.size(), recv_request.data(), recv_status.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store data
  for (int i = 0; i < source->size(); i++) {
    target_vec[i] = source_vec[i];
  }
  const int recv_offset = source->size();
  for (int i = 0; i < recv_buffer.size(); i++) {
    target_vec[i + recv_offset] = recv_buffer[i];
  }
}

void particle_geometry::ghost_forward(vec_type source, vec_type target) {
  const int unit_length = 3;
  int num_target_num = source->size();
  for (int i = 0; i < ghost_in_num.size(); i++) {
    num_target_num += ghost_in_num[i];
  }

  target->resize(num_target_num);

  auto &source_vec = *source;
  auto &target_vec = *target;

  vector<MPI_Request> send_request;
  vector<MPI_Request> recv_request;

  vector<MPI_Status> send_status;
  vector<MPI_Status> recv_status;

  send_request.resize(ghost_out_graph.size());
  recv_request.resize(ghost_in_graph.size());
  send_status.resize(ghost_out_graph.size());
  recv_status.resize(ghost_in_graph.size());

  vector<double> send_buffer, recv_buffer;
  send_buffer.resize(ghost_out_offset[ghost_out_graph.size()] * unit_length);
  recv_buffer.resize(ghost_in_offset[ghost_in_graph.size()] * unit_length);

  // prepare send buffer
  for (int i = 0; i < ghost_map.size(); i++) {
    for (int j = 0; j < unit_length; j++)
      send_buffer[i * unit_length + j] = source_vec[ghost_map[i]][j];
  }

  // send and recv data buffer
  for (int i = 0; i < ghost_out_graph.size(); i++) {
    MPI_Isend(send_buffer.data() + ghost_out_offset[i] * unit_length,
              ghost_out_num[i] * unit_length, MPI_DOUBLE, ghost_out_graph[i], 0,
              MPI_COMM_WORLD, send_request.data() + i);
  }

  for (int i = 0; i < ghost_in_graph.size(); i++) {
    MPI_Irecv(recv_buffer.data() + ghost_in_offset[i] * unit_length,
              ghost_in_num[i] * unit_length, MPI_DOUBLE, ghost_in_graph[i], 0,
              MPI_COMM_WORLD, recv_request.data() + i);
  }

  MPI_Waitall(send_request.size(), send_request.data(), send_status.data());
  MPI_Waitall(recv_request.size(), recv_request.data(), recv_status.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store data
  for (int i = 0; i < source->size(); i++) {
    target_vec[i] = source_vec[i];
  }
  const int recv_offset = source->size();
  for (int i = 0; i < ghost_in_offset[ghost_in_graph.size()]; i++) {
    for (int j = 0; j < unit_length; j++)
      target_vec[i + recv_offset][j] = recv_buffer[i * unit_length + j];
  }
}

void particle_geometry::ghost_forward(std::vector<int> &source_vec,
                                      std::vector<int> &target_vec) {
  int num_target_num = source_vec.size();
  for (int i = 0; i < ghost_in_num.size(); i++) {
    num_target_num += ghost_in_num[i];
  }

  target_vec.resize(num_target_num);

  vector<MPI_Request> send_request;
  vector<MPI_Request> recv_request;

  vector<MPI_Status> send_status;
  vector<MPI_Status> recv_status;

  send_request.resize(ghost_out_graph.size());
  recv_request.resize(ghost_in_graph.size());
  send_status.resize(ghost_out_graph.size());
  recv_status.resize(ghost_in_graph.size());

  vector<int> send_buffer, recv_buffer;
  send_buffer.resize(ghost_out_offset[ghost_out_graph.size()]);
  recv_buffer.resize(ghost_in_offset[ghost_in_graph.size()]);

  // prepare send buffer
  for (int i = 0; i < ghost_map.size(); i++) {
    send_buffer[i] = source_vec[ghost_map[i]];
  }

  // send and recv data buffer
  for (int i = 0; i < ghost_out_graph.size(); i++) {
    MPI_Isend(send_buffer.data() + ghost_out_offset[i], ghost_out_num[i],
              MPI_INT, ghost_out_graph[i], 0, MPI_COMM_WORLD,
              send_request.data() + i);
  }

  for (int i = 0; i < ghost_in_graph.size(); i++) {
    MPI_Irecv(recv_buffer.data() + ghost_in_offset[i], ghost_in_num[i], MPI_INT,
              ghost_in_graph[i], 0, MPI_COMM_WORLD, recv_request.data() + i);
  }

  MPI_Waitall(send_request.size(), send_request.data(), send_status.data());
  MPI_Waitall(recv_request.size(), recv_request.data(), recv_status.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store data
  for (int i = 0; i < source_vec.size(); i++) {
    target_vec[i] = source_vec[i];
  }
  const int recv_offset = source_vec.size();
  for (int i = 0; i < recv_buffer.size(); i++) {
    target_vec[i + recv_offset] = recv_buffer[i];
  }
}

void particle_geometry::ghost_forward(std::vector<double> &source_vec,
                                      std::vector<double> &target_vec) {
  int num_target_num = source_vec.size();
  for (int i = 0; i < ghost_in_num.size(); i++) {
    num_target_num += ghost_in_num[i];
  }

  target_vec.resize(num_target_num);

  vector<MPI_Request> send_request;
  vector<MPI_Request> recv_request;

  vector<MPI_Status> send_status;
  vector<MPI_Status> recv_status;

  send_request.resize(ghost_out_graph.size());
  recv_request.resize(ghost_in_graph.size());
  send_status.resize(ghost_out_graph.size());
  recv_status.resize(ghost_in_graph.size());

  vector<double> send_buffer, recv_buffer;
  send_buffer.resize(ghost_out_offset[ghost_out_graph.size()]);
  recv_buffer.resize(ghost_in_offset[ghost_in_graph.size()]);

  // prepare send buffer
  for (int i = 0; i < ghost_map.size(); i++) {
    send_buffer[i] = source_vec[ghost_map[i]];
  }

  // send and recv data buffer
  for (int i = 0; i < ghost_out_graph.size(); i++) {
    MPI_Isend(send_buffer.data() + ghost_out_offset[i], ghost_out_num[i],
              MPI_DOUBLE, ghost_out_graph[i], 0, MPI_COMM_WORLD,
              send_request.data() + i);
  }

  for (int i = 0; i < ghost_in_graph.size(); i++) {
    MPI_Irecv(recv_buffer.data() + ghost_in_offset[i], ghost_in_num[i],
              MPI_DOUBLE, ghost_in_graph[i], 0, MPI_COMM_WORLD,
              recv_request.data() + i);
  }

  MPI_Waitall(send_request.size(), send_request.data(), send_status.data());
  MPI_Waitall(recv_request.size(), recv_request.data(), recv_status.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store data
  for (int i = 0; i < source_vec.size(); i++) {
    target_vec[i] = source_vec[i];
  }
  const int recv_offset = source_vec.size();
  for (int i = 0; i < recv_buffer.size(); i++) {
    target_vec[i + recv_offset] = recv_buffer[i];
  }
}

void particle_geometry::ghost_forward(std::vector<vec3> &source_vec,
                                      std::vector<vec3> &target_vec) {
  const int unit_length = 3;
  int num_target_num = source_vec.size();
  for (int i = 0; i < ghost_in_num.size(); i++) {
    num_target_num += ghost_in_num[i];
  }

  target_vec.resize(num_target_num);

  vector<MPI_Request> send_request;
  vector<MPI_Request> recv_request;

  vector<MPI_Status> send_status;
  vector<MPI_Status> recv_status;

  send_request.resize(ghost_out_graph.size());
  recv_request.resize(ghost_in_graph.size());
  send_status.resize(ghost_out_graph.size());
  recv_status.resize(ghost_in_graph.size());

  vector<double> send_buffer, recv_buffer;
  send_buffer.resize(ghost_out_offset[ghost_out_graph.size()] * unit_length);
  recv_buffer.resize(ghost_in_offset[ghost_in_graph.size()] * unit_length);

  // prepare send buffer
  for (int i = 0; i < ghost_map.size(); i++) {
    for (int j = 0; j < unit_length; j++)
      send_buffer[i * unit_length + j] = source_vec[ghost_map[i]][j];
  }

  // send and recv data buffer
  for (int i = 0; i < ghost_out_graph.size(); i++) {
    MPI_Isend(send_buffer.data() + ghost_out_offset[i] * unit_length,
              ghost_out_num[i] * unit_length, MPI_DOUBLE, ghost_out_graph[i], 0,
              MPI_COMM_WORLD, send_request.data() + i);
  }

  for (int i = 0; i < ghost_in_graph.size(); i++) {
    MPI_Irecv(recv_buffer.data() + ghost_in_offset[i] * unit_length,
              ghost_in_num[i] * unit_length, MPI_DOUBLE, ghost_in_graph[i], 0,
              MPI_COMM_WORLD, recv_request.data() + i);
  }

  MPI_Waitall(send_request.size(), send_request.data(), send_status.data());
  MPI_Waitall(recv_request.size(), recv_request.data(), recv_status.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store data
  for (int i = 0; i < source_vec.size(); i++) {
    target_vec[i] = source_vec[i];
  }
  const int recv_offset = source_vec.size();
  for (int i = 0; i < ghost_in_offset[ghost_in_graph.size()]; i++) {
    for (int j = 0; j < unit_length; j++)
      target_vec[i + recv_offset][j] = recv_buffer[i * unit_length + j];
  }
}

void particle_geometry::ghost_forward(vector<vector<double>> &source_chunk,
                                      vector<vector<double>> &target_chunk,
                                      size_t unit_length) {
  int num_target_num = source_chunk.size();
  for (int i = 0; i < ghost_in_num.size(); i++) {
    num_target_num += ghost_in_num[i];
  }

  target_chunk.resize(num_target_num);

  vector<MPI_Request> send_request;
  vector<MPI_Request> recv_request;

  vector<MPI_Status> send_status;
  vector<MPI_Status> recv_status;

  send_request.resize(ghost_out_graph.size());
  recv_request.resize(ghost_in_graph.size());
  send_status.resize(ghost_out_graph.size());
  recv_status.resize(ghost_in_graph.size());

  vector<double> send_buffer, recv_buffer;
  send_buffer.resize(ghost_out_offset[ghost_out_graph.size()] * unit_length);
  recv_buffer.resize(ghost_in_offset[ghost_in_graph.size()] * unit_length);

  // prepare send buffer
  for (int i = 0; i < ghost_map.size(); i++) {
    for (int j = 0; j < unit_length; j++)
      send_buffer[i * unit_length + j] = source_chunk[ghost_map[i]][j];
  }

  // send and recv data buffer
  for (int i = 0; i < ghost_out_graph.size(); i++) {
    MPI_Isend(send_buffer.data() + ghost_out_offset[i] * unit_length,
              ghost_out_num[i] * unit_length, MPI_DOUBLE, ghost_out_graph[i], 0,
              MPI_COMM_WORLD, send_request.data() + i);
  }

  for (int i = 0; i < ghost_in_graph.size(); i++) {
    MPI_Irecv(recv_buffer.data() + ghost_in_offset[i] * unit_length,
              ghost_in_num[i] * unit_length, MPI_DOUBLE, ghost_in_graph[i], 0,
              MPI_COMM_WORLD, recv_request.data() + i);
  }

  MPI_Waitall(send_request.size(), send_request.data(), send_status.data());
  MPI_Waitall(recv_request.size(), recv_request.data(), recv_status.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store data
  for (int i = 0; i < source_chunk.size(); i++) {
    target_chunk[i] = source_chunk[i];
  }
  const int recv_offset = source_chunk.size();
  for (int i = 0; i < ghost_in_offset[ghost_in_graph.size()]; i++) {
    target_chunk[i + recv_offset].resize(unit_length);
    for (int j = 0; j < unit_length; j++)
      target_chunk[i + recv_offset][j] = recv_buffer[i * unit_length + j];
  }
}

void particle_geometry::ghost_clll_forward(int_type source, int_type target) {
  int num_target_num = reserve_clll_map.size();
  for (int i = 0; i < ghost_clll_in_num.size(); i++) {
    num_target_num += ghost_clll_in_num[i];
  }

  target->resize(num_target_num);

  auto &source_vec = *source;
  auto &target_vec = *target;

  vector<MPI_Request> send_request;
  vector<MPI_Request> recv_request;

  vector<MPI_Status> send_status;
  vector<MPI_Status> recv_status;

  send_request.resize(ghost_clll_out_graph.size());
  recv_request.resize(ghost_clll_in_graph.size());
  send_status.resize(ghost_clll_out_graph.size());
  recv_status.resize(ghost_clll_in_graph.size());

  vector<int> send_buffer, recv_buffer;
  send_buffer.resize(ghost_clll_out_offset[ghost_clll_out_graph.size()]);
  recv_buffer.resize(ghost_clll_in_offset[ghost_clll_in_graph.size()]);

  // prepare send buffer
  for (int i = 0; i < ghost_clll_map.size(); i++) {
    send_buffer[i] = source_vec[ghost_clll_map[i]];
  }

  // send and recv data buffer
  for (int i = 0; i < ghost_clll_out_graph.size(); i++) {
    MPI_Isend(send_buffer.data() + ghost_clll_out_offset[i],
              ghost_clll_out_num[i], MPI_INT, ghost_clll_out_graph[i], 0,
              MPI_COMM_WORLD, send_request.data() + i);
  }

  for (int i = 0; i < ghost_clll_in_graph.size(); i++) {
    MPI_Irecv(recv_buffer.data() + ghost_clll_in_offset[i],
              ghost_clll_in_num[i], MPI_INT, ghost_clll_in_graph[i], 0,
              MPI_COMM_WORLD, recv_request.data() + i);
  }

  MPI_Waitall(send_request.size(), send_request.data(), send_status.data());
  MPI_Waitall(recv_request.size(), recv_request.data(), recv_status.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store data
  for (int i = 0; i < reserve_clll_map.size(); i++) {
    target_vec[i] = source_vec[reserve_clll_map[i]];
  }
  const int recv_offset = reserve_clll_map.size();
  for (int i = 0; i < recv_buffer.size(); i++) {
    target_vec[i + recv_offset] = recv_buffer[i];
  }
}

void particle_geometry::ghost_clll_forward(real_type source, real_type target) {
  int num_target_num = reserve_clll_map.size();
  for (int i = 0; i < ghost_clll_in_num.size(); i++) {
    num_target_num += ghost_clll_in_num[i];
  }

  target->resize(num_target_num);

  auto &source_vec = *source;
  auto &target_vec = *target;

  vector<MPI_Request> send_request;
  vector<MPI_Request> recv_request;

  vector<MPI_Status> send_status;
  vector<MPI_Status> recv_status;

  send_request.resize(ghost_clll_out_graph.size());
  recv_request.resize(ghost_clll_in_graph.size());
  send_status.resize(ghost_clll_out_graph.size());
  recv_status.resize(ghost_clll_in_graph.size());

  vector<double> send_buffer, recv_buffer;
  send_buffer.resize(ghost_clll_out_offset[ghost_clll_out_graph.size()]);
  recv_buffer.resize(ghost_clll_in_offset[ghost_clll_in_graph.size()]);

  // prepare send buffer
  for (int i = 0; i < ghost_clll_map.size(); i++) {
    send_buffer[i] = source_vec[ghost_clll_map[i]];
  }

  // send and recv data buffer
  for (int i = 0; i < ghost_clll_out_graph.size(); i++) {
    MPI_Isend(send_buffer.data() + ghost_clll_out_offset[i],
              ghost_clll_out_num[i], MPI_DOUBLE, ghost_clll_out_graph[i], 0,
              MPI_COMM_WORLD, send_request.data() + i);
  }

  for (int i = 0; i < ghost_clll_in_graph.size(); i++) {
    MPI_Irecv(recv_buffer.data() + ghost_clll_in_offset[i],
              ghost_clll_in_num[i], MPI_DOUBLE, ghost_clll_in_graph[i], 0,
              MPI_COMM_WORLD, recv_request.data() + i);
  }

  MPI_Waitall(send_request.size(), send_request.data(), send_status.data());
  MPI_Waitall(recv_request.size(), recv_request.data(), recv_status.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store data
  for (int i = 0; i < reserve_clll_map.size(); i++) {
    target_vec[i] = source_vec[reserve_clll_map[i]];
  }
  const int recv_offset = reserve_clll_map.size();
  for (int i = 0; i < recv_buffer.size(); i++) {
    target_vec[i + recv_offset] = recv_buffer[i];
  }
}

void particle_geometry::ghost_clll_forward(vec_type source, vec_type target) {
  const int unit_length = 3;
  int num_target_num = reserve_clll_map.size();
  for (int i = 0; i < ghost_clll_in_num.size(); i++) {
    num_target_num += ghost_clll_in_num[i];
  }

  target->resize(num_target_num);

  auto &source_vec = *source;
  auto &target_vec = *target;

  vector<MPI_Request> send_request;
  vector<MPI_Request> recv_request;

  vector<MPI_Status> send_status;
  vector<MPI_Status> recv_status;

  send_request.resize(ghost_clll_out_graph.size());
  recv_request.resize(ghost_clll_in_graph.size());
  send_status.resize(ghost_clll_out_graph.size());
  recv_status.resize(ghost_clll_in_graph.size());

  vector<double> send_buffer, recv_buffer;
  send_buffer.resize(ghost_clll_out_offset[ghost_clll_out_graph.size()] *
                     unit_length);
  recv_buffer.resize(ghost_clll_in_offset[ghost_clll_in_graph.size()] *
                     unit_length);

  // prepare send buffer
  for (int i = 0; i < ghost_clll_map.size(); i++) {
    for (int j = 0; j < unit_length; j++)
      send_buffer[i * unit_length + j] = source_vec[ghost_clll_map[i]][j];
  }

  // send and recv data buffer
  for (int i = 0; i < ghost_clll_out_graph.size(); i++) {
    MPI_Isend(send_buffer.data() + ghost_clll_out_offset[i] * unit_length,
              ghost_clll_out_num[i] * unit_length, MPI_DOUBLE,
              ghost_clll_out_graph[i], 0, MPI_COMM_WORLD,
              send_request.data() + i);
  }

  for (int i = 0; i < ghost_clll_in_graph.size(); i++) {
    MPI_Irecv(recv_buffer.data() + ghost_clll_in_offset[i] * unit_length,
              ghost_clll_in_num[i] * unit_length, MPI_DOUBLE,
              ghost_clll_in_graph[i], 0, MPI_COMM_WORLD,
              recv_request.data() + i);
  }

  MPI_Waitall(send_request.size(), send_request.data(), send_status.data());
  MPI_Waitall(recv_request.size(), recv_request.data(), recv_status.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store data
  for (int i = 0; i < reserve_clll_map.size(); i++) {
    target_vec[i] = source_vec[reserve_clll_map[i]];
  }
  const int recv_offset = reserve_clll_map.size();
  for (int i = 0; i < ghost_clll_in_offset[ghost_clll_in_graph.size()]; i++) {
    for (int j = 0; j < unit_length; j++)
      target_vec[i + recv_offset][j] = recv_buffer[i * unit_length + j];
  }
}

void particle_geometry::ghost_llcl_forward(int_type source, int_type target) {
  int num_target_num = reserve_llcl_map.size();
  for (int i = 0; i < ghost_llcl_in_num.size(); i++) {
    num_target_num += ghost_llcl_in_num[i];
  }

  target->resize(num_target_num);

  auto &source_vec = *source;
  auto &target_vec = *target;

  vector<MPI_Request> send_request;
  vector<MPI_Request> recv_request;

  vector<MPI_Status> send_status;
  vector<MPI_Status> recv_status;

  send_request.resize(ghost_llcl_out_graph.size());
  recv_request.resize(ghost_llcl_in_graph.size());
  send_status.resize(ghost_llcl_out_graph.size());
  recv_status.resize(ghost_llcl_in_graph.size());

  vector<int> send_buffer, recv_buffer;
  send_buffer.resize(ghost_llcl_out_offset[ghost_llcl_out_graph.size()]);
  recv_buffer.resize(ghost_llcl_in_offset[ghost_llcl_in_graph.size()]);

  // prepare send buffer
  for (int i = 0; i < ghost_llcl_map.size(); i++) {
    send_buffer[i] = source_vec[ghost_llcl_map[i]];
  }

  // send and recv data buffer
  for (int i = 0; i < ghost_llcl_out_graph.size(); i++) {
    MPI_Isend(send_buffer.data() + ghost_llcl_out_offset[i],
              ghost_llcl_out_num[i], MPI_INT, ghost_llcl_out_graph[i], 0,
              MPI_COMM_WORLD, send_request.data() + i);
  }

  for (int i = 0; i < ghost_llcl_in_graph.size(); i++) {
    MPI_Irecv(recv_buffer.data() + ghost_llcl_in_offset[i],
              ghost_llcl_in_num[i], MPI_INT, ghost_llcl_in_graph[i], 0,
              MPI_COMM_WORLD, recv_request.data() + i);
  }

  MPI_Waitall(send_request.size(), send_request.data(), send_status.data());
  MPI_Waitall(recv_request.size(), recv_request.data(), recv_status.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store data
  for (int i = 0; i < reserve_llcl_map.size(); i++) {
    target_vec[i] = source_vec[reserve_llcl_map[i]];
  }
  const int recv_offset = reserve_llcl_map.size();
  for (int i = 0; i < recv_buffer.size(); i++) {
    target_vec[i + recv_offset] = recv_buffer[i];
  }
}

void particle_geometry::ghost_llcl_forward(real_type source, real_type target) {
  int num_target_num = reserve_llcl_map.size();
  for (int i = 0; i < ghost_llcl_in_num.size(); i++) {
    num_target_num += ghost_llcl_in_num[i];
  }

  target->resize(num_target_num);

  auto &source_vec = *source;
  auto &target_vec = *target;

  vector<MPI_Request> send_request;
  vector<MPI_Request> recv_request;

  vector<MPI_Status> send_status;
  vector<MPI_Status> recv_status;

  send_request.resize(ghost_llcl_out_graph.size());
  recv_request.resize(ghost_llcl_in_graph.size());
  send_status.resize(ghost_llcl_out_graph.size());
  recv_status.resize(ghost_llcl_in_graph.size());

  vector<double> send_buffer, recv_buffer;
  send_buffer.resize(ghost_llcl_out_offset[ghost_llcl_out_graph.size()]);
  recv_buffer.resize(ghost_llcl_in_offset[ghost_llcl_in_graph.size()]);

  // prepare send buffer
  for (int i = 0; i < ghost_llcl_map.size(); i++) {
    send_buffer[i] = source_vec[ghost_llcl_map[i]];
  }

  // send and recv data buffer
  for (int i = 0; i < ghost_llcl_out_graph.size(); i++) {
    MPI_Isend(send_buffer.data() + ghost_llcl_out_offset[i],
              ghost_llcl_out_num[i], MPI_DOUBLE, ghost_llcl_out_graph[i], 0,
              MPI_COMM_WORLD, send_request.data() + i);
  }

  for (int i = 0; i < ghost_llcl_in_graph.size(); i++) {
    MPI_Irecv(recv_buffer.data() + ghost_llcl_in_offset[i],
              ghost_llcl_in_num[i], MPI_DOUBLE, ghost_llcl_in_graph[i], 0,
              MPI_COMM_WORLD, recv_request.data() + i);
  }

  MPI_Waitall(send_request.size(), send_request.data(), send_status.data());
  MPI_Waitall(recv_request.size(), recv_request.data(), recv_status.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store data
  for (int i = 0; i < reserve_llcl_map.size(); i++) {
    target_vec[i] = source_vec[reserve_llcl_map[i]];
  }
  const int recv_offset = reserve_llcl_map.size();
  for (int i = 0; i < recv_buffer.size(); i++) {
    target_vec[i + recv_offset] = recv_buffer[i];
  }
}

void particle_geometry::ghost_llcl_forward(vec_type source, vec_type target) {
  const int unit_length = 3;
  int num_target_num = reserve_llcl_map.size();
  for (int i = 0; i < ghost_llcl_in_num.size(); i++) {
    num_target_num += ghost_llcl_in_num[i];
  }

  target->resize(num_target_num);

  auto &source_vec = *source;
  auto &target_vec = *target;

  vector<MPI_Request> send_request;
  vector<MPI_Request> recv_request;

  vector<MPI_Status> send_status;
  vector<MPI_Status> recv_status;

  send_request.resize(ghost_llcl_out_graph.size());
  recv_request.resize(ghost_llcl_in_graph.size());
  send_status.resize(ghost_llcl_out_graph.size());
  recv_status.resize(ghost_llcl_in_graph.size());

  vector<double> send_buffer, recv_buffer;
  send_buffer.resize(ghost_llcl_out_offset[ghost_llcl_out_graph.size()] *
                     unit_length);
  recv_buffer.resize(ghost_llcl_in_offset[ghost_llcl_in_graph.size()] *
                     unit_length);

  // prepare send buffer
  for (int i = 0; i < ghost_llcl_map.size(); i++) {
    for (int j = 0; j < unit_length; j++)
      send_buffer[i * unit_length + j] = source_vec[ghost_llcl_map[i]][j];
  }

  // send and recv data buffer
  for (int i = 0; i < ghost_llcl_out_graph.size(); i++) {
    MPI_Isend(send_buffer.data() + ghost_llcl_out_offset[i] * unit_length,
              ghost_llcl_out_num[i] * unit_length, MPI_DOUBLE,
              ghost_llcl_out_graph[i], 0, MPI_COMM_WORLD,
              send_request.data() + i);
  }

  for (int i = 0; i < ghost_llcl_in_graph.size(); i++) {
    MPI_Irecv(recv_buffer.data() + ghost_llcl_in_offset[i] * unit_length,
              ghost_llcl_in_num[i] * unit_length, MPI_DOUBLE,
              ghost_llcl_in_graph[i], 0, MPI_COMM_WORLD,
              recv_request.data() + i);
  }

  MPI_Waitall(send_request.size(), send_request.data(), send_status.data());
  MPI_Waitall(recv_request.size(), recv_request.data(), recv_status.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // store data
  for (int i = 0; i < reserve_llcl_map.size(); i++) {
    target_vec[i] = source_vec[reserve_llcl_map[i]];
  }
  const int recv_offset = reserve_llcl_map.size();
  for (int i = 0; i < ghost_llcl_in_offset[ghost_llcl_in_graph.size()]; i++) {
    for (int j = 0; j < unit_length; j++)
      target_vec[i + recv_offset][j] = recv_buffer[i * unit_length + j];
  }
}

void particle_geometry::refine(vector<int> &split_tag) {
  if (refinement_type == UNIFORM_REFINE) {
    uniform_refine();
  } else if (refinement_type == ADAPTIVE_REFINE) {
    adaptive_refine(split_tag);
  }

  index_particle();

  balance_workload();

  // restore current particle to last particle
  last_local_work_particle_coord.reset();
  last_local_work_particle_normal.reset();
  last_local_work_particle_spacing.reset();
  last_local_work_particle_volume.reset();
  last_local_work_particle_index.reset();
  last_local_work_particle_type.reset();
  last_local_work_particle_adaptive_level.reset();

  last_local_work_ghost_particle_coord.reset();
  last_local_work_ghost_particle_index.reset();

  last_local_work_particle_coord = move(current_local_work_particle_coord);
  last_local_work_particle_normal = move(current_local_work_particle_normal);
  last_local_work_particle_spacing = move(current_local_work_particle_spacing);
  last_local_work_particle_volume = move(current_local_work_particle_volume);
  last_local_work_particle_index = move(current_local_work_particle_index);
  last_local_work_particle_type = move(current_local_work_particle_type);
  last_local_work_particle_adaptive_level =
      move(current_local_work_particle_adaptive_level);

  last_local_work_ghost_particle_coord =
      move(current_local_work_ghost_particle_coord);
  last_local_work_ghost_particle_index =
      move(current_local_work_ghost_particle_index);

  current_local_work_particle_coord = make_shared<vector<vec3>>();
  current_local_work_particle_normal = make_shared<vector<vec3>>();
  current_local_work_particle_p_spacing = make_shared<vector<vec3>>();
  current_local_work_particle_spacing = make_shared<vector<double>>();
  current_local_work_particle_volume = make_shared<vector<double>>();
  current_local_work_particle_index = make_shared<vector<int>>();
  current_local_work_particle_type = make_shared<vector<int>>();
  current_local_work_particle_adaptive_level = make_shared<vector<int>>();
  current_local_work_particle_new_added = make_shared<vector<int>>();
  current_local_work_particle_attached_rigid_body = make_shared<vector<int>>();

  current_local_work_ghost_particle_coord = make_shared<vector<vec3>>();
  current_local_work_ghost_particle_volume = make_shared<vector<double>>();
  current_local_work_ghost_particle_index = make_shared<vector<int>>();

  mitigate_forward(current_local_managing_particle_coord,
                   current_local_work_particle_coord);
  mitigate_forward(current_local_managing_particle_normal,
                   current_local_work_particle_normal);
  mitigate_forward(current_local_managing_particle_p_spacing,
                   current_local_work_particle_p_spacing);
  mitigate_forward(current_local_managing_particle_spacing,
                   current_local_work_particle_spacing);
  mitigate_forward(current_local_managing_particle_volume,
                   current_local_work_particle_volume);
  mitigate_forward(current_local_managing_particle_type,
                   current_local_work_particle_type);
  mitigate_forward(current_local_managing_particle_adaptive_level,
                   current_local_work_particle_adaptive_level);
  mitigate_forward(current_local_managing_particle_new_added,
                   current_local_work_particle_new_added);
  mitigate_forward(current_local_managing_particle_attached_rigid_body,
                   current_local_work_particle_attached_rigid_body);

  index_work_particle();

  build_ghost();

  ghost_forward(current_local_work_particle_coord,
                current_local_work_ghost_particle_coord);
  ghost_forward(current_local_work_particle_volume,
                current_local_work_ghost_particle_volume);
  ghost_forward(current_local_work_particle_index,
                current_local_work_ghost_particle_index);

  build_ghost_from_last_level();
  build_ghost_for_last_level();

  clll_particle_coord.reset();
  clll_particle_index.reset();
  clll_particle_type.reset();
  llcl_particle_coord.reset();
  llcl_particle_index.reset();
  llcl_particle_type.reset();

  clll_particle_coord = make_shared<vector<vec3>>();
  clll_particle_index = make_shared<vector<int>>();
  clll_particle_type = make_shared<vector<int>>();
  llcl_particle_coord = make_shared<vector<vec3>>();
  llcl_particle_index = make_shared<vector<int>>();
  llcl_particle_type = make_shared<vector<int>>();

  ghost_clll_forward(last_local_work_particle_coord, clll_particle_coord);
  ghost_clll_forward(last_local_work_particle_index, clll_particle_index);
  ghost_clll_forward(last_local_work_particle_type, clll_particle_type);
  ghost_llcl_forward(current_local_work_particle_coord, llcl_particle_coord);
  ghost_llcl_forward(current_local_work_particle_index, llcl_particle_index);
  ghost_llcl_forward(current_local_work_particle_type, llcl_particle_type);
}

void particle_geometry::init_domain_boundary() {
  if (dim == 3) {
    // six faces as boundary
    // 0 front face
    // 1 right face
    // 2 back face
    // 3 bottom face
    // 4 left face
    // 5 top face
    domain_boundary_type.resize(6);
    if (abs(domain[1][0] - bounding_box[1][0]) < 1e-6) {
      domain_boundary_type[0] = 1;
    } else {
      domain_boundary_type[0] = 0;
    }

    if (abs(domain[1][1] - bounding_box[1][1]) < 1e-6) {
      domain_boundary_type[1] = 1;
    } else {
      domain_boundary_type[1] = 0;
    }

    if (abs(domain[0][0] - bounding_box[0][0]) < 1e-6) {
      domain_boundary_type[2] = 1;
    } else {
      domain_boundary_type[2] = 0;
    }

    if (abs(domain[0][2] - bounding_box[0][2]) < 1e-6) {
      domain_boundary_type[3] = 1;
    } else {
      domain_boundary_type[3] = 0;
    }

    if (abs(domain[0][1] - bounding_box[0][1]) < 1e-6) {
      domain_boundary_type[4] = 1;
    } else {
      domain_boundary_type[4] = 0;
    }

    if (abs(domain[1][2] - bounding_box[1][2]) < 1e-6) {
      domain_boundary_type[5] = 1;
    } else {
      domain_boundary_type[5] = 0;
    }
  }
  if (dim == 2) {
    // four edges as boundary
    // 0 down edge
    // 1 right edge
    // 2 up edge
    // 3 left edge
    domain_boundary_type.resize(4);
    if (abs(domain[0][0] - bounding_box[0][0]) < 1e-6) {
      domain_boundary_type[3] = 1;
    } else {
      domain_boundary_type[3] = 0;
    }

    if (abs(domain[0][1] - bounding_box[0][1]) < 1e-6) {
      domain_boundary_type[0] = 1;
    } else {
      domain_boundary_type[0] = 0;
    }

    if (abs(domain[1][0] - bounding_box[1][0]) < 1e-6) {
      domain_boundary_type[1] = 1;
    } else {
      domain_boundary_type[1] = 0;
    }

    if (abs(domain[1][1] - bounding_box[1][1]) < 1e-6) {
      domain_boundary_type[2] = 1;
    } else {
      domain_boundary_type[2] = 0;
    }
  }
}

void particle_geometry::generate_field_particle() {
  double pos_x, pos_y, pos_z;
  vec3 normal = vec3(1.0, 0.0, 0.0);
  vec3 boundary_normal;

  if (dim == 2) {
    pos_z = 0.0;
    double vol = spacing * spacing;

    // down
    if (domain_boundary_type[0] != 0) {
      pos_x = domain[0][0];
      pos_y = domain[0][1];
      if (domain_boundary_type[3] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        boundary_normal = vec3(sqrt(2) / 2.0, sqrt(2) / 2.0, 0.0);
        insert_particle(_pos, 1, spacing, boundary_normal, 0, vol);
      }
      pos_x += 0.5 * spacing;

      while (pos_x < domain[1][0] - 1e-5) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        boundary_normal = vec3(0.0, 1.0, 0.0);
        insert_particle(_pos, 2, spacing, boundary_normal, 0, vol);
        pos_x += spacing;
      }

      if (domain_boundary_type[1] != 0) {
        pos_x = domain[1][0];
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        boundary_normal = vec3(-sqrt(2) / 2.0, sqrt(2) / 2.0, 0.0);
        insert_particle(_pos, 1, spacing, boundary_normal, 0, vol);
      }
    }

    // fluid particle
    pos_y = domain[0][1] + spacing / 2.0;
    while (pos_y < domain[1][1] - 1e-5) {
      // left
      if (domain_boundary_type[3] != 0) {
        pos_x = domain[0][0];
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        boundary_normal = vec3(1.0, 0.0, 0.0);
        insert_particle(_pos, 2, spacing, boundary_normal, 0, vol);
      }

      pos_x = domain[0][0] + spacing / 2.0;
      while (pos_x < domain[1][0] - 1e-5) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        insert_particle(_pos, 0, spacing, normal, 0, vol);
        pos_x += spacing;
      }

      // right
      if (domain_boundary_type[1] != 0) {
        pos_x = domain[1][0];
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        boundary_normal = vec3(-1.0, 0.0, 0.0);
        insert_particle(_pos, 2, spacing, boundary_normal, 0, vol);
      }

      pos_y += spacing;
    }

    // up
    if (domain_boundary_type[2] != 0) {
      pos_x = domain[0][0];
      pos_y = domain[1][1];
      if (domain_boundary_type[3] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        boundary_normal = vec3(sqrt(2) / 2.0, -sqrt(2) / 2.0, 0.0);
        insert_particle(_pos, 1, spacing, boundary_normal, 0, vol);
      }
      pos_x += 0.5 * spacing;

      while (pos_x < domain[1][0] - 1e-5) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        boundary_normal = vec3(0.0, -1.0, 0.0);
        insert_particle(_pos, 2, spacing, boundary_normal, 0, vol);
        pos_x += spacing;
      }

      pos_x = domain[1][0];
      if (domain_boundary_type[1] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        boundary_normal = vec3(-sqrt(2) / 2.0, -sqrt(2) / 2.0, 0.0);
        insert_particle(_pos, 1, spacing, boundary_normal, 0, vol);
      }
    }
  }
  if (dim == 3) {
    double vol = spacing * spacing * spacing;

    // x-y, z=-z0 face
    if (domain_boundary_type[3] != 0) {
      pos_z = domain[0][2];

      pos_x = domain[0][0];
      pos_y = domain[0][1];
      if (domain_boundary_type[2] != 0 && domain_boundary_type[4] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(sqrt(3) / 3.0, sqrt(3) / 3.0, sqrt(3) / 3.0);
        insert_particle(_pos, 1, spacing, normal, 0, vol);
      }

      pos_y += 0.5 * spacing;
      if (domain_boundary_type[2] != 0) {
        while (pos_y < domain[1][1] - 1e-5) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(sqrt(2.0) / 2.0, 0.0, sqrt(2.0) / 2.0);
          insert_particle(_pos, 2, spacing, normal, 0, vol);
          pos_y += spacing;
        }
      }

      pos_y = domain[1][1];
      if (domain_boundary_type[1] != 0 && domain_boundary_type[2] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(sqrt(3) / 3.0, -sqrt(3) / 3.0, sqrt(3) / 3.0);
        insert_particle(_pos, 1, spacing, normal, 0, vol);
      }

      pos_x += 0.5 * spacing;
      while (pos_x < domain[1][0] - 1e-5) {
        pos_y = domain[0][1];
        if (domain_boundary_type[4] != 0) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(0.0, sqrt(2.0) / 2.0, sqrt(2.0) / 2.0);
          insert_particle(_pos, 2, spacing, normal, 0, vol);
        }

        pos_y += 0.5 * spacing;
        while (pos_y < domain[1][1] - 1e-5) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(0.0, 0.0, 1.0);
          insert_particle(_pos, 3, spacing, normal, 0, vol);
          pos_y += spacing;
        }

        pos_y = domain[1][1];
        if (domain_boundary_type[1] != 0) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(0.0, -sqrt(2.0) / 2.0, sqrt(2.0) / 2.0);
          insert_particle(_pos, 2, spacing, normal, 0, vol);
        }

        pos_x += spacing;
      }

      pos_x = domain[1][0];
      pos_y = domain[0][1];
      if (domain_boundary_type[0] != 0 && domain_boundary_type[4] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(-sqrt(3) / 3.0, sqrt(3) / 3.0, sqrt(3) / 3.0);
        insert_particle(_pos, 1, spacing, normal, 0, vol);
      }

      pos_y += 0.5 * spacing;
      if (domain_boundary_type[0] != 0) {
        while (pos_y < domain[1][1] - 1e-5) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(-sqrt(2.0) / 2.0, 0.0, sqrt(2.0) / 2.0);
          insert_particle(_pos, 2, spacing, normal, 0, vol);
          pos_y += spacing;
        }
      }

      pos_y = domain[1][1];
      if (domain_boundary_type[0] != 0 && domain_boundary_type[1] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(-sqrt(3) / 3.0, -sqrt(3) / 3.0, sqrt(3) / 3.0);
        insert_particle(_pos, 1, spacing, normal, 0, vol);
      }
    }

    pos_z = domain[0][2] + spacing / 2.0;
    while (pos_z < domain[1][2] - 1e-5) {
      pos_y = domain[0][1];
      pos_x = domain[0][0];
      if (domain_boundary_type[2] != 0 && domain_boundary_type[4] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(sqrt(2.0) / 2.0, sqrt(2.0) / 2.0, 0.0);
        insert_particle(_pos, 2, spacing, normal, 0, vol);
      }

      pos_y += 0.5 * spacing;
      if (domain_boundary_type[2] != 0) {
        while (pos_y < domain[1][1] - 1e-5) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(1.0, 0.0, 0.0);
          insert_particle(_pos, 3, spacing, normal, 0, vol);
          pos_y += spacing;
        }
      }

      pos_y = domain[1][1];
      if (domain_boundary_type[1] != 0 && domain_boundary_type[2] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0, 0.0);
        insert_particle(_pos, 2, spacing, normal, 0, vol);
      }

      pos_x += 0.5 * spacing;
      while (pos_x < domain[1][0] - 1e-5) {
        pos_y = domain[0][1];
        if (domain_boundary_type[4] != 0) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(0.0, 1.0, 0.0);
          insert_particle(_pos, 3, spacing, normal, 0, vol);
        }

        pos_y += spacing / 2.0;
        while (pos_y < domain[1][1] - 1e-5) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(1.0, 0.0, 0.0);
          insert_particle(_pos, 0, spacing, normal, 0, vol);
          pos_y += spacing;
        }

        pos_y = domain[1][1];
        if (domain_boundary_type[1] != 0) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(0.0, -1.0, 0.0);
          insert_particle(_pos, 3, spacing, normal, 0, vol);
        }

        pos_x += spacing;
      }

      pos_y = domain[0][1];
      pos_x = domain[1][0];
      if (domain_boundary_type[0] != 0 && domain_boundary_type[4] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(-sqrt(2.0) / 2.0, sqrt(2.0) / 2.0, 0.0);
        insert_particle(_pos, 2, spacing, normal, 0, vol);
      }

      pos_y += 0.5 * spacing;
      if (domain_boundary_type[0] != 0) {
        while (pos_y < domain[1][1] - 1e-5) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(-1.0, 0.0, 0.0);
          insert_particle(_pos, 3, spacing, normal, 0, vol);
          pos_y += spacing;
        }
      }

      pos_y = domain[1][1];
      if (domain_boundary_type[0] != 0 && domain_boundary_type[1] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(-sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0, 0.0);
        insert_particle(_pos, 2, spacing, normal, 0, vol);
      }

      pos_z += spacing;
    }

    // x-y, z=+z0 face
    if (domain_boundary_type[5] != 0) {
      pos_z = domain[1][2];

      pos_x = domain[0][0];
      pos_y = domain[0][1];
      if (domain_boundary_type[2] != 0 && domain_boundary_type[4] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(sqrt(3) / 3.0, sqrt(3) / 3.0, -sqrt(3) / 3.0);
        insert_particle(_pos, 1, spacing, normal, 0, vol);
      }

      pos_y += 0.5 * spacing;
      if (domain_boundary_type[2] != 0) {
        while (pos_y < domain[1][1] - 1e-5) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(sqrt(2.0) / 2.0, 0.0, -sqrt(2.0) / 2.0);
          insert_particle(_pos, 2, spacing, normal, 0, vol);
          pos_y += spacing;
        }
      }

      pos_y = domain[1][1];
      if (domain_boundary_type[1] != 0 && domain_boundary_type[2] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(sqrt(3) / 3.0, -sqrt(3) / 3.0, -sqrt(3) / 3.0);
        insert_particle(_pos, 1, spacing, normal, 0, vol);
      }

      pos_x += 0.5 * spacing;
      while (pos_x < domain[1][0] - 1e-5) {
        pos_y = domain[0][1];
        if (domain_boundary_type[4] != 0) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(0.0, sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0);
          insert_particle(_pos, 2, spacing, normal, 0, vol);
        }

        pos_y += 0.5 * spacing;
        while (pos_y < domain[1][1] - 1e-5) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(0.0, 0.0, -1.0);
          insert_particle(_pos, 3, spacing, normal, 0, vol);
          pos_y += spacing;
        }

        pos_y = domain[1][1];
        if (domain_boundary_type[1] != 0) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(0.0, -sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0);
          insert_particle(_pos, 2, spacing, normal, 0, vol);
        }

        pos_x += spacing;
      }

      pos_x = domain[1][0];
      pos_y = domain[0][1];
      if (domain_boundary_type[0] != 0 && domain_boundary_type[4] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(-sqrt(3) / 3.0, sqrt(3) / 3.0, -sqrt(3) / 3.0);
        insert_particle(_pos, 1, spacing, normal, 0, vol);
      }

      pos_y += 0.5 * spacing;
      if (domain_boundary_type[0] != 0) {
        while (pos_y < domain[1][1] - 1e-5) {
          vec3 _pos = vec3(pos_x, pos_y, pos_z);
          normal = vec3(-sqrt(2.0) / 2.0, 0.0, -sqrt(2.0) / 2.0);
          insert_particle(_pos, 2, spacing, normal, 0, vol);
          pos_y += spacing;
        }
      }

      pos_y = domain[1][1];
      if (domain_boundary_type[0] != 0 && domain_boundary_type[1] != 0) {
        vec3 _pos = vec3(pos_x, pos_y, pos_z);
        normal = vec3(-sqrt(3) / 3.0, -sqrt(3) / 3.0, -sqrt(3) / 3.0);
        insert_particle(_pos, 1, spacing, normal, 0, vol);
      }
    }
  }
}

void particle_geometry::generate_rigid_body_surface_particle() {
  auto &rigid_body_coord = rb_mgr->get_position();
  auto &rigid_body_orientation = rb_mgr->get_orientation();
  auto &rigid_body_size = rb_mgr->get_rigid_body_size();
  auto &rigid_body_type = rb_mgr->get_rigid_body_type();

  if (dim == 3) {
    double h = spacing;
    double vol = pow(h, 3);
    double a = pow(h, 2);

    for (size_t n = 0; n < rigid_body_coord.size(); n++) {
      double r = rigid_body_size[n];
      int M_theta = round(r * M_PI / h);
      double d_theta = r * M_PI / M_theta;
      double d_phi = a / d_theta;

      for (int i = 0; i < M_theta; ++i) {
        double theta = M_PI * (i + 0.5) / M_theta;
        int M_phi = round(2 * M_PI * r * sin(theta) / d_phi);
        for (int j = 0; j < M_phi; ++j) {
          double phi = 2 * M_PI * (j + 0.5) / M_phi;

          double theta0 = M_PI * i / M_theta;
          double theta1 = M_PI * (i + 1) / M_theta;
          // double d_phi = 2 * M_PI / M_phi;
          // double area = pow(r, 2.0) * (cos(theta0) - cos(theta1)) * d_phi;

          vec3 p_spacing = vec3(d_theta, d_phi, 0);

          vec3 p_coord = vec3(theta, phi, 0.0);
          vec3 normal =
              vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
          vec3 pos = normal * r + rigid_body_coord[n];
          if (pos[0] >= domain[0][0] && pos[0] < domain[1][0] &&
              pos[1] >= domain[0][1] && pos[1] < domain[1][1] &&
              pos[2] >= domain[0][2] && pos[2] < domain[1][2])
            insert_particle(pos, 5, spacing, normal, 0, vol, true, n, p_coord,
                            p_spacing);
        }
      }
    }
  }

  if (dim == 2) {

    for (size_t n = 0; n < rigid_body_coord.size(); n++) {
      switch (rigid_body_type[n]) {
      case 1:
        // circle
        {
          double h = spacing;
          double vol = pow(h, 2);

          double r = rigid_body_size[n];

          int M_theta = round(2 * M_PI * r / h);
          double d_theta = 2 * M_PI * r / M_theta;

          vec3 p_spacing = vec3(d_theta, 0, 0);

          for (int i = 0; i < M_theta; ++i) {
            double theta = 2 * M_PI * (i + 0.5) / M_theta;
            vec3 p_coord = vec3(theta, 0.0, 0.0);
            vec3 normal = vec3(cos(theta), sin(theta), 0.0);
            vec3 pos = normal * r + rigid_body_coord[n];
            if (pos[0] >= domain[0][0] && pos[0] < domain[1][0] &&
                pos[1] >= domain[0][1] && pos[1] < domain[1][1])
              insert_particle(pos, 5, spacing, normal, 0, vol, true, n, p_coord,
                              p_spacing);
          }
        }

        break;

      case 2:
        // square
        {
          double half_side_length = rigid_body_size[n];
          double theta = rigid_body_orientation[n][0];

          int particle_num_per_size = rigid_body_size[n] * 2.0 / spacing;

          double h = rigid_body_size[n] * 2.0 / particle_num_per_size;
          vec3 particleSize = vec3(h, h, 0.0);
          double vol = pow(h, 2.0);

          double xPos, yPos;
          xPos = -half_side_length;
          yPos = -half_side_length;

          vec3 norm = vec3(-sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0, 0.0);
          vec3 normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                             sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
          vec3 pos = vec3(cos(theta) * xPos - sin(theta) * yPos,
                          sin(theta) * xPos + cos(theta) * yPos, 0.0) +
                     rigid_body_coord[n];
          vec3 p_coord = vec3(0.0, 0.0, 0.0);
          vec3 p_spacing = vec3(h, 0.0, 0.0);
          if (pos[0] >= domain[0][0] && pos[0] < domain[1][0] &&
              pos[1] >= domain[0][1] && pos[1] < domain[1][1])
            insert_particle(pos, 4, spacing, normal, 0, vol, true, n, p_coord,
                            p_spacing);

          xPos += 0.5 * h;
          norm = vec3(0.0, -1.0, 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
          while (xPos < half_side_length) {
            pos = vec3(cos(theta) * xPos - sin(theta) * yPos,
                       sin(theta) * xPos + cos(theta) * yPos, 0.0) +
                  rigid_body_coord[n];
            if (pos[0] >= domain[0][0] && pos[0] < domain[1][0] &&
                pos[1] >= domain[0][1] && pos[1] < domain[1][1])
              insert_particle(pos, 5, spacing, normal, 0, vol, true, n, p_coord,
                              p_spacing);
            xPos += h;
          }

          xPos = half_side_length;
          norm = vec3(sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0, 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
          pos = vec3(cos(theta) * xPos - sin(theta) * yPos,
                     sin(theta) * xPos + cos(theta) * yPos, 0.0) +
                rigid_body_coord[n];
          if (pos[0] >= domain[0][0] && pos[0] < domain[1][0] &&
              pos[1] >= domain[0][1] && pos[1] < domain[1][1])
            insert_particle(pos, 4, spacing, normal, 0, vol, true, n, p_coord,
                            p_spacing);

          yPos += 0.5 * h;
          norm = vec3(1.0, 0.0, 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
          while (yPos < half_side_length) {
            pos = vec3(cos(theta) * xPos - sin(theta) * yPos,
                       sin(theta) * xPos + cos(theta) * yPos, 0.0) +
                  rigid_body_coord[n];
            if (pos[0] >= domain[0][0] && pos[0] < domain[1][0] &&
                pos[1] >= domain[0][1] && pos[1] < domain[1][1])
              insert_particle(pos, 5, spacing, normal, 0, vol, true, n, p_coord,
                              p_spacing);
            yPos += h;
          }

          yPos = half_side_length;
          norm = vec3(sqrt(2.0) / 2.0, sqrt(2.0) / 2.0, 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
          pos = vec3(cos(theta) * xPos - sin(theta) * yPos,
                     sin(theta) * xPos + cos(theta) * yPos, 0.0) +
                rigid_body_coord[n];
          if (pos[0] >= domain[0][0] && pos[0] < domain[1][0] &&
              pos[1] >= domain[0][1] && pos[1] < domain[1][1])
            insert_particle(pos, 4, spacing, normal, 0, vol, true, n, p_coord,
                            p_spacing);

          xPos -= 0.5 * h;
          norm = vec3(0.0, 1.0, 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
          while (xPos > -half_side_length) {
            pos = vec3(cos(theta) * xPos - sin(theta) * yPos,
                       sin(theta) * xPos + cos(theta) * yPos, 0.0) +
                  rigid_body_coord[n];
            if (pos[0] >= domain[0][0] && pos[0] < domain[1][0] &&
                pos[1] >= domain[0][1] && pos[1] < domain[1][1])
              insert_particle(pos, 5, spacing, normal, 0, vol, true, n, p_coord,
                              p_spacing);
            xPos -= h;
          }

          xPos = -half_side_length;
          norm = vec3(-sqrt(2.0) / 2.0, sqrt(2.0) / 2.0, 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
          pos = vec3(cos(theta) * xPos - sin(theta) * yPos,
                     sin(theta) * xPos + cos(theta) * yPos, 0.0) +
                rigid_body_coord[n];
          if (pos[0] >= domain[0][0] && pos[0] < domain[1][0] &&
              pos[1] >= domain[0][1] && pos[1] < domain[1][1])
            insert_particle(pos, 4, spacing, normal, 0, vol, true, n, p_coord,
                            p_spacing);

          yPos -= 0.5 * h;
          norm = vec3(-1.0, 0.0, 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
          while (yPos > -half_side_length) {
            pos = vec3(cos(theta) * xPos - sin(theta) * yPos,
                       sin(theta) * xPos + cos(theta) * yPos, 0.0) +
                  rigid_body_coord[n];
            if (pos[0] >= domain[0][0] && pos[0] < domain[1][0] &&
                pos[1] >= domain[0][1] && pos[1] < domain[1][1])
              insert_particle(pos, 5, spacing, normal, 0, vol, true, n, p_coord,
                              p_spacing);
            yPos -= h;
          }
        }

        break;

      case 3: {
        double theta = rigid_body_orientation[n][0];
        double side_length = rigid_body_size[n];
        int side_step = side_length / spacing;
        double h = side_length / side_step;
        double vol = pow(h, 2.0);
        vec3 particleSize = vec3(h, h, 0.0);
        vec3 increase_normal;
        vec3 start_point;
        vec3 normal;
        vec3 norm;
        vec3 p_coord = vec3(0.0, 0.0, 0.0);
        vec3 p_spacing = vec3(h, 0.0, 0.0);
        vec3 translation = vec3(0.0, -sqrt(3) / 6.0 * side_length, 0.0);
        // first side
        {
          vec3 pos = vec3(0.0, 0.5 * sqrt(3) * side_length, 0.0) + translation;
          // rotate
          vec3 new_pos = vec3(cos(theta) * pos[0] - sin(theta) * pos[1],
                              sin(theta) * pos[0] + cos(theta) * pos[1], 0.0) +
                         rigid_body_coord[n];

          norm = vec3(0.0, 1.0, 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
          if (new_pos[0] >= domain[0][0] && new_pos[0] < domain[1][0] &&
              new_pos[1] >= domain[0][1] && new_pos[1] < domain[1][1])
            insert_particle(new_pos, 4, spacing, normal, 0, vol, true, n,
                            p_coord, p_spacing);
        }

        increase_normal = vec3(cos(M_PI / 3), -sin(M_PI / 3), 0.0);
        start_point = vec3(0.0, sqrt(3) / 2.0 * side_length, 0.0);
        norm = vec3(cos(M_PI / 6.0), sin(M_PI / 6.0), 0.0);
        normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                      sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
        for (int i = 0; i < side_step; i++) {
          vec3 pos =
              start_point + increase_normal * ((i + 0.5) * h) + translation;
          // rotate
          vec3 new_pos = vec3(cos(theta) * pos[0] - sin(theta) * pos[1],
                              sin(theta) * pos[0] + cos(theta) * pos[1], 0.0) +
                         rigid_body_coord[n];

          if (new_pos[0] >= domain[0][0] && new_pos[0] < domain[1][0] &&
              new_pos[1] >= domain[0][1] && new_pos[1] < domain[1][1])
            insert_particle(new_pos, 5, spacing, normal, 0, vol, true, n,
                            p_coord, p_spacing);
        }

        // second side
        {
          vec3 pos = vec3(0.5 * side_length, 0.0, 0.0) + translation;
          // rotate
          vec3 new_pos = vec3(cos(theta) * pos[0] - sin(theta) * pos[1],
                              sin(theta) * pos[0] + cos(theta) * pos[1], 0.0) +
                         rigid_body_coord[n];

          norm = vec3(cos(M_PI / 6.0), -sin(M_PI / 6.0), 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);

          if (new_pos[0] >= domain[0][0] && new_pos[0] < domain[1][0] &&
              new_pos[1] >= domain[0][1] && new_pos[1] < domain[1][1])
            insert_particle(new_pos, 4, spacing, normal, 0, vol, true, n,
                            p_coord, p_spacing);
        }

        increase_normal = vec3(-1.0, 0.0, 0.0);
        start_point = vec3(0.5 * side_length, 0.0, 0.0);
        norm = vec3(0.0, -1.0, 0.0);
        normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                      sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
        for (int i = 0; i < side_step; i++) {
          vec3 pos =
              start_point + increase_normal * ((i + 0.5) * h) + translation;
          // rotate
          vec3 new_pos = vec3(cos(theta) * pos[0] - sin(theta) * pos[1],
                              sin(theta) * pos[0] + cos(theta) * pos[1], 0.0) +
                         rigid_body_coord[n];

          if (new_pos[0] >= domain[0][0] && new_pos[0] < domain[1][0] &&
              new_pos[1] >= domain[0][1] && new_pos[1] < domain[1][1])
            insert_particle(new_pos, 5, spacing, normal, 0, vol, true, n,
                            p_coord, p_spacing);
        }

        // third side
        {
          vec3 pos = vec3(-0.5 * side_length, 0.0, 0.0) + translation;
          // rotate
          vec3 new_pos = vec3(cos(theta) * pos[0] - sin(theta) * pos[1],
                              sin(theta) * pos[0] + cos(theta) * pos[1], 0.0) +
                         rigid_body_coord[n];

          norm = vec3(-cos(M_PI / 6.0), -sin(M_PI / 6.0), 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);

          if (new_pos[0] >= domain[0][0] && new_pos[0] < domain[1][0] &&
              new_pos[1] >= domain[0][1] && new_pos[1] < domain[1][1])
            insert_particle(new_pos, 4, spacing, normal, 0, vol, true, n,
                            p_coord, p_spacing);
        }

        increase_normal = vec3(cos(M_PI / 3), sin(M_PI / 3), 0.0);
        start_point = vec3(-0.5 * side_length, 0.0, 0.0);
        norm = vec3(-cos(M_PI / 6.0), sin(M_PI / 6.0), 0.0);
        normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                      sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
        for (int i = 0; i < side_step; i++) {
          vec3 pos =
              start_point + increase_normal * ((i + 0.5) * h) + translation;
          // rotate
          vec3 new_pos = vec3(cos(theta) * pos[0] - sin(theta) * pos[1],
                              sin(theta) * pos[0] + cos(theta) * pos[1], 0.0) +
                         rigid_body_coord[n];

          if (new_pos[0] >= domain[0][0] && new_pos[0] < domain[1][0] &&
              new_pos[1] >= domain[0][1] && new_pos[1] < domain[1][1])
            insert_particle(new_pos, 5, spacing, normal, 0, vol, true, n,
                            p_coord, p_spacing);
        }
      }

      break;
      }
    }
  }
}

void particle_geometry::uniform_refine() {}

void particle_geometry::adaptive_refine(vector<int> &split_tag) {
  vector<int> managing_split_tag;
  vector<int> managing_work_index;
  mitigate_backward(split_tag, managing_split_tag);
  mitigate_backward(*current_local_work_particle_index, managing_work_index);

  auto &coord = *current_local_managing_particle_coord;
  auto &spacing = *current_local_managing_particle_spacing;
  auto &particle_type = *current_local_managing_particle_type;
  auto &adaptive_level = *current_local_managing_particle_adaptive_level;
  auto &new_added = *current_local_managing_particle_new_added;

  auto &gap_coord = *local_managing_gap_particle_coord;
  auto &gap_spacing = *local_managing_gap_particle_spacing;
  auto &gap_adaptive_level = *local_managing_gap_particle_adaptive_level;

  unsigned int num_source_coord = coord.size();
  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> source_coord_device(
      "source coordinates", num_source_coord, 3);
  Kokkos::View<double **>::HostMirror source_coord_host =
      Kokkos::create_mirror_view(source_coord_device);

  for (size_t i = 0; i < num_source_coord; i++) {
    for (int j = 0; j < 3; j++) {
      source_coord_host(i, j) = coord[i][j];
    }
  }

  unsigned int num_target_coord = gap_coord.size();
  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> target_coord_device(
      "target coordinates", num_target_coord, 3);
  Kokkos::View<double **>::HostMirror target_coord_host =
      Kokkos::create_mirror_view(target_coord_device);

  for (size_t i = 0; i < num_target_coord; i++) {
    for (int j = 0; j < 3; j++) {
      target_coord_host(i, j) = gap_coord[i][j];
    }
  }

  Kokkos::deep_copy(source_coord_device, source_coord_host);
  Kokkos::deep_copy(target_coord_device, target_coord_host);

  auto point_cloud_search(CreatePointCloudSearch(source_coord_host, dim));

  int estimated_num_neighbor_max = 2 * pow(2, dim);

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> neighbor_list_device(
      "neighbor lists", num_target_coord, estimated_num_neighbor_max);
  Kokkos::View<int **>::HostMirror neighbor_list_host =
      Kokkos::create_mirror_view(neighbor_list_device);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> epsilon_device(
      "h supports", num_target_coord);
  Kokkos::View<double *>::HostMirror epsilon_host =
      Kokkos::create_mirror_view(epsilon_device);

  point_cloud_search.generateNeighborListsFromKNNSearch(
      false, target_coord_host, neighbor_list_host, epsilon_host, pow(2, dim),
      1.0);

  vector<int> gap_split_tag;
  gap_split_tag.resize(num_target_coord);
  for (int i = 0; i < num_target_coord; i++) {
    int counter = 0;
    double min_dis = 1.0;
    for (int j = 0; j < neighbor_list_host(i, 0); j++) {
      // find the nearest particle
      vec3 dis = coord[neighbor_list_host(i, j + 1)] - gap_coord[i];
      if (particle_type[neighbor_list_host(i, j + 1)] >= 4)
        if (dis.mag() < min_dis) {
          min_dis = dis.mag();
          counter = neighbor_list_host(i, j + 1);
        }
    }

    if ((gap_adaptive_level[i] < adaptive_level[counter]) ||
        (gap_adaptive_level[i] == adaptive_level[counter] &&
         managing_split_tag[counter] == 1)) {
      gap_split_tag[i] = 1;
    } else {
      gap_split_tag[i] = 0;
    }
  }

  for (int i = 0; i < new_added.size(); i++) {
    new_added[i] = managing_work_index[i];
  }

  vector<int> rigid_body_surface_particle_split_tag;
  vector<int> field_particle_split_tag;

  for (int i = 0; i < managing_split_tag.size(); i++) {
    if (managing_split_tag[i] != 0) {
      if (particle_type[i] < 4) {
        field_particle_split_tag.push_back(i);
      } else {
        rigid_body_surface_particle_split_tag.push_back(i);
      }
    }
  }

  split_rigid_body_surface_particle(rigid_body_surface_particle_split_tag);
  split_field_particle(field_particle_split_tag);
  split_gap_particle(gap_split_tag);
}

void particle_geometry::insert_particle(const vec3 &_pos, int _particle_type,
                                        const double _spacing,
                                        const vec3 &_normal,
                                        int _adaptive_level, double _volume,
                                        bool _rigid_body_particle,
                                        int _rigid_body_index, vec3 _p_coord,
                                        vec3 _p_spacing) {
  int idx = is_gap_particle(_pos, _spacing, _rigid_body_index);

  if (idx == -2) {
    current_local_managing_particle_coord->push_back(_pos);
    current_local_managing_particle_normal->push_back(_normal);
    current_local_managing_particle_p_coord->push_back(_p_coord);
    current_local_managing_particle_p_spacing->push_back(_p_spacing);
    current_local_managing_particle_spacing->push_back(_spacing);
    current_local_managing_particle_volume->push_back(_volume);
    current_local_managing_particle_type->push_back(_particle_type);
    current_local_managing_particle_adaptive_level->push_back(_adaptive_level);
    current_local_managing_particle_new_added->push_back(-1);
    current_local_managing_particle_attached_rigid_body->push_back(
        _rigid_body_index);
  } else if (idx > -1) {
    local_managing_gap_particle_coord->push_back(_pos);
    local_managing_gap_particle_normal->push_back(_normal);
    local_managing_gap_particle_p_coord->push_back(_p_coord);
    local_managing_gap_particle_volume->push_back(_volume);
    local_managing_gap_particle_spacing->push_back(_spacing);
    local_managing_gap_particle_particle_type->push_back(_particle_type);
    local_managing_gap_particle_adaptive_level->push_back(_adaptive_level);
  }
}

void particle_geometry::split_field_particle(vector<int> &split_tag) {
  auto &coord = *current_local_managing_particle_coord;
  auto &normal = *current_local_managing_particle_normal;
  auto &p_coord = *current_local_managing_particle_p_coord;
  auto &p_spacing = *current_local_managing_particle_p_spacing;
  auto &spacing = *current_local_managing_particle_spacing;
  auto &volume = *current_local_managing_particle_volume;
  auto &particle_type = *current_local_managing_particle_type;
  auto &adaptive_level = *current_local_managing_particle_adaptive_level;
  auto &new_added = *current_local_managing_particle_new_added;
  auto &attached_rigid_body_index =
      *current_local_managing_particle_attached_rigid_body;

  for (int i = 0; i < split_tag.size(); i++) {
    auto tag = split_tag[i];
    if (particle_type[tag] == 0) {
      // inner particle
      if (dim == 2) {
        vec3 origin = coord[tag];
        const double x_delta = spacing[tag] * 0.25;
        const double y_delta = spacing[tag] * 0.25;
        spacing[tag] /= 2.0;
        volume[tag] /= 4.0;
        bool insert = false;
        new_added[tag] = -1;
        adaptive_level[tag]++;
        for (int i = -1; i < 2; i += 2) {
          for (int j = -1; j < 2; j += 2) {
            vec3 new_pos = origin + vec3(i * x_delta, j * y_delta, 0.0);
            if (!insert) {
              int idx = is_gap_particle(new_pos, x_delta, -1);
              if (idx == -2) {
                coord[tag] = new_pos;

                insert = true;
              } else if (idx > -1) {
                local_managing_gap_particle_coord->push_back(new_pos);
                local_managing_gap_particle_normal->push_back(normal[tag]);
                local_managing_gap_particle_p_coord->push_back(p_coord[tag]);
                local_managing_gap_particle_volume->push_back(volume[tag]);
                local_managing_gap_particle_spacing->push_back(spacing[tag]);
                local_managing_gap_particle_particle_type->push_back(
                    particle_type[tag]);
                local_managing_gap_particle_adaptive_level->push_back(
                    adaptive_level[tag]);
              }
            } else {
              insert_particle(new_pos, particle_type[tag], spacing[tag],
                              normal[tag], adaptive_level[tag], volume[tag]);
            }
          }
        }
      }

      if (dim == 3) {
        vec3 origin = coord[tag];
        const double x_delta = spacing[tag] * 0.25;
        const double y_delta = spacing[tag] * 0.25;
        const double z_delta = spacing[tag] * 0.25;
        spacing[tag] /= 2.0;
        volume[tag] /= 8.0;
        bool insert = false;
        new_added[tag] = -1;
        for (int i = -1; i < 2; i += 2) {
          for (int j = -1; j < 2; j += 2) {
            for (int k = -1; k < 2; k += 2) {
              vec3 new_pos =
                  origin + vec3(i * x_delta, j * y_delta, k * z_delta);
              if (!insert) {
                int idx = is_gap_particle(new_pos, x_delta, -1);
                if (idx == -2) {
                  coord[tag] = new_pos;
                  adaptive_level[tag]++;

                  insert = true;
                } else if (idx > -1) {
                  local_managing_gap_particle_coord->push_back(new_pos);
                  local_managing_gap_particle_normal->push_back(normal[tag]);
                  local_managing_gap_particle_p_coord->push_back(p_coord[tag]);
                  local_managing_gap_particle_volume->push_back(volume[tag]);
                  local_managing_gap_particle_spacing->push_back(spacing[tag]);
                  local_managing_gap_particle_particle_type->push_back(
                      particle_type[tag]);
                  local_managing_gap_particle_adaptive_level->push_back(
                      adaptive_level[tag]);
                }
              } else {
                insert_particle(new_pos, particle_type[tag], spacing[tag],
                                normal[tag], adaptive_level[tag], volume[tag]);
              }
            }
          }
        }
      }
    }

    if (particle_type[tag] > 0) {
      // boundary particle
      if (dim == 2) {
        if (particle_type[tag] == 1) {
          // corner particle
          spacing[tag] /= 2.0;
          volume[tag] /= 4.0;
          adaptive_level[tag]++;
        } else {
          spacing[tag] /= 2.0;
          volume[tag] /= 4.0;
          adaptive_level[tag]++;
          new_added[tag] = -1;

          vec3 origin = coord[tag];

          bool insert = false;
          for (int i = -1; i < 2; i += 2) {
            vec3 new_pos = origin + vec3(normal[tag][1], -normal[tag][0], 0.0) *
                                        i * spacing[tag] * 0.5;

            if (!insert) {
              coord[tag] = new_pos;

              insert = true;
            } else {
              double vol = volume[tag];
              insert_particle(new_pos, particle_type[tag], spacing[tag],
                              normal[tag], adaptive_level[tag], vol);
            }
          }
        }
      }
      if (dim == 3) {
        if (particle_type[tag] == 1) {
          // corner particle
          spacing[tag] /= 2.0;
          volume[tag] /= 8.0;
          adaptive_level[tag]++;
        } else if (particle_type[tag] == 2) {
          // line particle
          spacing[tag] /= 2.0;
          volume[tag] /= 8.0;
          adaptive_level[tag]++;
          new_added[tag] = -1;

          vec3 origin = coord[tag];

          int x_direction = (normal[tag][0] == 0.0) ? 1 : 0;
          int y_direction = (normal[tag][1] == 0.0) ? 1 : 0;
          int z_direction = (normal[tag][2] == 0.0) ? 1 : 0;

          bool insert = false;
          for (int i = -1; i < 2; i += 2) {
            vec3 new_pos =
                origin + vec3(x_direction, y_direction, z_direction) * i *
                             spacing[tag] * 0.5;

            if (!insert) {
              coord[tag] = new_pos;

              insert = true;
            } else {
              double vol = volume[tag];
              insert_particle(new_pos, particle_type[tag], spacing[tag],
                              normal[tag], adaptive_level[tag], vol);
            }
          }
        } else {
          // plane particle
          spacing[tag] /= 2.0;
          volume[tag] /= 8.0;
          adaptive_level[tag]++;
          new_added[tag] = -1;

          vec3 origin = coord[tag];

          vec3 direction1, direction2;
          if (normal[tag][0] != 0) {
            direction1 = vec3(0.0, 1.0, 0.0);
            direction2 = vec3(0.0, 0.0, 1.0);
          }
          if (normal[tag][1] != 0) {
            direction1 = vec3(1.0, 0.0, 0.0);
            direction2 = vec3(0.0, 0.0, 1.0);
          }
          if (normal[tag][2] != 0) {
            direction1 = vec3(1.0, 0.0, 0.0);
            direction2 = vec3(0.0, 1.0, 0.0);
          }

          bool insert = false;
          for (int i = -1; i < 2; i += 2) {
            for (int j = -1; j < 2; j += 2) {
              vec3 new_pos = origin + direction1 * i * spacing[tag] * 0.5 +
                             direction2 * j * spacing[tag] * 0.5;

              if (!insert) {
                coord[tag] = new_pos;

                insert = true;
              } else {
                double vol = volume[tag];
                insert_particle(new_pos, particle_type[tag], spacing[tag],
                                normal[tag], adaptive_level[tag], vol);
              }
            }
          }
        }
      }
    }
  }
}

void particle_geometry::split_rigid_body_surface_particle(
    vector<int> &split_tag) {
  auto &coord = *current_local_managing_particle_coord;
  auto &normal = *current_local_managing_particle_normal;
  auto &p_coord = *current_local_managing_particle_p_coord;
  auto &p_spacing = *current_local_managing_particle_p_spacing;
  auto &spacing = *current_local_managing_particle_spacing;
  auto &volume = *current_local_managing_particle_volume;
  auto &particle_type = *current_local_managing_particle_type;
  auto &adaptive_level = *current_local_managing_particle_adaptive_level;
  auto &new_added = *current_local_managing_particle_new_added;
  auto &attached_rigid_body_index =
      *current_local_managing_particle_attached_rigid_body;

  auto &rigid_body_coord = rb_mgr->get_position();
  auto &rigid_body_orientation = rb_mgr->get_orientation();
  auto &rigid_body_size = rb_mgr->get_rigid_body_size();
  auto &rigid_body_type = rb_mgr->get_rigid_body_type();

  if (dim == 3) {
    for (auto tag : split_tag) {
      double theta = p_coord[tag][0];
      double phi = p_coord[tag][1];
      double r = rigid_body_size[attached_rigid_body_index[tag]];

      const double old_delta_theta = p_spacing[tag][0] / r;
      const double delta_theta = 0.5 * old_delta_theta;

      double d_theta = 0.5 * p_spacing[tag][0];
      double d_phi = 0.5 * p_spacing[tag][1];

      const int old_M_phi =
          round(2 * M_PI * r * sin(theta) / p_spacing[tag][1]);

      const double old_delta_phi = 2 * M_PI / old_M_phi;
      const double old_phi0 = phi - 0.5 * old_delta_phi;
      const double old_phi1 = phi + 0.5 * old_delta_phi;

      bool insert = false;
      for (int i = -1; i < 2; i += 2) {
        double new_theta = theta + i * delta_theta * 0.5;
        int M_phi = round(2 * M_PI * r * sin(new_theta) / d_phi);
        for (int j = 0; j < M_phi; j++) {
          double new_phi = 2 * M_PI * (j + 0.5) / M_phi;
          if (new_phi >= old_phi0 && new_phi <= old_phi1) {
            double theta0 = new_theta - 0.5 * delta_theta;
            double theta1 = new_theta + 0.5 * delta_theta;

            double d_phi = 2 * M_PI / M_phi;
            double area = pow(r, 2.0) * (cos(theta0) - cos(theta1)) * d_phi;

            vec3 new_p_spacing = vec3(d_theta, area / d_theta, 0.0);
            vec3 new_normal =
                vec3(sin(new_theta) * cos(new_phi),
                     sin(new_theta) * sin(new_phi), cos(new_theta));
            vec3 new_pos = new_normal * r +
                           rigid_body_coord[attached_rigid_body_index[tag]];

            if (!insert) {
              coord[tag] = new_pos;
              volume[tag] /= 8.0;
              normal[tag] = new_normal;
              spacing[tag] /= 2.0;
              p_coord[tag] = vec3(new_theta, new_phi, 0.0);
              p_spacing[tag] = new_p_spacing;
              adaptive_level[tag]++;
              new_added[tag] = -1;

              insert = true;
            } else {
              double vol = volume[tag];
              insert_particle(new_pos, particle_type[tag], spacing[tag],
                              new_normal, adaptive_level[tag], vol, true,
                              attached_rigid_body_index[tag],
                              vec3(new_theta, new_phi, 0.0), new_p_spacing);
            }
          }
        }
      }
    }
  }

  if (dim == 2) {
    for (auto tag : split_tag) {
      switch (rigid_body_type[attached_rigid_body_index[tag]]) {
      case 1:
        // cicle
        {
          double r = rigid_body_size[attached_rigid_body_index[tag]];
          const double delta_theta = 0.25 * p_spacing[tag][0] / r;

          double theta = p_coord[tag][0];

          vec3 new_normal = vec3(
              cos(theta) * cos(delta_theta) - sin(theta) * sin(delta_theta),
              cos(theta) * sin(delta_theta) + sin(theta) * cos(delta_theta),
              0.0);
          vec3 new_pos =
              new_normal * rigid_body_size[attached_rigid_body_index[tag]] +
              rigid_body_coord[attached_rigid_body_index[tag]];

          coord[tag] = new_pos;
          spacing[tag] /= 2.0;
          volume[tag] /= 4.0;
          normal[tag] = new_normal;
          p_coord[tag] = vec3(theta + delta_theta, 0.0, 0.0);
          p_spacing[tag] = vec3(p_spacing[tag][0] / 2.0, 0.0, 0.0);
          adaptive_level[tag]++;
          new_added[tag] = -1;

          new_normal = vec3(
              cos(theta) * cos(-delta_theta) - sin(theta) * sin(-delta_theta),
              cos(theta) * sin(-delta_theta) + sin(theta) * cos(-delta_theta),
              0.0);
          new_pos =
              new_normal * rigid_body_size[attached_rigid_body_index[tag]] +
              rigid_body_coord[attached_rigid_body_index[tag]];

          insert_particle(new_pos, particle_type[tag], spacing[tag], new_normal,
                          adaptive_level[tag], volume[tag], true,
                          attached_rigid_body_index[tag],
                          vec3(theta - delta_theta, 0.0, 0.0), p_spacing[tag]);
        }

        break;
      case 2:
        // square
        {
          if (particle_type[tag] == 4) {
            // corner particle
            spacing[tag] /= 2.0;
            volume[tag] /= 4.0;
            adaptive_level[tag]++;
            p_spacing[tag] = vec3(p_spacing[tag][0] / 2.0, 0.0, 0.0);
          } else {
            // side particle
            spacing[tag] /= 2.0;
            volume[tag] /= 4.0;
            adaptive_level[tag]++;
            new_added[tag] = -1;
            p_spacing[tag] = vec3(p_spacing[tag][0] / 2.0, 0.0, 0.0);

            vec3 old_pos = coord[tag];

            vec3 delta =
                vec3(-normal[tag][1], normal[tag][0], 0.0) * 0.5 * spacing[tag];
            coord[tag] = old_pos + delta;

            vec3 new_pos = old_pos - delta;

            insert_particle(new_pos, particle_type[tag], spacing[tag],
                            normal[tag], adaptive_level[tag], volume[tag], true,
                            attached_rigid_body_index[tag], p_coord[tag],
                            p_spacing[tag]);
          }
        }

        break;

      case 3: {
        if (particle_type[tag] == 4) {
          // corner particle
          spacing[tag] /= 2.0;
          volume[tag] /= 4.0;
          adaptive_level[tag]++;
          p_spacing[tag] = vec3(p_spacing[tag][0] / 2.0, 0.0, 0.0);
        } else {
          // side particle
          spacing[tag] /= 2.0;
          volume[tag] /= 4.0;
          adaptive_level[tag]++;
          new_added[tag] = -1;
          p_spacing[tag] = vec3(p_spacing[tag][0] / 2.0, 0.0, 0.0);

          vec3 old_pos = coord[tag];

          vec3 delta =
              vec3(-normal[tag][1], normal[tag][0], 0.0) * 0.5 * spacing[tag];
          coord[tag] = old_pos + delta;

          vec3 new_pos = old_pos - delta;

          insert_particle(new_pos, particle_type[tag], spacing[tag],
                          normal[tag], adaptive_level[tag], volume[tag], true,
                          attached_rigid_body_index[tag], p_coord[tag],
                          p_spacing[tag]);
        }
      }

      break;
      }
    }
  }
}

void particle_geometry::split_gap_particle(vector<int> &split_tag) {
  auto gap_coord = move(*local_managing_gap_particle_coord);
  auto gap_normal = move(*local_managing_gap_particle_normal);
  auto gap_p_coord = move(*local_managing_gap_particle_p_coord);
  auto gap_volume = move(*local_managing_gap_particle_volume);
  auto gap_spacing = move(*local_managing_gap_particle_spacing);
  auto gap_particle_type = move(*local_managing_gap_particle_particle_type);
  auto gap_adaptive_level = move(*local_managing_gap_particle_adaptive_level);

  local_managing_gap_particle_coord = make_shared<vector<vec3>>();
  local_managing_gap_particle_normal = make_shared<vector<vec3>>();
  local_managing_gap_particle_p_coord = make_shared<vector<vec3>>();
  local_managing_gap_particle_volume = make_shared<vector<double>>();
  local_managing_gap_particle_spacing = make_shared<vector<double>>();
  local_managing_gap_particle_particle_type = make_shared<vector<int>>();
  local_managing_gap_particle_adaptive_level = make_shared<vector<int>>();

  if (dim == 3) {
    for (int tag = 0; tag < split_tag.size(); tag++) {
      if (split_tag[tag] == 0) {
        insert_particle(gap_coord[tag], gap_particle_type[tag],
                        gap_spacing[tag], gap_normal[tag],
                        gap_adaptive_level[tag], gap_volume[tag]);
      } else {
        vec3 origin = gap_coord[tag];
        const double x_delta = gap_spacing[tag] * 0.25;
        const double y_delta = gap_spacing[tag] * 0.25;
        const double z_delta = gap_spacing[tag] * 0.25;
        for (int i = -1; i < 2; i += 2) {
          for (int j = -1; j < 2; j += 2) {
            for (int k = -1; k < 2; k += 2) {
              double new_spacing = gap_spacing[tag] * 0.5;
              vec3 new_pos =
                  origin + vec3(i * x_delta, j * y_delta, k * z_delta);
              double new_volume = gap_volume[tag] / 8.0;
              insert_particle(new_pos, gap_particle_type[tag], new_spacing,
                              gap_normal[tag], gap_adaptive_level[tag] + 1,
                              new_volume);
            }
          }
        }
      }
    }

    for (int tag = split_tag.size(); tag < gap_coord.size(); tag++) {
      insert_particle(gap_coord[tag], gap_particle_type[tag], gap_spacing[tag],
                      gap_normal[tag], gap_adaptive_level[tag],
                      gap_volume[tag]);
    }
  }

  if (dim == 2) {
    for (int tag = 0; tag < split_tag.size(); tag++) {
      if (split_tag[tag] == 0) {
        insert_particle(gap_coord[tag], gap_particle_type[tag],
                        gap_spacing[tag], gap_normal[tag],
                        gap_adaptive_level[tag], gap_volume[tag]);
      } else {
        vec3 origin = gap_coord[tag];
        const double x_delta = gap_spacing[tag] * 0.25;
        const double y_delta = gap_spacing[tag] * 0.25;
        for (int i = -1; i < 2; i += 2) {
          for (int j = -1; j < 2; j += 2) {
            double new_spacing = gap_spacing[tag] * 0.5;
            vec3 new_pos = origin + vec3(i * x_delta, j * y_delta, 0.0);
            double new_volume = gap_volume[tag] / 4.0;
            insert_particle(new_pos, gap_particle_type[tag], new_spacing,
                            gap_normal[tag], gap_adaptive_level[tag] + 1,
                            new_volume);
          }
        }
      }
    }

    for (int tag = split_tag.size(); tag < gap_coord.size(); tag++) {
      insert_particle(gap_coord[tag], gap_particle_type[tag], gap_spacing[tag],
                      gap_normal[tag], gap_adaptive_level[tag],
                      gap_volume[tag]);
    }
  }
}

int particle_geometry::is_gap_particle(const vec3 &_pos, double _spacing,
                                       int _attached_rigid_body_index) {
  int rigid_body_num = rb_mgr->get_rigid_body_num();
  for (size_t idx = 0; idx < rigid_body_num; idx++) {
    int rigid_body_type = rb_mgr->get_rigid_body_type(idx);
    vec3 rigid_body_pos = rb_mgr->get_position(idx);
    vec3 rigid_body_ori = rb_mgr->get_orientation(idx);
    double rigid_body_size = rb_mgr->get_rigid_body_size(idx);
    switch (rigid_body_type) {
    case 1:
      // circle in 2d, sphere in 3d
      {
        vec3 dis = _pos - rigid_body_pos;
        if (_attached_rigid_body_index >= 0) {
          // this is a particle on the rigid body surface
        } else {
          // this is a fluid particle

          if (dis.mag() < rigid_body_size - 1.5 * _spacing) {
            return -1;
          }
          if (dis.mag() <= rigid_body_size + 0.1 * _spacing) {
            return idx;
          }

          if (dis.mag() < rigid_body_size + 1.5 * _spacing) {
            double min_dis = bounding_box_size[0];
            for (int i = 0; i < rigid_body_surface_particle.size(); i++) {
              vec3 rci = _pos - rigid_body_surface_particle[i];
              if (min_dis > rci.mag()) {
                min_dis = rci.mag();
              }
            }

            if (min_dis < 0.25 * _spacing) {
              // this is a gap particle near the surface of the colloids
              return idx;
            }
          }
        }
      }
      break;

    case 2:
      // square in 2d, cubic in 3d
      {
        if (dim == 2) {
          double half_side_length = rigid_body_size;
          double theta = rigid_body_ori[0];

          vec3 abs_dis = _pos - rigid_body_pos;
          // rotate back
          vec3 dis =
              vec3(cos(theta) * abs_dis[0] + sin(theta) * abs_dis[1],
                   -sin(theta) * abs_dis[0] + cos(theta) * abs_dis[1], 0.0);
          if (_attached_rigid_body_index >= 0) {
            // this is a particle on the rigid body surface
          } else {
            if (abs(dis[0]) < half_side_length - 1.5 * _spacing &&
                abs(dis[1]) < half_side_length - 1.5 * _spacing) {
              return -1;
            }
            if (abs(dis[0]) < half_side_length + 0.1 * _spacing &&
                abs(dis[1]) < half_side_length + 0.1 * _spacing) {
              return idx;
            }
            if (abs(dis[0]) < half_side_length + 1.0 * _spacing &&
                abs(dis[1]) < half_side_length + 1.0 * _spacing) {
              double min_dis = bounding_box_size[0];
              for (int i = 0; i < rigid_body_surface_particle.size(); i++) {
                vec3 rci = _pos - rigid_body_surface_particle[i];
                if (min_dis > rci.mag()) {
                  min_dis = rci.mag();
                }
              }

              if (min_dis < 0.25 * _spacing) {
                // this is a gap particle near the surface of the colloids
                return idx;
              }
            }
          }
        }
        if (dim == 3) {
        }
      }
      break;
    case 3:
      // triangle in 2d, tetrahedron in 3d
      if (dim == 2) {
        double side_length = rigid_body_size;
        double height = (sqrt(3) / 2.0) * side_length;
        double theta = rigid_body_ori[0];

        vec3 translation = vec3(0.0, sqrt(3) / 6.0 * side_length, 0.0);
        vec3 abs_dis = _pos - rigid_body_pos;
        // rotate back
        vec3 dis =
            vec3(cos(theta) * abs_dis[0] + sin(theta) * abs_dis[1],
                 -sin(theta) * abs_dis[0] + cos(theta) * abs_dis[1], 0.0) +
            translation;

        bool possible_gap_particle = false;
        bool gap_particle = false;

        double enlarged_xlim_low = -0.5 * side_length - 0.5 * _spacing;
        double enlarged_xlim_high = 0.5 * side_length + 0.5 * _spacing;
        double enlarged_ylim_low = -0.5 * _spacing;
        double enlarged_ylim_high = height + 0.5 * _spacing;
        double exact_xlim_low = -0.5 * side_length;
        double exact_xlim_high = 0.5 * side_length;
        double exact_ylim_low = 0.0;
        double exact_ylim_high = height;
        if (_attached_rigid_body_index >= 0) {
          // this is a particle on the rigid body surface
        } else {
          if ((dis[0] > enlarged_xlim_low && dis[0] < enlarged_xlim_high) &&
              (dis[1] > enlarged_ylim_low && dis[1] < enlarged_ylim_high)) {
            // this is a possible particle in the gap region of the triangle
            double dis_x = min(abs(dis[0] - exact_xlim_low),
                               abs(dis[0] - exact_xlim_high));
            double dis_y = sqrt(3) * dis_x;
            if (dis[1] < 0 && dis[1] > -0.1 * _spacing) {
              gap_particle = true;
            } else if (dis[1] > 0) {
              if (dis[0] < exact_xlim_low || dis[0] > exact_xlim_high) {
                possible_gap_particle = true;
              } else if (dis[1] < dis_y + 0.1 * _spacing) {
                gap_particle = true;
              } else if (dis[1] < dis_y + 0.5 * _spacing) {
                possible_gap_particle = true;
              }
            }
            possible_gap_particle = true;
          }

          if (gap_particle) {
            return idx;
          } else if (possible_gap_particle) {
            double min_dis = bounding_box_size[0];
            for (int i = 0; i < rigid_body_surface_particle.size(); i++) {
              vec3 rci = _pos - rigid_body_surface_particle[i];
              if (min_dis > rci.mag()) {
                min_dis = rci.mag();
              }
            }

            if (min_dis < 0.25 * _spacing) {
              // this is a gap particle near the surface of the colloids
              return idx;
            }
          }
        }
      }
      if (dim == 3) {
      }
      break;
    }
  }

  return -2;
}

void particle_geometry::index_particle() {
  int local_particle_num = current_local_managing_particle_coord->size();
  current_local_managing_particle_index->resize(local_particle_num);

  vector<int> particle_offset(size + 1);
  vector<int> particle_num(size);
  MPI_Allgather(&local_particle_num, 1, MPI_INT, particle_num.data(), 1,
                MPI_INT, MPI_COMM_WORLD);

  particle_offset[0] = 0;
  for (int i = 0; i < size; i++) {
    particle_offset[i + 1] = particle_offset[i] + particle_num[i];
  }

  vector<long long> &index = *current_local_managing_particle_index;
  for (int i = 0; i < local_particle_num; i++) {
    index[i] = i + particle_offset[rank];
  }
}

void particle_geometry::index_work_particle() {
  int local_particle_num = current_local_work_particle_coord->size();
  current_local_work_particle_index->resize(local_particle_num);

  vector<int> particle_offset(size + 1);
  vector<int> particle_num(size);
  MPI_Allgather(&local_particle_num, 1, MPI_INT, particle_num.data(), 1,
                MPI_INT, MPI_COMM_WORLD);

  particle_offset[0] = 0;
  for (int i = 0; i < size; i++) {
    particle_offset[i + 1] = particle_offset[i] + particle_num[i];
  }

  auto &index = *current_local_work_particle_index;
  for (int i = 0; i < local_particle_num; i++) {
    index[i] = i + particle_offset[rank];
  }

  // particle distribution summary
  int min_local_particle_num, max_local_particle_num, global_particle_num;
  MPI_Allreduce(&local_particle_num, &min_local_particle_num, 1, MPI_INT,
                MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_particle_num, &max_local_particle_num, 1, MPI_INT,
                MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&local_particle_num, &global_particle_num, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  if (rank == 0) {
    cout << "number of total particles: " << global_particle_num << endl;
    cout << "workload imbalance: "
         << (double)(max_local_particle_num) / (double)(min_local_particle_num)
         << endl;
  }
}

void particle_geometry::balance_workload() {
  // use zoltan2 to build a solution to partition
  vector<int> result;
  partitioner.partition(*current_local_managing_particle_index,
                        *current_local_managing_particle_coord, result);

  // use the solution to build the mitigation graph
  vector<int> whole_mitigation_out_num, whole_mitigation_in_num;
  whole_mitigation_in_num.resize(size);
  whole_mitigation_out_num.resize(size);
  for (int i = 0; i < size; i++) {
    whole_mitigation_out_num[i] = 0;
  }

  int local_particle_num = result.size();
  for (int i = 0; i < local_particle_num; i++) {
    if (result[i] != rank) {
      whole_mitigation_out_num[result[i]]++;
    }
  }

  for (int i = 0; i < size; i++) {
    int out_num = whole_mitigation_out_num[i];
    MPI_Gather(&out_num, 1, MPI_INT, whole_mitigation_in_num.data(), 1, MPI_INT,
               i, MPI_COMM_WORLD);
  }

  mitigation_in_graph.clear();
  mitigation_out_graph.clear();
  mitigation_out_num.clear();
  mitigation_in_num.clear();

  for (int i = 0; i < size; i++) {
    if (whole_mitigation_out_num[i] != 0) {
      mitigation_out_graph.push_back(i);
      mitigation_out_num.push_back(whole_mitigation_out_num[i]);
    }
  }

  for (int i = 0; i < size; i++) {
    if (whole_mitigation_in_num[i] != 0) {
      mitigation_in_graph.push_back(i);
      mitigation_in_num.push_back(whole_mitigation_in_num[i]);
    }
  }

  mitigation_in_offset.resize(mitigation_in_graph.size() + 1);
  mitigation_out_offset.resize(mitigation_out_graph.size() + 1);

  mitigation_in_offset[0] = 0;
  for (int i = 0; i < mitigation_in_num.size(); i++) {
    mitigation_in_offset[i + 1] =
        mitigation_in_offset[i] + mitigation_in_num[i];
  }

  mitigation_out_offset[0] = 0;
  for (int i = 0; i < mitigation_out_num.size(); i++) {
    mitigation_out_offset[i + 1] =
        mitigation_out_offset[i] + mitigation_out_num[i];
  }

  local_reserve_map.clear();
  local_mitigation_map.resize(mitigation_out_offset[mitigation_out_num.size()]);
  vector<int> mitigation_map_idx;
  mitigation_map_idx.resize(mitigation_out_num.size());
  for (int i = 0; i < mitigation_out_num.size(); i++) {
    mitigation_map_idx[i] = mitigation_out_offset[i];
  }
  for (int i = 0; i < local_particle_num; i++) {
    if (result[i] != rank) {
      auto ite = (size_t)(lower_bound(mitigation_out_graph.begin(),
                                      mitigation_out_graph.end(), result[i]) -
                          mitigation_out_graph.begin());
      local_mitigation_map[mitigation_map_idx[ite]] = i;
      mitigation_map_idx[ite]++;
    } else {
      local_reserve_map.push_back(i);
    }
  }
}

void particle_geometry::build_ghost() {
  vec3 work_domain_low, work_domain_high;
  work_domain_low = bounding_box[1];
  work_domain_high = bounding_box[0];

  int local_particle_num = current_local_work_particle_coord->size();
  auto coord = *current_local_work_particle_coord;
  for (int i = 0; i < local_particle_num; i++) {
    if (work_domain_high[0] < coord[i][0]) {
      work_domain_high[0] = coord[i][0];
    }
    if (work_domain_high[1] < coord[i][1]) {
      work_domain_high[1] = coord[i][1];
    }
    if (work_domain_high[2] < coord[i][2]) {
      work_domain_high[2] = coord[i][2];
    }

    if (work_domain_low[0] > coord[i][0]) {
      work_domain_low[0] = coord[i][0];
    }
    if (work_domain_low[1] > coord[i][1]) {
      work_domain_low[1] = coord[i][1];
    }
    if (work_domain_low[2] > coord[i][2]) {
      work_domain_low[2] = coord[i][2];
    }
  }

  vector<double> whole_work_domain;
  whole_work_domain.resize(size * 6);
  MPI_Allgather(&work_domain_low[0], 1, MPI_DOUBLE, whole_work_domain.data(), 1,
                MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Allgather(&work_domain_low[1], 1, MPI_DOUBLE,
                whole_work_domain.data() + size, 1, MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Allgather(&work_domain_low[2], 1, MPI_DOUBLE,
                whole_work_domain.data() + size * 2, 1, MPI_DOUBLE,
                MPI_COMM_WORLD);
  MPI_Allgather(&work_domain_high[0], 1, MPI_DOUBLE,
                whole_work_domain.data() + size * 3, 1, MPI_DOUBLE,
                MPI_COMM_WORLD);
  MPI_Allgather(&work_domain_high[1], 1, MPI_DOUBLE,
                whole_work_domain.data() + size * 4, 1, MPI_DOUBLE,
                MPI_COMM_WORLD);
  MPI_Allgather(&work_domain_high[2], 1, MPI_DOUBLE,
                whole_work_domain.data() + size * 5, 1, MPI_DOUBLE,
                MPI_COMM_WORLD);

  vector<vec3> whole_ghost_domain_low;
  vector<vec3> whole_ghost_domain_high;
  whole_ghost_domain_low.resize(size);
  whole_ghost_domain_high.resize(size);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < 3; j++) {
      whole_ghost_domain_low[i][j] =
          whole_work_domain[i + j * size] - cutoff_distance;
      whole_ghost_domain_high[i][j] =
          whole_work_domain[i + (j + 3) * size] + cutoff_distance;
    }
  }

  vector<vector<int>> whole_ghost_out_map;
  whole_ghost_out_map.resize(size);
  for (int i = 0; i < local_particle_num; i++) {
    for (int j = 0; j < size; j++) {
      if (j != rank) {
        if (dim == 2) {
          if (coord[i][0] > whole_ghost_domain_low[j][0] &&
              coord[i][1] > whole_ghost_domain_low[j][1] &&
              coord[i][0] < whole_ghost_domain_high[j][0] &&
              coord[i][1] < whole_ghost_domain_high[j][1]) {
            whole_ghost_out_map[j].push_back(i);
          }
        } else if (dim == 3) {
          if (coord[i][0] > whole_ghost_domain_low[j][0] &&
              coord[i][1] > whole_ghost_domain_low[j][1] &&
              coord[i][2] > whole_ghost_domain_low[j][2] &&
              coord[i][0] < whole_ghost_domain_high[j][0] &&
              coord[i][1] < whole_ghost_domain_high[j][1] &&
              coord[i][2] < whole_ghost_domain_high[j][2]) {
            whole_ghost_out_map[j].push_back(i);
          }
        }
      }
    }
  }

  vector<int> whole_ghost_in_num;
  whole_ghost_in_num.resize(size);

  for (int i = 0; i < size; i++) {
    int out_num = whole_ghost_out_map[i].size();
    MPI_Gather(&out_num, 1, MPI_INT, whole_ghost_in_num.data(), 1, MPI_INT, i,
               MPI_COMM_WORLD);
  }

  ghost_out_graph.clear();
  ghost_in_graph.clear();
  ghost_out_num.clear();
  ghost_in_num.clear();

  for (int i = 0; i < size; i++) {
    if (whole_ghost_out_map[i].size() != 0) {
      ghost_out_graph.push_back(i);
      ghost_out_num.push_back(whole_ghost_out_map[i].size());
    }
  }

  for (int i = 0; i < size; i++) {
    if (whole_ghost_in_num[i] != 0) {
      ghost_in_graph.push_back(i);
      ghost_in_num.push_back(whole_ghost_in_num[i]);
    }
  }

  ghost_in_offset.resize(ghost_in_graph.size() + 1);
  ghost_out_offset.resize(ghost_out_graph.size() + 1);

  ghost_in_offset[0] = 0;
  for (int i = 0; i < ghost_in_num.size(); i++) {
    ghost_in_offset[i + 1] = ghost_in_offset[i] + ghost_in_num[i];
  }

  ghost_out_offset[0] = 0;
  for (int i = 0; i < ghost_out_num.size(); i++) {
    ghost_out_offset[i + 1] = ghost_out_offset[i] + ghost_out_num[i];
  }

  ghost_map.resize(ghost_out_offset[ghost_out_num.size()]);
  for (int i = 0; i < size; i++) {
    auto ite = (size_t)(
        lower_bound(ghost_out_graph.begin(), ghost_out_graph.end(), i) -
        ghost_out_graph.begin());
    for (int j = 0; j < whole_ghost_out_map[i].size(); j++) {
      ghost_map[ghost_out_offset[ite] + j] = whole_ghost_out_map[i][j];
    }
  }
}

void particle_geometry::build_ghost_from_last_level() {
  vec3 work_domain_low, work_domain_high;
  work_domain_low = bounding_box[1];
  work_domain_high = bounding_box[0];

  int target_local_particle_num = current_local_work_particle_coord->size();
  int source_local_particle_num = last_local_work_particle_coord->size();
  auto target_coord = *current_local_work_particle_coord;
  auto source_coord = *last_local_work_particle_coord;
  for (int i = 0; i < target_local_particle_num; i++) {
    if (work_domain_high[0] < target_coord[i][0]) {
      work_domain_high[0] = target_coord[i][0];
    }
    if (work_domain_high[1] < target_coord[i][1]) {
      work_domain_high[1] = target_coord[i][1];
    }
    if (work_domain_high[2] < target_coord[i][2]) {
      work_domain_high[2] = target_coord[i][2];
    }

    if (work_domain_low[0] > target_coord[i][0]) {
      work_domain_low[0] = target_coord[i][0];
    }
    if (work_domain_low[1] > target_coord[i][1]) {
      work_domain_low[1] = target_coord[i][1];
    }
    if (work_domain_low[2] > target_coord[i][2]) {
      work_domain_low[2] = target_coord[i][2];
    }
  }

  vector<double> whole_work_domain;
  whole_work_domain.resize(size * 6);
  MPI_Allgather(&work_domain_low[0], 1, MPI_DOUBLE, whole_work_domain.data(), 1,
                MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Allgather(&work_domain_low[1], 1, MPI_DOUBLE,
                whole_work_domain.data() + size, 1, MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Allgather(&work_domain_low[2], 1, MPI_DOUBLE,
                whole_work_domain.data() + size * 2, 1, MPI_DOUBLE,
                MPI_COMM_WORLD);
  MPI_Allgather(&work_domain_high[0], 1, MPI_DOUBLE,
                whole_work_domain.data() + size * 3, 1, MPI_DOUBLE,
                MPI_COMM_WORLD);
  MPI_Allgather(&work_domain_high[1], 1, MPI_DOUBLE,
                whole_work_domain.data() + size * 4, 1, MPI_DOUBLE,
                MPI_COMM_WORLD);
  MPI_Allgather(&work_domain_high[2], 1, MPI_DOUBLE,
                whole_work_domain.data() + size * 5, 1, MPI_DOUBLE,
                MPI_COMM_WORLD);

  vector<vec3> whole_ghost_domain_low;
  vector<vec3> whole_ghost_domain_high;
  whole_ghost_domain_low.resize(size);
  whole_ghost_domain_high.resize(size);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < 3; j++) {
      whole_ghost_domain_low[i][j] =
          whole_work_domain[i + j * size] - cutoff_distance;
      whole_ghost_domain_high[i][j] =
          whole_work_domain[i + (j + 3) * size] + cutoff_distance;
    }
  }

  vector<vector<int>> whole_ghost_clll_out_map;
  whole_ghost_clll_out_map.resize(size);
  for (int i = 0; i < source_local_particle_num; i++) {
    for (int j = 0; j < size; j++) {
      if (dim == 2) {
        if (source_coord[i][0] > whole_ghost_domain_low[j][0] &&
            source_coord[i][1] > whole_ghost_domain_low[j][1] &&
            source_coord[i][0] < whole_ghost_domain_high[j][0] &&
            source_coord[i][1] < whole_ghost_domain_high[j][1]) {
          whole_ghost_clll_out_map[j].push_back(i);
        }
      } else if (dim == 3) {
        if (source_coord[i][0] > whole_ghost_domain_low[j][0] &&
            source_coord[i][1] > whole_ghost_domain_low[j][1] &&
            source_coord[i][2] > whole_ghost_domain_low[j][2] &&
            source_coord[i][0] < whole_ghost_domain_high[j][0] &&
            source_coord[i][1] < whole_ghost_domain_high[j][1] &&
            source_coord[i][2] < whole_ghost_domain_high[j][2]) {
          whole_ghost_clll_out_map[j].push_back(i);
        }
      }
    }
  }

  vector<int> whole_ghost_clll_in_num;
  whole_ghost_clll_in_num.resize(size);

  for (int i = 0; i < size; i++) {
    int out_num = whole_ghost_clll_out_map[i].size();
    if (i == rank)
      out_num = 0;
    MPI_Gather(&out_num, 1, MPI_INT, whole_ghost_clll_in_num.data(), 1, MPI_INT,
               i, MPI_COMM_WORLD);
  }

  ghost_clll_out_graph.clear();
  ghost_clll_in_graph.clear();
  ghost_clll_out_num.clear();
  ghost_clll_in_num.clear();

  for (int i = 0; i < size; i++) {
    if (whole_ghost_clll_out_map[i].size() != 0 && i != rank) {
      ghost_clll_out_graph.push_back(i);
      ghost_clll_out_num.push_back(whole_ghost_clll_out_map[i].size());
    }
  }

  for (int i = 0; i < size; i++) {
    if (whole_ghost_clll_in_num[i] != 0) {
      ghost_clll_in_graph.push_back(i);
      ghost_clll_in_num.push_back(whole_ghost_clll_in_num[i]);
    }
  }

  ghost_clll_in_offset.resize(ghost_clll_in_graph.size() + 1);
  ghost_clll_out_offset.resize(ghost_clll_out_graph.size() + 1);

  ghost_clll_in_offset[0] = 0;
  for (int i = 0; i < ghost_clll_in_num.size(); i++) {
    ghost_clll_in_offset[i + 1] =
        ghost_clll_in_offset[i] + ghost_clll_in_num[i];
  }

  ghost_clll_out_offset[0] = 0;
  for (int i = 0; i < ghost_clll_out_num.size(); i++) {
    ghost_clll_out_offset[i + 1] =
        ghost_clll_out_offset[i] + ghost_clll_out_num[i];
  }

  ghost_clll_map.resize(ghost_clll_out_offset[ghost_clll_out_num.size()]);
  for (int i = 0; i < size; i++) {
    if (i != rank) {
      auto ite = (size_t)(lower_bound(ghost_clll_out_graph.begin(),
                                      ghost_clll_out_graph.end(), i) -
                          ghost_clll_out_graph.begin());
      for (int j = 0; j < whole_ghost_clll_out_map[i].size(); j++) {
        ghost_clll_map[ghost_clll_out_offset[ite] + j] =
            whole_ghost_clll_out_map[i][j];
      }
    }
  }

  reserve_clll_map.resize(whole_ghost_clll_out_map[rank].size());
  for (int i = 0; i < whole_ghost_clll_out_map[rank].size(); i++) {
    reserve_clll_map[i] = whole_ghost_clll_out_map[rank][i];
  }
}

void particle_geometry::build_ghost_for_last_level() {
  vec3 work_domain_low, work_domain_high;
  work_domain_low = bounding_box[1];
  work_domain_high = bounding_box[0];

  int target_local_particle_num = last_local_work_particle_coord->size();
  int source_local_particle_num = current_local_work_particle_coord->size();
  auto target_coord = *last_local_work_particle_coord;
  auto source_coord = *current_local_work_particle_coord;
  for (int i = 0; i < target_local_particle_num; i++) {
    if (work_domain_high[0] < target_coord[i][0]) {
      work_domain_high[0] = target_coord[i][0];
    }
    if (work_domain_high[1] < target_coord[i][1]) {
      work_domain_high[1] = target_coord[i][1];
    }
    if (work_domain_high[2] < target_coord[i][2]) {
      work_domain_high[2] = target_coord[i][2];
    }

    if (work_domain_low[0] > target_coord[i][0]) {
      work_domain_low[0] = target_coord[i][0];
    }
    if (work_domain_low[1] > target_coord[i][1]) {
      work_domain_low[1] = target_coord[i][1];
    }
    if (work_domain_low[2] > target_coord[i][2]) {
      work_domain_low[2] = target_coord[i][2];
    }
  }

  vector<double> whole_work_domain;
  whole_work_domain.resize(size * 6);
  MPI_Allgather(&work_domain_low[0], 1, MPI_DOUBLE, whole_work_domain.data(), 1,
                MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Allgather(&work_domain_low[1], 1, MPI_DOUBLE,
                whole_work_domain.data() + size, 1, MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Allgather(&work_domain_low[2], 1, MPI_DOUBLE,
                whole_work_domain.data() + size * 2, 1, MPI_DOUBLE,
                MPI_COMM_WORLD);
  MPI_Allgather(&work_domain_high[0], 1, MPI_DOUBLE,
                whole_work_domain.data() + size * 3, 1, MPI_DOUBLE,
                MPI_COMM_WORLD);
  MPI_Allgather(&work_domain_high[1], 1, MPI_DOUBLE,
                whole_work_domain.data() + size * 4, 1, MPI_DOUBLE,
                MPI_COMM_WORLD);
  MPI_Allgather(&work_domain_high[2], 1, MPI_DOUBLE,
                whole_work_domain.data() + size * 5, 1, MPI_DOUBLE,
                MPI_COMM_WORLD);

  vector<vec3> whole_ghost_domain_low;
  vector<vec3> whole_ghost_domain_high;
  whole_ghost_domain_low.resize(size);
  whole_ghost_domain_high.resize(size);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < 3; j++) {
      whole_ghost_domain_low[i][j] =
          whole_work_domain[i + j * size] - cutoff_distance;
      whole_ghost_domain_high[i][j] =
          whole_work_domain[i + (j + 3) * size] + cutoff_distance;
    }
  }

  vector<vector<int>> whole_ghost_llcl_out_map;
  whole_ghost_llcl_out_map.resize(size);
  for (int i = 0; i < source_local_particle_num; i++) {
    for (int j = 0; j < size; j++) {
      if (dim == 2) {
        if (source_coord[i][0] > whole_ghost_domain_low[j][0] &&
            source_coord[i][1] > whole_ghost_domain_low[j][1] &&
            source_coord[i][0] < whole_ghost_domain_high[j][0] &&
            source_coord[i][1] < whole_ghost_domain_high[j][1]) {
          whole_ghost_llcl_out_map[j].push_back(i);
        }
      } else if (dim == 3) {
        if (source_coord[i][0] > whole_ghost_domain_low[j][0] &&
            source_coord[i][1] > whole_ghost_domain_low[j][1] &&
            source_coord[i][2] > whole_ghost_domain_low[j][2] &&
            source_coord[i][0] < whole_ghost_domain_high[j][0] &&
            source_coord[i][1] < whole_ghost_domain_high[j][1] &&
            source_coord[i][2] < whole_ghost_domain_high[j][2]) {
          whole_ghost_llcl_out_map[j].push_back(i);
        }
      }
    }
  }

  vector<int> whole_ghost_llcl_in_num;
  whole_ghost_llcl_in_num.resize(size);

  for (int i = 0; i < size; i++) {
    int out_num = whole_ghost_llcl_out_map[i].size();
    if (i == rank)
      out_num = 0;
    MPI_Gather(&out_num, 1, MPI_INT, whole_ghost_llcl_in_num.data(), 1, MPI_INT,
               i, MPI_COMM_WORLD);
  }

  ghost_llcl_out_graph.clear();
  ghost_llcl_in_graph.clear();
  ghost_llcl_out_num.clear();
  ghost_llcl_in_num.clear();

  for (int i = 0; i < size; i++) {
    if (whole_ghost_llcl_out_map[i].size() != 0 && i != rank) {
      ghost_llcl_out_graph.push_back(i);
      ghost_llcl_out_num.push_back(whole_ghost_llcl_out_map[i].size());
    }
  }

  for (int i = 0; i < size; i++) {
    if (whole_ghost_llcl_in_num[i] != 0) {
      ghost_llcl_in_graph.push_back(i);
      ghost_llcl_in_num.push_back(whole_ghost_llcl_in_num[i]);
    }
  }

  ghost_llcl_in_offset.resize(ghost_llcl_in_graph.size() + 1);
  ghost_llcl_out_offset.resize(ghost_llcl_out_graph.size() + 1);

  ghost_llcl_in_offset[0] = 0;
  for (int i = 0; i < ghost_llcl_in_num.size(); i++) {
    ghost_llcl_in_offset[i + 1] =
        ghost_llcl_in_offset[i] + ghost_llcl_in_num[i];
  }

  ghost_llcl_out_offset[0] = 0;
  for (int i = 0; i < ghost_llcl_out_num.size(); i++) {
    ghost_llcl_out_offset[i + 1] =
        ghost_llcl_out_offset[i] + ghost_llcl_out_num[i];
  }

  ghost_llcl_map.resize(ghost_llcl_out_offset[ghost_llcl_out_num.size()]);
  for (int i = 0; i < size; i++) {
    if (i != rank) {
      auto ite = (size_t)(lower_bound(ghost_llcl_out_graph.begin(),
                                      ghost_llcl_out_graph.end(), i) -
                          ghost_llcl_out_graph.begin());
      for (int j = 0; j < whole_ghost_llcl_out_map[i].size(); j++) {
        ghost_llcl_map[ghost_llcl_out_offset[ite] + j] =
            whole_ghost_llcl_out_map[i][j];
      }
    }
  }

  reserve_llcl_map.resize(whole_ghost_llcl_out_map[rank].size());
  for (int i = 0; i < whole_ghost_llcl_out_map[rank].size(); i++) {
    reserve_llcl_map[i] = whole_ghost_llcl_out_map[rank][i];
  }
}