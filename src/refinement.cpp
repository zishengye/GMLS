#include <limits>

#include "DivergenceFree.hpp"
#include "gmls_solver.hpp"

using namespace std;
using namespace Compadre;

struct ErrorComb {
  double error;
  int rank;
};

bool pair_compare(const std::pair<int, double> &firstElem,
                  const std::pair<int, double> &secondElem) {
  return firstElem.second > secondElem.second;
}

bool gmls_solver::refinement() {
  double global_error = equation_mgr->get_estimated_error();

  PetscPrintf(PETSC_COMM_WORLD,
              "Total error for gradient: %f, with tolerance: %f\n",
              global_error, refinement_tolerance);

  static vector<vec3> old_rigid_body_velocity;
  static vector<vec3> old_rigid_body_angular_velocity;

  auto &rigid_body_position = rb_mgr->get_position();
  const auto num_rigid_body = rb_mgr->get_rigid_body_num();
  vector<double> &rigid_body_size = rb_mgr->get_rigid_body_size();
  vector<vec3> &rigid_body_velocity = rb_mgr->get_velocity();
  vector<vec3> &rigid_body_angular_velocity = rb_mgr->get_angular_velocity();
  if (isnan(global_error) || global_error < refinement_tolerance) {
    return false;
  }

  if (current_refinement_step >= max_refinement_level)
    return false;

  vector<double> &error = equation_mgr->get_error();

  auto &local_spacing = *(geo_mgr->get_current_work_particle_spacing());

  // mark stage
  double alpha = 0.7;

  double min_h = 1.0;
  for (int i = 0; i < local_spacing.size(); i++) {
    if (min_h > local_spacing[i])
      min_h = local_spacing[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, &min_h, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  double min_dis = 1.0;
  for (int i = 0; i < rigid_body_position.size(); i++) {
    for (int j = i + 1; j < rigid_body_position.size(); j++) {
      vec3 dist = rigid_body_position[i] - rigid_body_position[j];
      if (min_dis > dist.mag() - rigid_body_size[i] - rigid_body_size[j]) {
        min_dis = dist.mag() - rigid_body_size[i] - rigid_body_size[j];
      }
    }
  }

  PetscPrintf(PETSC_COMM_WORLD, "alpha: %f, min distance: %f, min h: %f\n",
              alpha, min_dis, min_h);

  vector<pair<int, double>> chopper;
  pair<int, double> to_add;

  const int local_particle_num = error.size();

  double local_error = 0.0;
  for (int i = 0; i < local_particle_num; i++) {
    to_add = pair<int, double>(i, pow(error[i], 2.0));
    chopper.push_back(to_add);
    local_error += pow(error[i], 2.0);
  }
  MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  std::sort(chopper.begin(), chopper.end(), pair_compare);

  vector<int> split_tag;

  // parallel selection
  int split_max_index = 0;

  double error_max, error_min, current_error_split, next_error;
  error_max = chopper[0].second;
  error_min = chopper[local_particle_num - 1].second;

  MPI_Allreduce(MPI_IN_PLACE, &error_max, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &error_min, 1, MPI_DOUBLE, MPI_MIN,
                MPI_COMM_WORLD);

  current_error_split = (error_max + error_min) / 2.0;
  bool selection_finished = false;
  while (!selection_finished) {
    int ite = 0;
    double error_sum = 0.0;
    while (ite < local_particle_num) {
      if (chopper[ite].second > current_error_split) {
        error_sum += chopper[ite].second;
        next_error = error_min;
        ite++;
      } else {
        next_error = chopper[ite].second;
        break;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &error_sum, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &next_error, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    if ((error_sum < alpha * global_error) &&
        (error_sum + next_error >= alpha * global_error)) {
      selection_finished = true;
      split_max_index = ite;
    } else if (error_sum < alpha * global_error) {
      error_max = current_error_split;
      current_error_split = (error_min + error_max) / 2.0;
    } else {
      error_min = current_error_split;
      current_error_split = (error_min + error_max) / 2.0;
    }
  }

  next_error = 0.9 * next_error;
  for (int i = 0; i < local_particle_num; i++) {
    if (chopper[i].second < next_error)
      break;
  }

  split_tag.resize(local_particle_num);
  for (int i = 0; i < local_particle_num; i++) {
    split_tag[i] = 0;
  }
  for (int i = 0; i < split_max_index; i++) {
    split_tag[chopper[i].first] = 1;
  }

  if (write_data)
    write_refinement_data(split_tag);

  // prevent over splitting
  vector<int> candidate_split_tag, ghost_split_tag;

  auto &source_coord = *(geo_mgr->get_current_work_ghost_particle_coord());
  auto &coord = *(geo_mgr->get_current_work_particle_coord());
  auto &spacing = *(geo_mgr->get_current_work_particle_spacing());
  auto &adaptive_level = *(geo_mgr->get_current_work_particle_adaptive_level());
  auto &particle_type = *(geo_mgr->get_current_work_particle_type());
  vector<int> source_adaptive_level;
  vector<int> source_particle_type;
  geo_mgr->ghost_forward(adaptive_level, source_adaptive_level);
  geo_mgr->ghost_forward(particle_type, source_particle_type);

  int num_source_coord = source_coord.size();
  int num_target_coord = coord.size();

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

  for (int i = 0; i < local_particle_num; i++) {
    for (int j = 0; j < 3; j++) {
      target_coord_host(i, j) = coord[i][j];
    }
  }

  Kokkos::deep_copy(source_coord_device, source_coord_host);
  Kokkos::deep_copy(target_coord_device, target_coord_host);

  auto point_cloud_search(CreatePointCloudSearch(source_coord_host, dim));

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> temp_neighbor_list_device(
      "temp neighbor lists", num_target_coord, 1);
  Kokkos::View<int **>::HostMirror temp_neighbor_list_host =
      Kokkos::create_mirror_view(temp_neighbor_list_device);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> epsilon_device(
      "h supports", num_target_coord);
  Kokkos::View<double *>::HostMirror epsilon_host =
      Kokkos::create_mirror_view(epsilon_device);

  auto &epsilon = equation_mgr->get_epsilon();

  for (int i = 0; i < num_target_coord; i++) {
    epsilon_host(i) = epsilon[i];
  }

  size_t max_num_neighbor =
      point_cloud_search.generate2DNeighborListsFromRadiusSearch(
          true, target_coord_host, temp_neighbor_list_host, epsilon_host, 0.0,
          0.0) +
      2;

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> neighbor_list_device(
      "neighbor lists", num_target_coord, max_num_neighbor);
  Kokkos::View<int **>::HostMirror neighbor_list_host =
      Kokkos::create_mirror_view(neighbor_list_device);

  point_cloud_search.generate2DNeighborListsFromRadiusSearch(
      false, target_coord_host, neighbor_list_host, epsilon_host, 0.0, 0.0);

  int min_num_neighbor = Compadre::GMLS::getNP(
      polynomial_order, dim, DivergenceFreeVectorTaylorPolynomial);

  int iteration_finished = 1;
  while (iteration_finished != 0) {
    geo_mgr->ghost_forward(split_tag, ghost_split_tag);
    candidate_split_tag = split_tag;
    int local_change = 0;
    for (int i = 0; i < num_target_coord; i++) {
      if (particle_type[i] == 0) {
        if (candidate_split_tag[i] == 0) {
          //
          for (int j = 0; j < neighbor_list_host(i, 0); j++) {
            int neighbor_index = neighbor_list_host(i, j + 1);
            if (ghost_split_tag[neighbor_index] == 1 &&
                source_adaptive_level[neighbor_index] - adaptive_level[i] > 0) {
              split_tag[i] = 1;
              local_change++;
            }
          }

          int num_split = 0;
          for (int j = 0; j < neighbor_list_host(i, 0); j++) {
            int neighbor_index = neighbor_list_host(i, j + 1);
            if ((source_adaptive_level[neighbor_index] > adaptive_level[i]) ||
                (ghost_split_tag[neighbor_index] == 1 &&
                 source_adaptive_level[neighbor_index] == adaptive_level[i])) {
              num_split++;
            }
          }
          if (num_split > 0.6 * (neighbor_list_host(i, 0) - 1)) {
            split_tag[i] = 1;
            local_change++;
          }
        }
      }
    }

    MPI_Allreduce(&local_change, &iteration_finished, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
  }

  geo_mgr->ghost_forward(split_tag, ghost_split_tag);
  candidate_split_tag = split_tag;
  for (int i = 0; i < num_target_coord; i++) {
    if (particle_type[i] != 0) {
      if (candidate_split_tag[i] == 0) {
        for (int j = 0; j < neighbor_list_host(i, 0); j++) {
          int neighbor_index = neighbor_list_host(i, j + 1);
          if (ghost_split_tag[neighbor_index] == 1 &&
              source_adaptive_level[neighbor_index] - adaptive_level[i] == 0 &&
              source_particle_type[neighbor_index] == 0) {
            split_tag[i] = 1;
          }
        }
      }
    }
  }

  // geo_mgr->ghost_forward(split_tag, ghost_split_tag);
  // candidate_split_tag = split_tag;
  // int local_change = 0;
  // for (int i = 0; i < num_target_coord; i++) {
  //   if (particle_type[i] == 0) {
  //     if (candidate_split_tag[i] == 1) {
  //       for (int j = 0; j < neighbor_list_host(i, 0); j++) {
  //         int neighbor_index = neighbor_list_host(i, j + 1);
  //         if (ghost_split_tag[neighbor_index] == 0 &&
  //             source_particle_type[neighbor_index] >= 4 &&
  //             adaptive_level[i] == source_adaptive_level[neighbor_index]) {
  //           split_tag[i] = 0;
  //         }
  //       }
  //     }
  //   }
  // }

  // for (int i = 0; i < local_particle_num; i++) {
  //   split_tag[i] = 0;

  //   double theta = atan2(coord[i][1], coord[i][0]);
  //   int n = theta / (M_PI / 34);

  //   if (coord[i].mag() < 0.21 && n % 2 == 0)
  //     split_tag[i] = 1;
  //   if (coord[i].mag() < 0.22 && particle_type[i] == 0) {
  //     if (n % 2 == 1 || n % 2 == -1)
  //       split_tag[i] = 1;
  //   }
  //   // if (coord[i].mag() < 0.22)
  //   //   split_tag[i] = 1;
  // }

  geo_mgr->refine(split_tag);

  current_refinement_step++;

  // get new number of particles
  MPI_Barrier(MPI_COMM_WORLD);
  {
    auto &new_coord = *(geo_mgr->get_current_work_particle_coord());
    int new_local_particle_num = new_coord.size();
    int new_global_particle_num;

    MPI_Allreduce(&new_local_particle_num, &new_global_particle_num, 1, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);

    if (new_global_particle_num > max_particle_num) {
      PetscPrintf(PETSC_COMM_WORLD,
                  "next refinement level has %d particles exceeds the maximum "
                  "particle num %d\n",
                  new_global_particle_num, (int)max_particle_num);
      return false;
    }
  }

  return true;
}