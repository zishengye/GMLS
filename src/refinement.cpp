#include <limits>

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
  double estimated_global_error = equation_mgr->get_estimated_error();

  PetscPrintf(PETSC_COMM_WORLD,
              "Total error for gradient: %f, with tolerance: %f\n",
              estimated_global_error, refinement_tolerance);

  static vector<vec3> old_rigid_body_velocity;
  static vector<vec3> old_rigid_body_angular_velocity;

  auto &rigid_body_position = rb_mgr->get_position();
  const auto num_rigid_body = rb_mgr->get_rigid_body_num();
  vector<vec3> &rigid_body_velocity = rb_mgr->get_velocity();
  vector<vec3> &rigid_body_angular_velocity = rb_mgr->get_angular_velocity();

  vector<int> split_tag;
  vector<double> &error = equation_mgr->get_error();

  auto &local_spacing = *(geo_mgr->get_current_work_particle_spacing());

  // mark stage
  double alpha = 0.8;

  PetscPrintf(PETSC_COMM_WORLD, "alpha: %f\n", alpha);

  vector<pair<int, double>> chopper;
  pair<int, double> to_add;

  const int local_particle_num = error.size();

  double local_error = 0.0;
  double global_error;
  for (int i = 0; i < local_particle_num; i++) {
    to_add = pair<int, double>(i, pow(error[i], 2.0));
    chopper.push_back(to_add);
    local_error += pow(error[i], 2.0);
  }
  MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  std::sort(chopper.begin(), chopper.end(), pair_compare);

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
  int ite_counter = 0;
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

  MPI_Allreduce(MPI_IN_PLACE, &next_error, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  for (int i = 0; i < local_particle_num; i++) {
    if (chopper[i].second < next_error) {
      split_max_index = i;
      break;
    }
  }

  split_tag.resize(local_particle_num);
  for (int i = 0; i < local_particle_num; i++) {
    split_tag[i] = 0;
  }
  for (int i = 0; i < split_max_index; i++) {
    split_tag[chopper[i].first] = 1;
  }

  // The split tag at this point is from error estimator.  We should keep this
  // one.

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

  for (int i = 0; i < num_target_coord; i++) {
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

  int iteration_finished = 1;
  ite_counter = 0;
  while (iteration_finished != 0) {
    ite_counter++;
    geo_mgr->ghost_forward(split_tag, ghost_split_tag);
    candidate_split_tag = split_tag;
    int local_change = 0;

    vector<int> cross_refinement_level, ghost_cross_refinement_level;
    cross_refinement_level.resize(num_target_coord);
    for (int i = 0; i < num_target_coord; i++) {
      cross_refinement_level[i] = -1;
    }

    for (int i = 0; i < num_target_coord; i++) {
      // ensure boundary particles at least have the same level of refinement
      // compared to their nearest interior particles
      if (particle_type[i] != 0) {
        double distance = 1.0;
        int nearest_index = 0;
        for (int j = 1; j < neighbor_list_host(i, 0); j++) {
          int neighbor_index = neighbor_list_host(i, j + 1);
          vec3 difference = coord[i] - source_coord[neighbor_index];
          if (particle_type[neighbor_index] == 0 &&
              difference.mag() < distance) {
            distance = difference.mag();
            nearest_index = neighbor_index;
          }
        }
        if (ghost_split_tag[nearest_index] == 1 &&
            source_adaptive_level[nearest_index] - adaptive_level[i] >= 0 &&
            split_tag[i] == 0) {
          split_tag[i] = 1;
          local_change++;
        }
      }

      int min_refinement_level = 100;
      int max_refinement_level = 0;
      for (int j = 1; j < neighbor_list_host(i, 0); j++) {
        int neighbor_index = neighbor_list_host(i, j + 1);
        if (source_adaptive_level[neighbor_index] +
                ghost_split_tag[neighbor_index] <
            min_refinement_level) {
          min_refinement_level = source_adaptive_level[neighbor_index] +
                                 ghost_split_tag[neighbor_index];
        }
        if (source_adaptive_level[neighbor_index] +
                ghost_split_tag[neighbor_index] >
            max_refinement_level) {
          max_refinement_level = source_adaptive_level[neighbor_index] +
                                 ghost_split_tag[neighbor_index];
        }
      }

      if (max_refinement_level - min_refinement_level > 1) {
        cross_refinement_level[i] = adaptive_level[i] + split_tag[i];
      }
    }

    geo_mgr->ghost_forward(cross_refinement_level,
                           ghost_cross_refinement_level);

    for (int i = 0; i < num_target_coord; i++) {
      for (int j = 1; j < neighbor_list_host(i, 0); j++) {
        int neighbor_index = neighbor_list_host(i, j + 1);
        if (ghost_cross_refinement_level[neighbor_index] >= 0) {
          if (adaptive_level[i] <
                  ghost_cross_refinement_level[neighbor_index] &&
              split_tag[i] == 0) {
            split_tag[i] = 1;
            local_change++;
          }
        }
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, &local_change, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    if (local_change == 0) {
      iteration_finished = 0;
    } else {
      PetscPrintf(PETSC_COMM_WORLD, "number of local change: %d\n",
                  local_change);
    }
  }

  vector<double> h_gradient;
  write_refinement_data(split_tag, h_gradient);

  if (isnan(estimated_global_error) ||
      estimated_global_error < refinement_tolerance) {
    return false;
  }

  if (current_refinement_step >= max_refinement_level)
    return false;

  PetscPrintf(PETSC_COMM_WORLD, "start of adaptive refinement\n");

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