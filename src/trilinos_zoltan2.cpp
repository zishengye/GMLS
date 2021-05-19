#include "trilinos_zoltan2.hpp"
#include "kd_tree.hpp"

using namespace std;

void trilinos_rcp_partitioner::partition(vector<long long> &index,
                                         vector<vec3> &coord,
                                         vector<int> &result) {
  // call zoltan2 to build the solution
  int local_particle_num = index.size();
  int global_particle_num;

  int disable_repartition = 0;
  if (local_particle_num < 10) {
    disable_repartition = 1;
  }
  MPI_Allreduce(MPI_IN_PLACE, &disable_repartition, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(&local_particle_num, &global_particle_num, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  if (disable_repartition != 1) {
    vector<double> x, y, z;
    x.resize(local_particle_num);
    y.resize(local_particle_num);
    z.resize(local_particle_num);

    for (int i = 0; i < local_particle_num; i++) {
      x[i] = coord[i][0];
      y[i] = coord[i][1];
      z[i] = coord[i][2];
    }

    inputAdapter_t *ia =
        new inputAdapter_t(local_particle_num, index.data(), x.data(), y.data(),
                           z.data(), 1, 1, 1);

    Zoltan2::PartitioningProblem<inputAdapter_t> *problem =
        new Zoltan2::PartitioningProblem<inputAdapter_t>(ia, &params);

    problem->solve();

    auto &solution = problem->getSolution();

    const int *ptr = solution.getPartListView();

    if (ptr != NULL) {
      result.resize(local_particle_num);

      vector<double> flatted_mass_center;
      flatted_mass_center.resize(size * dim);

      for (int i = 0; i < size * dim; i++) {
        flatted_mass_center[i] = 0.0;
      }
      for (int i = 0; i < local_particle_num; i++) {
        for (int k = 0; k < dim; k++) {
          flatted_mass_center[ptr[i] * dim + k] += coord[i][k];
        }
      }
      MPI_Allreduce(MPI_IN_PLACE, flatted_mass_center.data(), size * dim,
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      vector<int> part_count;
      part_count.resize(size);
      for (int i = 0; i < size; i++) {
        part_count[i] = 0;
      }
      for (int i = 0; i < local_particle_num; i++) {
        part_count[ptr[i]]++;
      }
      MPI_Allreduce(MPI_IN_PLACE, part_count.data(), size, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);

      for (int i = 0; i < size; i++) {
        for (int k = 0; k < dim; k++) {
          flatted_mass_center[i * dim + k] /= part_count[i];
        }
      }

      shared_ptr<vector<vec3>> mass_center = make_shared<vector<vec3>>();
      mass_center->resize(size);

      for (int i = 0; i < size; i++) {
        for (int k = 0; k < dim; k++) {
          (*mass_center)[i][k] = flatted_mass_center[i * dim + k];
        }
      }

      KDTree point_cloud(mass_center, dim, 2);
      point_cloud.generateKDTree();

      vector<int> part_map;
      point_cloud.getIndex(part_map);

      for (int i = 0; i < local_particle_num; i++) {
        result[i] = part_map[ptr[i]];
      }
    } else {
      result.resize(local_particle_num);
      for (int i = 0; i < local_particle_num; i++) {
        result[i] = rank;
      }
    }

    delete problem;
    delete ia;
  } else {
    result.resize(local_particle_num);
    for (int i = 0; i < local_particle_num; i++) {
      result[i] = rank;
    }
  }
}