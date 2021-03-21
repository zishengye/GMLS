#include "trilinos_zoltan2.hpp"

using namespace std;

void trilinos_rcp_partitioner::partition(vector<long long> &index,
                                         vector<vec3> &coord,
                                         vector<int> &result) {
  // call zoltan2 to build the solution
  int local_particle_num = index.size();

  int disable_repartition = 0;
  if (local_particle_num < 10) {
    disable_repartition = 1;
  }
  MPI_Allreduce(MPI_IN_PLACE, &disable_repartition, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  if (disable_repartition < 0.5 * size) {
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
      for (int i = 0; i < local_particle_num; i++) {
        result[i] = ptr[i];
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