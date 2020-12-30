#include "trilinos_zoltan2.hpp"

using namespace std;

void trilinos_rcp_partitioner::partition(vector<long long> &index,
                                         vector<vec3> &coord,
                                         vector<int> &result) {
  // call zoltan2 to build the solution
  int local_particle_num = index.size();

  vector<double> x, y, z;
  x.resize(local_particle_num);
  y.resize(local_particle_num);
  z.resize(local_particle_num);

  for (int i = 0; i < local_particle_num; i++) {
    x[i] = coord[i][0];
    y[i] = coord[i][1];
    z[i] = coord[i][2];
  }

  cout << local_particle_num << endl;

  inputAdapter_t *ia = new inputAdapter_t(
      local_particle_num, index.data(), x.data(), y.data(), z.data(), 1, 1, 1);

  Zoltan2::PartitioningProblem<inputAdapter_t> *problem =
      new Zoltan2::PartitioningProblem<inputAdapter_t>(ia, &params);

  problem->solve();

  auto &solution = problem->getSolution();

  quality_t *metricObject1 = new quality_t(ia, &params, // problem1->getComm(),
                                           &problem->getSolution());
  // Check the solution.

  if (rank == 0) {
    metricObject1->printMetrics(std::cout);
  }

  if (rank == 0) {
    scalar_t imb = metricObject1->getObjectCountImbalance();
    if (imb <= tolerance)
      std::cout << "pass: " << imb << std::endl;
    else
      std::cout << "fail: " << imb << std::endl;
    std::cout << std::endl;
  }
  delete metricObject1;

  const int *ptr = solution.getPartListView();

  if (ptr != NULL) {
    result.resize(local_particle_num);
    for (int i = 0; i < local_particle_num; i++) {
      result[i] = ptr[i];
    }
  } else {
    result.resize(local_particle_num);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (int i = 0; i < local_particle_num; i++) {
      result[i] = rank;
    }
  }

  delete problem;
  delete ia;
}