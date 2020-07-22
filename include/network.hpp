#ifndef _NETWORK_HPP_
#define _NETWORK_HPP_

#include <mpi.h>

class network {
private:
  int _id, _proc_num;

public:
  network() {
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Comm_size(MPI_COMM_WORLD, &_proc_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &_id);
  }

  int get_id() { return _id; }

  int get_proc_num() { return _proc_num; }
};

#endif