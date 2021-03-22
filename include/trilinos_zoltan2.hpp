#ifndef _TRILINOS_ZOLTAN2_HPP_
#define _TRILINOS_ZOLTAN2_HPP_

#include <Tpetra_Map.hpp>
#include <Zoltan2_BasicVectorAdapter.hpp>
#include <Zoltan2_InputTraits.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_PartitioningSolution.hpp>
#include <cstdlib>
#include <vector>

#include "vec3.hpp"

// from trilinos example
// Trilinos/packages/zoltan2/example/geometric/rcb_C.cpp

class trilinos_rcp_partitioner {
public:
  typedef Tpetra::Map<> Map_t;
  typedef Map_t::local_ordinal_type localId_t;
  typedef Map_t::global_ordinal_type globalId_t;

  typedef Tpetra::Details::DefaultTypes::scalar_type scalar_t;
  typedef Zoltan2::BasicUserTypes<scalar_t, localId_t, globalId_t> myTypes;

  // TODO explain
  typedef Zoltan2::BasicVectorAdapter<myTypes> inputAdapter_t;
  typedef Zoltan2::EvaluatePartition<inputAdapter_t> quality_t;
  typedef inputAdapter_t::part_t part_t;

private:
  Teuchos::RCP<const Teuchos::Comm<int>> comm;
  Teuchos::ParameterList params;

  double tolerance;

  int rank, size;

public:
  trilinos_rcp_partitioner() : params("zoltan2 params") {
    comm = Tpetra::getDefaultComm();

    rank = comm->getRank();
    size = comm->getSize();

    tolerance = 1.1;

    params.set("debug_level", "no_status");
    params.set("debug_procs", "0");
    params.set("error_check_level", "no_assertions");
    params.set("debug_output_stream", "null");

    params.set("algorithm", "rcb");
    params.set("imbalance_tolerance", tolerance);
    params.set("num_global_parts", size);
  }

  void partition(std::vector<long long> &index, std::vector<vec3> &coord,
                 std::vector<int> &result);
};

#endif