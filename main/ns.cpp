#include "Kokkos_Core.hpp"
#include "get_input_file.hpp"
#include "gmls_solver.hpp"
#include "search_command.hpp"
#include "trilinos_wrapper.hpp"
#include <mpi.h>

using namespace std;
using namespace Compadre;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  // get info from input file
  string inputFileName;
  vector<char *> cstrings;
  vector<string> strings;
  SearchCommand<string>(argc, argv, "-input", inputFileName);

  for (int i = 1; i < argc; i++) {
    cstrings.push_back(argv[i]);
  }

  if (!GetInputFile(inputFileName, strings, cstrings)) {
    return -1;
  }

  int inputCommandCount = cstrings.size();
  char **inputCommand = cstrings.data();

  PetscErrorCode ierr;

  ierr = PetscInitialize(&inputCommandCount, &inputCommand, NULL, NULL);
  if (ierr)
    return ierr;

  Kokkos::initialize(argc, argv);

  int mpiRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  if (mpiRank == 0)
    Kokkos::print_configuration(std::cout, true);

  double tStart, tEnd;
  tStart = MPI_Wtime();

  Tpetra::ScopeGuard tscope(&inputCommandCount, &inputCommand);

  {
    gmls_solver ns(inputCommandCount, inputCommand);

    if (!ns.is_initialized()) {
      return -1;
    }

    ns.time_integration();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();

  Kokkos::finalize();

  PetscPrintf(MPI_COMM_WORLD, "Program execution duration: %fs\n",
              tEnd - tStart);

  PetscFinalize();

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}