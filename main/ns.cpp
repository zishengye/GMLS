#include "get_input_file.hpp"
#include "gmls_solver.hpp"
#include "search_command.hpp"

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

  double tStart, tEnd;
  tStart = MPI_Wtime();

  Kokkos::initialize(inputCommandCount, inputCommand);

  {
    GMLS_Solver ns(inputCommandCount, inputCommand);

    if (!ns.IsSuccessInit()) {
      return -1;
    }

    ns.TimeIntegration();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  tEnd = MPI_Wtime();

  PetscPrintf(MPI_COMM_WORLD, "Program execution duration: %fs\n",
              tEnd - tStart);

  Kokkos::finalize();

  PetscFinalize();

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}