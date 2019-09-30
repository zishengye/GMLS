#include "GMLS_solver.h"

using namespace std;
using namespace Compadre;

int main(int argc, char *argv[]) {
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL);
  if (ierr)
    return ierr;

  double tStart, tEnd;
  tStart = MPI_Wtime();

  Kokkos::initialize(argc, argv);

  GMLS_Solver ns(argc, argv);

  if (!ns.IsSuccessInit()) {
    return -1;
  }

  ns.TimeIntegration();

  tEnd = MPI_Wtime();

  PetscPrintf(MPI_COMM_WORLD, "Program execution duration: %fs\n",
              tEnd - tStart);

  Kokkos::finalize();

  PetscFinalize();
}