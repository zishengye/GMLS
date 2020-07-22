#include "parser.hpp"
#include "solver.hpp"

using namespace std;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  parser par(argc, argv);

  int full_argc = par.size();
  char **full_argv = par.data();

  PetscInitialize(&full_argc, &full_argv, NULL, NULL);

  PetscFinalize();

  network net;
  solver s(make_shared<network>(net));

  s.attach_parser(make_shared<parser>(par));

  MPI_Finalize();

  return 0;
}