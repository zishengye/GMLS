#include "petsc_wrapper.hpp"

using namespace std;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL);
  if (ierr)
    return ierr;

  petsc_block_matrix mat;
  mat.resize(2, 2);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto A = mat.get_reference(0, 0);
  A.resize(10, 10, 10 * size);

  A.graph_assemble();

  PetscFinalize();

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}