#include "StokesEquation.hpp"

#include <gtest/gtest.h>

int globalArgc;
char **globalArgv;

TEST(PoissonEquationTest, 2DLinearSystemSolving) {
  Kokkos::initialize(globalArgc, globalArgv);
  PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.yaml",
                  PETSC_NULL);

  {
    StokesEquation equation;
    equation.SetErrorTolerance(1e-3);
    equation.SetInitialDiscretizationResolution(0.1);
  }

  PetscFinalize();
  Kokkos::finalize();
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);

  globalArgc = argc;
  globalArgv = argv;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ::testing::TestEventListeners &listeners =
      ::testing::UnitTest::GetInstance()->listeners();
  if (rank != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }

  auto result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}