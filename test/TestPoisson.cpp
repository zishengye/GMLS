#include "PoissonEquation.hpp"

#include <gtest/gtest.h>

int globalArgc;
char **globalArgv;

TEST(PoissonEquationTest, LinearSystemSovling) {
  Kokkos::initialize(globalArgc, globalArgv);
  PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.yaml",
                  PETSC_NULL);

  {
    PoissonEquation equation;
    equation.SetErrorTolerance(1e-3);
    equation.SetInitialDiscretizationResolution(0.1);

    std::vector<double> size(2);
    size[0] = 2.0;
    size[1] = 2.0;

    equation.SetDimension(2);
    equation.SetDomainSize(size);
    equation.SetDomainType(Box);
    equation.SetMaxRefinementIteration(1);
    equation.SetOutputLevel(1);
    equation.SetRefinementMarkRatio();

    equation.Init();
    equation.Update();
  }

  {
    PoissonEquation equation;
    equation.SetErrorTolerance(1e-3);
    equation.SetInitialDiscretizationResolution(0.05);

    std::vector<double> size(2);
    size[0] = 2.0;
    size[1] = 2.0;

    equation.SetDimension(2);
    equation.SetDomainSize(size);
    equation.SetDomainType(Box);
    equation.SetMaxRefinementIteration(1);
    equation.SetOutputLevel(1);
    equation.SetRefinementMarkRatio();

    equation.Init();
    equation.Update();
  }

  PetscFinalize();
  Kokkos::finalize();
}

TEST(PoissonEquationTest, AdaptiveRefinement) {
  Kokkos::initialize(globalArgc, globalArgv);
  PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.yaml",
                  PETSC_NULL);

  // {
  //   PoissonEquation equation;
  //   equation.SetErrorTolerance(1e-3);
  //   equation.SetInitialDiscretizationResolution(0.1);

  //   std::vector<double> size(2);
  //   size[0] = 2.0;
  //   size[1] = 2.0;

  //   equation.SetDimension(2);
  //   equation.SetDomainSize(size);
  //   equation.SetDomainType(Box);
  //   equation.SetMaxRefinementIteration(10);
  //   equation.SetOutputLevel(1);
  //   equation.SetRefinementMarkRatio();

  //   equation.Init();
  //   equation.Update();
  // }

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