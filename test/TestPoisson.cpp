#include "PoissonEquation.hpp"

#include <gtest/gtest.h>

int globalArgc;
char **globalArgv;

TEST(PoissonEquationTest, 2DLinearSystemSolving) {
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
    equation.SetInteriorRhs(
        [](double x, double y, double z) { return 2.0 * cos(x) * cos(y); });
    equation.SetBoundaryRhs(
        [](double x, double y, double z) { return cos(x) * cos(y); });

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
    equation.SetInteriorRhs(
        [](double x, double y, double z) { return 2.0 * cos(x) * cos(y); });
    equation.SetBoundaryRhs(
        [](double x, double y, double z) { return cos(x) * cos(y); });

    equation.Init();
    equation.Update();
  }

  PetscFinalize();
  Kokkos::finalize();
}

TEST(PoissonEquationTest, 2DAdaptiveRefinement) {
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
    equation.SetMaxRefinementIteration(10);
    equation.SetOutputLevel(1);
    equation.SetRefinementMarkRatio();
    equation.SetInteriorRhs(
        [](double x, double y, double z) { return 2.0 * cos(x) * cos(y); });
    equation.SetBoundaryRhs(
        [](double x, double y, double z) { return cos(x) * cos(y); });

    equation.Init();
    equation.Update();
  }

  PetscFinalize();
  Kokkos::finalize();
}

TEST(PoissonEquationTest, 2DPeriodicBoundaryCondition) {
  Kokkos::initialize(globalArgc, globalArgv);
  PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.yaml",
                  PETSC_NULL);
}

TEST(PoissonEquationTest, 3DLinearSystemSolving) {
  Kokkos::initialize(globalArgc, globalArgv);
  PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.yaml",
                  PETSC_NULL);

  {
    PoissonEquation equation;
    equation.SetErrorTolerance(1e-3);
    equation.SetInitialDiscretizationResolution(0.05);

    std::vector<double> size(3);
    size[0] = 2.0;
    size[1] = 2.0;
    size[2] = 2.0;

    equation.SetDimension(3);
    equation.SetDomainSize(size);
    equation.SetDomainType(Box);
    equation.SetMaxRefinementIteration(1);
    equation.SetOutputLevel(1);
    equation.SetRefinementMarkRatio();
    equation.SetInteriorRhs([](double x, double y, double z) {
      return 3.0 * cos(x) * cos(y) * cos(z);
    });
    equation.SetBoundaryRhs(
        [](double x, double y, double z) { return cos(x) * cos(y) * cos(z); });

    equation.Init();
    equation.Update();
  }

  PetscFinalize();
  Kokkos::finalize();
}

TEST(PoissonEquationTest, 3DAdaptiveRefinement) {
  Kokkos::initialize(globalArgc, globalArgv);
  PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.yaml",
                  PETSC_NULL);

  {
    PoissonEquation equation;
    equation.SetErrorTolerance(1e-3);
    equation.SetInitialDiscretizationResolution(0.05);

    std::vector<double> size(3);
    size[0] = 2.0;
    size[1] = 2.0;
    size[2] = 2.0;

    equation.SetDimension(3);
    equation.SetDomainSize(size);
    equation.SetDomainType(Box);
    equation.SetMaxRefinementIteration(10);
    equation.SetOutputLevel(0);
    equation.SetRefinementMarkRatio();
    equation.SetInteriorRhs([](double x, double y, double z) {
      return 3.0 * cos(x) * cos(y) * cos(z);
    });
    equation.SetBoundaryRhs(
        [](double x, double y, double z) { return cos(x) * cos(y) * cos(z); });

    equation.Init();
    equation.Update();
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