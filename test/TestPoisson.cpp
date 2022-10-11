#include "Equation/Poisson/PoissonEquation.hpp"

#include <gtest/gtest.h>

int globalArgc;
char **globalArgv;

TEST(PoissonEquationTest, 2DLinearSystemSolving) {
  Kokkos::initialize(globalArgc, globalArgv);
  PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.txt",
                  PETSC_NULL);

  {
    Equation::PoissonEquation equation;
    equation.SetErrorTolerance(1e-3);
    equation.SetInitialDiscretizationResolution(0.1);

    std::vector<double> size(2);
    size[0] = 2.0;
    size[1] = 2.0;

    equation.SetDimension(2);
    equation.SetDomainSize(size);
    equation.SetDomainType(Geometry::Box);
    equation.SetMaxRefinementIteration(1);
    equation.SetOutputLevel(1);
    equation.SetRefinementMarkRatio();
    equation.SetInteriorRhs([](const double x, const double y, const double z) {
      return 2.0 * cos(x) * cos(y);
    });
    equation.SetBoundaryRhs([](const double x, const double y, const double z) {
      return cos(x) * cos(y);
    });
    equation.SetKappa(
        [](const double x, const double y, const double z) { return 1.0; });

    equation.Init();
    equation.Update();
  }

  {
    Equation::PoissonEquation equation;
    equation.SetErrorTolerance(1e-3);
    equation.SetInitialDiscretizationResolution(0.05);

    std::vector<double> size(2);
    size[0] = 2.0;
    size[1] = 2.0;

    equation.SetDimension(2);
    equation.SetDomainSize(size);
    equation.SetDomainType(Geometry::Box);
    equation.SetMaxRefinementIteration(1);
    equation.SetOutputLevel(1);
    equation.SetRefinementMarkRatio();
    equation.SetInteriorRhs([](const double x, const double y, const double z) {
      return 2.0 * cos(x) * cos(y);
    });
    equation.SetBoundaryRhs([](const double x, const double y, const double z) {
      return cos(x) * cos(y);
    });
    equation.SetKappa(
        [](const double x, const double y, const double z) { return 1.0; });

    equation.Init();
    equation.Update();
  }

  PetscFinalize();
  Kokkos::finalize();
}

TEST(PoissonEquationTest, 2DAdaptiveRefinement) {
  Kokkos::initialize(globalArgc, globalArgv);
  PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.txt",
                  PETSC_NULL);

  {
    Equation::PoissonEquation equation;
    equation.SetErrorTolerance(1e-3);
    equation.SetInitialDiscretizationResolution(0.2);

    std::vector<double> size(2);
    size[0] = 2.0;
    size[1] = 2.0;

    equation.SetDimension(2);
    equation.SetDomainSize(size);
    equation.SetDomainType(Geometry::Box);
    equation.SetMaxRefinementIteration(10);
    equation.SetOutputLevel(0);
    equation.SetRefinementMarkRatio();
    equation.SetInteriorRhs([](const double x, const double y, const double z) {
      return 2.0 * cos(x) * cos(y);
    });
    equation.SetBoundaryRhs([](const double x, const double y, const double z) {
      return cos(x) * cos(y);
    });
    equation.SetKappa(
        [](const double x, const double y, const double z) { return 1.0; });
    equation.SetAnalyticalFieldSolution(
        [](const double x, const double y, const double z) {
          return cos(x) * cos(y);
        });
    equation.SetAnalyticalFieldGradientSolution(
        [](const double x, const double y, const double z,
           const unsigned int i) {
          double grad[2];
          grad[0] = -sin(x) * cos(y);
          grad[1] = -cos(x) * sin(y);
          return grad[i];
        });

    equation.Init();
    equation.Update();
  }

  PetscFinalize();
  Kokkos::finalize();
}

TEST(PoissonEquationTest, 2DKappaDifference) {
  Kokkos::initialize(globalArgc, globalArgv);
  PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.txt",
                  PETSC_NULL);

  {
    Equation::PoissonEquation equation;
    equation.SetErrorTolerance(1e-3);
    equation.SetInitialDiscretizationResolution(0.02);

    std::vector<double> size(2);
    size[0] = 2.0;
    size[1] = 2.0;

    equation.SetDimension(2);
    equation.SetDomainSize(size);
    equation.SetDomainType(Geometry::Box);
    equation.SetMaxRefinementIteration(10);
    equation.SetOutputLevel(1);
    equation.SetRefinementMarkRatio();
    equation.SetInteriorRhs(
        [](const double x, const double y, const double z) { return 0.01; });
    equation.SetBoundaryRhs(
        [](const double x, const double y, const double z) { return 0; });
    equation.SetKappa([](const double x, const double y, const double z) {
      if (x < 0)
        return 1.0;
      else
        return 1.0;
    });

    equation.Init();
    equation.Update();
  }

  PetscFinalize();
  Kokkos::finalize();
}

TEST(PoissonEquationTest, 2DPeriodicBoundaryCondition) {
  Kokkos::initialize(globalArgc, globalArgv);
  PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.txt",
                  PETSC_NULL);
}

TEST(PoissonEquationTest, 3DLinearSystemSolving) {
  Kokkos::initialize(globalArgc, globalArgv);
  PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.txt",
                  PETSC_NULL);

  {
    Equation::PoissonEquation equation;
    equation.SetErrorTolerance(1e-3);
    equation.SetInitialDiscretizationResolution(0.05);

    std::vector<double> size(3);
    size[0] = 2.0;
    size[1] = 2.0;
    size[2] = 2.0;

    equation.SetDimension(3);
    equation.SetDomainSize(size);
    equation.SetDomainType(Geometry::Box);
    equation.SetMaxRefinementIteration(1);
    equation.SetOutputLevel(0);
    equation.SetRefinementMarkRatio();
    equation.SetInteriorRhs([](const double x, const double y, const double z) {
      return 3.0 * cos(x) * cos(y) * cos(z);
    });
    equation.SetBoundaryRhs([](const double x, const double y, const double z) {
      return cos(x) * cos(y) * cos(z);
    });
    equation.SetKappa(
        [](const double x, const double y, const double z) { return 1.0; });

    equation.Init();
    equation.Update();
  }

  PetscFinalize();
  Kokkos::finalize();
}

TEST(PoissonEquationTest, 3DAdaptiveRefinement) {
  Kokkos::initialize(globalArgc, globalArgv);
  PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.txt",
                  PETSC_NULL);

  {
    Equation::PoissonEquation equation;
    equation.SetErrorTolerance(1e-3);
    equation.SetInitialDiscretizationResolution(0.2);

    std::vector<double> size(3);
    size[0] = 2.0;
    size[1] = 2.0;
    size[2] = 2.0;

    equation.SetDimension(3);
    equation.SetDomainSize(size);
    equation.SetDomainType(Geometry::Box);
    equation.SetMaxRefinementIteration(10);
    equation.SetOutputLevel(0);
    equation.SetRefinementMarkRatio();
    equation.SetInteriorRhs([](const double x, const double y, const double z) {
      return 3.0 * cos(x) * cos(y) * cos(z);
    });
    equation.SetBoundaryRhs([](const double x, const double y, const double z) {
      return cos(x) * cos(y) * cos(z);
    });
    equation.SetKappa(
        [](const double x, const double y, const double z) { return 1.0; });
    equation.SetAnalyticalFieldSolution(
        [](const double x, const double y, const double z) {
          return cos(x) * cos(y) * cos(z);
        });
    equation.SetAnalyticalFieldGradientSolution(
        [](const double x, const double y, const double z,
           const unsigned int i) {
          double grad[3];
          grad[0] = -sin(x) * cos(y) * cos(z);
          grad[1] = -cos(x) * sin(y) * cos(z);
          grad[2] = -cos(x) * cos(y) * sin(z);
          return grad[i];
        });

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