#include "StokesEquation.hpp"

#include <gtest/gtest.h>

#include <cmath>

int globalArgc;
char **globalArgv;

TEST(PoissonEquationTest, 2DLinearSystemSolving) {
  Kokkos::initialize(globalArgc, globalArgv);
  PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.txt",
                  PETSC_NULL);

  {
    StokesEquation equation;
    equation.SetErrorTolerance(1e-3);
    equation.SetInitialDiscretizationResolution(0.02);

    std::vector<double> size(2);
    size[0] = 2.0;
    size[1] = 2.0;

    equation.SetDimension(2);
    equation.SetDomainSize(size);
    equation.SetDomainType(Box);
    equation.SetMaxRefinementIteration(1);
    equation.SetOutputLevel(1);
    equation.SetRefinementMarkRatio();
    equation.SetVelocityInteriorRhs([](const double x, const double y,
                                       const double z, const unsigned int i) {
      double rhs[2];
      rhs[0] = 2.0 * pow(M_PI, 2.0) * sin(M_PI * x) * cos(M_PI * y) +
               2.0 * M_PI * sin(2.0 * M_PI * x);
      rhs[1] = -2.0 * pow(M_PI, 2.0) * cos(M_PI * x) * sin(M_PI * y) +
               2.0 * M_PI * sin(2.0 * M_PI * y);

      return rhs[i];
    });
    equation.SetVelocityBoundaryRhs([](const double x, const double y,
                                       const double z, const unsigned int i) {
      double rhs[2];
      rhs[0] = sin(M_PI * x) * cos(M_PI * y);
      rhs[1] = -cos(M_PI * x) * sin(M_PI * y);

      return rhs[i];
    });
    equation.SetPressureInteriorRhs(
        [](const double x, const double y, const double z) {
          return -4.0 * pow(M_PI, 2.0) *
                 (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y));
        });

    equation.Init();
    equation.Update();
  }

  PetscFinalize();
  Kokkos::finalize();
}

TEST(PoissonEquationTest, 3DLinearSystemSolving) {
  Kokkos::initialize(globalArgc, globalArgv);
  PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.txt",
                  PETSC_NULL);

  {
    StokesEquation equation;
    equation.SetErrorTolerance(1e-3);
    equation.SetInitialDiscretizationResolution(0.1);

    std::vector<double> size(3);
    size[0] = 2.0;
    size[1] = 2.0;
    size[2] = 2.0;

    equation.SetDimension(3);
    equation.SetDomainSize(size);
    equation.SetDomainType(Box);
    equation.SetMaxRefinementIteration(1);
    equation.SetOutputLevel(0);
    equation.SetRefinementMarkRatio();
    equation.SetVelocityInteriorRhs([](const double x, const double y,
                                       const double z, const unsigned int i) {
      double rhs[3];
      rhs[0] =
          3.0 * pow(M_PI, 2) * sin(M_PI * x) * cos(M_PI * y) * cos(M_PI * z) +
          2.0 * M_PI * sin(2.0 * M_PI * x);
      rhs[1] =
          -6.0 * pow(M_PI, 2) * cos(M_PI * x) * sin(M_PI * y) * cos(M_PI * z) +
          2.0 * M_PI * sin(2.0 * M_PI * y);
      rhs[2] =
          3.0 * pow(M_PI, 2) * cos(M_PI * x) * cos(M_PI * y) * sin(M_PI * z) +
          2.0 * M_PI * sin(2.0 * M_PI * z);

      return rhs[i];
    });
    equation.SetVelocityBoundaryRhs([](const double x, const double y,
                                       const double z, const unsigned int i) {
      double rhs[3];
      rhs[0] = sin(M_PI * x) * cos(M_PI * y) * cos(M_PI * z);
      rhs[1] = -2 * cos(M_PI * x) * sin(M_PI * y) * cos(M_PI * z);
      rhs[2] = cos(M_PI * x) * cos(M_PI * y) * sin(M_PI * z);

      return rhs[i];
    });
    equation.SetPressureInteriorRhs([](const double x, const double y,
                                       const double z) {
      return -4.0 * pow(M_PI, 2.0) *
             (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y) + cos(2.0 * M_PI * z));
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