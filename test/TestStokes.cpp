#include "Equation/Stokes/StokesEquation.hpp"
#include "Kokkos_Core.hpp"

#include <gtest/gtest.h>

#include <cmath>

TEST(StokesEquationTest, 2DLinearSystemSolving) {
  {
    Equation::StokesEquation equation;
    equation.SetErrorTolerance(1e-3);
    equation.SetInitialDiscretizationResolution(0.02);

    std::vector<double> size(2);
    size[0] = 2.0;
    size[1] = 2.0;

    equation.SetDimension(2);
    equation.SetDomainSize(size);
    equation.SetDomainType(Geometry::Box);
    equation.SetMaxRefinementIteration(4);
    equation.SetOutputLevel(0);
    equation.SetRefinementMarkRatio(0.9);
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
                                       const double z,
                                       const unsigned int axes) {
      double rhs[2];
      rhs[0] = sin(M_PI * x) * cos(M_PI * y);
      rhs[1] = -cos(M_PI * x) * sin(M_PI * y);

      return rhs[axes];
    });
    equation.SetPressureInteriorRhs(
        [](const double x, const double y, const double z) {
          return -4.0 * pow(M_PI, 2.0) *
                 (cos(2.0 * M_PI * x) + cos(2.0 * M_PI * y));
        });

    equation.SetAnalyticalVelocitySolution([](const double x, const double y,
                                              const double z,
                                              const unsigned int axes) {
      double rhs[2];
      rhs[0] = sin(M_PI * x) * cos(M_PI * y);
      rhs[1] = -cos(M_PI * x) * sin(M_PI * y);

      return rhs[axes];
    });
    equation.SetAnalyticalPressureSolution(
        [](const double x, const double y, const double z) {
          return -cos(2.0 * M_PI * x) - cos(2.0 * M_PI * y);
        });

    equation.Init();
    equation.Update();
  }
}

TEST(StokesEquationTest, 3DLinearSystemSolving) {
  {
    Equation::StokesEquation equation;
    equation.SetErrorTolerance(1e-3);
    equation.SetInitialDiscretizationResolution(0.1);

    std::vector<double> size(3);
    size[0] = 2.0;
    size[1] = 2.0;
    size[2] = 2.0;

    equation.SetDimension(3);
    equation.SetDomainSize(size);
    equation.SetDomainType(Geometry::Box);
    equation.SetMaxRefinementIteration(3);
    equation.SetOutputLevel(0);
    equation.SetRefinementMarkRatio(0.9);
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
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  PetscInitialize(&argc, &argv, "build/petsc_setup.txt", PETSC_NULL);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ::testing::TestEventListeners &listeners =
      ::testing::UnitTest::GetInstance()->listeners();
  if (rank != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }

  Kokkos::initialize(argc, argv);

  auto result = RUN_ALL_TESTS();

  Kokkos::finalize();

  PetscFinalize();

  return result;
}