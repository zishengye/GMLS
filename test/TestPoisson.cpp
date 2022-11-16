#include "Equation/Poisson/PoissonEquation.hpp"

#include <gtest/gtest.h>

int globalArgc;
char **globalArgv;

// TEST(PoissonEquationTest, 2DLinearSystemSolving) {
//   Kokkos::initialize(globalArgc, globalArgv);
//   PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.txt",
//                   PETSC_NULL);

//   {
//     Equation::PoissonEquation equation;
//     equation.SetErrorTolerance(1e-3);
//     equation.SetInitialDiscretizationResolution(0.1);

//     std::vector<double> size(2);
//     size[0] = 2.0;
//     size[1] = 2.0;

//     equation.SetDimension(2);
//     equation.SetDomainSize(size);
//     equation.SetDomainType(Geometry::Box);
//     equation.SetMaxRefinementIteration(1);
//     equation.SetOutputLevel(1);
//     equation.SetRefinementMarkRatio();
//     equation.SetInteriorRhs([](const double x, const double y, const double
//     z) {
//       return 2.0 * cos(x) * cos(y);
//     });
//     equation.SetBoundaryRhs([](const double x, const double y, const double
//     z) {
//       return cos(x) * cos(y);
//     });
//     equation.SetKappa(
//         [](const double x, const double y, const double z) { return 1.0; });

//     equation.Init();
//     equation.Update();
//   }

//   {
//     Equation::PoissonEquation equation;
//     equation.SetErrorTolerance(1e-3);
//     equation.SetInitialDiscretizationResolution(0.05);

//     std::vector<double> size(2);
//     size[0] = 2.0;
//     size[1] = 2.0;

//     equation.SetDimension(2);
//     equation.SetDomainSize(size);
//     equation.SetDomainType(Geometry::Box);
//     equation.SetMaxRefinementIteration(1);
//     equation.SetOutputLevel(1);
//     equation.SetRefinementMarkRatio();
//     equation.SetInteriorRhs([](const double x, const double y, const double
//     z) {
//       return 2.0 * cos(x) * cos(y);
//     });
//     equation.SetBoundaryRhs([](const double x, const double y, const double
//     z) {
//       return cos(x) * cos(y);
//     });
//     equation.SetKappa(
//         [](const double x, const double y, const double z) { return 1.0; });

//     equation.Init();
//     equation.Update();
//   }

//   PetscFinalize();
//   Kokkos::finalize();
// }

// TEST(PoissonEquationTest, 2DAdaptiveRefinement) {
//   Kokkos::initialize(globalArgc, globalArgv);
//   PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.txt",
//                   PETSC_NULL);

//   {
//     Equation::PoissonEquation equation;
//     equation.SetErrorTolerance(1e-3);
//     equation.SetInitialDiscretizationResolution(0.2);

//     std::vector<double> size(2);
//     size[0] = 2.0;
//     size[1] = 2.0;

//     equation.SetDimension(2);
//     equation.SetDomainSize(size);
//     equation.SetDomainType(Geometry::Box);
//     equation.SetMaxRefinementIteration(10);
//     equation.SetOutputLevel(0);
//     equation.SetRefinementMarkRatio();
//     equation.SetInteriorRhs([](const double x, const double y, const double
//     z) {
//       return 2.0 * cos(x) * cos(y);
//     });
//     equation.SetBoundaryRhs([](const double x, const double y, const double
//     z) {
//       return cos(x) * cos(y);
//     });
//     equation.SetKappa(
//         [](const double x, const double y, const double z) { return 1.0; });
//     equation.SetAnalyticalFieldSolution(
//         [](const double x, const double y, const double z) {
//           return cos(x) * cos(y);
//         });
//     equation.SetAnalyticalFieldGradientSolution(
//         [](const double x, const double y, const double z,
//            const unsigned int i) {
//           double grad[2];
//           grad[0] = -sin(x) * cos(y);
//           grad[1] = -cos(x) * sin(y);
//           return grad[i];
//         });

//     equation.Init();
//     equation.Update();
//   }

//   PetscFinalize();
//   Kokkos::finalize();
// }

// TEST(PoissonEquationTest, 2DKappaDifference) {
//   Kokkos::initialize(globalArgc, globalArgv);
//   PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.txt",
//                   PETSC_NULL);

//   {
//     Equation::PoissonEquation equation;
//     equation.SetErrorTolerance(1e-4);
//     equation.SetInitialDiscretizationResolution(0.2);

//     std::vector<double> size(2);
//     size[0] = 2.0;
//     size[1] = 2.0;

//     equation.SetDimension(2);
//     equation.SetDomainSize(size);
//     equation.SetDomainType(Geometry::Box);
//     equation.SetMaxRefinementIteration(20);
//     equation.SetOutputLevel(1);
//     equation.SetRefinementMarkRatio(1.0);
//     equation.SetInteriorRhs([](const double x, const double y, const double
//     z) {
//       if (x < 0)
//         return 2.0 * cos(x) * cos(y);
//       else
//         return 0.2 * cos(x) * cos(y);
//     });
//     equation.SetBoundaryRhs([](const double x, const double y, const double
//     z) {
//       return cos(x) * cos(y);
//     });
//     equation.SetKappa([](const double x, const double y, const double z) {
//       if (x < 0)
//         return 1.0;
//       else
//         return 0.1;
//     });
//     equation.SetAnalyticalFieldSolution(
//         [](const double x, const double y, const double z) {
//           return cos(x) * cos(y);
//         });
//     equation.SetAnalyticalFieldGradientSolution(
//         [](const double x, const double y, const double z,
//            const unsigned int i) {
//           double grad[2];
//           grad[0] = -sin(x) * cos(y);
//           grad[1] = -cos(x) * sin(y);
//           return grad[i];
//         });

//     equation.Init();
//     equation.Update();
//   }

//   PetscFinalize();
//   Kokkos::finalize();
// }

TEST(PoissonEquationTest, 2DKappaDifference) {
  Kokkos::initialize(globalArgc, globalArgv);
  PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.txt",
                  PETSC_NULL);

  {
    Equation::PoissonEquation equation;
    equation.SetErrorTolerance(1e-3);
    equation.SetInitialDiscretizationResolution(0.2);

    std::vector<double> size(2);
    size[0] = 10.0;
    size[1] = 10.0;

    const double sigmaF = 100.0;
    const double sigmaP = 0.1;
    const double a = 2.0;
    const double coeff = (sigmaF - sigmaP) / (sigmaF + sigmaP);

    equation.SetPolyOrder(2);
    equation.SetDimension(2);
    equation.SetDomainSize(size);
    equation.SetDomainType(Geometry::Box);
    equation.SetMaxRefinementIteration(4);
    equation.SetOutputLevel(1);
    equation.SetRefinementMarkRatio(1.0);
    equation.SetBoundaryType(
        [](const double x, const double y, const double z) { return true; });
    equation.SetInteriorRhs(
        [](const double x, const double y, const double z) { return 0.0; });
    equation.SetBoundaryRhs(
        [=](const double x, const double y, const double z) {
          double r = sqrt(x * x + y * y);
          return (1 - coeff * a * a / r / r) * x;
        });
    equation.SetKappa([=](const double x, const double y, const double z) {
      double r = sqrt(x * x + y * y);
      if (r < a)
        return sigmaF;
      else
        return sigmaP;
    });
    equation.SetAnalyticalFieldSolution(
        [=](const double x, const double y, const double z) {
          double r = sqrt(x * x + y * y);
          if (r < a)
            return (1 - coeff) * x;
          else
            return (1 - coeff * a * a / r / r) * x;
        });
    equation.SetAnalyticalFieldGradientSolution(
        [=](const double x, const double y, const double z,
            const unsigned int i) {
          double gradPolar[2];
          double theta = std::atan2(y, x);
          double r = sqrt(x * x + y * y);
          if (r < a) {
            gradPolar[0] = (1 - coeff) * cos(theta);
            gradPolar[1] = -(1 - coeff) * sin(theta);
          } else {
            gradPolar[0] = (1 + coeff * a * a / r / r) * cos(theta);
            gradPolar[1] = -(1 - coeff * a * a / r / r) * sin(theta);
          }

          double grad[2];
          grad[0] = cos(theta) * gradPolar[0] - sin(theta) * gradPolar[1];
          grad[1] = sin(theta) * gradPolar[0] + cos(theta) * gradPolar[1];

          return grad[i];
        });

    equation.Init();
    equation.Update();
  }

  PetscFinalize();
  Kokkos::finalize();
}

// TEST(PoissonEquationTest, 2DHybridBoundaryCondition) {
//   Kokkos::initialize(globalArgc, globalArgv);
//   PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.txt",
//                   PETSC_NULL);

//   {
//     Equation::PoissonEquation equation;
//     equation.SetErrorTolerance(1e-3);
//     equation.SetInitialDiscretizationResolution(0.01);

//     std::vector<double> size(2);
//     size[0] = 1.0;
//     size[1] = 1.0;

//     equation.SetPolyOrder(1);
//     equation.SetDimension(2);
//     equation.SetDomainSize(size);
//     equation.SetDomainType(Geometry::Box);
//     equation.SetMaxRefinementIteration(1);
//     equation.SetOutputLevel(1);
//     equation.SetRefinementMarkRatio(0.95);
//     equation.SetBoundaryType(
//         [](const double x, const double y, const double z) {
//           if ((abs(y) < 0.5 && x < 0) || (abs(x) < 0.5 && y > 0))
//             return true;
//           else
//             return false;
//         });
//     equation.SetInteriorRhs(
//         [](const double x, const double y, const double z) { return 1.0; });
//     equation.SetBoundaryRhs(
//         [](const double x, const double y, const double z) { return 0.0; });
//     equation.SetKappa(
//         [](const double x, const double y, const double z) { return 0.5; });

//     equation.Init();
//     equation.Update();
//   }

//   PetscFinalize();
//   Kokkos::finalize();
// }

// TEST(PoissonEquationTest, 3DLinearSystemSolving) {
//   Kokkos::initialize(globalArgc, globalArgv);
//   PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.txt",
//                   PETSC_NULL);

//   {
//     Equation::PoissonEquation equation;
//     equation.SetErrorTolerance(1e-3);
//     equation.SetInitialDiscretizationResolution(0.05);

//     std::vector<double> size(3);
//     size[0] = 2.0;
//     size[1] = 2.0;
//     size[2] = 2.0;

//     equation.SetDimension(3);
//     equation.SetDomainSize(size);
//     equation.SetDomainType(Geometry::Box);
//     equation.SetMaxRefinementIteration(1);
//     equation.SetOutputLevel(0);
//     equation.SetRefinementMarkRatio();
//     equation.SetInteriorRhs([](const double x, const double y, const double
//     z) {
//       return 3.0 * cos(x) * cos(y) * cos(z);
//     });
//     equation.SetBoundaryRhs([](const double x, const double y, const double
//     z) {
//       return cos(x) * cos(y) * cos(z);
//     });
//     equation.SetKappa(
//         [](const double x, const double y, const double z) { return 1.0; });

//     equation.Init();
//     equation.Update();
//   }

//   PetscFinalize();
//   Kokkos::finalize();
// }

// TEST(PoissonEquationTest, 3DAdaptiveRefinement) {
//   Kokkos::initialize(globalArgc, globalArgv);
//   PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.txt",
//                   PETSC_NULL);

//   {
//     Equation::PoissonEquation equation;
//     equation.SetErrorTolerance(1e-3);
//     equation.SetInitialDiscretizationResolution(0.2);

//     std::vector<double> size(3);
//     size[0] = 2.0;
//     size[1] = 2.0;
//     size[2] = 2.0;

//     equation.SetDimension(3);
//     equation.SetDomainSize(size);
//     equation.SetDomainType(Geometry::Box);
//     equation.SetMaxRefinementIteration(10);
//     equation.SetOutputLevel(0);
//     equation.SetRefinementMarkRatio();
//     equation.SetInteriorRhs([](const double x, const double y, const double
//     z) {
//       return 3.0 * cos(x) * cos(y) * cos(z);
//     });
//     equation.SetBoundaryRhs([](const double x, const double y, const double
//     z) {
//       return cos(x) * cos(y) * cos(z);
//     });
//     equation.SetKappa(
//         [](const double x, const double y, const double z) { return 1.0; });
//     equation.SetAnalyticalFieldSolution(
//         [](const double x, const double y, const double z) {
//           return cos(x) * cos(y) * cos(z);
//         });
//     equation.SetAnalyticalFieldGradientSolution(
//         [](const double x, const double y, const double z,
//            const unsigned int i) {
//           double grad[3];
//           grad[0] = -sin(x) * cos(y) * cos(z);
//           grad[1] = -cos(x) * sin(y) * cos(z);
//           grad[2] = -cos(x) * cos(y) * sin(z);
//           return grad[i];
//         });

//     equation.Init();
//     equation.Update();
//   }

//   PetscFinalize();
//   Kokkos::finalize();
// }

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