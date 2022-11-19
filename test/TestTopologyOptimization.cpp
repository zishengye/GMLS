#include "Equation/Poisson/PoissonEquation.hpp"
#include "TopologyOptimization/GeneralizedBendersDecomposition.hpp"
#include "TopologyOptimization/SolidIsotropicMicrostructurePenalization.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

int globalArgc;
char **globalArgv;

TEST(TopologyOptimizationTest, SolidIsotropicMicrostructurePenalization) {
  Kokkos::initialize(globalArgc, globalArgv);
  PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.txt",
                  PETSC_NULL);

  {
    Equation::PoissonEquation equation;
    equation.SetErrorTolerance(1e-2);
    equation.SetInitialDiscretizationResolution(0.01);

    equation.SetPolyOrder(1);
    equation.SetMaxRefinementIteration(10);
    equation.SetOutputLevel(0);
    equation.SetRefinementMarkRatio(0.9);

    equation.SetBoundaryType(
        [](const double x, const double y, const double z) {
          if ((abs(y) < 0.5 && x < 0) || (abs(x) < 0.5 && y < 0))
            return true;
          else
            return false;
        });
    equation.SetInteriorRhs(
        [](const double x, const double y, const double z) { return 0.1; });
    equation.SetBoundaryRhs([](const double x, const double y, const double z) {
      if ((abs(y) < 0.5 && x < 0) || (abs(x) < 0.5 && y < 0))
        return 0.0;
      else
        return 1.0;
    });

    std::vector<double> size(2);
    size[0] = 1.0;
    size[1] = 1.0;

    TopologyOptimization::SolidIsotropicMicrostructurePenalization simpTo;

    simpTo.AddEquation(equation);

    simpTo.SetDimension(2);
    simpTo.SetDomainSize(size);
    simpTo.SetDomainType(Geometry::Box);
    simpTo.SetInitialDiscretizationResolution(0.01);
    simpTo.SetVolumeFraction(0.4);
    simpTo.SetMaxIteration(500);

    simpTo.Init();
    simpTo.Optimize();
  }

  PetscFinalize();
  Kokkos::finalize();
}

// TEST(TopologyOptimizationTest, GeneralizedBendersDecomposition) {
//   Kokkos::initialize(globalArgc, globalArgv);
//   PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.txt",
//                   PETSC_NULL);

//   {
//     Equation::PoissonEquation equation;
//     equation.SetErrorTolerance(1e-2);
//     equation.SetInitialDiscretizationResolution(0.01);

//     equation.SetPolyOrder(1);
//     equation.SetMaxRefinementIteration(4);
//     equation.SetOutputLevel(0);
//     equation.SetRefinementMarkRatio(1.0);

//     equation.SetBoundaryType(
//         [](const double x, const double y, const double z) {
//           if ((abs(y) < 0.5 && x < 0) || (abs(x) < 0.5 && y < 0))
//             return true;
//           else
//             return false;
//         });
//     equation.SetInteriorRhs(
//         [](const double x, const double y, const double z) { return 0.1; });
//     equation.SetBoundaryRhs([](const double x, const double y, const double
//     z) {
//       if ((abs(y) < 0.5 && x < 0) || (abs(x) < 0.5 && y < 0))
//         return 0.0;
//       else
//         return 1.0;
//     });

//     std::vector<double> size(2);
//     size[0] = 1.0;
//     size[1] = 1.0;

//     TopologyOptimization::GeneralizedBendersDecomposition gbdTo;

//     gbdTo.AddEquation(equation);

//     gbdTo.SetDimension(2);
//     gbdTo.SetDomainSize(size);
//     gbdTo.SetDomainType(Geometry::Box);
//     gbdTo.SetInitialDiscretizationResolution(0.01);
//     gbdTo.SetVolumeFraction(0.40);

//     gbdTo.Init();
//     gbdTo.Optimize();
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