#include "Equation/Poisson/PoissonEquation.hpp"
#include "TopologyOptimization/SolidIsotropicMicrostructurePenalization.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

int globalArgc;
char **globalArgv;

TEST(PoissonEquationTest, SolidIsotropicMicrostructurePenalization) {
  Kokkos::initialize(globalArgc, globalArgv);
  PetscInitialize(&globalArgc, &globalArgv, "build/petsc_setup.txt",
                  PETSC_NULL);

  {
    Equation::PoissonEquation equation;
    equation.SetErrorTolerance(1e-3);
    equation.SetInitialDiscretizationResolution(0.005);

    equation.SetBoundaryType(
        [](const double x, const double y, const double z) {
          if ((abs(y) < 0.5 && x < 0) || (abs(x) < 0.5 && y > 0))
            return true;
          else
            return false;
        });
    equation.SetInteriorRhs(
        [](const double x, const double y, const double z) { return 1.0; });
    equation.SetBoundaryRhs(
        [](const double x, const double y, const double z) { return 0.0; });

    std::vector<double> size(2);
    size[0] = 1.0;
    size[1] = 1.0;

    TopologyOptimization::SolidIsotropicMicrostructurePenalization simpTo;

    simpTo.AddEquation(equation);

    simpTo.SetDimension(2);
    simpTo.SetDomainSize(size);
    simpTo.SetDomainType(Geometry::Box);
    simpTo.SetInitialDiscretizationResolution(0.005);
    simpTo.SetVolumeFraction(0.4);
    simpTo.SetMaxIteration(50);

    simpTo.Init();
    simpTo.Optimize();
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