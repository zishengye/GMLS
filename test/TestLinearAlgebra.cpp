#include "LinearAlgebra/LinearAlgebra.hpp"

#include <gtest/gtest.h>

int globalArgc;
char **globalArgv;

TEST(LinearAlgebraVectorTest, VectorOperation) {
  LinearAlgebra::LinearAlgebraInitialize<DefaultLinearAlgebraBackend>(
      &globalArgc, &globalArgv, "build/petsc_setup.txt", nullptr);

  {
    LinearAlgebra::Vector<DefaultLinearAlgebraBackend> vec1;
    LinearAlgebra::Vector<DefaultLinearAlgebraBackend> vec2(10);
    LinearAlgebra::Vector<DefaultLinearAlgebraBackend> vec3;

    vec1.Clear();
    vec1.Create(vec2);

    vec2(2) = 1;
    vec1 = vec2;
    vec3 = vec1;

    EXPECT_EQ(vec1(2), 1);
  }

  LinearAlgebra::LinearAlgebraFinalize<DefaultLinearAlgebraBackend>();
}

TEST(LinearAlgebraMatrixTest, MatrixOperation) {
  LinearAlgebra::LinearAlgebraInitialize<DefaultLinearAlgebraBackend>(
      &globalArgc, &globalArgv, "build/petsc_setup.txt", nullptr);

  {
    LinearAlgebra::Matrix<DefaultLinearAlgebraBackend> mat1;
    LinearAlgebra::Matrix<DefaultLinearAlgebraBackend> mat2;
    LinearAlgebra::Matrix<DefaultLinearAlgebraBackend> mat3;

    LinearAlgebra::Vector<DefaultLinearAlgebraBackend> vec1;
    LinearAlgebra::Vector<DefaultLinearAlgebraBackend> vec2;
    LinearAlgebra::Vector<DefaultLinearAlgebraBackend> vec3;

    mat1.Resize(10, 10);
  }

  LinearAlgebra::LinearAlgebraFinalize<DefaultLinearAlgebraBackend>();
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