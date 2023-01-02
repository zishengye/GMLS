#include "LinearAlgebra/Impl/Default/Default.hpp"
#include "LinearAlgebra/Impl/Default/DefaultLinearSolver.hpp"
#include "LinearAlgebra/Impl/Default/DefaultMatrix.hpp"
#include "LinearAlgebra/Impl/Default/DefaultVector.hpp"
#include "LinearAlgebra/Impl/Default/SquareMatrix.hpp"
#include "LinearAlgebra/LinearSolverDescriptor.hpp"

#include <mpi.h>

#include <gtest/gtest.h>

TEST(DefaultPackageTest, UpperTriangleMatrix) {
  LinearAlgebra::Impl::UpperTriangleMatrix triMat(3);
  HostRealVector b, x;

  Kokkos::resize(b, 3);
  Kokkos::resize(x, 3);

  triMat(0, 0) = 3;
  triMat(0, 1) = 2;
  triMat(0, 2) = 1;
  triMat(1, 1) = 5;
  triMat(1, 2) = 4;
  triMat(2, 2) = 6;

  b(0) = 1;
  b(1) = 1;
  b(2) = 1;

  triMat.Solve(b, x);

  EXPECT_NEAR(x(0), 0.233333, 1e-6);
  EXPECT_NEAR(x(1), 0.066667, 1e-6);
  EXPECT_NEAR(x(2), 0.166667, 1e-6);
}

// TEST(DefaultPackageTest, SparseMatMultiplication) {
//   LinearAlgebra::Impl::DefaultMatrix mat;
//   LinearAlgebra::Impl::DefaultVector b, x, y;

//   int mpiRank;
//   MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

//   int N = 200;

//   mat.Resize(N, N);
//   for (int i = 0; i < N; i++) {
//     std::vector<std::size_t> index;
//     index.resize(2);
//     index[0] = i;
//     index[1] = i + N;
//     mat.SetColIndex(i, index);
//   }
//   mat.GraphAssemble();

//   for (int i = 0; i < N; i++) {
//     mat.Increment(i, i, i + 1 + mpiRank);
//     mat.Increment(i, i + N, 2 * i + 2 + mpiRank);
//   }
//   mat.Assemble();

//   x.Resize(N);
//   y.Resize(N);
//   b.Resize(N);

//   x.Ones();
//   mat.MatrixVectorMultiplication(x, b);

//   for (int i = 0; i < N; i++)
//     EXPECT_NEAR(b(i), 3 * i + 3 + 2 * mpiRank, 1e-3);

//   LinearAlgebra::Impl::DefaultLinearSolver ls;
//   LinearAlgebra::LinearSolverDescriptor<LinearAlgebra::Impl::DefaultBackend>
//       descriptor;

//   descriptor.outerIteration = 1;
//   descriptor.spd = -1;
//   descriptor.maxIter = 500;
//   descriptor.relativeTol = 1e-10;
//   descriptor.setFromDatabase = true;

//   ls.AddLinearSystem(std::make_shared<LinearAlgebra::Impl::DefaultMatrix>(mat),
//                      descriptor);
//   ls.Solve(b, y);

//   for (int i = 0; i < N; i++)
//     EXPECT_NEAR(y(i), x(i), 1e-3);
// }

TEST(DefaultPackageTest, FivePointFiniteDifference) {
  LinearAlgebra::Impl::DefaultMatrix mat;
  LinearAlgebra::Impl::DefaultVector b, x;

  int mpiRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

  int N = 10;
  int Nx = N;
  int Ny = 2 * N;
  mat.Resize(Nx * Ny, Nx * Ny);

  std::vector<GlobalIndex> index;
  for (int i = 0; i < Nx; i++) {
    for (int j = 0; j < Ny; j++) {
      if (j == 0 || j == Ny - 1 || (i == 0 && mpiRank == 0) ||
          (i == Nx - 1 && mpiRank == 1)) {
        // boundary point
        index.resize(1);
        GlobalIndex Ni = i + Nx * mpiRank;
        GlobalIndex Nj = j;
        index[0] = Ni * Ny + Nj;
        mat.SetColIndex(i * Ny + j, index);
      } else {
        index.resize(5);
        GlobalIndex Ni = i + Nx * mpiRank;
        GlobalIndex Nj = j;
        index[0] = (Ni - 1) * Ny + Nj;
        index[1] = Ni * Ny + Nj - 1;
        index[2] = Ni * Ny + Nj;
        index[3] = Ni * Ny + Nj + 1;
        index[4] = (Ni + 1) * Ny + Nj;
        std::sort(index.begin(), index.end());
        mat.SetColIndex(i * Ny + j, index);
      }
    }
  }
  mat.GraphAssemble();

  for (int i = 0; i < Nx; i++) {
    for (int j = 0; j < Ny; j++) {
      if (j == 0 || j == Ny - 1 || (i == 0 && mpiRank == 0) ||
          ((i == Nx - 1) && mpiRank == 1)) {
        // boundary point
        index.resize(1);
        GlobalIndex Ni = i + Nx * mpiRank;
        GlobalIndex Nj = j;
        index[0] = Ni * Ny + Nj;
        mat.Increment(i * Ny + j, index[0], 1);
      } else {
        index.resize(5);
        GlobalIndex Ni = i + Nx * mpiRank;
        GlobalIndex Nj = j;
        index[0] = (Ni - 1) * Ny + Nj;
        index[1] = Ni * Ny + Nj - 1;
        index[2] = Ni * Ny + Nj;
        index[3] = Ni * Ny + Nj + 1;
        index[4] = (Ni + 1) * Ny + Nj;

        mat.Increment(i * Ny + j, index[0], 1.0);
        mat.Increment(i * Ny + j, index[1], 1.0);
        mat.Increment(i * Ny + j, index[2], -4.0);
        mat.Increment(i * Ny + j, index[3], 1.0);
        mat.Increment(i * Ny + j, index[4], 1.0);
      }
    }
  }
  mat.Assemble();

  // boundary condition
  b.Resize(Nx * Ny);
  for (int i = 0; i < Nx; i++) {
    for (int j = 0; j < Ny; j++) {
      if (j == 0 || j == Ny - 1 || (i == 0 && mpiRank == 0) ||
          (i == Nx - 1 && mpiRank == 1)) {
        b(i * Ny + j) = 0;
      } else {
        b(i * Ny + j) = 1;
      }
    }
  }

  for (int j = 0; j < Ny; j++) {
    if (mpiRank == 0)
      for (int i = 0; i < Nx; i++)
        printf("%6.4f  ", b(i * Ny + j));

    if (mpiRank == 0)
      printf("\n");
  }

  if (mpiRank == 0)
    printf("**************\n");

  x.Resize(Nx * Ny);
  LinearAlgebra::Impl::DefaultLinearSolver ls;
  LinearAlgebra::LinearSolverDescriptor<LinearAlgebra::Impl::DefaultBackend>
      descriptor;

  descriptor.outerIteration = 1;
  descriptor.spd = -1;
  descriptor.maxIter = 500;
  descriptor.relativeTol = 1e-6;
  descriptor.setFromDatabase = true;

  ls.AddLinearSystem(std::make_shared<LinearAlgebra::Impl::DefaultMatrix>(mat),
                     descriptor);
  ls.Solve(b, x);

  for (int j = 0; j < Ny; j++) {
    if (mpiRank == 0)
      for (int i = 0; i < Nx; i++)
        printf("%6.4f  ", x(i * Ny + j));

    if (mpiRank == 0)
      printf("\n");
  }
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  Kokkos::initialize(argc, argv);

  auto result = RUN_ALL_TESTS();

  Kokkos::finalize();

  MPI_Finalize();

  return result;
}