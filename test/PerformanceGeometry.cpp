#include "DomainGeometry.hpp"

#include <gtest/gtest.h>

#include <cstdlib>

int globalArgc;
char **globalArgv;

std::vector<double> random_number;
const int random_number_size = 10000;

TEST(DomainGeometryTest, IsInterior2D) {
  Kokkos::initialize(globalArgc, globalArgv);

  {
    std::vector<double> size(2);
    size[0] = 2.0;
    size[1] = 2.0;

    DomainGeometry geometry;
    geometry.SetDimension(2);
    geometry.SetType(Box);
    geometry.SetSize(size);

    const int N = 10000000;

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace> coords(
        "source coordinates", N, 3);

    Kokkos::View<bool *, Kokkos::DefaultExecutionSpace> results(
        "interior check results", N);

    for (int ite = 0; ite < 10; ite++) {
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,
                                                             coords.extent(0)),
          KOKKOS_LAMBDA(const int i) {
            coords(i, 0) =
                2.0 * random_number[2 * i % random_number_size] - 1.0;
            coords(i, 1) =
                2.0 * random_number[(2 * i + 1) % random_number_size] - 1.0;
          });
      Kokkos::fence();

      geometry.IsInterior(coords, results);

      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,
                                                             coords.extent(0)),
          KOKKOS_LAMBDA(const int i) { EXPECT_EQ(results(i), true); });
      Kokkos::fence();
    }
  }

  Kokkos::finalize();
}

TEST(DomainGeometryTest, IsInterior3D) {
  Kokkos::initialize(globalArgc, globalArgv);

  {
    std::vector<double> size(3);
    size[0] = 2.0;
    size[1] = 2.0;
    size[2] = 2.0;

    DomainGeometry geometry;
    geometry.SetDimension(3);
    geometry.SetType(Box);
    geometry.SetSize(size);

    EXPECT_EQ(geometry.IsInterior(0.0, 0.0, 0.0), true);
    EXPECT_EQ(geometry.IsInterior(0.1, -0.1, 0.1), true);
    EXPECT_EQ(geometry.IsInterior(1.1, -0.1, 2.0), false);

    const int N = 10000000;

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace> coords(
        "source coordinates", N, 3);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, coords.extent(0)),
        KOKKOS_LAMBDA(const int i) {
          coords(i, 0) = 2.0 * random_number[3 * i % random_number_size] - 1.0;
          coords(i, 1) =
              2.0 * random_number[(3 * i + 1) % random_number_size] - 1.0;
          coords(i, 2) =
              2.0 * random_number[(3 * i + 2) % random_number_size] - 1.0;
        });
    Kokkos::fence();

    Kokkos::View<bool *, Kokkos::DefaultExecutionSpace> results(
        "interior check results", N);

    geometry.IsInterior(coords, results);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, coords.extent(0)),
        KOKKOS_LAMBDA(const int i) { EXPECT_EQ(results(i), true); });
    Kokkos::fence();
  }

  Kokkos::finalize();
}

int main(int argc, char *argv[]) {
  MPI_Init(&globalArgc, &globalArgv);
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

  random_number.resize(random_number_size);
  for (int i = 0; i < random_number_size; i++) {
    random_number[i] =
        static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  auto result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}