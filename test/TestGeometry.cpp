#include "Geometry/Geometry.hpp"

#include <gtest/gtest.h>

int globalArgc;
char **globalArgv;

TEST(DomainGeometryTest, IsInterior2D) {
  Kokkos::initialize(globalArgc, globalArgv);

  {
    std::vector<double> size(2);
    size[0] = 2.0;
    size[1] = 2.0;

    Geometry::DomainGeometry geometry;
    geometry.SetDimension(2);
    geometry.SetType(Geometry::Box);
    geometry.SetSize(size);

    EXPECT_EQ(geometry.IsInterior(0.0, 0.0, 0.0), true);
    EXPECT_EQ(geometry.IsInterior(0.1, -0.1, 0.0), true);
    EXPECT_EQ(geometry.IsInterior(1.1, -0.1, 0.0), false);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace> coords(
        "source coordinates", 10, 3);
    auto hostCoords = Kokkos::create_mirror_view(coords);

    for (int i = 0; i < 10; i++) {
      hostCoords(i, 0) = 0.1 * i - 0.5;
      hostCoords(i, 1) = 0.1 * i - 0.3;
    }

    Kokkos::deep_copy(hostCoords, coords);

    Kokkos::View<bool *, Kokkos::DefaultExecutionSpace> results(
        "interior check results", 10);

    geometry.IsInterior(coords, results);

    auto hostResults = Kokkos::create_mirror_view(results);
    Kokkos::deep_copy(results, hostResults);
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

    Geometry::DomainGeometry geometry;
    geometry.SetDimension(3);
    geometry.SetType(Geometry::Box);
    geometry.SetSize(size);

    EXPECT_EQ(geometry.IsInterior(0.0, 0.0, 0.0), true);
    EXPECT_EQ(geometry.IsInterior(0.1, -0.1, 0.1), true);
    EXPECT_EQ(geometry.IsInterior(1.1, -0.1, 2.0), false);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace> coords(
        "source coordinates", 10, 3);
    auto hostCoords = Kokkos::create_mirror_view(coords);

    for (int i = 0; i < 10; i++) {
      hostCoords(i, 0) = 0.1 * i - 0.5;
      hostCoords(i, 1) = 0.1 * i - 0.3;
      hostCoords(i, 2) = 0.1 * i + 0.3;
    }

    Kokkos::deep_copy(hostCoords, coords);

    Kokkos::View<bool *, Kokkos::DefaultExecutionSpace> results(
        "interior check results", 10);

    geometry.IsInterior(coords, results);
  }

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