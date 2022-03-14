#include "domain_geometry.hpp"

#include <gtest/gtest.h>

int globalArgc;
char **globalArgv;

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

    EXPECT_EQ(geometry.IsInterior(0.0, 0.0, 0.0), true);
    EXPECT_EQ(geometry.IsInterior(0.1, -0.1, 0.0), true);
    EXPECT_EQ(geometry.IsInterior(1.1, -0.1, 0.0), false);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace> coords(
        "source coordinates", 10, 3);
    auto host_coords = Kokkos::create_mirror_view(coords);

    for (int i = 0; i < 10; i++) {
      host_coords(i, 0) = 0.1 * i - 0.5;
      host_coords(i, 1) = 0.1 * i - 0.3;
    }

    Kokkos::deep_copy(host_coords, coords);

    Kokkos::View<bool *, Kokkos::DefaultExecutionSpace> results;

    geometry.IsInterior(coords, results);
  }

  Kokkos::finalize();
}

TEST(DomainGeometryTest, IsInterior3D) {
  Kokkos::initialize(globalArgc, globalArgv);

  {
    std::vector<double> size(2);
    size[0] = 2.0;
    size[1] = 2.0;
    size[2] = 2.0;

    DomainGeometry geometry;
    geometry.SetDimension(2);
    geometry.SetType(Box);
    geometry.SetSize(size);

    EXPECT_EQ(geometry.IsInterior(0.0, 0.0, 0.0), true);
    EXPECT_EQ(geometry.IsInterior(0.1, -0.1, 0.1), true);
    EXPECT_EQ(geometry.IsInterior(1.1, -0.1, 2.0), false);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace> coords(
        "source coordinates", 10, 3);
    auto host_coords = Kokkos::create_mirror_view(coords);

    for (int i = 0; i < 10; i++) {
      host_coords(i, 0) = 0.1 * i - 0.5;
      host_coords(i, 1) = 0.1 * i - 0.3;
      host_coords(i, 2) = 0.1 * i + 0.3;
    }

    Kokkos::deep_copy(host_coords, coords);

    Kokkos::View<bool *, Kokkos::DefaultExecutionSpace> results;

    geometry.IsInterior(coords, results);
  }

  Kokkos::finalize();
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  globalArgc = argc;
  globalArgv = argv;

  return RUN_ALL_TESTS();
}