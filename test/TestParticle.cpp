#include "ParticleManager.hpp"

#include <gtest/gtest.h>

int globalArgc;
char **globalArgv;

TEST(ParticleSetTest, CartesianCoordinates) {
  Kokkos::initialize(globalArgc, globalArgv);

  {
    ParticleSet particleSet2D;
    particleSet2D.SetDimension(2);

    EXPECT_EQ(particleSet2D.GetLocalParticleNum(), 0);

    particleSet2D.Resize(10);
    EXPECT_EQ(particleSet2D.GetLocalParticleNum(), 10);

    // check reference
    auto coords = particleSet2D.GetParticleCoords();
    coords(1, 0) = 1.0;
    coords(5, 1) = -1.0;
    EXPECT_EQ(particleSet2D.GetParticleCoords(1, 0), 1.0);
    EXPECT_EQ(particleSet2D.GetParticleCoords(5, 1), -1.0);

    int mpiSize;
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    EXPECT_EQ(particleSet2D.GetGlobalParticleNum(), 10 * mpiSize);
  }

  Kokkos::finalize();
}

TEST(ParticleManagerTest, ParticleManager) {
  Kokkos::initialize(globalArgc, globalArgv);

  {
    std::vector<double> size(2);
    size[0] = 2.0;
    size[1] = 3.0;

    ParticleManager particleMgr;
    particleMgr.SetDimension(2);
    particleMgr.SetDomainType(Box);
    particleMgr.SetSize(size);
    particleMgr.SetSpacing(0.02);

    particleMgr.Init();
    particleMgr.Output();
  }

  Kokkos::finalize();
}

TEST(ParticleManagerTest, HierarchicalParticleManager) {
  Kokkos::initialize(globalArgc, globalArgv);

  {
    std::vector<double> size(2);
    size[0] = 2.0;
    size[1] = 3.0;

    HierarchicalParticleManager particleMgr;
    particleMgr.SetDimension(2);
    particleMgr.SetDomainType(Box);
    particleMgr.SetSize(size);
    particleMgr.SetSpacing(0.02);

    particleMgr.Init();
    particleMgr.Output();
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