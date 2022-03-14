#include "hydrogen.hpp"

#include <gtest/gtest.h>

using namespace std;
using namespace hydrogen;

int gArgc = 0;
char **gArgv = nullptr;

TEST(HostArrayTest, Initialization) {
  HostSpaceParameter parameter;
  HostSpace space(parameter);
}

TEST(DeviceArrayTest, Initialization) {
  HostSpaceParameter parameter;
  HostSpace space(parameter);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  gArgc = argc;
  gArgv = argv;

  return RUN_ALL_TESTS();
}