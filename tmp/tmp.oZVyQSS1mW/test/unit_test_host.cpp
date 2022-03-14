#include "hydrogen.hpp"

#include <gtest/gtest.h>

using namespace std;
using namespace hydrogen;

int gArgc;
char **gArgv;

TEST(HostSpaceTest, Initialization) {
  proton::Option option;
  option.addOptionDatabase(gArgc, gArgv);

  HostSpaceParameter parameter;
  HostSpace space(parameter);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  gArgc = argc;
  gArgv = argv;

  return RUN_ALL_TESTS();
}