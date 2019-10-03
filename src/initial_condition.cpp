#include "GMLS_solver.h"
#include "manifold.h"
#include "sparse_matrix.h"

using namespace std;
using namespace Compadre;

#define PI 3.1415926

void GMLS_Solver::InitialCondition() {
  __fluid.us_old.resize(__fluid.localParticleNum);
  for (int i = 0; i < __fluid.localParticleNum; i++) {
    __fluid.us_old[i] = cos(__fluid.X_origin[i][1]);
  }
}