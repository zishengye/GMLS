#include "GMLS_solver.h"
#include "manifold.h"
#include "sparse_matrix.h"

using namespace std;
using namespace Compadre;

#define PI 3.1415926

void GMLS_Solver::InitialCondition() {
  __particle.us_old.resize(__particle.localParticleNum);
  for (int i = 0; i < __particle.localParticleNum; i++) {
    __particle.us_old[i] = cos(__particle.X_origin[i][1]);
  }
}