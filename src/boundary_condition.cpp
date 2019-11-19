#include "GMLS_solver.h"

using namespace std;

void GMLS_Solver::EmposeBoundaryCondition() {
  __eq.F.resize(__particle.localParticleNum);
  __eq.gradP.resize(__particle.localParticleNum);
  __eq.dP.resize(__particle.localParticleNum);

  for (int i = 0; i < __particle.localParticleNum; i++) {
    __eq.F[i] = vec3(0.0, 0.0, 0.0);
    __eq.gradP[i] = vec3(0, 0, 0);
    __eq.dP[i] = 0.0;
  }
}