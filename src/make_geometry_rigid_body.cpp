#include "GMLS_solver.h"

void GMLS_Solver::InitRigidBody() {
  // initialize data storage
  int Nr = 0;

  __rigidBody.Ci_X.resize(Nr);
  __rigidBody.Ci_Theta.resize(Nr);
  __rigidBody.Ci_V.resize(Nr);
  __rigidBody.Ci_Omega.resize(Nr);
  __rigidBody.Ci_F.resize(Nr);
  __rigidBody.Ci_Torque.resize(Nr);
  __rigidBody.Ci_R.resize(Nr);
  __rigidBody.type.resize(Nr);

  for (int i = 0; i < Nr; i++) {
  }
}