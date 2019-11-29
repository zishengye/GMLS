#include "GMLS_solver.h"

#include <cmath>

void GMLS_Solver::InitRigidBody() {
  // initialize data storage
  int Nr = 27;

  __rigidBody.Ci_X.resize(Nr);
  __rigidBody.Ci_Theta.resize(Nr);
  __rigidBody.Ci_V.resize(Nr);
  __rigidBody.Ci_Omega.resize(Nr);
  __rigidBody.Ci_F.resize(Nr);
  __rigidBody.Ci_Torque.resize(Nr);
  __rigidBody.Ci_R.resize(Nr);
  __rigidBody.type.resize(Nr);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        const int index = i * 9 + j * 3 + k;
        __rigidBody.Ci_X[index][0] = (i - 1) * 4;
        __rigidBody.Ci_X[index][1] = (j - 1) * 4;
        __rigidBody.Ci_X[index][2] = (k - 1) * 4;
        __rigidBody.Ci_R[index] = 1.0;
      }
    }
  }
}
int GMLS_Solver::IsInRigidBody(vec3 &pos) {
  for (size_t i = 0; i < __rigidBody.Ci_X.size(); i++) {
    vec3 dis = pos - __rigidBody.Ci_X[i];
    if (dis.mag() < __rigidBody.Ci_R[i]) {
      return -1;
    } else if (dis.mag() < __rigidBody.Ci_R[i] + 0.5 * __particleSize0[0]) {
      return i;
    }
  }

  return -2;
}

void GMLS_Solver::InitRigidBodySurfaceParticle() {
  int localIndex = __particle.X.size();
  double h = __particleSize0[0];
  double vol = pow(h, 3);
  double a = pow(h, 2);
  for (size_t n = 0; n < __rigidBody.Ci_X.size(); n++) {
    int M_theta = std::round(M_PI / h);
    double d_theta = M_PI / M_theta;
    double d_phi = a / d_theta;
    double r = __rigidBody.Ci_R[n];
    for (int i = 0; i < M_theta; ++i) {
      double theta = M_PI * (i + 0.5) / M_theta;
      int M_phi = std::round(2 * M_PI * std::sin(theta) / d_phi);
      for (int j = 0; j < M_phi; ++j) {
        double phi = 2 * M_PI * j / M_phi;
        vec3 normal = vec3(std::sin(theta) * std::cos(phi),
                           std::sin(theta) * std::sin(phi), cos(theta));
        vec3 pos = normal * r + __rigidBody.Ci_X[n];
        InsertParticle(pos, 4, __particleSize0, normal, localIndex++, vol);
      }
    }
  }
}