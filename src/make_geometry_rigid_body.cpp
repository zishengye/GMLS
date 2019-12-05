#include "GMLS_solver.h"

#include <cmath>
#include <iostream>

using namespace std;

void GMLS_Solver::InitRigidBody() {
  // initialize data storage
  int Nr = 1;

  __rigidBody.Ci_X.resize(Nr);
  __rigidBody.Ci_Theta.resize(Nr);
  __rigidBody.Ci_V.resize(Nr);
  __rigidBody.Ci_Omega.resize(Nr);
  __rigidBody.Ci_F.resize(Nr);
  __rigidBody.Ci_Torque.resize(Nr);
  __rigidBody.Ci_R.resize(Nr);
  __rigidBody.type.resize(Nr);

  // int index = 0;
  // for (int i = 0; i < 2; i++) {
  //   for (int j = 0; j < 2; j++) {
  //     for (int k = 0; k < 3; k++) {
  //       __rigidBody.Ci_X[index][0] = 6 * i - 3;
  //       __rigidBody.Ci_X[index][1] = 6 * j - 3;
  //       __rigidBody.Ci_X[index][2] = (k - 1) * 3;
  //       __rigidBody.Ci_R[index] = 1.0;
  //       index++;
  //     }
  //   }
  // }

  __rigidBody.Ci_X[0][0] = 0;
  __rigidBody.Ci_X[0][1] = 0;
  __rigidBody.Ci_X[0][2] = 0;

  __rigidBody.Ci_R[0] = 1.0;
}

int GMLS_Solver::IsInRigidBody(vec3 &pos) {
  for (size_t i = 0; i < __rigidBody.Ci_X.size(); i++) {
    vec3 dis = pos - __rigidBody.Ci_X[i];
    if (dis.mag() < __rigidBody.Ci_R[i] - 0.5 * __particleSize0[0]) {
      return -1;
    } else if ((dis.mag() + 1e-15) <
               (__rigidBody.Ci_R[i] + 0.75 * __particleSize0[0])) {
      return i;
    }
  }

  return -2;
}

void GMLS_Solver::InitRigidBodySurfaceParticle() {
  int localIndex = __particle.X.size();
  double h = __particleSize0[0] / 2.0;
  double vol = pow(h, 3);
  double a = pow(h, 2);

  vec3 particleSize = vec3(h, h, h);

  for (size_t n = 0; n < __rigidBody.Ci_X.size(); n++) {
    int M_theta = round(M_PI / h);
    double d_theta = M_PI / M_theta;
    double d_phi = a / d_theta;
    double r = __rigidBody.Ci_R[n];
    for (int i = 0; i < M_theta; ++i) {
      double theta = M_PI * (i + 0.5) / M_theta;
      int M_phi = round(2 * M_PI * sin(theta) / d_phi);
      for (int j = 0; j < M_phi; ++j) {
        double phi = 2 * M_PI * j / M_phi;
        vec3 normal =
            vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
        vec3 pos = normal * r + __rigidBody.Ci_X[n];
        if (pos[0] >= __domain[0][0] && pos[0] < __domain[1][0] &&
            pos[1] >= __domain[0][1] && pos[1] < __domain[1][1] &&
            pos[2] >= __domain[0][2] && pos[2] < __domain[1][2])
          InsertParticle(pos, 4, particleSize, normal, localIndex, vol, true,
                         n);
      }
    }
  }
}