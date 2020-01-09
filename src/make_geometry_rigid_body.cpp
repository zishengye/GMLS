#include "gmls_solver.h"

#include <cmath>
#include <iostream>

using namespace std;

void GMLS_Solver::InitRigidBody() {
  vector<vec3> &rigidBodyPosition = __rigidBody.vector.Register("position");
  vector<vec3> &rigidBodyOrientation =
      __rigidBody.vector.Register("orientation");
  vector<vec3> &rigidBodyVelocity = __rigidBody.vector.Register("velocity");
  vector<vec3> &rigidBodyAngularVelocity =
      __rigidBody.vector.Register("angular velocity");
  vector<double> &rigidBodySize = __rigidBody.scalar.Register("size");
  // initialize data storage
  int Nr = 1;

  rigidBodyPosition.resize(Nr);
  rigidBodyOrientation.resize(Nr);
  rigidBodyVelocity.resize(Nr);
  rigidBodyAngularVelocity.resize(Nr);
  rigidBodySize.resize(Nr);

  rigidBodyPosition[0][0] = 0.0;
  rigidBodyPosition[0][1] = 1.0;
  rigidBodyPosition[0][2] = 0.0;
  rigidBodySize[0] = 1.1;

  // int index = 0;
  // for (int i = 0; i < 3; i++) {
  //   for (int j = 0; j < 3; j++) {
  //     for (int k = 0; k < 3; k++) {
  //       rigidBodyPosition[index][0] = (i - 1) * 3;
  //       rigidBodyPosition[index][1] = (j - 1) * 3;
  //       rigidBodyPosition[index][2] = (k - 1) * 3;
  //       rigidBodySize[index] = 1.0;
  //       index++;
  //     }
  //   }
  // }

  // rigidBodyCoord[0][0] = 0;
  // rigidBodyCoord[0][1] = 0;
  // rigidBodyCoord[0][2] = 0;

  // rigidBodySize[0] = 1.0;
}

int GMLS_Solver::IsInRigidBody(vec3 &pos) {
  static vector<vec3> &rigidBodyCoord =
      __rigidBody.vector.GetHandle("position");
  static vector<double> &rigidBodySize = __rigidBody.scalar.GetHandle("size");
  for (size_t i = 0; i < rigidBodyCoord.size(); i++) {
    vec3 dis = pos - rigidBodyCoord[i];
    if (dis.mag() < rigidBodySize[i] - 0.5 * __particleSize0[0]) {
      return -1;
    } else if ((dis.mag() + 1e-15) <
               (rigidBodySize[i] + 0.25 * __particleSize0[0])) {
      return i;
    }
  }

  return -2;
}

void GMLS_Solver::InitRigidBodySurfaceParticle() {
  static vector<vec3> &rigidBodyCoord =
      __rigidBody.vector.GetHandle("position");
  static vector<double> &rigidBodySize = __rigidBody.scalar.GetHandle("size");
  static vector<vec3> &fluidCoord = __field.vector.GetHandle("coord");

  int localIndex = fluidCoord.size();
  if (__dim == 3) {
    double h = __particleSize0[0] / 2.0;
    double vol = pow(h, 3);
    double a = pow(h, 2);

    vec3 particleSize = vec3(h, h, h);

    for (size_t n = 0; n < rigidBodyCoord.size(); n++) {
      int M_theta = round(M_PI / h);
      double d_theta = M_PI / M_theta;
      double d_phi = a / d_theta;
      double r = rigidBodySize[n];
      for (int i = 0; i < M_theta; ++i) {
        double theta = M_PI * (i + 0.5) / M_theta;
        int M_phi = round(2 * M_PI * sin(theta) / d_phi);
        for (int j = 0; j < M_phi; ++j) {
          double phi = 2 * M_PI * j / M_phi;
          vec3 normal =
              vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
          vec3 pos = normal * r + rigidBodyCoord[n];
          if (pos[0] >= __domain[0][0] && pos[0] < __domain[1][0] &&
              pos[1] >= __domain[0][1] && pos[1] < __domain[1][1] &&
              pos[2] >= __domain[0][2] && pos[2] < __domain[1][2])
            InsertParticle(pos, 4, particleSize, normal, localIndex, vol, true,
                           n);
        }
      }
    }
  }

  if (__dim == 2) {
    double h = __particleSize0[0] / 2.0;
    double vol = pow(h, 2);

    vec3 particleSize = vec3(h, h, 0);

    for (size_t n = 0; n < rigidBodyCoord.size(); n++) {
      double r = rigidBodySize[n];
      int M_theta = round(2 * M_PI * r / h);
      double d_theta = 2 * M_PI * r / M_theta;
      for (int i = 0; i < M_theta; ++i) {
        double theta = 2 * M_PI * (i + 0.5) / M_theta;
        vec3 normal = vec3(cos(theta), sin(theta), 0.0);
        vec3 pos = normal * r + rigidBodyCoord[n];
        if (pos[0] >= __domain[0][0] && pos[0] < __domain[1][0] &&
            pos[1] >= __domain[0][1] && pos[1] < __domain[1][1])
          InsertParticle(pos, 4, particleSize, normal, localIndex, vol, true,
                         n);
      }
    }
  }
}