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
  int Nr = 2;

  rigidBodyPosition.resize(Nr);
  rigidBodyOrientation.resize(Nr);
  rigidBodyVelocity.resize(Nr);
  rigidBodyAngularVelocity.resize(Nr);
  rigidBodySize.resize(Nr);

  // rigidBodyPosition[0][0] = 0.0;
  // rigidBodyPosition[0][1] = 0.5;
  // rigidBodyPosition[0][2] = 0.0;
  // rigidBodySize[0] = 0.1;

  rigidBodyPosition[0][0] = -1.0;
  rigidBodyPosition[0][1] = 1.0;
  rigidBodyPosition[0][2] = 1.0;
  rigidBodySize[0] = 1.0;

  rigidBodyPosition[1][0] = 0.0;
  rigidBodyPosition[1][1] = -1.0;
  rigidBodyPosition[1][2] = -1.5;
  rigidBodySize[1] = 1.0;

  // rigidBodyCoord[0][0] = 0;
  // rigidBodyCoord[0][1] = 0;
  // rigidBodyCoord[0][2] = 0;

  // rigidBodySize[0] = 1.0;
}

int GMLS_Solver::IsInRigidBody(vec3 &pos, double h) {
  static vector<vec3> &rigidBodyCoord =
      __rigidBody.vector.GetHandle("position");
  static vector<double> &rigidBodySize = __rigidBody.scalar.GetHandle("size");
  for (size_t i = 0; i < rigidBodyCoord.size(); i++) {
    vec3 dis = pos - rigidBodyCoord[i];
    if (dis.mag() < rigidBodySize[i] - 0.5 * __particleSize0[0]) {
      return -1;
    } else if ((dis.mag() + 1e-15) < (rigidBodySize[i] + 0.75 * h)) {
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

    for (size_t n = 0; n < rigidBodyCoord.size(); n++) {
      int M_theta = round(M_PI / h);
      double d_theta = M_PI / M_theta;
      double d_phi = a / d_theta;
      double r = rigidBodySize[n];

      vec3 particleSize = vec3(d_theta, d_phi, 0.0);

      for (int i = 0; i < M_theta; ++i) {
        double theta = M_PI * (i + 0.5) / M_theta;
        int M_phi = round(2 * M_PI * sin(theta) / d_phi);
        for (int j = 0; j < M_phi; ++j) {
          double phi = 2 * M_PI * j / M_phi;
          vec3 pCoord = vec3(theta, phi, 0.0);
          vec3 normal =
              vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
          vec3 pos = normal * r + rigidBodyCoord[n];
          if (pos[0] >= __domain[0][0] && pos[0] < __domain[1][0] &&
              pos[1] >= __domain[0][1] && pos[1] < __domain[1][1] &&
              pos[2] >= __domain[0][2] && pos[2] < __domain[1][2])
            InsertParticle(pos, 4, particleSize, normal, localIndex, vol, true,
                           n, pCoord);
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

void GMLS_Solver::SplitRigidBodySurfaceParticle(vector<int> &splitTag) {
  static auto &coord = __field.vector.GetHandle("coord");
  static auto &normal = __field.vector.GetHandle("normal");
  static auto &particleSize = __field.vector.GetHandle("size");
  static auto &pCoord = __field.vector.GetHandle("parameter coordinate");
  static auto &globalIndex = __field.index.GetHandle("global index");
  static auto &particleType = __field.index.GetHandle("particle type");
  static auto &attachedRigidBodyIndex =
      __field.index.GetHandle("attached rigid body index");

  static vector<vec3> &rigidBodyCoord =
      __rigidBody.vector.GetHandle("position");
  static vector<double> &rigidBodySize = __rigidBody.scalar.GetHandle("size");

  int localIndex = coord.size();

  if (__dim == 3) {
    for (auto tag : splitTag) {
      const double thetaDelta = particleSize[tag][0] * 0.25;
      const double phiDelta = particleSize[tag][1] * 0.25;

      double theta = pCoord[tag][0];
      double phi = pCoord[tag][1];

      bool insert = false;
      for (int i = -1; i < 2; i += 2) {
        for (int j = -1; j < 2; j += 2) {
          vec3 newNormal =
              vec3(sin(theta + i * thetaDelta) * cos(phi + j * phiDelta),
                   sin(theta + i * thetaDelta) * sin(phi + j * phiDelta),
                   cos(theta + i * thetaDelta));
          vec3 newPos = newNormal * rigidBodySize[attachedRigidBodyIndex[tag]] +
                        rigidBodyCoord[attachedRigidBodyIndex[tag]];

          if (!insert) {
            coord[tag] = newPos;
            particleSize[tag][0] /= 2.0;
            particleSize[tag][1] /= 2.0;
            normal[tag] = newNormal;
            pCoord[tag] = vec3(theta + i * thetaDelta, phi + j * phiDelta, 0.0);

            insert = true;
          } else {
            double vol = particleSize[tag][0] * particleSize[tag][1];
            InsertParticle(
                newPos, particleType[tag], particleSize[tag], newNormal,
                localIndex, vol, attachedRigidBodyIndex[tag],
                rigidBodySize[attachedRigidBodyIndex[tag]],
                vec3(theta + i * thetaDelta, phi + j * phiDelta, 0.0));
          }
        }
      }
    }
  }
}