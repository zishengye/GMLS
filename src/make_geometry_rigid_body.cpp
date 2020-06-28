#include "gmls_solver.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

void GMLS_Solver::InitRigidBody() {
  vector<vec3> &rigidBodyPosition = __rigidBody.vector.Register("position");
  vector<vec3> &rigidBodyOrientation =
      __rigidBody.vector.Register("orientation");
  vector<vec3> &rigidBodyVelocity = __rigidBody.vector.Register("velocity");
  vector<vec3> &rigidBodyAngularVelocity =
      __rigidBody.vector.Register("angular velocity");
  vector<double> &rigidBodySize = __rigidBody.scalar.Register("size");

  if (__rigidBodyInclusion) {
    ifstream input(__rigidBodyInputFileName, ios::in);
    if (!input.is_open()) {
      PetscPrintf(PETSC_COMM_WORLD, "rigid body input file not exist\n");
      return;
    }

    while (!input.eof()) {
      vec3 xyz;
      double size;
      int type;
      for (int i = 0; i < __dim; i++) {
        input >> xyz[i];
      }
      rigidBodyPosition.push_back(xyz);
      input >> size;
      input >> type;
      rigidBodySize.push_back(size);

      rigidBodyOrientation.push_back(vec3(0.0, 0.0, 0.0));
      rigidBodyVelocity.push_back(vec3(0.0, 0.0, 0.0));
      rigidBodyAngularVelocity.push_back(vec3(0.0, 0.0, 0.0));
    }

    _multi.set_num_rigid_body(rigidBodyPosition.size());

    MPI_Barrier(MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "==> Number of rigid body: %d\n",
                rigidBodyPosition.size());
  }
}

int GMLS_Solver::IsInRigidBody(const vec3 &pos, double h) {
  static vector<vec3> &rigidBodyCoord =
      __rigidBody.vector.GetHandle("position");
  static vector<double> &rigidBodySize = __rigidBody.scalar.GetHandle("size");

  for (size_t i = 0; i < rigidBodyCoord.size(); i++) {
    vec3 dis = pos - rigidBodyCoord[i];
    if (dis.mag() < rigidBodySize[i] - 1.5 * h) {
      return -1;
    }
    if (dis.mag() < rigidBodySize[i] + 0.25 * h) {
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
    double h = __particleSize0[0];
    double vol = pow(h, 3);
    double a = pow(h, 2);

    for (size_t n = 0; n < rigidBodyCoord.size(); n++) {
      double r = rigidBodySize[n];
      int M_theta = round(r * M_PI / h);
      double d_theta = r * M_PI / M_theta;
      double d_phi = a / d_theta;

      vec3 particleSize = vec3(d_theta, d_phi, 0.0);

      for (int i = 0; i < M_theta; ++i) {
        double theta = M_PI * (i + 0.5) / M_theta;
        int M_phi = round(2 * M_PI * r * sin(theta) / d_phi);
        for (int j = 0; j < M_phi; ++j) {
          double phi = 2 * M_PI * j / M_phi;
          vec3 pCoord = vec3(theta, phi, 0.0);
          vec3 normal =
              vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
          vec3 pos = normal * r + rigidBodyCoord[n];
          if (pos[0] >= __domain[0][0] && pos[0] < __domain[1][0] &&
              pos[1] >= __domain[0][1] && pos[1] < __domain[1][1] &&
              pos[2] >= __domain[0][2] && pos[2] < __domain[1][2])
            InsertParticle(pos, 4, particleSize, normal, localIndex, 0, vol,
                           true, n, pCoord);
        }
      }
    }
  }

  if (__dim == 2) {
    double h = __particleSize0[0];
    double vol = pow(h, 2);

    for (size_t n = 0; n < rigidBodyCoord.size(); n++) {
      double r = rigidBodySize[n];
      int M_theta = round(2 * M_PI * r / h);
      if (M_theta % 2 == 1)
        M_theta++;
      double d_theta = 2 * M_PI * r / M_theta;

      vec3 particleSize = vec3(d_theta, 0, 0);

      for (int i = 0; i < M_theta; ++i) {
        double theta = 2 * M_PI * (i + 0.5) / M_theta;
        vec3 pCoord = vec3(theta, 0.0, 0.0);
        vec3 normal = vec3(cos(theta), sin(theta), 0.0);
        vec3 pos = normal * r + rigidBodyCoord[n];
        if (pos[0] >= __domain[0][0] && pos[0] < __domain[1][0] &&
            pos[1] >= __domain[0][1] && pos[1] < __domain[1][1])
          InsertParticle(pos, 4, particleSize, normal, localIndex, 0, vol, true,
                         n, pCoord);
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
  static auto &adaptive_level = __field.index.GetHandle("adaptive level");
  static auto &particleType = __field.index.GetHandle("particle type");
  static auto &attachedRigidBodyIndex =
      __field.index.GetHandle("attached rigid body index");
  static auto &volume = __field.scalar.GetHandle("volume");

  static auto &rigidBodyCoord = __rigidBody.vector.GetHandle("position");
  static auto &rigidBodySize = __rigidBody.scalar.GetHandle("size");

  int localIndex = coord.size();

  if (__dim == 3) {
    for (auto tag : splitTag) {
      splitList[tag].clear();
      const double thetaDelta = particleSize[tag][0] * 0.25 /
                                rigidBodySize[attachedRigidBodyIndex[tag]];
      const double phiDelta = particleSize[tag][1] * 0.25 /
                              rigidBodySize[attachedRigidBodyIndex[tag]];

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
            volume[tag] /= 8.0;
            normal[tag] = newNormal;
            pCoord[tag] = vec3(theta + i * thetaDelta, phi + j * phiDelta, 0.0);

            splitList[tag].push_back(tag);

            insert = true;
          } else {
            double vol = volume[tag];
            int newParticle = InsertParticle(
                newPos, particleType[tag], particleSize[tag], newNormal,
                localIndex, __adaptive_step, vol, true,
                attachedRigidBodyIndex[tag],
                vec3(theta + i * thetaDelta, phi + j * phiDelta, 0.0));

            if (newParticle == 0) {
              splitList[tag].push_back(localIndex - 1);
            }
          }
        }
      }
    }
  }

  if (__dim == 2) {
    for (auto tag : splitTag) {
      splitList[tag].clear();

      const double thetaDelta = particleSize[tag][0] * 0.25 /
                                rigidBodySize[attachedRigidBodyIndex[tag]];

      double theta = pCoord[tag][0];

      vec3 newNormal = vec3(
          cos(theta) * cos(thetaDelta) - sin(theta) * sin(thetaDelta),
          cos(theta) * sin(thetaDelta) + sin(theta) * cos(thetaDelta), 0.0);
      vec3 newPos = newNormal * rigidBodySize[attachedRigidBodyIndex[tag]] +
                    rigidBodyCoord[attachedRigidBodyIndex[tag]];

      coord[tag] = newPos;
      particleSize[tag][0] /= 2.0;
      volume[tag] /= 4.0;
      normal[tag] = newNormal;
      pCoord[tag] = vec3(theta + thetaDelta, 0.0, 0.0);
      adaptive_level[tag] = __adaptive_step;

      splitList[tag].push_back(tag);

      newNormal = vec3(
          cos(theta) * cos(-thetaDelta) - sin(theta) * sin(-thetaDelta),
          cos(theta) * sin(-thetaDelta) + sin(theta) * cos(-thetaDelta), 0.0);
      newPos = newNormal * rigidBodySize[attachedRigidBodyIndex[tag]] +
               rigidBodyCoord[attachedRigidBodyIndex[tag]];

      InsertParticle(newPos, particleType[tag], particleSize[tag], newNormal,
                     localIndex, __adaptive_step, volume[tag], true,
                     attachedRigidBodyIndex[tag],
                     vec3(theta - thetaDelta, 0.0, 0.0));

      splitList[tag].push_back(localIndex - 1);
    }
  }
}