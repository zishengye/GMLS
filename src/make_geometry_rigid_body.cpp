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
  vector<int> &rigidBodyType = __rigidBody.index.Register("type");

  if (__rigidBodyInclusion) {
    ifstream input(__rigidBodyInputFileName, ios::in);
    if (!input.is_open()) {
      PetscPrintf(PETSC_COMM_WORLD, "rigid body input file not exist\n");
      return;
    }

    while (!input.eof()) {
      vec3 xyz;
      vec3 rxyz;
      double size;
      int type;
      input >> type;
      input >> size;
      for (int i = 0; i < __dim; i++) {
        input >> xyz[i];
      }

      int rotation_dof = (__dim == 3) ? 3 : 1;
      for (int i = 0; i < rotation_dof; i++) {
        input >> rxyz[i];
      }

      rigidBodyType.push_back(type);
      rigidBodySize.push_back(size);
      rigidBodyPosition.push_back(xyz);
      rigidBodyOrientation.push_back(rxyz);

      rigidBodyVelocity.push_back(vec3(0.0, 0.0, 0.0));
      rigidBodyAngularVelocity.push_back(vec3(0.0, 0.0, 0.0));

      /* rigid body type
      type 1: circle in 2d, sphere in 3d
      type 2: square in 2d, cubic in 3d
      type 3: equilateral triangle in 2d, tetrahedron in 3d
       */
    }

    _multi.set_num_rigid_body(rigidBodyPosition.size());

    MPI_Barrier(MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "==> Number of rigid body: %d\n",
                rigidBodyPosition.size());
  }
}

int GMLS_Solver::IsInRigidBody(const vec3 &pos, double h,
                               int attachedRigidBodyIndex) {
  static vector<vec3> &rigidBodyCoord =
      __rigidBody.vector.GetHandle("position");
  static vector<vec3> &rigidBodyOrientation =
      __rigidBody.vector.GetHandle("orientation");
  static vector<double> &rigidBodySize = __rigidBody.scalar.GetHandle("size");
  static auto &rigidBodyType = __rigidBody.index.GetHandle("type");

  for (size_t i = 0; i < rigidBodyCoord.size(); i++) {
    switch (rigidBodyType[i]) {
    case 1:
      // circle in 2d, sphere in 3d
      {
        vec3 dis = pos - rigidBodyCoord[i];
        if (dis.mag() < rigidBodySize[i] - h) {
          return -1;
        }
        if (dis.mag() <= rigidBodySize[i]) {
          if (attachedRigidBodyIndex != i)
            return i;
        }
        // for 2d case
        if (__dim == 2) {
          if (dis.mag() < rigidBodySize[i] + h) {
            double r = rigidBodySize[i];
            double h0 = __particleSize0[0];
            int M_theta = round(2 * M_PI * r / h0) * (__adaptive_step + 1);
            double d_theta = 2 * M_PI / M_theta;
            double theta = atan2(dis[1], dis[0]);
            if (theta < 0) {
              theta += 2 * M_PI;
            }
            int theta0 = floor(theta / d_theta);
            int theta1 = theta0 + 1;
            vec3 pos1 = vec3(r * cos(theta0 * d_theta),
                             r * sin(theta0 * d_theta), 0.0) +
                        rigidBodyCoord[i];
            vec3 pos2 = vec3(r * cos(theta1 * d_theta),
                             r * sin(theta1 * d_theta), 0.0) +
                        rigidBodyCoord[i];
            vec3 dis1 = pos - pos1;
            vec3 dis2 = pos - pos2;
            if (attachedRigidBodyIndex < 0) {
              if (dis1.mag() < 0.5 * h || dis2.mag() < 0.5 * h) {
                return i;
              }
            } else {
              if (dis1.mag() < 0.5 * h || dis2.mag() < 0.5 * h) {
                if (attachedRigidBodyIndex != i) {
                  return i;
                }
              }
            }
          }
        }
      }
      break;

    case 2:
      // square in 2d, cubic in 3d
      {
        if (__dim == 2) {
          double half_side_length = rigidBodySize[i];
          double theta = rigidBodyOrientation[i][0];

          vec3 abs_dis = pos - rigidBodyCoord[i];
          // rotate back
          vec3 dis =
              vec3(cos(theta) * abs_dis[0] + sin(theta) * abs_dis[1],
                   -sin(theta) * abs_dis[0] + cos(theta) * abs_dis[1], 0.0);

          if (abs(dis[0]) < half_side_length - 1.5 * h &&
              abs(dis[1]) < half_side_length - 1.5 * h) {
            return -1;
          }
          if (abs(dis[0]) < half_side_length + 0.25 * h &&
              abs(dis[1]) < half_side_length + 0.25 * h) {
            return i;
          }
        }
        if (__dim == 3) {
        }
      }
      break;
    }
  }

  return -2;
}

void GMLS_Solver::InitRigidBodySurfaceParticle() {
  static vector<vec3> &rigidBodyCoord =
      __rigidBody.vector.GetHandle("position");
  static vector<vec3> &rigidBodyOrientation =
      __rigidBody.vector.GetHandle("orientation");
  static vector<double> &rigidBodySize = __rigidBody.scalar.GetHandle("size");
  static auto &rigidBodyType = __rigidBody.index.GetHandle("type");
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

      for (int i = 0; i < M_theta; ++i) {
        double theta = M_PI * (i + 0.5) / M_theta;
        int M_phi = round(2 * M_PI * r * sin(theta) / d_phi);
        for (int j = 0; j < M_phi; ++j) {
          double phi = 2 * M_PI * (j + 0.5) / M_phi;

          vec3 particleSize =
              vec3(d_theta, 2 * M_PI / M_phi * r * sin(theta), 0.0);
          vec3 pCoord = vec3(theta, phi, 0.0);
          vec3 normal =
              vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
          vec3 pos = normal * r + rigidBodyCoord[n];
          if (pos[0] >= __domain[0][0] && pos[0] < __domain[1][0] &&
              pos[1] >= __domain[0][1] && pos[1] < __domain[1][1] &&
              pos[2] >= __domain[0][2] && pos[2] < __domain[1][2])
            InsertParticle(pos, 5, particleSize, normal, localIndex, 0, vol,
                           true, n, pCoord);
        }
      }
    }
  }

  if (__dim == 2) {

    for (size_t n = 0; n < rigidBodyCoord.size(); n++) {
      switch (rigidBodyType[n]) {
      case 1:
        // circle
        {
          double h = __particleSize0[0];
          double vol = pow(h, 2);

          double r = rigidBodySize[n];

          int M_theta = round(2 * M_PI * r / h);
          double d_theta = 2 * M_PI * r / M_theta;

          vec3 particleSize = vec3(d_theta, 0, 0);

          for (int i = 0; i < M_theta; ++i) {
            double theta = 2 * M_PI * (i + 0.5) / M_theta;
            vec3 pCoord = vec3(theta, 0.0, 0.0);
            vec3 normal = vec3(cos(theta), sin(theta), 0.0);
            vec3 pos = normal * r + rigidBodyCoord[n];
            if (pos[0] >= __domain[0][0] && pos[0] < __domain[1][0] &&
                pos[1] >= __domain[0][1] && pos[1] < __domain[1][1])
              InsertParticle(pos, 5, particleSize, normal, localIndex, 0, vol,
                             true, n, pCoord);
          }
        }

        break;

      case 2:
        // square
        {
          double half_side_length = rigidBodySize[n];
          double theta = rigidBodyOrientation[n][0];

          int particleNumPerSize = rigidBodySize[n] * 2.0 / __particleSize0[0];

          double h = rigidBodySize[n] * 2.0 / particleNumPerSize;
          vec3 particleSize = vec3(h, h, 0.0);
          double vol = pow(h, 2.0);

          double xPos, yPos;
          xPos = -half_side_length;
          yPos = -half_side_length;

          vec3 norm = vec3(-sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0, 0.0);
          vec3 normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                             sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
          vec3 pos = vec3(cos(theta) * xPos - sin(theta) * yPos,
                          sin(theta) * xPos + cos(theta) * yPos, 0.0) +
                     rigidBodyCoord[n];
          vec3 pCoord = vec3(0.0, 0.0, 0.0);
          if (pos[0] >= __domain[0][0] && pos[0] < __domain[1][0] &&
              pos[1] >= __domain[0][1] && pos[1] < __domain[1][1])
            InsertParticle(pos, 4, particleSize, normal, localIndex, 0, vol,
                           true, n, pCoord);

          xPos += 0.5 * h;
          norm = vec3(0.0, -1.0, 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
          while (xPos < half_side_length) {
            pos = vec3(cos(theta) * xPos - sin(theta) * yPos,
                       sin(theta) * xPos + cos(theta) * yPos, 0.0) +
                  rigidBodyCoord[n];
            if (pos[0] >= __domain[0][0] && pos[0] < __domain[1][0] &&
                pos[1] >= __domain[0][1] && pos[1] < __domain[1][1])
              InsertParticle(pos, 5, particleSize, normal, localIndex, 0, vol,
                             true, n, pCoord);
            xPos += h;
          }

          xPos = half_side_length;
          norm = vec3(sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0, 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
          pos = vec3(cos(theta) * xPos - sin(theta) * yPos,
                     sin(theta) * xPos + cos(theta) * yPos, 0.0) +
                rigidBodyCoord[n];
          if (pos[0] >= __domain[0][0] && pos[0] < __domain[1][0] &&
              pos[1] >= __domain[0][1] && pos[1] < __domain[1][1])
            InsertParticle(pos, 4, particleSize, normal, localIndex, 0, vol,
                           true, n, pCoord);

          yPos += 0.5 * h;
          norm = vec3(1.0, 0.0, 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
          while (yPos < half_side_length) {
            pos = vec3(cos(theta) * xPos - sin(theta) * yPos,
                       sin(theta) * xPos + cos(theta) * yPos, 0.0) +
                  rigidBodyCoord[n];
            if (pos[0] >= __domain[0][0] && pos[0] < __domain[1][0] &&
                pos[1] >= __domain[0][1] && pos[1] < __domain[1][1])
              InsertParticle(pos, 5, particleSize, normal, localIndex, 0, vol,
                             true, n, pCoord);
            yPos += h;
          }

          yPos = half_side_length;
          norm = vec3(sqrt(2.0) / 2.0, sqrt(2.0) / 2.0, 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
          pos = vec3(cos(theta) * xPos - sin(theta) * yPos,
                     sin(theta) * xPos + cos(theta) * yPos, 0.0) +
                rigidBodyCoord[n];
          if (pos[0] >= __domain[0][0] && pos[0] < __domain[1][0] &&
              pos[1] >= __domain[0][1] && pos[1] < __domain[1][1])
            InsertParticle(pos, 4, particleSize, normal, localIndex, 0, vol,
                           true, n, pCoord);

          xPos -= 0.5 * h;
          norm = vec3(0.0, 1.0, 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
          while (xPos > -half_side_length) {
            pos = vec3(cos(theta) * xPos - sin(theta) * yPos,
                       sin(theta) * xPos + cos(theta) * yPos, 0.0) +
                  rigidBodyCoord[n];
            if (pos[0] >= __domain[0][0] && pos[0] < __domain[1][0] &&
                pos[1] >= __domain[0][1] && pos[1] < __domain[1][1])
              InsertParticle(pos, 5, particleSize, normal, localIndex, 0, vol,
                             true, n, pCoord);
            xPos -= h;
          }

          xPos = -half_side_length;
          norm = vec3(-sqrt(2.0) / 2.0, sqrt(2.0) / 2.0, 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
          pos = vec3(cos(theta) * xPos - sin(theta) * yPos,
                     sin(theta) * xPos + cos(theta) * yPos, 0.0) +
                rigidBodyCoord[n];
          if (pos[0] >= __domain[0][0] && pos[0] < __domain[1][0] &&
              pos[1] >= __domain[0][1] && pos[1] < __domain[1][1])
            InsertParticle(pos, 4, particleSize, normal, localIndex, 0, vol,
                           true, n, pCoord);

          yPos -= 0.5 * h;
          norm = vec3(-1.0, 0.0, 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
          while (yPos > -half_side_length) {
            pos = vec3(cos(theta) * xPos - sin(theta) * yPos,
                       sin(theta) * xPos + cos(theta) * yPos, 0.0) +
                  rigidBodyCoord[n];
            if (pos[0] >= __domain[0][0] && pos[0] < __domain[1][0] &&
                pos[1] >= __domain[0][1] && pos[1] < __domain[1][1])
              InsertParticle(pos, 5, particleSize, normal, localIndex, 0, vol,
                             true, n, pCoord);
            yPos -= h;
          }
        }

        break;
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
  static auto &rigidBodyType = __rigidBody.index.GetHandle("type");

  int localIndex = coord.size();

  if (__dim == 3) {
    for (auto tag : splitTag) {
      splitList[tag].clear();
      double theta = pCoord[tag][0];
      double phi = pCoord[tag][1];

      const double thetaDelta = particleSize[tag][0] * 0.25 /
                                rigidBodySize[attachedRigidBodyIndex[tag]];
      const double phiDelta = particleSize[tag][1] * 0.25 /
                              rigidBodySize[attachedRigidBodyIndex[tag]] /
                              sin(theta);

      vec3 oldParticleSize = particleSize[tag];

      bool insert = false;
      for (int i = -1; i < 2; i += 2) {
        for (int j = -1; j < 2; j += 2) {
          double ratio = sin(theta + i * thetaDelta) / sin(theta);
          vec3 newParticleSize = vec3(oldParticleSize[0] / 2.0,
                                      oldParticleSize[1] / 2.0 * ratio, 0.0);
          vec3 newNormal =
              vec3(sin(theta + i * thetaDelta) * cos(phi + j * phiDelta),
                   sin(theta + i * thetaDelta) * sin(phi + j * phiDelta),
                   cos(theta + i * thetaDelta));
          vec3 newPos = newNormal * rigidBodySize[attachedRigidBodyIndex[tag]] +
                        rigidBodyCoord[attachedRigidBodyIndex[tag]];

          if (!insert) {
            coord[tag] = newPos;
            volume[tag] /= 8.0;
            normal[tag] = newNormal;
            particleSize[tag] = newParticleSize;
            pCoord[tag] = vec3(theta + i * thetaDelta, phi + j * phiDelta, 0.0);
            adaptive_level[tag] = __adaptive_step;

            splitList[tag].push_back(tag);

            insert = true;
          } else {
            double vol = volume[tag];
            int newParticle = InsertParticle(
                newPos, particleType[tag], newParticleSize, newNormal,
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
      switch (rigidBodyType[attachedRigidBodyIndex[tag]]) {
      case 1:
        // cicle
        {
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
              cos(theta) * sin(-thetaDelta) + sin(theta) * cos(-thetaDelta),
              0.0);
          newPos = newNormal * rigidBodySize[attachedRigidBodyIndex[tag]] +
                   rigidBodyCoord[attachedRigidBodyIndex[tag]];

          int return_val = InsertParticle(
              newPos, particleType[tag], particleSize[tag], newNormal,
              localIndex, __adaptive_step, volume[tag], true,
              attachedRigidBodyIndex[tag], vec3(theta - thetaDelta, 0.0, 0.0));

          if (return_val == 0)
            splitList[tag].push_back(localIndex - 1);
        }

        break;
      case 2:
        // square
        {
          splitList[tag].clear();
          if (particleType[tag] == 4) {
            // corner particle
            splitList[tag].push_back(tag);

            particleSize[tag] *= 0.5;
            volume[tag] /= 4.0;
            adaptive_level[tag] = __adaptive_step;
          } else {
            // side particle
            splitList[tag].push_back(tag);

            particleSize[tag] *= 0.5;
            volume[tag] /= 4.0;
            adaptive_level[tag] = __adaptive_step;

            vec3 oldPos = coord[tag];

            vec3 delta = vec3(-normal[tag][1], normal[tag][0], 0.0) * 0.5 *
                         particleSize[tag][0];
            coord[tag] = oldPos + delta;

            vec3 newPos = oldPos - delta;

            InsertParticle(newPos, particleType[tag], particleSize[tag],
                           normal[tag], localIndex, __adaptive_step,
                           volume[tag], true, attachedRigidBodyIndex[tag],
                           pCoord[tag]);

            splitList[tag].push_back(localIndex - 1);
          }
        }

        break;
      }
    }
  }
}

void GMLS_Solver::SplitGapRigidBodyParticle(vector<int> &splitTag) {
  static auto &coord = __field.vector.GetHandle("coord");

  int localIndex = coord.size();

  // gap particles on rigid body surface
  static auto &gapRigidBodyCoord =
      __gap.vector.GetHandle("rigid body surface coord");
  static auto &gapRigidBodyNormal =
      __gap.vector.GetHandle("rigid body surface normal");
  static auto &gapRigidBodySize =
      __gap.vector.GetHandle("rigid body surface size");
  static auto &gapRigidBodyPCoord =
      __gap.vector.GetHandle("rigid body surface parameter coordinate");
  static auto &gapRigidBodyVolume =
      __gap.scalar.GetHandle("rigid body surface volume");
  static auto &gapRigidBodyParticleType =
      __gap.index.GetHandle("rigid body surface particle type");
  static auto &gapRigidBodyAdaptiveLevel =
      __gap.index.GetHandle("rigid body surface adaptive level");
  static auto &gapRigidBodyAttachedRigidBodyIndex =
      __gap.index.GetHandle("rigid body surface attached rigid body index");

  static auto &rigidBodyCoord = __rigidBody.vector.GetHandle("position");
  static auto &rigidBodySize = __rigidBody.scalar.GetHandle("size");
  static auto &rigidBodyType = __rigidBody.index.GetHandle("type");

  auto oldGapRigidBodyCoord = move(gapRigidBodyCoord);
  auto oldGapRigidBodyNormal = move(gapRigidBodyNormal);
  auto oldGapRigidBodySize = move(gapRigidBodySize);
  auto oldGapRigidBodyPCoord = move(gapRigidBodyPCoord);
  auto oldGapRigidBodyVolume = move(gapRigidBodyVolume);
  auto oldGapRigidBodyParticleType = move(gapRigidBodyParticleType);
  auto oldGapRigidBodyAdaptiveLevel = move(gapRigidBodyAdaptiveLevel);
  auto oldGapRigidBodyAttachedRigidBodyIndex =
      move(gapRigidBodyAttachedRigidBodyIndex);

  if (__dim == 2) {
    for (int tag = 0; tag < splitTag.size(); tag++) {
      int attachedRigidBodyIndex = oldGapRigidBodyAttachedRigidBodyIndex[tag];
      switch (rigidBodyType[attachedRigidBodyIndex]) {
      case 1:
        // cicle
        {
          const double thetaDelta = oldGapRigidBodySize[tag][0] * 0.25 /
                                    rigidBodySize[attachedRigidBodyIndex];

          double theta = oldGapRigidBodyPCoord[tag][0];

          vec3 newNormal = vec3(
              cos(theta) * cos(thetaDelta) - sin(theta) * sin(thetaDelta),
              cos(theta) * sin(thetaDelta) + sin(theta) * cos(thetaDelta), 0.0);
          vec3 newPos = newNormal * rigidBodySize[attachedRigidBodyIndex] +
                        rigidBodyCoord[attachedRigidBodyIndex];
          vec3 newParticleSize = oldGapRigidBodySize[tag];
          newParticleSize[0] /= 2.0;

          InsertParticle(
              newPos, oldGapRigidBodyParticleType[tag], newParticleSize,
              newNormal, localIndex, oldGapRigidBodyAdaptiveLevel[tag] + 1,
              oldGapRigidBodyVolume[tag] / 4.0, true, attachedRigidBodyIndex,
              vec3(theta + thetaDelta, 0.0, 0.0));

          newNormal = vec3(
              cos(theta) * cos(-thetaDelta) - sin(theta) * sin(-thetaDelta),
              cos(theta) * sin(-thetaDelta) + sin(theta) * cos(-thetaDelta),
              0.0);
          newPos = newNormal * rigidBodySize[attachedRigidBodyIndex] +
                   rigidBodyCoord[attachedRigidBodyIndex];

          InsertParticle(
              newPos, oldGapRigidBodyParticleType[tag], newParticleSize,
              newNormal, localIndex, oldGapRigidBodyAdaptiveLevel[tag] + 1,
              oldGapRigidBodyVolume[tag] / 4.0, true, attachedRigidBodyIndex,
              vec3(theta - thetaDelta, 0.0, 0.0));
        }

        break;
      case 2:
        // square
        {}

        break;
      }
    }

    for (int tag = splitTag.size(); tag < oldGapRigidBodyCoord.size(); tag++) {
      InsertParticle(
          oldGapRigidBodyCoord[tag], oldGapRigidBodyParticleType[tag],
          oldGapRigidBodySize[tag], oldGapRigidBodyNormal[tag], localIndex,
          oldGapRigidBodyAdaptiveLevel[tag], oldGapRigidBodyVolume[tag], true,
          oldGapRigidBodyAttachedRigidBodyIndex[tag],
          oldGapRigidBodyPCoord[tag]);
    }
  }
}