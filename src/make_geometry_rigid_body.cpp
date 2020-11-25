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
        if (attachedRigidBodyIndex >= 0) {
          // this is a particle on the rigid body surface
        } else {
          // this is a fluid particle

          if (dis.mag() < rigidBodySize[i] - 1.5 * h) {
            return -1;
          }
          if (dis.mag() <= rigidBodySize[i] + 0.1 * h) {
            return i;
          }

          if (dis.mag() < rigidBodySize[i] + 1.5 * h) {
            double min_dis = __boundingBoxSize[0];
            for (int i = 0; i < __rigidBodySurfaceParticle.size(); i++) {
              vec3 rci = pos - __rigidBodySurfaceParticle[i];
              if (min_dis > rci.mag()) {
                min_dis = rci.mag();
              }
            }

            if (min_dis < 0.25 * h) {
              // this is a gap particle near the surface of the colloids
              return i;
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
          if (attachedRigidBodyIndex >= 0) {
            // this is a particle on the rigid body surface
          } else {
            if (abs(dis[0]) < half_side_length - 1.5 * h &&
                abs(dis[1]) < half_side_length - 1.5 * h) {
              return -1;
            }
            if (abs(dis[0]) < half_side_length + 0.1 * h &&
                abs(dis[1]) < half_side_length + 0.1 * h) {
              return i;
            }
            if (abs(dis[0]) < half_side_length + 1.0 * h &&
                abs(dis[1]) < half_side_length + 1.0 * h) {
              double min_dis = __boundingBoxSize[0];
              for (int i = 0; i < __rigidBodySurfaceParticle.size(); i++) {
                vec3 rci = pos - __rigidBodySurfaceParticle[i];
                if (min_dis > rci.mag()) {
                  min_dis = rci.mag();
                }
              }

              if (min_dis < 0.25 * h) {
                // this is a gap particle near the surface of the colloids
                return i;
              }
            }
          }
        }
        if (__dim == 3) {
        }
      }
      break;
    case 3:
      // triangle in 2d, tetrahedron in 3d
      if (__dim == 2) {
        double side_length = rigidBodySize[i];
        double height = (sqrt(3) / 2.0) * side_length;
        double theta = rigidBodyOrientation[i][0];

        vec3 translation = vec3(0.0, sqrt(3) / 6.0 * side_length, 0.0);
        vec3 abs_dis = pos - rigidBodyCoord[i];
        // rotate back
        vec3 dis =
            vec3(cos(theta) * abs_dis[0] + sin(theta) * abs_dis[1],
                 -sin(theta) * abs_dis[0] + cos(theta) * abs_dis[1], 0.0) +
            translation;

        bool possible_gap_particle = false;
        bool gap_particle = false;

        double enlarged_xlim_low = -0.5 * side_length - 0.5 * h;
        double enlarged_xlim_high = 0.5 * side_length + 0.5 * h;
        double enlarged_ylim_low = -0.5 * h;
        double enlarged_ylim_high = height + 0.5 * h;
        double exact_xlim_low = -0.5 * side_length;
        double exact_xlim_high = 0.5 * side_length;
        double exact_ylim_low = 0.0;
        double exact_ylim_high = height;
        if (attachedRigidBodyIndex >= 0) {
          // this is a particle on the rigid body surface
        } else {
          if ((dis[0] > enlarged_xlim_low && dis[0] < enlarged_xlim_high) &&
              (dis[1] > enlarged_ylim_low && dis[1] < enlarged_ylim_high)) {
            // this is a possible particle in the gap region of the triangle
            double dis_x = min(abs(dis[0] - exact_xlim_low),
                               abs(dis[0] - exact_xlim_high));
            double dis_y = sqrt(3) * dis_x;
            if (dis[1] < 0 && dis[1] > -0.1 * h) {
              gap_particle = true;
            } else if (dis[1] > 0) {
              if (dis[0] < exact_xlim_low || dis[0] > exact_xlim_high) {
                possible_gap_particle = true;
              } else if (dis[1] < dis_y + 0.1 * h) {
                gap_particle = true;
              } else if (dis[1] < dis_y + 0.5 * h) {
                possible_gap_particle = true;
              }
            }
            possible_gap_particle = true;
          }

          if (gap_particle) {
            return i;
          } else if (possible_gap_particle) {
            double min_dis = __boundingBoxSize[0];
            for (int i = 0; i < __rigidBodySurfaceParticle.size(); i++) {
              vec3 rci = pos - __rigidBodySurfaceParticle[i];
              if (min_dis > rci.mag()) {
                min_dis = rci.mag();
              }
            }

            if (min_dis < 0.25 * h) {
              // this is a gap particle near the surface of the colloids
              return i;
            }
          }
        }
      }
      if (__dim == 3) {
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

          double theta0 = M_PI * i / M_theta;
          double theta1 = M_PI * (i + 1) / M_theta;
          double dPhi = 2 * M_PI / M_phi;
          double area = pow(r, 2.0) * (cos(theta0) - cos(theta1)) * dPhi;

          vec3 particleSize = vec3(d_theta, area / d_theta, 0.0);
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

      case 3: {
        double theta = rigidBodyOrientation[n][0];
        double side_length = rigidBodySize[n];
        int side_step = side_length / __particleSize0[0];
        double h = side_length / side_step;
        double vol = pow(h, 2.0);
        vec3 particleSize = vec3(h, h, 0.0);
        vec3 increase_normal;
        vec3 start_point;
        vec3 normal;
        vec3 norm;
        vec3 pCoord = vec3(0.0, 0.0, 0.0);
        vec3 translation = vec3(0.0, -sqrt(3) / 6.0 * side_length, 0.0);
        // first side
        {
          vec3 pos = vec3(0.0, 0.5 * sqrt(3) * side_length, 0.0) + translation;
          // rotate
          vec3 newPos = vec3(cos(theta) * pos[0] - sin(theta) * pos[1],
                             sin(theta) * pos[0] + cos(theta) * pos[1], 0.0) +
                        rigidBodyCoord[n];

          norm = vec3(0.0, 1.0, 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
          if (newPos[0] >= __domain[0][0] && newPos[0] < __domain[1][0] &&
              newPos[1] >= __domain[0][1] && newPos[1] < __domain[1][1])
            InsertParticle(newPos, 4, particleSize, normal, localIndex, 0, vol,
                           true, n, pCoord);
        }

        increase_normal = vec3(cos(M_PI / 3), -sin(M_PI / 3), 0.0);
        start_point = vec3(0.0, sqrt(3) / 2.0 * side_length, 0.0);
        norm = vec3(cos(M_PI / 6.0), sin(M_PI / 6.0), 0.0);
        normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                      sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
        for (int i = 0; i < side_step; i++) {
          vec3 pos =
              start_point + increase_normal * ((i + 0.5) * h) + translation;
          // rotate
          vec3 newPos = vec3(cos(theta) * pos[0] - sin(theta) * pos[1],
                             sin(theta) * pos[0] + cos(theta) * pos[1], 0.0) +
                        rigidBodyCoord[n];

          if (newPos[0] >= __domain[0][0] && newPos[0] < __domain[1][0] &&
              newPos[1] >= __domain[0][1] && newPos[1] < __domain[1][1])
            InsertParticle(newPos, 5, particleSize, normal, localIndex, 0, vol,
                           true, n, pCoord);
        }

        // second side
        {
          vec3 pos = vec3(0.5 * side_length, 0.0, 0.0) + translation;
          // rotate
          vec3 newPos = vec3(cos(theta) * pos[0] - sin(theta) * pos[1],
                             sin(theta) * pos[0] + cos(theta) * pos[1], 0.0) +
                        rigidBodyCoord[n];

          norm = vec3(cos(M_PI / 6.0), -sin(M_PI / 6.0), 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);

          if (newPos[0] >= __domain[0][0] && newPos[0] < __domain[1][0] &&
              newPos[1] >= __domain[0][1] && newPos[1] < __domain[1][1])
            InsertParticle(newPos, 4, particleSize, normal, localIndex, 0, vol,
                           true, n, pCoord);
        }

        increase_normal = vec3(-1.0, 0.0, 0.0);
        start_point = vec3(0.5 * side_length, 0.0, 0.0);
        norm = vec3(0.0, -1.0, 0.0);
        normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                      sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
        for (int i = 0; i < side_step; i++) {
          vec3 pos =
              start_point + increase_normal * ((i + 0.5) * h) + translation;
          // rotate
          vec3 newPos = vec3(cos(theta) * pos[0] - sin(theta) * pos[1],
                             sin(theta) * pos[0] + cos(theta) * pos[1], 0.0) +
                        rigidBodyCoord[n];

          if (newPos[0] >= __domain[0][0] && newPos[0] < __domain[1][0] &&
              newPos[1] >= __domain[0][1] && newPos[1] < __domain[1][1])
            InsertParticle(newPos, 5, particleSize, normal, localIndex, 0, vol,
                           true, n, pCoord);
        }

        // third side
        {
          vec3 pos = vec3(-0.5 * side_length, 0.0, 0.0) + translation;
          // rotate
          vec3 newPos = vec3(cos(theta) * pos[0] - sin(theta) * pos[1],
                             sin(theta) * pos[0] + cos(theta) * pos[1], 0.0) +
                        rigidBodyCoord[n];

          norm = vec3(-cos(M_PI / 6.0), -sin(M_PI / 6.0), 0.0);
          normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                        sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);

          if (newPos[0] >= __domain[0][0] && newPos[0] < __domain[1][0] &&
              newPos[1] >= __domain[0][1] && newPos[1] < __domain[1][1])
            InsertParticle(newPos, 4, particleSize, normal, localIndex, 0, vol,
                           true, n, pCoord);
        }

        increase_normal = vec3(cos(M_PI / 3), sin(M_PI / 3), 0.0);
        start_point = vec3(-0.5 * side_length, 0.0, 0.0);
        norm = vec3(-cos(M_PI / 6.0), sin(M_PI / 6.0), 0.0);
        normal = vec3(cos(theta) * norm[0] - sin(theta) * norm[1],
                      sin(theta) * norm[0] + cos(theta) * norm[1], 0.0);
        for (int i = 0; i < side_step; i++) {
          vec3 pos =
              start_point + increase_normal * ((i + 0.5) * h) + translation;
          // rotate
          vec3 newPos = vec3(cos(theta) * pos[0] - sin(theta) * pos[1],
                             sin(theta) * pos[0] + cos(theta) * pos[1], 0.0) +
                        rigidBodyCoord[n];

          if (newPos[0] >= __domain[0][0] && newPos[0] < __domain[1][0] &&
              newPos[1] >= __domain[0][1] && newPos[1] < __domain[1][1])
            InsertParticle(newPos, 5, particleSize, normal, localIndex, 0, vol,
                           true, n, pCoord);
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
  static auto &newAdded = __field.index.GetHandle("new added particle flag");

  static auto &rigidBodyCoord = __rigidBody.vector.GetHandle("position");
  static auto &rigidBodySize = __rigidBody.scalar.GetHandle("size");
  static auto &rigidBodyType = __rigidBody.index.GetHandle("type");

  int localIndex = coord.size();

  if (__dim == 3) {
    for (auto tag : splitTag) {
      splitList[tag].clear();
      double theta = pCoord[tag][0];
      double phi = pCoord[tag][1];
      double r = rigidBodySize[attachedRigidBodyIndex[tag]];

      const double oldDeltaTheta = particleSize[tag][0] / r;
      const double oldArea = particleSize[tag][0] * particleSize[tag][1];

      const double thetaDelta = 0.5 * oldDeltaTheta;

      vec3 oldParticleSize = particleSize[tag];

      double d_theta = oldParticleSize[0] * 0.5;
      double d_phi = d_theta;

      bool insert = false;
      for (int i = -1; i < 2; i += 2) {
        double newTheta = theta + i * thetaDelta * 0.5;
        int M_phi = round(2 * M_PI * r * sin(newTheta) / d_phi);

        const int old_M_phi =
            round(2 * M_PI * r * sin(theta) / oldParticleSize[1]);
        const double oldDeltaPhi = 2 * M_PI / old_M_phi;
        const double oldPhi0 = phi - 0.5 * oldDeltaPhi;
        const double oldPhi1 = phi + 0.5 * oldDeltaPhi;
        for (int j = 0; j < M_phi; j++) {
          double newPhi = 2 * M_PI * (j + 0.5) / M_phi;
          if (newPhi >= oldPhi0 && newPhi <= oldPhi1) {
            double theta0 = newTheta - 0.5 * thetaDelta;
            double theta1 = newTheta + 0.5 * thetaDelta;

            double dPhi = 2 * M_PI / M_phi;
            double area = pow(r, 2.0) * (cos(theta0) - cos(theta1)) * dPhi;

            vec3 newParticleSize = vec3(d_theta, area / d_theta, 0.0);
            vec3 newNormal = vec3(sin(newTheta) * cos(newPhi),
                                  sin(newTheta) * sin(newPhi), cos(newTheta));
            vec3 newPos =
                newNormal * r + rigidBodyCoord[attachedRigidBodyIndex[tag]];

            if (!insert) {
              coord[tag] = newPos;
              volume[tag] /= 8.0;
              normal[tag] = newNormal;
              particleSize[tag] = newParticleSize;
              pCoord[tag] = vec3(newTheta, newPhi, 0.0);
              adaptive_level[tag]++;
              newAdded[tag] = 1;

              splitList[tag].push_back(tag);

              insert = true;
            } else {
              double vol = volume[tag];
              int newParticle = InsertParticle(
                  newPos, particleType[tag], newParticleSize, newNormal,
                  localIndex, adaptive_level[tag], vol, true,
                  attachedRigidBodyIndex[tag], vec3(newTheta, newPhi, 0.0));

              if (newParticle == 0) {
                splitList[tag].push_back(localIndex - 1);
              }
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
          adaptive_level[tag]++;
          newAdded[tag] = 1;

          splitList[tag].push_back(tag);

          newNormal = vec3(
              cos(theta) * cos(-thetaDelta) - sin(theta) * sin(-thetaDelta),
              cos(theta) * sin(-thetaDelta) + sin(theta) * cos(-thetaDelta),
              0.0);
          newPos = newNormal * rigidBodySize[attachedRigidBodyIndex[tag]] +
                   rigidBodyCoord[attachedRigidBodyIndex[tag]];

          int return_val = InsertParticle(
              newPos, particleType[tag], particleSize[tag], newNormal,
              localIndex, adaptive_level[tag], volume[tag], true,
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
            adaptive_level[tag]++;
          } else {
            // side particle
            splitList[tag].push_back(tag);

            particleSize[tag] *= 0.5;
            volume[tag] /= 4.0;
            adaptive_level[tag]++;
            newAdded[tag] = 1;

            vec3 oldPos = coord[tag];

            vec3 delta = vec3(-normal[tag][1], normal[tag][0], 0.0) * 0.5 *
                         particleSize[tag][0];
            coord[tag] = oldPos + delta;

            vec3 newPos = oldPos - delta;

            InsertParticle(newPos, particleType[tag], particleSize[tag],
                           normal[tag], localIndex, adaptive_level[tag],
                           volume[tag], true, attachedRigidBodyIndex[tag],
                           pCoord[tag]);

            splitList[tag].push_back(localIndex - 1);
          }
        }

        break;

      case 3: {
        splitList[tag].clear();
        if (particleType[tag] == 4) {
          // corner particle
          splitList[tag].push_back(tag);

          particleSize[tag] *= 0.5;
          volume[tag] /= 4.0;
          adaptive_level[tag]++;
        } else {
          // side particle
          splitList[tag].push_back(tag);

          particleSize[tag] *= 0.5;
          volume[tag] /= 4.0;
          adaptive_level[tag]++;
          newAdded[tag] = 1;

          vec3 oldPos = coord[tag];

          vec3 delta = vec3(-normal[tag][1], normal[tag][0], 0.0) * 0.5 *
                       particleSize[tag][0];
          coord[tag] = oldPos + delta;

          vec3 newPos = oldPos - delta;

          InsertParticle(newPos, particleType[tag], particleSize[tag],
                         normal[tag], localIndex, adaptive_level[tag],
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

void GMLS_Solver::UpdateRigidBodySurfaceParticlePointCloudSearch() {
  static auto &particleType = __field.index.GetHandle("particle type");
  static auto &particleSize = __field.vector.GetHandle("size");
  static vector<vec3> &backgroundSourceCoord =
      __background.vector.GetHandle("source coord");

  vector<int> recvParticleType;
  DataSwapAmongNeighbor(particleType, recvParticleType);
  vector<int> backgroundParticleType;

  backgroundParticleType.insert(backgroundParticleType.end(),
                                particleType.begin(), particleType.end());

  backgroundParticleType.insert(backgroundParticleType.end(),
                                recvParticleType.begin(),
                                recvParticleType.end());

  vector<vec3> recvParticleSize;
  DataSwapAmongNeighbor(particleSize, recvParticleSize);
  vector<vec3> backgroundParticleSize;

  backgroundParticleSize.insert(backgroundParticleSize.end(),
                                particleSize.begin(), particleSize.end());

  backgroundParticleSize.insert(backgroundParticleSize.end(),
                                recvParticleSize.begin(),
                                recvParticleSize.end());

  __rigidBodySurfaceParticle.clear();
  __rigidBodySurfaceParticleSize.clear();

  for (int i = 0; i < backgroundParticleType.size(); i++) {
    if (backgroundParticleType[i] >= 4) {
      __rigidBodySurfaceParticle.push_back(backgroundSourceCoord[i]);
      __rigidBodySurfaceParticleSize.push_back(backgroundParticleSize[i]);
    }
  }
}

bool GMLS_Solver::IsAcceptableRigidBodyPosition() {
  vector<vec3> &rigidBodyPosition = __rigidBody.vector.Register("position");
  vector<double> &rigidBodySize = __rigidBody.scalar.Register("size");

  for (int i = 0; i < rigidBodyPosition.size(); i++) {
    for (int j = i + 1; j < rigidBodyPosition.size(); j++) {
      vec3 dis = rigidBodyPosition[i] - rigidBodyPosition[j];
      if (dis.mag() < rigidBodySize[i] + rigidBodySize[j])
        return false;
    }
  }

  return true;
}