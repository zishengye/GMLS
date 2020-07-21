#include "gmls_solver.h"
#include "manifold.h"

#include <iostream>

using namespace std;

#define PI 3.1415926

void GMLS_Solver::SetBoundingBoxManifold() {
  __coordinateSystem = 2;

  __boundingBoxSize[0] = 2 * PI;
  __boundingBoxSize[1] = PI;
  __boundingBoxSize[2] = 0.0;

  if (__coordinateSystem == 1) {
    __boundingBox.push_back(
        vec3(-__boundingBoxSize[0] / 2.0, -__boundingBoxSize[1] / 2.0, 0.0));
    __boundingBox.push_back(
        vec3(__boundingBoxSize[0] / 2.0, __boundingBoxSize[1] / 2.0, 0.0));
  } else {
    __boundingBox.push_back(vec3(0.0, 0.0, 0.0));
    __boundingBox.push_back(
        vec3(__boundingBoxSize[0], __boundingBoxSize[1], 0.0));
  }
}

void GMLS_Solver::SetBoundingBoxBoundaryManifold() {
  __boundingBoxBoundaryType.resize(4);

  for (int i = 0; i < 4; i++) {
    __boundingBoxBoundaryType[i] = 0;
  }
}

void GMLS_Solver::SetDomainBoundaryManifold() {
  // four edges as boundary
  // 0 down edge
  // 1 right edge
  // 2 up edge
  // 3 left edge
  __domainBoundaryType.resize(4);
  if (abs(__domain[0][0] - __boundingBox[0][0]) < 1e-6) {
    __domainBoundaryType[3] = __boundingBoxBoundaryType[3];
  } else {
    __domainBoundaryType[3] = 0;
  }

  if (abs(__domain[0][1] - __boundingBox[0][1]) < 1e-6) {
    __domainBoundaryType[0] = __boundingBoxBoundaryType[0];
  } else {
    __domainBoundaryType[0] = 0;
  }

  if (abs(__domain[1][0] - __boundingBox[1][0]) < 1e-6) {
    __domainBoundaryType[1] = __boundingBoxBoundaryType[1];
  } else {
    __domainBoundaryType[1] = 0;
  }

  if (abs(__domain[1][1] - __boundingBox[1][1]) < 1e-6) {
    __domainBoundaryType[2] = __boundingBoxBoundaryType[2];
  } else {
    __domainBoundaryType[2] = 0;
  }
}

void GMLS_Solver::InitDomainDecompositionManifold() {
  ProcessSplit(__nX, __nY, __nI, __nJ, __MPISize, __myID);
  // BoundingBoxSplit(__boundingBoxSize, __boundingBoxCount, __boundingBox,
  //                  __particleSize0, __domainBoundingBox, __domainCount,
  //                  __domain, __nX, __nY, __nI, __nJ, 0.5);

  SetDomainBoundaryManifold();

  InitNeighborListManifold();
}

void GMLS_Solver::InitUniformParticleManifoldField() {
  static vector<vec3> &coord = __field.vector.GetHandle("coord");
  static vector<int> &globalIndex = __field.index.GetHandle("global index");
  static vector<int> &particleNum = __field.index.GetHandle("particle number");
  ClearParticle();

  InitFieldParticleManifold();
  InitFieldBoundaryParticleManifold();

  SerialOperation([this]() {
    cout << "[Proc " << this->__myID << "]: generated " << coord.size()
         << " particles." << endl;
  });

  particleNum.resize(2 + __MPISize);

  int &localParticleNum = particleNum[0];
  int &globalParticleNum = particleNum[1];
  localParticleNum = coord.size();
  MPI_Allreduce(&localParticleNum, &globalParticleNum, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  vector<int> particleOffset;
  particleNum.resize(__MPISize);
  particleOffset.resize(__MPISize + 1);
  MPI_Allgather(&localParticleNum, 1, MPI_INT, particleNum.data() + 2, 1,
                MPI_INT, MPI_COMM_WORLD);
  particleOffset[0] = 0;
  for (int i = 0; i < __MPISize; i++) {
    particleOffset[i + 1] = particleOffset[i] + particleNum[i + 2];
  }

  for (size_t i = 0; i < globalIndex.size(); i++) {
    globalIndex[i] += particleOffset[__myID];
  }
}

void GMLS_Solver::InitFieldParticleManifold() {
  __cutoffDistance = 4.5 * std::max(__particleSize0[0], __particleSize0[1]);

  // double xPos, yPos;
  // vec3 normal;

  // double vol = __particleSize0[0] * __particleSize0[1];
  // int localIndex = __particle.X.size();
  // // fluid particle
  // yPos = __domain[0][1] + __particleSize0[1] / 2.0;
  // for (int j = 0; j < __domainCount[1]; j++) {
  //   xPos = __domain[0][0] + __particleSize0[0] / 2.0;
  //   for (int i = 0; i < __domainCount[0]; i++) {
  //     vec3 pos = Manifold(xPos, yPos);
  //     normal = ManifoldNorm(xPos, yPos);
  //     InsertParticle(pos, 0, __particleSize0, normal, localIndex, vol);
  //     __particle.X_origin.push_back(vec3(xPos, yPos, 0.0));
  //     xPos += __particleSize0[0];
  //   }
  //   yPos += __particleSize0[1];
  // }

  double xPos, yPos, zPos;
  int localIndex = 0;

  double dz = 2.0 / __boundingBoxCount[1];
  double dTheta = 2.0 * PI / __boundingBoxCount[0];
  int M_theta = std::round(2.0 * PI / dTheta);
  for (int i = 0; i < M_theta; ++i) {
    double theta = 2 * PI * (i + 0.5) / M_theta;
    for (int j = 0; j < __boundingBoxCount[1]; ++j) {
      xPos = std::cos(theta);
      yPos = std::sin(theta);
      zPos = dz * (j + 0.5);
      vec3 normal = vec3(yPos, -xPos, 0.0);
      vec3 pos = vec3(xPos, yPos, zPos);
      InsertParticle(pos, 0, __particleSize0, normal, localIndex, 0, 1.0);
    }
  }
}

void GMLS_Solver::InitFieldBoundaryParticleManifold() {
  static vector<vec3> &coord = __field.vector.GetHandle("coord");
  static vector<vec3> &chartCoord = __field.vector.GetHandle("chart coord");

  double xPos, yPos;
  vec3 normal;
  double vol = __particleSize0[0] * __particleSize0[1];
  int localIndex = coord.size();
  // down
  if (__domainBoundaryType[0] != 0) {
    xPos = __domain[0][0];
    yPos = __domain[0][1];
    if (__domainBoundaryType[3] != 0) {
      vec3 pos = Manifold(xPos, yPos);
      normal = ManifoldNorm(xPos, yPos);
      InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
      chartCoord.push_back(vec3(xPos, yPos, 0.0));
      xPos += __particleSize0[0];
    }

    while (xPos < __domain[1][0] - 1e-5) {
      vec3 pos = Manifold(xPos, yPos);
      normal = ManifoldNorm(xPos, yPos);
      InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
      chartCoord.push_back(vec3(xPos, yPos, 0.0));
      xPos += __particleSize0[0];
    }
  }

  // right
  if (__domainBoundaryType[1] != 0) {
    xPos = __domain[1][0];
    yPos = __domain[0][1];
    if (__domainBoundaryType[0] != 0) {
      vec3 pos = Manifold(xPos, yPos);
      normal = ManifoldNorm(xPos, yPos);
      InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
      chartCoord.push_back(vec3(xPos, yPos, 0.0));
      yPos += __particleSize0[1];
    }

    while (yPos < __domain[1][1] - 1e-5) {
      vec3 pos = Manifold(xPos, yPos);
      normal = ManifoldNorm(xPos, yPos);
      InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
      chartCoord.push_back(vec3(xPos, yPos, 0.0));
      yPos += __particleSize0[1];
    }
  }

  // up
  if (__domainBoundaryType[2] != 0) {
    xPos = __domain[1][0];
    yPos = __domain[1][1];
    if (__domainBoundaryType[1] != 0) {
      vec3 pos = Manifold(xPos, yPos);
      normal = ManifoldNorm(xPos, yPos);
      InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
      chartCoord.push_back(vec3(xPos, yPos, 0.0));
      xPos -= __particleSize0[0];
    }

    while (xPos > __domain[0][0] + 1e-5) {
      vec3 pos = Manifold(xPos, yPos);
      normal = ManifoldNorm(xPos, yPos);
      InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
      chartCoord.push_back(vec3(xPos, yPos, 0.0));
      xPos -= __particleSize0[0];
    }
  }

  // left
  if (__domainBoundaryType[3] != 0) {
    xPos = __domain[0][0];
    yPos = __domain[1][1];
    if (__domainBoundaryType[2] != 0) {
      vec3 pos = Manifold(xPos, yPos);
      normal = ManifoldNorm(xPos, yPos);
      InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
      chartCoord.push_back(vec3(xPos, yPos, 0.0));
      yPos -= __particleSize0[1];
    }

    while (yPos > __domain[0][1] + 1e-5) {
      vec3 pos = Manifold(xPos, yPos);
      normal = ManifoldNorm(xPos, yPos);
      InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
      chartCoord.push_back(vec3(xPos, yPos, 0.0));
      yPos -= __particleSize0[1];
    }
  }
}