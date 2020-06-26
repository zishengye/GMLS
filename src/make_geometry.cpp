#include "domain_decomposition.h"
#include "gmls_solver.h"

#include <iostream>

using namespace std;
using namespace Compadre;

void GMLS_Solver::SetBoundingBox() {
  if (__dim == 3) {
    __boundingBoxSize[0] = 2.0;
    __boundingBoxSize[1] = 2.0;
    __boundingBoxSize[2] = 2.0;

    __boundingBox.push_back(vec3(-__boundingBoxSize[0] / 2.0,
                                 -__boundingBoxSize[1] / 2.0,
                                 -__boundingBoxSize[2] / 2.0));
    __boundingBox.push_back(vec3(__boundingBoxSize[0] / 2.0,
                                 __boundingBoxSize[1] / 2.0,
                                 __boundingBoxSize[2] / 2.0));
  } else if (__dim == 2) {
    __boundingBoxSize[0] = 2.0;
    __boundingBoxSize[1] = 2.0;
    __boundingBoxSize[2] = 0.0;

    __boundingBox.push_back(
        vec3(-__boundingBoxSize[0] / 2.0, -__boundingBoxSize[1] / 2.0, 0.0));
    __boundingBox.push_back(
        vec3(__boundingBoxSize[0] / 2.0, __boundingBoxSize[1] / 2.0, 0.0));
  }
}

void GMLS_Solver::SetBoundingBoxBoundary() {
  if (__dim == 2) {
    __boundingBoxBoundaryType.resize(4);

    for (int i = 0; i < 4; i++) {
      __boundingBoxBoundaryType[i] = 1;
    }
  } else if (__dim == 3) {
    __boundingBoxBoundaryType.resize(12);

    for (int i = 0; i < 12; i++) {
      __boundingBoxBoundaryType[i] = 1;
    }
  }
}

void GMLS_Solver::SetDomainBoundary() {
  if (__dim == 3) {
    // six faces as boundary
    // 0 front face
    // 1 right face
    // 2 back face
    // 3 bottom face
    // 4 left face
    // 5 top face
    __domainBoundaryType.resize(6);
    if (abs(__domain[1][0] - __boundingBox[1][0]) < 1e-6) {
      __domainBoundaryType[0] = __boundingBoxBoundaryType[0];
    } else {
      __domainBoundaryType[0] = 0;
    }

    if (abs(__domain[1][1] - __boundingBox[1][1]) < 1e-6) {
      __domainBoundaryType[1] = __boundingBoxBoundaryType[1];
    } else {
      __domainBoundaryType[1] = 0;
    }

    if (abs(__domain[0][0] - __boundingBox[0][0]) < 1e-6) {
      __domainBoundaryType[2] = __boundingBoxBoundaryType[2];
    } else {
      __domainBoundaryType[2] = 0;
    }

    if (abs(__domain[0][2] - __boundingBox[0][2]) < 1e-6) {
      __domainBoundaryType[3] = __boundingBoxBoundaryType[3];
    } else {
      __domainBoundaryType[3] = 0;
    }

    if (abs(__domain[0][1] - __boundingBox[0][1]) < 1e-6) {
      __domainBoundaryType[4] = __boundingBoxBoundaryType[4];
    } else {
      __domainBoundaryType[4] = 0;
    }

    if (abs(__domain[1][2] - __boundingBox[1][2]) < 1e-6) {
      __domainBoundaryType[5] = __boundingBoxBoundaryType[5];
    } else {
      __domainBoundaryType[5] = 0;
    }
  }
  if (__dim == 2) {
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
}

void GMLS_Solver::InitDomainDecomposition() {
  if (__dim == 3) {
    ProcessSplit(__nX, __nY, __nZ, __nI, __nJ, __nK, __MPISize, __myID);
  } else if (__dim == 2) {
    ProcessSplit(__nX, __nY, __nI, __nJ, __MPISize, __myID);
  }

  InitNeighborList();
}

void GMLS_Solver::FinalizeDomainDecomposition() {
  MPI_Win_free(&__neighborWinCount);
  MPI_Win_free(&__neighborWinIndex);
  MPI_Win_free(&__neighborWinOffset);
}

void GMLS_Solver::InitParticle() {
  __background.vector.Register("coord");
  __background.vector.Register("source coord");
  __background.index.Register("index");
  __background.index.Register("source index");

  __field.vector.Register("coord");
  __field.vector.Register("normal");
  __field.vector.Register("size");
  __field.vector.Register("parameter coordinate");
  __field.scalar.Register("volume");
  __field.scalar.Register("error");
  __field.index.Register("particle type");
  __field.index.Register("global index");
  __field.index.Register("adaptive level");
  __field.index.Register("attached rigid body index");
  __field.index.Register("particle number");

  __gap.vector.Register("coord");
  __gap.vector.Register("normal");
  __gap.vector.Register("size");
  __gap.vector.Register("volume");
  __gap.index.Register("particle type");
  __gap.index.Register("adaptive level");
}

void GMLS_Solver::ClearParticle() {
  static auto &backgroundCoord = __background.vector.GetHandle("coord");
  static auto &sourceCoord = __background.vector.GetHandle("source coord");
  static auto &backgroundIndex = __background.index.GetHandle("index");
  static auto &sourceIndex = __background.index.GetHandle("source index");

  static auto &coord = __field.vector.GetHandle("coord");
  static auto &normal = __field.vector.GetHandle("normal");
  static auto &size = __field.vector.GetHandle("size");
  static auto &pCoord = __field.vector.GetHandle("parameter coordinate");
  static auto &volume = __field.scalar.GetHandle("volume");
  static auto &particleType = __field.index.GetHandle("particle type");
  static auto &globalIndex = __field.index.GetHandle("global index");
  static auto &adaptive_level = __field.index.GetHandle("adaptive level");
  static auto &attachedRigidBodyIndex =
      __field.index.GetHandle("attached rigid body index");
  static auto &particleNum = __field.index.GetHandle("particle number");

  static auto &gapCoord = __gap.vector.GetHandle("coord");
  static auto &gapNormal = __gap.vector.GetHandle("normal");
  static auto &gapParticleSize = __gap.vector.GetHandle("size");
  static auto &gapParticleType = __gap.index.GetHandle("particle type");

  backgroundCoord.clear();
  sourceCoord.clear();
  backgroundIndex.clear();
  sourceIndex.clear();

  coord.clear();
  normal.clear();
  size.clear();
  pCoord.clear();
  volume.clear();
  particleType.clear();
  globalIndex.clear();
  adaptive_level.clear();
  attachedRigidBodyIndex.clear();
  particleNum.clear();

  gapCoord.clear();
  gapNormal.clear();
  gapParticleSize.clear();
  gapParticleType.clear();
}

void GMLS_Solver::InitUniformParticleField() {
  static vector<vec3> &coord = __field.vector.GetHandle("coord");

  // adaptively adjust uniform particle distribution
  static vector<vec3> &rigidBodyPosition =
      __rigidBody.vector.GetHandle("position");
  static vector<double> &rigidBodySize = __rigidBody.scalar.GetHandle("size");

  // ensure enough particles between boundaries
  const int rigid_body_num = rigidBodyPosition.size();

  double minDistance = 2.0;
  // check between rigid body boundaries
  for (int i = 0; i < rigid_body_num - 1; i++) {
    for (int j = i + 1; j < rigid_body_num; j++) {
      auto dis = rigidBodyPosition[i] - rigidBodyPosition[j];
      if ((dis.mag() - rigidBodySize[i] - rigidBodySize[j]) < minDistance) {
        minDistance = (dis.mag() - rigidBodySize[i] - rigidBodySize[j]);
      }
    }
  }

  // check between rigid body and bounding box boundaries
  for (int i = 0; i < rigid_body_num; i++) {
    for (int j = 0; j < __dim; j++) {
      if (abs(rigidBodyPosition[i][j] - __boundingBox[0][j]) -
              rigidBodySize[i] <
          minDistance) {
        minDistance = abs(rigidBodyPosition[i][j] - __boundingBox[0][j]) -
                      rigidBodySize[i];
      }
      if (abs(__boundingBox[1][j] - rigidBodyPosition[i][j]) -
              rigidBodySize[i] <
          minDistance) {
        minDistance = abs(__boundingBox[1][j] - rigidBodyPosition[i][j]) -
                      rigidBodySize[i];
      }
    }
  }

  int addedLevel;
  if (__dim == 3) {
    BoundingBoxSplit(__boundingBoxSize, __boundingBoxCount, __boundingBox,
                     __particleSize0, __domainBoundingBox, __domainCount,
                     __domain, __nX, __nY, __nZ, __nI, __nJ, __nK,
                     0.5 * minDistance);
  } else if (__dim == 2) {
    addedLevel = BoundingBoxSplit(
        __boundingBoxSize, __boundingBoxCount, __boundingBox, __particleSize0,
        __domainBoundingBox, __domainCount, __domain, __nX, __nY, __nI, __nJ,
        0.5 * minDistance, __maxAdaptiveLevel);
  }

  __maxAdaptiveLevel = 4 - addedLevel;

  SetDomainBoundary();

  ClearParticle();

  InitFieldParticle();
  // InitFieldBoundaryParticle();
  InitRigidBodySurfaceParticle();

  ParticleIndex();
}

void GMLS_Solver::ParticleIndex() {
  static auto &coord = __field.vector.GetHandle("coord");
  static auto &normal = __field.vector.GetHandle("normal");
  static auto &particleSize = __field.vector.GetHandle("size");
  static auto &pCoord = __field.vector.GetHandle("parameter coordinate");
  static auto &volume = __field.scalar.GetHandle("volume");
  static auto &globalIndex = __field.index.GetHandle("global index");
  static auto &particleType = __field.index.GetHandle("particle type");
  static vector<int> &particleNum = __field.index.GetHandle("particle number");
  static auto &attachedRigidBodyIndex =
      __field.index.GetHandle("attached rigid body index");

  particleNum.resize(2 + __MPISize);

  int &localParticleNum = particleNum[0];
  int &globalParticleNum = particleNum[1];
  localParticleNum = coord.size();
  MPI_Allreduce(&localParticleNum, &globalParticleNum, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  vector<int> particleOffset;
  particleOffset.resize(__MPISize + 1);
  MPI_Allgather(&localParticleNum, 1, MPI_INT, particleNum.data() + 2, 1,
                MPI_INT, MPI_COMM_WORLD);
  particleOffset[0] = 0;
  for (int i = 0; i < __MPISize; i++) {
    particleOffset[i + 1] = particleOffset[i] + particleNum[i + 2];
  }

  for (size_t i = 0; i < globalIndex.size(); i++) {
    globalIndex[i] = i + particleOffset[__myID];
  }

  // SerialOperation([this]() {
  //   cout << "[Proc " << this->__myID << "]: generated " << coord.size()
  //        << " particles." << endl;
  // });

  PetscPrintf(PETSC_COMM_WORLD, "global particle number: %d\n",
              globalParticleNum);

  // check data consistence
  if (coord.size() != normal.size()) {
    cout << __myID << " normal size inconsistent!" << endl;
  }
  if (coord.size() != particleSize.size()) {
    cout << __myID << " particleSize size inconsistent!" << endl;
  }
  if (coord.size() != pCoord.size()) {
    cout << __myID << " pCoord size inconsistent!" << endl;
  }
  if (coord.size() != volume.size()) {
    cout << __myID << " volume size inconsistent!" << endl;
  }
  if (coord.size() != globalIndex.size()) {
    cout << __myID << " globalIndex size inconsistent!" << endl;
  }
  if (coord.size() != particleType.size()) {
    cout << __myID << " particleType size inconsistent!" << endl;
  }
  if (coord.size() != attachedRigidBodyIndex.size()) {
    cout << __myID << " attachedRigidBodyIndex size inconsistent!" << endl;
  }
}

bool GMLS_Solver::IsInGap(vec3 &xScalar) { return false; }

void GMLS_Solver::InitFieldParticle() {
  __cutoffDistance = (__polynomialOrder + 1.0) *
                         std::max(__particleSize0[0], __particleSize0[1]) +
                     1e-5;

  double xPos, yPos, zPos;
  vec3 normal = vec3(1.0, 0.0, 0.0);
  vec3 boundary_normal;

  if (__dim == 2) {
    zPos = 0.0;
    double vol = __particleSize0[0] * __particleSize0[1];
    int localIndex = 0;

    // down
    if (__domainBoundaryType[0] != 0) {
      xPos = __domain[0][0];
      yPos = __domain[0][1];
      if (__domainBoundaryType[3] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        boundary_normal = vec3(sqrt(2) / 2.0, sqrt(2) / 2.0, 0.0);
        InsertParticle(pos, 1, __particleSize0, boundary_normal, localIndex, 0,
                       vol);
      }
      xPos += 0.5 * __particleSize0[0];

      while (xPos < __domain[1][0] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        boundary_normal = vec3(0.0, 1.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, boundary_normal, localIndex, 0,
                       vol);
        xPos += __particleSize0[0];
      }

      if (__domainBoundaryType[1] != 0) {
        xPos = __domain[1][0];
        vec3 pos = vec3(xPos, yPos, zPos);
        boundary_normal = vec3(-sqrt(2) / 2.0, sqrt(2) / 2.0, 0.0);
        InsertParticle(pos, 1, __particleSize0, boundary_normal, localIndex, 0,
                       vol);
      }
    }

    // fluid particle
    yPos = __domain[0][1] + __particleSize0[1] / 2.0;
    while (yPos < __domain[1][1] - 1e-5) {
      // left
      if (__domainBoundaryType[3] != 0) {
        xPos = __domain[0][0];
        vec3 pos = vec3(xPos, yPos, zPos);
        boundary_normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, boundary_normal, localIndex, 0,
                       vol);
      }

      xPos = __domain[0][0] + __particleSize0[0] / 2.0;
      while (xPos < __domain[1][0] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        InsertParticle(pos, 0, __particleSize0, normal, localIndex, 0, vol);
        xPos += __particleSize0[0];
      }

      // right
      if (__domainBoundaryType[1] != 0) {
        xPos = __domain[1][0];
        vec3 pos = vec3(xPos, yPos, zPos);
        boundary_normal = vec3(-1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, boundary_normal, localIndex, 0,
                       vol);
      }

      yPos += __particleSize0[1];
    }

    // up
    if (__domainBoundaryType[2] != 0) {
      xPos = __domain[0][0];
      yPos = __domain[1][1];
      if (__domainBoundaryType[3] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        boundary_normal = vec3(sqrt(2) / 2.0, -sqrt(2) / 2.0, 0.0);
        InsertParticle(pos, 1, __particleSize0, boundary_normal, localIndex, 0,
                       vol);
      }
      xPos += 0.5 * __particleSize0[0];

      while (xPos < __domain[1][0] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        boundary_normal = vec3(0.0, -1.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, boundary_normal, localIndex, 0,
                       vol);
        xPos += __particleSize0[0];
      }

      xPos = __domain[1][0];
      if (__domainBoundaryType[1] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        boundary_normal = vec3(-sqrt(2) / 2.0, -sqrt(2) / 2.0, 0.0);
        InsertParticle(pos, 1, __particleSize0, boundary_normal, localIndex, 0,
                       vol);
      }
    }
  }
  if (__dim == 3) {
    double vol = __particleSize0[0] * __particleSize0[1] * __particleSize0[2];
    int localIndex = 0;

    // x-y, z=-z0 face
    if (__domainBoundaryType[3] != 0) {
      zPos = __domain[0][2];

      xPos = __domain[0][0];
      yPos = __domain[0][1];
      if (__domainBoundaryType[2] != 0 && __domainBoundaryType[4] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(sqrt(3) / 3.0, sqrt(3) / 3.0, sqrt(3) / 3.0);
        InsertParticle(pos, 1, __particleSize0, normal, localIndex, 0, vol);
      }

      yPos += 0.5 * __particleSize0[1];
      if (__domainBoundaryType[2] != 0) {
        while (yPos < __domain[1][1] - 1e-5) {
          vec3 pos = vec3(xPos, yPos, zPos);
          normal = vec3(sqrt(2.0) / 2.0, 0.0, sqrt(2.0) / 2.0);
          InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
          yPos += __particleSize0[1];
        }
      }

      yPos = __domain[1][1];
      if (__domainBoundaryType[1] != 0 && __domainBoundaryType[2] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(sqrt(3) / 3.0, -sqrt(3) / 3.0, sqrt(3) / 3.0);
        InsertParticle(pos, 1, __particleSize0, normal, localIndex, 0, vol);
      }

      xPos += 0.5 * __particleSize0[0];
      while (xPos < __domain[1][0] - 1e-5) {
        yPos = __domain[0][1];
        if (__domainBoundaryType[4] != 0) {
          vec3 pos = vec3(xPos, yPos, zPos);
          normal = vec3(0.0, sqrt(2.0) / 2.0, sqrt(2.0) / 2.0);
          InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        }

        yPos += 0.5 * __particleSize0[1];
        while (yPos < __domain[1][1] - 1e-5) {
          vec3 pos = vec3(xPos, yPos, zPos);
          normal = vec3(0.0, 0.0, 1.0);
          InsertParticle(pos, 3, __particleSize0, normal, localIndex, 0, vol);
          yPos += __particleSize0[1];
        }

        yPos = __domain[1][1];
        if (__domainBoundaryType[1] != 0) {
          vec3 pos = vec3(xPos, yPos, zPos);
          normal = vec3(0.0, -sqrt(2.0) / 2.0, sqrt(2.0) / 2.0);
          InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        }

        xPos += __particleSize0[0];
      }

      xPos = __domain[1][0];
      yPos = __domain[0][1];
      if (__domainBoundaryType[0] != 0 && __domainBoundaryType[4] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(-sqrt(3) / 3.0, sqrt(3) / 3.0, sqrt(3) / 3.0);
        InsertParticle(pos, 1, __particleSize0, normal, localIndex, 0, vol);
      }

      yPos += 0.5 * __particleSize0[1];
      if (__domainBoundaryType[0] != 0) {
        while (yPos < __domain[1][1] - 1e-5) {
          vec3 pos = vec3(xPos, yPos, zPos);
          normal = vec3(-sqrt(2.0) / 2.0, 0.0, sqrt(2.0) / 2.0);
          InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
          yPos += __particleSize0[1];
        }
      }

      yPos = __domain[1][1];
      if (__domainBoundaryType[0] != 0 && __domainBoundaryType[1] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(-sqrt(3) / 3.0, -sqrt(3) / 3.0, sqrt(3) / 3.0);
        InsertParticle(pos, 1, __particleSize0, normal, localIndex, 0, vol);
      }
    }

    zPos = __domain[0][2] + __particleSize0[2] / 2.0;
    while (zPos < __domain[1][2] - 1e-5) {
      yPos = __domain[0][1];
      xPos = __domain[0][0];
      if (__domainBoundaryType[2] != 0 && __domainBoundaryType[4] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(sqrt(2.0) / 2.0, sqrt(2.0) / 2.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
      }

      yPos += 0.5 * __particleSize0[1];
      if (__domainBoundaryType[2] != 0) {
        while (yPos < __domain[1][1] - 1e-5) {
          vec3 pos = vec3(xPos, yPos, zPos);
          normal = vec3(1.0, 0.0, 0.0);
          InsertParticle(pos, 3, __particleSize0, normal, localIndex, 0, vol);
          yPos += __particleSize0[1];
        }
      }

      yPos = __domain[1][1];
      if (__domainBoundaryType[1] != 0 && __domainBoundaryType[2] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
      }

      xPos += 0.5 * __particleSize0[0];
      while (xPos < __domain[1][0] - 1e-5) {
        yPos = __domain[0][1];
        if (__domainBoundaryType[4] != 0) {
          vec3 pos = vec3(xPos, yPos, zPos);
          normal = vec3(0.0, 1.0, 0.0);
          InsertParticle(pos, 3, __particleSize0, normal, localIndex, 0, vol);
        }

        yPos += __particleSize0[1] / 2.0;
        while (yPos < __domain[1][1] - 1e-5) {
          vec3 pos = vec3(xPos, yPos, zPos);
          normal = vec3(1.0, 0.0, 0.0);
          InsertParticle(pos, 0, __particleSize0, normal, localIndex, 0, vol);
          yPos += __particleSize0[1];
        }

        yPos = __domain[1][1];
        if (__domainBoundaryType[1] != 0) {
          vec3 pos = vec3(xPos, yPos, zPos);
          normal = vec3(0.0, -1.0, 0.0);
          InsertParticle(pos, 3, __particleSize0, normal, localIndex, 0, vol);
        }

        xPos += __particleSize0[0];
      }

      yPos = __domain[0][1];
      xPos = __domain[1][0];
      if (__domainBoundaryType[0] != 0 && __domainBoundaryType[4] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(-sqrt(2.0) / 2.0, sqrt(2.0) / 2.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
      }

      yPos += 0.5 * __particleSize0[1];
      if (__domainBoundaryType[0] != 0) {
        while (yPos < __domain[1][1] - 1e-5) {
          vec3 pos = vec3(xPos, yPos, zPos);
          normal = vec3(-1.0, 0.0, 0.0);
          InsertParticle(pos, 3, __particleSize0, normal, localIndex, 0, vol);
          yPos += __particleSize0[1];
        }
      }

      yPos = __domain[1][1];
      if (__domainBoundaryType[0] != 0 && __domainBoundaryType[1] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(-sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
      }

      zPos += __particleSize0[2];
    }

    // x-y, z=+z0 face
    if (__domainBoundaryType[5] != 0) {
      zPos = __domain[1][2];

      xPos = __domain[0][0];
      yPos = __domain[0][1];
      if (__domainBoundaryType[2] != 0 && __domainBoundaryType[4] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(sqrt(3) / 3.0, sqrt(3) / 3.0, -sqrt(3) / 3.0);
        InsertParticle(pos, 1, __particleSize0, normal, localIndex, 0, vol);
      }

      yPos += 0.5 * __particleSize0[1];
      if (__domainBoundaryType[2] != 0) {
        while (yPos < __domain[1][1] - 1e-5) {
          vec3 pos = vec3(xPos, yPos, zPos);
          normal = vec3(sqrt(2.0) / 2.0, 0.0, -sqrt(2.0) / 2.0);
          InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
          yPos += __particleSize0[1];
        }
      }

      yPos = __domain[1][1];
      if (__domainBoundaryType[1] != 0 && __domainBoundaryType[2] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(sqrt(3) / 3.0, -sqrt(3) / 3.0, -sqrt(3) / 3.0);
        InsertParticle(pos, 1, __particleSize0, normal, localIndex, 0, vol);
      }

      xPos += 0.5 * __particleSize0[0];
      while (xPos < __domain[1][0] - 1e-5) {
        yPos = __domain[0][1];
        if (__domainBoundaryType[4] != 0) {
          vec3 pos = vec3(xPos, yPos, zPos);
          normal = vec3(0.0, sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0);
          InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        }

        yPos += 0.5 * __particleSize0[1];
        while (yPos < __domain[1][1] - 1e-5) {
          vec3 pos = vec3(xPos, yPos, zPos);
          normal = vec3(0.0, 0.0, -1.0);
          InsertParticle(pos, 3, __particleSize0, normal, localIndex, 0, vol);
          yPos += __particleSize0[1];
        }

        yPos = __domain[1][1];
        if (__domainBoundaryType[1] != 0) {
          vec3 pos = vec3(xPos, yPos, zPos);
          normal = vec3(0.0, -sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0);
          InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        }

        xPos += __particleSize0[0];
      }

      xPos = __domain[1][0];
      yPos = __domain[0][1];
      if (__domainBoundaryType[0] != 0 && __domainBoundaryType[4] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(-sqrt(3) / 3.0, sqrt(3) / 3.0, -sqrt(3) / 3.0);
        InsertParticle(pos, 1, __particleSize0, normal, localIndex, 0, vol);
      }

      yPos += 0.5 * __particleSize0[1];
      if (__domainBoundaryType[0] != 0) {
        while (yPos < __domain[1][1] - 1e-5) {
          vec3 pos = vec3(xPos, yPos, zPos);
          normal = vec3(-sqrt(2.0) / 2.0, 0.0, -sqrt(2.0) / 2.0);
          InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
          yPos += __particleSize0[1];
        }
      }

      yPos = __domain[1][1];
      if (__domainBoundaryType[0] != 0 && __domainBoundaryType[1] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(-sqrt(3) / 3.0, -sqrt(3) / 3.0, -sqrt(3) / 3.0);
        InsertParticle(pos, 1, __particleSize0, normal, localIndex, 0, vol);
      }
    }
  }
}

void GMLS_Solver::InitFieldBoundaryParticle() {
  static vector<vec3> &coord = __field.vector.GetHandle("coord");

  double xPos, yPos, zPos;
  vec3 normal;
  if (__dim == 2) {
    double vol = __particleSize0[0] * __particleSize0[1];
    int localIndex = coord.size();
    zPos = 0.0;
    // down
    if (__domainBoundaryType[0] != 0) {
      xPos = __domain[0][0];
      yPos = __domain[0][1];
      if (__domainBoundaryType[3] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(sqrt(2) / 2.0, sqrt(2) / 2.0, 0.0);
        InsertParticle(pos, 1, __particleSize0, normal, localIndex, 0, vol);
      }
      xPos += 0.5 * __particleSize0[0];

      while (xPos < __domain[1][0] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(0.0, 1.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        xPos += __particleSize0[0];
      }
    }

    // right
    if (__domainBoundaryType[1] != 0) {
      xPos = __domain[1][0];
      yPos = __domain[0][1];
      if (__domainBoundaryType[0] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(-sqrt(2) / 2.0, sqrt(2) / 2.0, 0.0);
        InsertParticle(pos, 1, __particleSize0, normal, localIndex, 0, vol);
      }
      yPos += 0.5 * __particleSize0[1];

      while (yPos < __domain[1][1] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(-1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        yPos += __particleSize0[1];
      }
    }

    // up
    if (__domainBoundaryType[2] != 0) {
      xPos = __domain[1][0];
      yPos = __domain[1][1];
      if (__domainBoundaryType[1] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(-sqrt(2) / 2.0, -sqrt(2) / 2.0, 0.0);
        InsertParticle(pos, 1, __particleSize0, normal, localIndex, 0, vol);
      }
      xPos -= 0.5 * __particleSize0[0];

      while (xPos > __domain[0][0] + 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(0.0, -1.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        xPos -= __particleSize0[0];
      }
    }

    // left
    if (__domainBoundaryType[3] != 0) {
      xPos = __domain[0][0];
      yPos = __domain[1][1];
      if (__domainBoundaryType[2] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(sqrt(2) / 2.0, -sqrt(2) / 2.0, 0.0);
        InsertParticle(pos, 1, __particleSize0, normal, localIndex, 0, vol);
      }
      yPos -= 0.5 * __particleSize0[1];

      while (yPos > __domain[0][1] + 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        yPos -= __particleSize0[1];
      }
    }
  } // end of 2d construction
  if (__dim == 3) {
    double vol = __particleSize0[0] * __particleSize0[1] * __particleSize0[2];
    int localIndex = coord.size();

    vec3 startPos, endPos;
    // face
    // front
    if (__domainBoundaryType[0] != 0) {
      startPos = vec3(__domain[1][0], __domain[0][1] + __particleSize0[1] / 2,
                      __domain[0][2] + __particleSize0[2] / 2);
      endPos = vec3(__domain[1][0], __domain[1][1], __domain[1][2]);
      normal = vec3(-1.0, 0.0, 0.0);

      InitWallFaceParticle(
          startPos, endPos, [=](vec3 &pos) { pos[1] += __particleSize0[1]; },
          [=](vec3 &pos) { pos[2] += __particleSize0[2]; },
          [](vec3 &pos, vec3 &endPos) { return (pos[1] < endPos[1]); },
          [](vec3 &pos, vec3 &endPos) { return (pos[2] < endPos[2]); },
          localIndex, vol, normal);
    }

    // right
    if (__domainBoundaryType[1] != 0) {
      startPos = vec3(__domain[0][0] + __particleSize0[0] / 2, __domain[1][1],
                      __domain[0][2] + __particleSize0[2] / 2);
      endPos = vec3(__domain[1][0], __domain[1][1], __domain[1][2]);
      normal = vec3(0.0, -1.0, 0.0);

      InitWallFaceParticle(
          startPos, endPos, [=](vec3 &pos) { pos[0] += __particleSize0[0]; },
          [=](vec3 &pos) { pos[2] += __particleSize0[2]; },
          [](vec3 &pos, vec3 &endPos) { return (pos[0] < endPos[0]); },
          [](vec3 &pos, vec3 &endPos) { return (pos[2] < endPos[2]); },
          localIndex, vol, normal);
    }

    // back
    if (__domainBoundaryType[2] != 0) {
      startPos = vec3(__domain[0][0], __domain[0][1] + __particleSize0[1] / 2,
                      __domain[0][2] + __particleSize0[2] / 2);
      endPos = vec3(__domain[0][0], __domain[1][1], __domain[1][2]);
      normal = vec3(1.0, 0.0, 0.0);

      InitWallFaceParticle(
          startPos, endPos, [=](vec3 &pos) { pos[1] += __particleSize0[1]; },
          [=](vec3 &pos) { pos[2] += __particleSize0[2]; },
          [](vec3 &pos, vec3 &endPos) { return (pos[1] < endPos[1]); },
          [](vec3 &pos, vec3 &endPos) { return (pos[2] < endPos[2]); },
          localIndex, vol, normal);
    }

    // bottom
    if (__domainBoundaryType[3] != 0) {
      startPos = vec3(__domain[0][0] + __particleSize0[0] / 2,
                      __domain[0][1] + __particleSize0[1] / 2, __domain[0][2]);
      endPos = vec3(__domain[1][0], __domain[1][1], __domain[0][2]);
      normal = vec3(0.0, 0.0, 1.0);

      InitWallFaceParticle(
          startPos, endPos, [=](vec3 &pos) { pos[0] += __particleSize0[0]; },
          [=](vec3 &pos) { pos[1] += __particleSize0[1]; },
          [](vec3 &pos, vec3 &endPos) { return (pos[0] < endPos[0]); },
          [](vec3 &pos, vec3 &endPos) { return (pos[1] < endPos[1]); },
          localIndex, vol, normal);
    }

    // left
    if (__domainBoundaryType[4] != 0) {
      startPos = vec3(__domain[0][0] + __particleSize0[0] / 2, __domain[0][1],
                      __domain[0][2] + __particleSize0[2] / 2);
      endPos = vec3(__domain[1][0], __domain[0][1], __domain[1][2]);
      normal = vec3(0.0, 1.0, 0.0);

      InitWallFaceParticle(
          startPos, endPos, [=](vec3 &pos) { pos[0] += __particleSize0[0]; },
          [=](vec3 &pos) { pos[2] += __particleSize0[2]; },
          [](vec3 &pos, vec3 &endPos) { return (pos[0] < endPos[0]); },
          [](vec3 &pos, vec3 &endPos) { return (pos[2] < endPos[2]); },
          localIndex, vol, normal);
    }

    // top
    if (__domainBoundaryType[5] != 0) {
      startPos = vec3(__domain[0][0] + __particleSize0[0] / 2,
                      __domain[0][1] + __particleSize0[1] / 2, __domain[1][2]);
      endPos = vec3(__domain[1][0], __domain[1][1], __domain[1][2]);
      normal = vec3(0.0, 0.0, -1.0);

      InitWallFaceParticle(
          startPos, endPos, [=](vec3 &pos) { pos[0] += __particleSize0[0]; },
          [=](vec3 &pos) { pos[1] += __particleSize0[1]; },
          [](vec3 &pos, vec3 &endPos) { return (pos[0] < endPos[0]); },
          [](vec3 &pos, vec3 &endPos) { return (pos[1] < endPos[1]); },
          localIndex, vol, normal);
    }

    // edge and corner
    // bottom 4 edges
    if (__domainBoundaryType[4] != 0 && __domainBoundaryType[3] != 0) {
      xPos = __domain[0][0];
      yPos = __domain[0][1];
      zPos = __domain[0][2];
      if (__domainBoundaryType[2] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(sqrt(3.0) / 3.0, sqrt(3.0) / 3.0, sqrt(3.0) / 3.0);
        InsertParticle(pos, 3, __particleSize0, normal, localIndex, 0, vol);
      }
      xPos += 0.5 * __particleSize0[0];

      while (xPos < __domain[1][0] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(0.0, sqrt(2.0) / 2.0, sqrt(2.0) / 2.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        xPos += __particleSize0[0];
      }
    }

    if (__domainBoundaryType[0] != 0 && __domainBoundaryType[3] != 0) {
      xPos = __domain[1][0];
      yPos = __domain[0][1];
      zPos = __domain[0][2];
      if (__domainBoundaryType[4] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(-sqrt(3.0) / 3.0, sqrt(3.0) / 3.0, sqrt(3.0) / 3.0);
        InsertParticle(pos, 3, __particleSize0, normal, localIndex, 0, vol);
      }
      yPos += 0.5 * __particleSize0[1];

      while (yPos < __domain[1][1] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(-sqrt(2.0) / 2.0, 0.0, sqrt(2.0) / 2.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        yPos += __particleSize0[1];
      }
    }

    if (__domainBoundaryType[1] != 0 && __domainBoundaryType[3] != 0) {
      xPos = __domain[1][0];
      yPos = __domain[1][1];
      zPos = __domain[0][2];
      if (__domainBoundaryType[0] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(-sqrt(3.0) / 3.0, -sqrt(3.0) / 3.0, sqrt(3.0) / 3.0);
        InsertParticle(pos, 3, __particleSize0, normal, localIndex, 0, vol);
      }
      xPos -= 0.5 * __particleSize0[0];

      while (xPos > __domain[0][0] + 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(0.0, -sqrt(2.0) / 2.0, sqrt(2.0) / 2.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        xPos -= __particleSize0[0];
      }
    }

    if (__domainBoundaryType[2] != 0 && __domainBoundaryType[3] != 0) {
      xPos = __domain[0][0];
      yPos = __domain[1][1];
      zPos = __domain[0][2];
      if (__domainBoundaryType[1] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(sqrt(3.0) / 3.0, -sqrt(3.0) / 3.0, sqrt(3.0) / 3.0);
        InsertParticle(pos, 3, __particleSize0, normal, localIndex, 0, vol);
      }
      yPos -= 0.5 * __particleSize0[1];

      while (yPos > __domain[0][1] + 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(sqrt(2.0) / 2.0, 0.0, sqrt(2.0) / 2.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        yPos -= __particleSize0[1];
      }
    }

    // top 4 edges
    if (__domainBoundaryType[4] != 0 && __domainBoundaryType[5] != 0) {
      xPos = __domain[0][0];
      yPos = __domain[0][1];
      zPos = __domain[1][2];
      if (__domainBoundaryType[2] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(sqrt(3.0) / 3.0, sqrt(3.0) / 3.0, -sqrt(3.0) / 3.0);
        InsertParticle(pos, 3, __particleSize0, normal, localIndex, 0, vol);
      }
      xPos += 0.5 * __particleSize0[0];

      while (xPos < __domain[1][0] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(0.0, sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        xPos += __particleSize0[0];
      }
    }

    if (__domainBoundaryType[0] != 0 && __domainBoundaryType[5] != 0) {
      xPos = __domain[1][0];
      yPos = __domain[0][1];
      zPos = __domain[1][2];
      if (__domainBoundaryType[4] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(-sqrt(3.0) / 3.0, sqrt(3.0) / 3.0, -sqrt(3.0) / 3.0);
        InsertParticle(pos, 3, __particleSize0, normal, localIndex, 0, vol);
      }
      yPos += 0.5 * __particleSize0[1];

      while (yPos < __domain[1][1] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(-sqrt(2.0) / 2.0, 0.0, -sqrt(2.0) / 2.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        yPos += __particleSize0[1];
      }
    }

    if (__domainBoundaryType[1] != 0 && __domainBoundaryType[5] != 0) {
      xPos = __domain[1][0];
      yPos = __domain[1][1];
      zPos = __domain[1][2];
      if (__domainBoundaryType[0] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(-sqrt(3.0) / 3.0, -sqrt(3.0) / 3.0, -sqrt(3.0) / 3.0);
        InsertParticle(pos, 3, __particleSize0, normal, localIndex, 0, vol);
      }
      xPos -= 0.5 * __particleSize0[0];

      while (xPos > __domain[0][0] + 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(0.0, -sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        xPos -= __particleSize0[0];
      }
    }

    if (__domainBoundaryType[2] != 0 && __domainBoundaryType[5] != 0) {
      xPos = __domain[0][0];
      yPos = __domain[1][1];
      zPos = __domain[1][2];
      if (__domainBoundaryType[1] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(sqrt(3.0) / 3.0, -sqrt(3.0) / 3.0, -sqrt(3.0) / 3.0);
        InsertParticle(pos, 3, __particleSize0, normal, localIndex, 0, vol);
      }
      yPos -= 0.5 * __particleSize0[1];

      while (yPos > __domain[0][1] + 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(sqrt(2.0) / 2.0, 0.0, -sqrt(2.0) / 2.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        yPos -= __particleSize0[1];
      }
    }

    // middle 4 edges
    if (__domainBoundaryType[0] != 0 && __domainBoundaryType[1] != 0) {
      xPos = __domain[1][0];
      yPos = __domain[1][1];
      zPos = __domain[0][2] + 0.5 * __particleSize0[2];
      while (zPos < __domain[1][2] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(-sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        zPos += __particleSize0[2];
      }
    }

    if (__domainBoundaryType[1] != 0 && __domainBoundaryType[2] != 0) {
      xPos = __domain[0][0];
      yPos = __domain[1][1];
      zPos = __domain[0][2] + 0.5 * __particleSize0[2];
      while (zPos < __domain[1][2] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        zPos += __particleSize0[2];
      }
    }

    if (__domainBoundaryType[2] != 0 && __domainBoundaryType[4] != 0) {
      xPos = __domain[0][0];
      yPos = __domain[0][1];
      zPos = __domain[0][2] + 0.5 * __particleSize0[2];
      while (zPos < __domain[1][2] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(sqrt(2.0) / 2.0, sqrt(2.0) / 2.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        zPos += __particleSize0[2];
      }
    }

    if (__domainBoundaryType[0] != 0 && __domainBoundaryType[4] != 0) {
      xPos = __domain[1][0];
      yPos = __domain[0][1];
      zPos = __domain[0][2] + 0.5 * __particleSize0[2];
      while (zPos < __domain[1][2] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(-sqrt(2.0) / 2.0, sqrt(2.0) / 2.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex, 0, vol);
        zPos += __particleSize0[2];
      }
    }
  }
}

void GMLS_Solver::SplitParticle(vector<int> &splitTag) {
  auto &particleType = __field.index.GetHandle("particle type");
  auto &particleSize = __field.vector.GetHandle("size");
  auto &gapParticleSize = __gap.vector.GetHandle("size");

  vector<int> fieldSplitTag;
  vector<int> fieldBoundarySplitTag;
  vector<int> fieldRigidBodySurfaceSplitTag;

  sort(splitTag.begin(), splitTag.end());

  for (auto tag : splitTag) {
    if (particleType[tag] == 0) {
      fieldSplitTag.push_back(tag);
    } else if (particleType[tag] < 4) {
      fieldBoundarySplitTag.push_back(tag);
    } else {
      fieldRigidBodySurfaceSplitTag.push_back(tag);
    }
  }

  // split gap particle
  auto &gapCoord = __gap.vector.GetHandle("coord");

  static vector<vec3> &backgroundSourceCoord =
      __background.vector.GetHandle("source coord");
  static vector<int> &backgroundSourceIndex =
      __background.index.GetHandle("source index");

  int numSourceCoords = backgroundSourceCoord.size();
  int numTargetCoords = gapCoord.size();

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
      "source coordinates", numSourceCoords, 3);
  Kokkos::View<double **>::HostMirror sourceCoords =
      Kokkos::create_mirror_view(sourceCoordsDevice);

  for (size_t i = 0; i < backgroundSourceCoord.size(); i++) {
    for (int j = 0; j < 3; j++) {
      sourceCoords(i, j) = backgroundSourceCoord[i][j];
    }
  }

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> targetCoordsDevice(
      "target coordinates", numTargetCoords, 3);
  Kokkos::View<double **>::HostMirror targetCoords =
      Kokkos::create_mirror_view(targetCoordsDevice);

  for (int i = 0; i < numTargetCoords; i++) {
    for (int j = 0; j < 3; j++) {
      targetCoords(i, j) = gapCoord[i][j];
    }
  }

  Kokkos::deep_copy(sourceCoordsDevice, sourceCoords);
  Kokkos::deep_copy(targetCoordsDevice, targetCoords);

  auto pointCloudSearch(CreatePointCloudSearch(sourceCoords, __dim));

  int estimatedUpperBoundNumberNeighbors = 2 * pow(2, __dim);

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> neighborListsDevice(
      "neighbor lists", numTargetCoords, estimatedUpperBoundNumberNeighbors);
  Kokkos::View<int **>::HostMirror neighborLists =
      Kokkos::create_mirror_view(neighborListsDevice);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> epsilonDevice(
      "h supports", numTargetCoords);
  Kokkos::View<double *>::HostMirror epsilon =
      Kokkos::create_mirror_view(epsilonDevice);

  pointCloudSearch.generateNeighborListsFromKNNSearch(
      false, targetCoords, neighborLists, epsilon, pow(2, __dim), 1.0);

  Kokkos::deep_copy(neighborListsDevice, neighborLists);

  static vector<int> &particleNum = __field.index.GetHandle("particle number");
  int &localParticleNum = particleNum[0];

  fieldParticleSplitTag.resize(localParticleNum);

  for (int i = 0; i < localParticleNum; i++) {
    fieldParticleSplitTag[i] = 0;
  }

  for (auto tag : splitTag) {
    fieldParticleSplitTag[tag] = 1;
  }

  vector<int> recvSplitTag;
  vector<int> backgroundSplitTag;
  vector<int> recvParticleType;
  vector<int> backgroundParticleType;
  vector<vec3> recvParticleSize;
  vector<vec3> backgroundParticleSize;
  DataSwapAmongNeighbor(particleSize, recvParticleSize);
  DataSwapAmongNeighbor(particleType, recvParticleType);
  DataSwapAmongNeighbor(fieldParticleSplitTag, recvSplitTag);

  backgroundParticleSize.insert(backgroundParticleSize.end(),
                                particleSize.begin(), particleSize.end());
  backgroundParticleSize.insert(backgroundParticleSize.end(),
                                recvParticleSize.begin(),
                                recvParticleSize.end());

  backgroundParticleType.insert(backgroundParticleType.end(),
                                particleType.begin(), particleType.end());
  backgroundParticleType.insert(backgroundParticleType.end(),
                                recvParticleType.begin(),
                                recvParticleType.end());

  backgroundSplitTag.insert(backgroundSplitTag.end(),
                            fieldParticleSplitTag.begin(),
                            fieldParticleSplitTag.end());
  backgroundSplitTag.insert(backgroundSplitTag.end(), recvSplitTag.begin(),
                            recvSplitTag.end());

  vector<int> gapParticleSplitTag(numTargetCoords);
  for (int i = 0; i < numTargetCoords; i++) {
    int counter = 0;
    bool splitBasedOnNeighborSize = true;
    for (int j = 0; j < neighborLists(i, 0); j++) {
      if (backgroundSplitTag[neighborLists(i, j + 1)] == 1 &&
          backgroundParticleType[neighborLists(i, j + 1)] == 4) {
        counter++;
      }

      if ((gapParticleSize[i][0] <
           0.75 * backgroundParticleSize[neighborLists(i, j + 1)][0]) &&
          backgroundSplitTag[neighborLists(i, j + 1)] == 1) {
        splitBasedOnNeighborSize = false;
      }
    }

    bool splitBasedOnNeighborSplit = false;
    if (counter > 0) {
      splitBasedOnNeighborSplit = true;
    }

    if (splitBasedOnNeighborSplit && splitBasedOnNeighborSize) {
      gapParticleSplitTag[i] = 1;
    } else {
      gapParticleSplitTag[i] = 0;
    }
  }

  splitList.resize(localParticleNum);
  SplitFieldParticle(fieldSplitTag);
  SplitFieldBoundaryParticle(fieldBoundarySplitTag);
  SplitRigidBodySurfaceParticle(fieldRigidBodySurfaceSplitTag);
  SplitGapParticle(gapParticleSplitTag);

  MPI_Barrier(MPI_COMM_WORLD);
  ParticleIndex();
}

void GMLS_Solver::SplitFieldParticle(vector<int> &splitTag) {
  static auto &coord = __field.vector.GetHandle("coord");
  static auto &normal = __field.vector.GetHandle("normal");
  static auto &particleSize = __field.vector.GetHandle("size");
  static auto &globalIndex = __field.index.GetHandle("global index");
  static auto &adaptive_level = __field.index.GetHandle("adaptive level");
  static auto &particleType = __field.index.GetHandle("particle type");
  static auto &attachedRigidBodyIndex =
      __field.index.GetHandle("attached rigid body index");
  static auto &volume = __field.scalar.GetHandle("volume");

  static auto &_gapCoord = __gap.vector.GetHandle("coord");
  static auto &_gapNormal = __gap.vector.GetHandle("normal");
  static auto &_gapParticleSize = __gap.vector.GetHandle("size");
  static auto &_gapParticleType = __gap.index.GetHandle("particle type");

  int localIndex = coord.size();

  if (__dim == 2) {
    for (auto tag : splitTag) {
      vec3 origin = coord[tag];
      const double xDelta = particleSize[tag][0] * 0.25;
      const double yDelta = particleSize[tag][1] * 0.25;
      particleSize[tag][0] /= 2.0;
      particleSize[tag][1] /= 2.0;
      volume[tag] /= 4.0;
      bool insert = false;
      splitList[tag].clear();
      for (int i = -1; i < 2; i += 2) {
        for (int j = -1; j < 2; j += 2) {
          vec3 newPos = origin + vec3(i * xDelta, j * yDelta, 0.0);
          if (!insert) {
            int idx = IsInRigidBody(newPos, xDelta);
            if (idx == -2) {
              coord[tag] = newPos;
              adaptive_level[tag] = __adaptive_step;

              splitList[tag].push_back(tag);

              insert = true;
            } else if (idx > -1) {
              _gapCoord.push_back(newPos);
              _gapNormal.push_back(normal[tag]);
              _gapParticleSize.push_back(particleSize[tag]);
              _gapParticleType.push_back(particleType[tag]);
            }
          } else {
            double vol = volume[tag];
            int newParticle =
                InsertParticle(newPos, particleType[tag], particleSize[tag],
                               normal[tag], localIndex, __adaptive_step, vol);

            if (newParticle == 0) {
              splitList[tag].push_back(localIndex - 1);
            }
          }
        }
      }
    }
  }

  if (__dim == 3) {
    for (auto tag : splitTag) {
      vec3 origin = coord[tag];
      const double xDelta = particleSize[tag][0] * 0.25;
      const double yDelta = particleSize[tag][1] * 0.25;
      const double zDelta = particleSize[tag][2] * 0.25;
      bool insert = false;
      splitList[tag].clear();
      for (int i = -1; i < 2; i += 2) {
        for (int j = -1; j < 2; j += 2) {
          for (int k = -1; k < 2; k += 2) {
            vec3 newPos = origin + vec3(i * xDelta, j * yDelta, k * zDelta);
            if (!insert) {
              if (IsInRigidBody(newPos, xDelta) == -2) {
                coord[tag] = newPos;
                particleSize[tag][0] /= 2.0;
                particleSize[tag][1] /= 2.0;
                particleSize[tag][2] /= 2.0;
                volume[tag] /= 8.0;
                adaptive_level[tag] = __adaptive_step;

                splitList[tag].push_back(tag);

                insert = true;
              }
            } else {
              double vol = volume[tag];
              int newParticle =
                  InsertParticle(newPos, particleType[tag], particleSize[tag],
                                 normal[tag], localIndex, __adaptive_step, vol);

              if (newParticle == 0) {
                splitList[tag].push_back(localIndex - 1);
              }
            }
          }
        }
      }
    }
  }
}

void GMLS_Solver::SplitFieldBoundaryParticle(vector<int> &splitTag) {
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

  int localIndex = coord.size();

  if (__dim == 2) {
    for (auto tag : splitTag) {
      if (particleType[tag] == 1) {
        // corner particle
        splitList[tag].resize(1);
        splitList[tag][0] = tag;

        particleSize[tag][0] /= 2.0;
        particleSize[tag][1] /= 2.0;
        volume[tag] /= 4.0;
      } else {
        splitList[tag].clear();

        particleSize[tag][0] /= 2.0;
        particleSize[tag][1] /= 2.0;
        volume[tag] /= 4.0;
        adaptive_level[tag] = __adaptive_step;

        double theta = pCoord[tag][0];

        vec3 oldCoord = coord[tag];

        bool insert = false;
        for (int i = -1; i < 2; i += 2) {
          vec3 newPos = oldCoord + vec3(normal[tag][1], -normal[tag][0], 0.0) *
                                       i * particleSize[tag][0] * 0.5;

          if (!insert) {
            coord[tag] = newPos;

            splitList[tag].push_back(tag);

            insert = true;
          } else {
            double vol = volume[tag];
            InitWallFaceParticle(newPos, particleType[tag], particleSize[tag],
                                 normal[tag], localIndex, __adaptive_step, vol);
            splitList[tag].push_back(localIndex - 1);
          }
        }
      }
    }
  }
  if (__dim == 3) {
  }
}

void GMLS_Solver::SplitGapParticle(vector<int> &splitTag) {
  static auto &coord = __field.vector.GetHandle("coord");

  int localIndex = coord.size();

  // gap particles
  static auto &_gapCoord = __gap.vector.GetHandle("coord");
  static auto &_gapNormal = __gap.vector.GetHandle("normal");
  static auto &_gapParticleSize = __gap.vector.GetHandle("size");
  static auto &_gapParticleType = __gap.index.GetHandle("particle type");
  static auto &_gap_particle_adaptive_level =
      __gap.index.GetHandle("adaptive level");

  auto oldGapCoord = move(_gapCoord);
  auto oldGapNormal = move(_gapNormal);
  auto oldGapParticleSize = move(_gapParticleSize);
  auto oldGapParticleType = move(_gapParticleType);

  _gapCoord.clear();
  _gapNormal.clear();
  _gapParticleType.clear();
  _gapParticleSize.clear();
  _gap_particle_adaptive_level.clear();

  // gap particles
  if (__dim == 3) {
    for (auto tag : splitTag) {
      vec3 origin = oldGapCoord[tag];
      const double xDelta = oldGapParticleSize[tag][0] * 0.25;
      const double yDelta = oldGapParticleSize[tag][1] * 0.25;
      const double zDelta = oldGapParticleSize[tag][2] * 0.25;
      for (int i = -1; i < 2; i += 2) {
        for (int j = -1; j < 2; j += 2) {
          for (int k = -1; k < 2; k += 2) {
            vec3 newParticleSize = oldGapParticleSize[tag] * 0.5;
            vec3 newPos = origin + vec3(i * xDelta, j * yDelta, k * zDelta);
            double vol = oldGapParticleSize[tag][0] *
                         oldGapParticleSize[tag][1] *
                         oldGapParticleSize[tag][2] / 8.0;
            InsertParticle(newPos, oldGapParticleType[tag], newParticleSize,
                           oldGapNormal[tag], localIndex, __adaptive_step, vol);
          }
        }
      }
    }
  }

  if (__dim == 2) {
    for (int tag = 0; tag < splitTag.size(); tag++) {
      if (splitTag[tag] == 0) {
        InsertParticle(oldGapCoord[tag], oldGapParticleType[tag],
                       oldGapParticleSize[tag], oldGapNormal[tag], localIndex,
                       __adaptive_step,
                       oldGapParticleSize[tag][0] * oldGapParticleSize[tag][1]);
      } else {
        vec3 origin = oldGapCoord[tag];
        const double xDelta = oldGapParticleSize[tag][0] * 0.25;
        const double yDelta = oldGapParticleSize[tag][1] * 0.25;
        for (int i = -1; i < 2; i += 2) {
          for (int j = -1; j < 2; j += 2) {
            vec3 newParticleSize = oldGapParticleSize[tag] * 0.5;
            vec3 newPos = origin + vec3(i * xDelta, j * yDelta, 0.0);
            double vol = newParticleSize[0] * newParticleSize[1];
            InsertParticle(newPos, oldGapParticleType[tag], newParticleSize,
                           oldGapNormal[tag], localIndex, __adaptive_step, vol);
          }
        }
      }
    }

    for (int tag = splitTag.size(); tag < oldGapCoord.size(); tag++) {
      InsertParticle(oldGapCoord[tag], oldGapParticleType[tag],
                     oldGapParticleSize[tag], oldGapNormal[tag], localIndex,
                     __adaptive_step,
                     oldGapParticleSize[tag][0] * oldGapParticleSize[tag][1]);
    }
  }
}