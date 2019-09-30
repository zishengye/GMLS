#include "GMLS_solver.h"
#include "domain_decomposition.h"

#include <iostream>

using namespace std;

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
    if (abs(__domain[0][0] - __boundingBox[0][0]) < 1e-6) {
      __domainBoundaryType[4] = __boundingBoxBoundaryType[4];
    } else {
      __domainBoundaryType[4] = 0;
    }

    if (abs(__domain[0][1] - __boundingBox[0][1]) < 1e-6) {
      __domainBoundaryType[0] = __boundingBoxBoundaryType[0];
    } else {
      __domainBoundaryType[0] = 0;
    }

    if (abs(__domain[0][2] - __boundingBox[0][2]) < 1e-6) {
      __domainBoundaryType[3] = __boundingBoxBoundaryType[3];
    } else {
      __domainBoundaryType[3] = 0;
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

void GMLS_Solver::InitColloid() {
  int NC = 0;
  std::vector<vec3> Theta0, X0, V0, Omega0;
  Theta0.resize(NC);
  X0.resize(NC);
  V0.resize(NC);
  Omega0.resize(NC);

  int colloidType = 1;
  double R = 1.0;

  for (int i = 0; i < NC; i++) {
    __colloid.Ci_X.push_back(X0[i]);
    __colloid.Ci_Theta.push_back(Theta0[i]);
    __colloid.Ci_V.push_back(V0[i]);
    __colloid.Ci_Omega.push_back(Omega0[i]);
    __colloid.Ci_R.push_back(R);
    __colloid.type.push_back(colloidType);
  }
}

void GMLS_Solver::InitDomainDecomposition() {
  if (__dim == 3) {
    ProcessSplit(__nX, __nY, __nZ, __nI, __nJ, __nK, __MPISize, __myID);
    BoundingBoxSplit(__boundingBoxSize, __boundingBoxCount, __boundingBox,
                     __particleSize0, __domainBoundingBox, __domainCount,
                     __domain, __nX, __nY, __nZ, __nI, __nJ, __nK);
  } else if (__dim == 2) {
    ProcessSplit(__nX, __nY, __nI, __nJ, __MPISize, __myID);
    BoundingBoxSplit(__boundingBoxSize, __boundingBoxCount, __boundingBox,
                     __particleSize0, __domainBoundingBox, __domainCount,
                     __domain, __nX, __nY, __nI, __nJ);
  }

  SetDomainBoundary();

  InitNeighborList();
}

void GMLS_Solver::InitUniformParticleField() {
  ClearMemory();

  InitFluidParticle();
  InitWallParticle();

  SerialOperation([this]() {
    cout << "[Proc " << this->__myID << "]: generated "
         << this->__fluid.X.size() << " particles." << endl;
  });

  __fluid.localParticleNum = this->__fluid.X.size();
  MPI_Allreduce(&(__fluid.localParticleNum), &(__fluid.globalParticleNum), 1,
                MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  vector<int> particleNum;
  particleNum.resize(__MPISize);
  __fluid.particleOffset.resize(__MPISize + 1);
  MPI_Allgather(&(__fluid.localParticleNum), 1, MPI_INT, particleNum.data(), 1,
                MPI_INT, MPI_COMM_WORLD);
  __fluid.particleOffset[0] = 0;
  for (int i = 0; i < __MPISize; i++) {
    __fluid.particleOffset[i + 1] = __fluid.particleOffset[i] + particleNum[i];
  }

  for (size_t i = 0; i < __fluid.globalIndex.size(); i++) {
    __fluid.globalIndex[i] += __fluid.particleOffset[__myID];
  }
}

bool GMLS_Solver::IsInGap(vec3 &x) { return false; }

void GMLS_Solver::InitFluidParticle() {
  __cutoffDistance = 4.5 * std::max(__particleSize0[0], __particleSize0[1]);

  double xPos, yPos, zPos;
  vec3 normal = vec3(1.0, 0.0, 0.0);

  if (__dim == 2) {
    zPos = 0.0;
    double vol = __particleSize0[0] * __particleSize0[1];
    int localIndex = __fluid.X.size();
    // fluid particle
    yPos = __domain[0][1] + __particleSize0[1] / 2.0;
    for (int j = 0; j < __domainCount[1]; j++) {
      xPos = __domain[0][0] + __particleSize0[0] / 2.0;
      for (int i = 0; i < __domainCount[0]; i++) {
        vec3 pos = vec3(xPos, yPos, zPos);
        InsertParticle(pos, 0, __particleSize0, normal, localIndex++, vol);
        xPos += __particleSize0[0];
      }
      yPos += __particleSize0[1];
    }
  }
  if (__dim == 3) {
    double vol = __particleSize0[0] * __particleSize0[1] * __particleSize0[2];
    int localIndex = 0;
    double zPos = __domain[0][2] + __particleSize0[2] / 2.0;
    for (int k = 0; k < __domainCount[2]; k++) {
      double yPos = __domain[0][1] + __particleSize0[1] / 2.0;
      for (int j = 0; j < __domainCount[1]; j++) {
        double xPos = __domain[0][0] + __particleSize0[0] / 2.0;
        for (int i = 0; i < __domainCount[0]; i++) {
          vec3 pos = vec3(xPos, yPos, zPos);
          InsertParticle(pos, 0, __particleSize0, normal, localIndex++, vol);
          xPos += __particleSize0[0];
        }
        yPos += __particleSize0[1];
      }
      zPos += __particleSize0[2];
    }
  }
}

void GMLS_Solver::InitWallParticle() {
  double xPos, yPos, zPos;
  vec3 normal;
  if (__dim == 2) {
    double vol = __particleSize0[0] * __particleSize0[1];
    int localIndex = __fluid.X.size();
    zPos = 0.0;
    // down
    if (__domainBoundaryType[0] != 0) {
      xPos = __domain[0][0];
      yPos = __domain[0][1];
      if (__domainBoundaryType[3] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        xPos += __particleSize0[0];
      }

      while (xPos < __domain[1][0] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        xPos += __particleSize0[0];
      }
    }

    // right
    if (__domainBoundaryType[1] != 0) {
      xPos = __domain[1][0];
      yPos = __domain[0][1];
      if (__domainBoundaryType[0] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        yPos += __particleSize0[1];
      }

      while (yPos < __domain[1][1] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        yPos += __particleSize0[1];
      }
    }

    // up
    if (__domainBoundaryType[2] != 0) {
      xPos = __domain[1][0];
      yPos = __domain[1][1];
      if (__domainBoundaryType[1] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        xPos -= __particleSize0[0];
      }

      while (xPos > __domain[0][0] + 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        xPos -= __particleSize0[0];
      }
    }

    // left
    if (__domainBoundaryType[3] != 0) {
      xPos = __domain[0][0];
      yPos = __domain[1][1];
      if (__domainBoundaryType[2] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        yPos -= __particleSize0[1];
      }

      while (yPos > __domain[0][1] + 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        yPos -= __particleSize0[1];
      }
    }
  }
  if (__dim == 3) {
    double vol = __particleSize0[0] * __particleSize0[1] * __particleSize0[2];
    int localIndex = __fluid.X.size();

    vec3 startPos, endPos;
    // face
    // front
    if (__domainBoundaryType[0] != 0) {
      startPos = vec3(__domain[0][0] + __particleSize0[0] / 2, __domain[0][1],
                      __domain[0][2] + __particleSize0[2] / 2);
      endPos = vec3(__domain[1][0], __domain[0][1], __domain[1][2]);
      normal = vec3(1.0, 0.0, 0.0);

      InitWallFaceParticle(
          startPos, endPos, [=](vec3 &pos) { pos[0] += __particleSize0[0]; },
          [=](vec3 &pos) { pos[2] += __particleSize0[2]; },
          [](vec3 &pos, vec3 &endPos) { return (pos[0] < endPos[0]); },
          [](vec3 &pos, vec3 &endPos) { return (pos[2] < endPos[2]); },
          localIndex, vol, normal);
    }

    // right
    if (__domainBoundaryType[1] != 0) {
      startPos = vec3(__domain[1][0], __domain[0][1] + __particleSize0[1] / 2,
                      __domain[0][2] + __particleSize0[2] / 2);
      endPos = vec3(__domain[1][0], __domain[1][1], __domain[1][2]);
      normal = vec3(1.0, 0.0, 0.0);

      InitWallFaceParticle(
          startPos, endPos, [=](vec3 &pos) { pos[1] += __particleSize0[1]; },
          [=](vec3 &pos) { pos[2] += __particleSize0[2]; },
          [](vec3 &pos, vec3 &endPos) { return (pos[1] < endPos[1]); },
          [](vec3 &pos, vec3 &endPos) { return (pos[2] < endPos[2]); },
          localIndex, vol, normal);
    }

    // back
    if (__domainBoundaryType[2] != 0) {
      startPos = vec3(__domain[0][0] + __particleSize0[0] / 2, __domain[1][1],
                      __domain[0][2] + __particleSize0[2] / 2);
      endPos = vec3(__domain[1][0], __domain[1][1], __domain[1][2]);
      normal = vec3(1.0, 0.0, 0.0);

      InitWallFaceParticle(
          startPos, endPos, [=](vec3 &pos) { pos[0] += __particleSize0[0]; },
          [=](vec3 &pos) { pos[2] += __particleSize0[2]; },
          [](vec3 &pos, vec3 &endPos) { return (pos[0] < endPos[0]); },
          [](vec3 &pos, vec3 &endPos) { return (pos[2] < endPos[2]); },
          localIndex, vol, normal);
    }

    // bottom
    if (__domainBoundaryType[3] != 0) {
      startPos = vec3(__domain[0][0] + __particleSize0[0] / 2,
                      __domain[0][1] + __particleSize0[1] / 2, __domain[0][2]);
      endPos = vec3(__domain[1][0], __domain[1][1], __domain[0][2]);
      normal = vec3(1.0, 0.0, 0.0);

      InitWallFaceParticle(
          startPos, endPos, [=](vec3 &pos) { pos[0] += __particleSize0[0]; },
          [=](vec3 &pos) { pos[1] += __particleSize0[1]; },
          [](vec3 &pos, vec3 &endPos) { return (pos[0] < endPos[0]); },
          [](vec3 &pos, vec3 &endPos) { return (pos[1] < endPos[1]); },
          localIndex, vol, normal);
    }

    // left
    if (__domainBoundaryType[4] != 0) {
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

    // top
    if (__domainBoundaryType[5] != 0) {
      startPos = vec3(__domain[0][0] + __particleSize0[0] / 2,
                      __domain[0][1] + __particleSize0[1] / 2, __domain[1][2]);
      endPos = vec3(__domain[1][0], __domain[1][1], __domain[1][2]);
      normal = vec3(1.0, 0.0, 0.0);

      InitWallFaceParticle(
          startPos, endPos, [=](vec3 &pos) { pos[0] += __particleSize0[0]; },
          [=](vec3 &pos) { pos[1] += __particleSize0[1]; },
          [](vec3 &pos, vec3 &endPos) { return (pos[0] < endPos[0]); },
          [](vec3 &pos, vec3 &endPos) { return (pos[1] < endPos[1]); },
          localIndex, vol, normal);
    }

    // edge and corner
    // bottom 4 edges
    if (__domainBoundaryType[0] != 0 && __domainBoundaryType[3] != 0) {
      xPos = __domain[0][0];
      yPos = __domain[0][1];
      zPos = __domain[0][2];
      if (__domainBoundaryType[4] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 3, __particleSize0, normal, localIndex++, vol);
        xPos += __particleSize0[0];
      }

      while (xPos < __domain[1][0] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        xPos += __particleSize0[0];
      }
    }

    if (__domainBoundaryType[1] != 0 && __domainBoundaryType[3] != 0) {
      xPos = __domain[1][0];
      yPos = __domain[0][1];
      zPos = __domain[0][2];
      if (__domainBoundaryType[0] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 3, __particleSize0, normal, localIndex++, vol);
        yPos += __particleSize0[1];
      }

      while (yPos < __domain[1][1] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        yPos += __particleSize0[1];
      }
    }

    if (__domainBoundaryType[2] != 0 && __domainBoundaryType[3] != 0) {
      xPos = __domain[1][0];
      yPos = __domain[1][1];
      zPos = __domain[0][2];
      if (__domainBoundaryType[1] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 3, __particleSize0, normal, localIndex++, vol);
        xPos -= __particleSize0[0];
      }

      while (xPos > __domain[0][0] + 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        xPos -= __particleSize0[0];
      }
    }

    if (__domainBoundaryType[4] != 0 && __domainBoundaryType[3] != 0) {
      xPos = __domain[0][0];
      yPos = __domain[1][1];
      zPos = __domain[0][2];
      if (__domainBoundaryType[2] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 3, __particleSize0, normal, localIndex++, vol);
        yPos -= __particleSize0[1];
      }

      while (yPos > __domain[0][0] + 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        yPos -= __particleSize0[1];
      }
    }

    // top 4 edges
    if (__domainBoundaryType[0] != 0 && __domainBoundaryType[5] != 0) {
      xPos = __domain[0][0];
      yPos = __domain[0][1];
      zPos = __domain[1][2];
      if (__domainBoundaryType[4] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 3, __particleSize0, normal, localIndex++, vol);
        xPos += __particleSize0[0];
      }

      while (xPos < __domain[1][0] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        xPos += __particleSize0[0];
      }
    }

    if (__domainBoundaryType[1] != 0 && __domainBoundaryType[5] != 0) {
      xPos = __domain[1][0];
      yPos = __domain[0][1];
      zPos = __domain[1][2];
      if (__domainBoundaryType[0] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 3, __particleSize0, normal, localIndex++, vol);
        yPos += __particleSize0[1];
      }

      while (yPos < __domain[1][1] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        yPos += __particleSize0[1];
      }
    }

    if (__domainBoundaryType[2] != 0 && __domainBoundaryType[5] != 0) {
      xPos = __domain[1][0];
      yPos = __domain[1][1];
      zPos = __domain[1][2];
      if (__domainBoundaryType[1] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 3, __particleSize0, normal, localIndex++, vol);
        xPos -= __particleSize0[0];
      }

      while (xPos > __domain[0][0] + 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        xPos -= __particleSize0[0];
      }
    }

    if (__domainBoundaryType[4] != 0 && __domainBoundaryType[5] != 0) {
      xPos = __domain[0][0];
      yPos = __domain[1][1];
      zPos = __domain[1][2];
      if (__domainBoundaryType[2] != 0) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 3, __particleSize0, normal, localIndex++, vol);
        yPos -= __particleSize0[1];
      }

      while (yPos > __domain[0][0] + 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        yPos -= __particleSize0[1];
      }
    }

    // middle 4 edges
    if (__domainBoundaryType[0] != 0 && __domainBoundaryType[1] != 0) {
      xPos = __domain[1][0];
      yPos = __domain[0][1];
      zPos = __domain[0][2];
      if (__domainBoundaryType[3] != 0) {
        zPos += __particleSize0[2];
      }
      while (zPos < __domain[1][2] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        zPos += __particleSize0[2];
      }
    }

    if (__domainBoundaryType[1] != 0 && __domainBoundaryType[2] != 0) {
      xPos = __domain[1][0];
      yPos = __domain[1][1];
      zPos = __domain[0][2];
      if (__domainBoundaryType[3] != 0) {
        zPos += __particleSize0[2];
      }
      while (zPos < __domain[1][2] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        zPos += __particleSize0[2];
      }
    }

    if (__domainBoundaryType[2] != 0 && __domainBoundaryType[4] != 0) {
      xPos = __domain[0][0];
      yPos = __domain[1][1];
      zPos = __domain[0][2];
      if (__domainBoundaryType[3] != 0) {
        zPos += __particleSize0[2];
      }
      while (zPos < __domain[1][2] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        zPos += __particleSize0[2];
      }
    }

    if (__domainBoundaryType[0] != 0 && __domainBoundaryType[4] != 0) {
      xPos = __domain[0][0];
      yPos = __domain[0][1];
      zPos = __domain[0][2];
      if (__domainBoundaryType[3] != 0) {
        zPos += __particleSize0[2];
      }
      while (zPos < __domain[1][2] - 1e-5) {
        vec3 pos = vec3(xPos, yPos, zPos);
        normal = vec3(1.0, 0.0, 0.0);
        InsertParticle(pos, 2, __particleSize0, normal, localIndex++, vol);
        zPos += __particleSize0[2];
      }
    }
  }
}