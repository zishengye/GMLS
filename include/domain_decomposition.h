#ifndef DOMAIN_DECOMPOSITION_H
#define DOMAIN_DECOMPOSITION_H

#include <cmath>
#include <iostream>
#include <tgmath.h>
#include <vector>

#include "vec3.h"

static void ProcessSplit(int &nX, int &nY, int &nI, int &nJ, const int nProc,
                         const int ID) {
  nX = sqrt(nProc);
  while (nX > 0) {
    nY = nProc / nX;
    if (nProc == nX * nY) {
      break;
    } else {
      nX--;
    }
  }

  nI = ID % nX;
  nJ = ID / nX;

  return;
}

static void ProcessSplit(int &nX, int &nY, int &nZ, int &nI, int &nJ, int &nK,
                         const int nProc, const int ID) {
  nX = cbrt(nProc);
  bool splitFound = false;
  while (nX > 0 && splitFound == false) {
    nY = nX;
    while (nY > 0 && splitFound == false) {
      nZ = nProc / (nX * nY);
      if (nProc == (nX * nY * nZ)) {
        splitFound = true;
        break;
      } else {
        nY--;
      }
    }
    if (splitFound == false) {
      nX--;
    }
  }

  nI = (ID % (nX * nY)) % nX;
  nJ = (ID % (nX * nY)) / nX;
  nK = ID / (nX * nY);
}

static int BoundingBoxSplit(vec3 &boundingBoxSize,
                            triple<int> &boundingBoxCount,
                            std::vector<vec3> &boundingBox, vec3 &particleSize,
                            vec3 *domainBoundingBox, triple<int> &domainCount,
                            vec3 *domain, int nX, int nY, int nI, int nJ,
                            double minDis, int maxLevel) {
  for (int i = 0; i < 2; i++) {
    particleSize[i] = boundingBoxSize[i] / boundingBoxCount[i];
  }

  int countMultiplier = 1;
  int addedLevel = 0;
  while (particleSize[0] > minDis && addedLevel < maxLevel) {
    particleSize *= 0.5;
    countMultiplier *= 2;
    addedLevel++;
  }

  std::vector<int> _countX;
  std::vector<int> _countY;

  auto actualBoundingBoxCount = boundingBoxCount;
  actualBoundingBoxCount *= countMultiplier;

  for (int i = 0; i < nX; i++) {
    if (actualBoundingBoxCount[0] % nX > i) {
      _countX.push_back(actualBoundingBoxCount[0] / nX + 1);
    } else {
      _countX.push_back(actualBoundingBoxCount[0] / nX);
    }
  }

  for (int i = 0; i < nY; i++) {
    if (actualBoundingBoxCount[1] % nY > i) {
      _countY.push_back(actualBoundingBoxCount[1] / nY + 1);
    } else {
      _countY.push_back(actualBoundingBoxCount[1] / nY);
    }
  }

  domainCount[0] = _countX[nI];
  domainCount[1] = _countY[nJ];

  double xStart = boundingBox[0][0];
  double yStart = boundingBox[0][1];
  for (int i = 0; i < nI; i++) {
    xStart += _countX[i] * particleSize[0];
  }
  for (int i = 0; i < nJ; i++) {
    yStart += _countY[i] * particleSize[1];
  }

  double xEnd = xStart + _countX[nI] * particleSize[0];
  double yEnd = yStart + _countY[nJ] * particleSize[1];

  domain[0][0] = xStart;
  domain[0][1] = yStart;
  domain[1][0] = xEnd;
  domain[1][1] = yEnd;

  domainBoundingBox[0][0] = boundingBoxSize[0] / nX * nI + boundingBox[0][0];
  domainBoundingBox[0][1] = boundingBoxSize[1] / nY * nJ + boundingBox[0][1];
  domainBoundingBox[1][0] =
      boundingBoxSize[0] / nX * (nI + 1) + boundingBox[0][0];
  domainBoundingBox[1][1] =
      boundingBoxSize[1] / nY * (nJ + 1) + boundingBox[0][1];

  return addedLevel;
}

static int BoundingBoxSplit(vec3 &boundingBoxSize,
                            triple<int> &boundingBoxCount,
                            std::vector<vec3> &boundingBox, vec3 &particleSize,
                            vec3 *domainBoundingBox, triple<int> &domainCount,
                            vec3 *domain, int nX, int nY, int nZ, int nI,
                            int nJ, int nK, double minDis) {
  for (int i = 0; i < 3; i++) {
    particleSize[i] = boundingBoxSize[i] / boundingBoxCount[i];
  }

  // while (particleSize[0] > minDis)
  //   particleSize *= 0.5;

  std::vector<int> _countX;
  std::vector<int> _countY;
  std::vector<int> _countZ;

  for (int i = 0; i < nX; i++) {
    if (boundingBoxCount[0] % nX > i) {
      _countX.push_back(boundingBoxCount[0] / nX + 1);
    } else {
      _countX.push_back(boundingBoxCount[0] / nX);
    }
  }

  for (int i = 0; i < nY; i++) {
    if (boundingBoxCount[1] % nY > i) {
      _countY.push_back(boundingBoxCount[1] / nY + 1);
    } else {
      _countY.push_back(boundingBoxCount[1] / nY);
    }
  }

  for (int i = 0; i < nZ; i++) {
    if (boundingBoxCount[2] % nZ > i) {
      _countZ.push_back(boundingBoxCount[2] / nZ + 1);
    } else {
      _countZ.push_back(boundingBoxCount[2] / nZ);
    }
  }

  domainCount[0] = _countX[nI];
  domainCount[1] = _countY[nJ];
  domainCount[2] = _countZ[nK];

  double xStart = boundingBox[0][0];
  double yStart = boundingBox[0][1];
  double zStart = boundingBox[0][2];
  for (int i = 0; i < nI; i++) {
    xStart += _countX[i] * particleSize[0];
  }
  for (int i = 0; i < nJ; i++) {
    yStart += _countY[i] * particleSize[1];
  }
  for (int i = 0; i < nK; i++) {
    zStart += _countZ[i] * particleSize[2];
  }

  double xEnd = xStart + _countX[nI] * particleSize[0];
  double yEnd = yStart + _countY[nJ] * particleSize[1];
  double zEnd = zStart + _countZ[nK] * particleSize[2];

  domain[0][0] = xStart;
  domain[0][1] = yStart;
  domain[0][2] = zStart;
  domain[1][0] = xEnd;
  domain[1][1] = yEnd;
  domain[1][2] = zStart;

  domainBoundingBox[0][0] = boundingBoxSize[0] / nX * nI + boundingBox[0][0];
  domainBoundingBox[0][1] = boundingBoxSize[1] / nY * nJ + boundingBox[0][1];
  domainBoundingBox[0][2] = boundingBoxSize[2] / nZ * nK + boundingBox[0][2];
  domainBoundingBox[1][0] =
      boundingBoxSize[0] / nX * (nI + 1) + boundingBox[0][0];
  domainBoundingBox[1][1] =
      boundingBoxSize[1] / nY * (nJ + 1) + boundingBox[0][1];
  domainBoundingBox[1][2] =
      boundingBoxSize[2] / nZ * (nK + 1) + boundingBox[0][2];

  return 0;
}

#endif