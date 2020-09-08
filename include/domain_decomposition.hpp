#ifndef _DOMAIN_DECOMPOSITION_HPP_
#define _DOMAIN_DECOMPOSITION_HPP_

#include <cmath>
#include <iostream>
#include <tgmath.h>
#include <vector>

#include "vec3.h"

void process_split(triple<int> &global_block, triple<int> &local_coord,
                   const int num_proc, const int id, const int dimension) {
  if (dimension == 2) {
    int x, y, i, j;
    x = sqrt(num_proc);
    while (x > 0) {
      y = num_proc / x;
      if (num_proc == x * y) {
        break;
      } else {
        x--;
      }
    }

    i = id % x;
    j = id / x;

    global_block[0] = x;
    global_block[1] = y;

    local_coord[0] = i;
    local_coord[1] = j;
  }
  if (dimension == 3) {
    int x, y, z, i, j, k;
    x = cbrt(num_proc);
    bool split_found = false;
    while (x > 0 && split_found == false) {
      y = x;
      while (y > 0 && split_found == false) {
        z = num_proc / (x * y);
        if (num_proc == (x * y * z)) {
          split_found = true;
          break;
        } else {
          y--;
        }
      }

      if (split_found == false) {
        x--;
      }
    }

    i = (id % (x * y)) % x;
    j = (id % (x * y)) / x;
    k = id / (x * y);

    global_block[0] = x;
    global_block[1] = y;
    global_block[2] = z;

    local_coord[0] = i;
    local_coord[1] = j;
    local_coord[2] = k;
  }
}

void bounding_box_split(vec3 *global_bounding_box, vec3 *local_bounding_box,
                        triple<int> &global_particle_num,
                        triple<int> &local_particle_num, vec3 &particle_size,
                        triple<int> &global_block_size,
                        triple<int> &local_block_coord, int dimension) {
  vec3 bounding_box_size;
  bounding_box_size[0] = global_bounding_box[1][0] - global_bounding_box[0][0];
  bounding_box_size[1] = global_bounding_box[1][1] - global_bounding_box[0][1];
  bounding_box_size[2] = global_bounding_box[1][2] - global_bounding_box[0][2];

  for (int i = 0; i < 2; i++) {
    particle_size[i] = bounding_box_size[i] / global_particle_num[i];
  }

  if (dimension == 2) {
    std::vector<int> count_x;
    std::vector<int> count_y;

    for (int i = 0; i < global_block_size[0]; i++) {
      if (global_particle_num[0] % global_block_size[0] > i) {
        count_x.push_back(global_particle_num[0] / global_block_size[0] + 1);
      } else {
        count_x.push_back(global_particle_num[0] / global_block_size[0]);
      }
    }

    for (int i = 0; i < global_block_size[1]; i++) {
      if (global_particle_num[1] % global_block_size[1] > i) {
        count_y.push_back(global_particle_num[1] / global_block_size[0] + 1);
      } else {
        count_y.push_back(global_particle_num[1] / global_block_size[1]);
      }
    }

    local_particle_num[0] = count_x[local_block_coord[0]];
    local_particle_num[1] = count_y[local_block_coord[1]];

    double x_start = global_bounding_box[0][0];
    double y_start = global_bounding_box[0][1];
    for (int i = 0; i < local_block_coord[0]; i++) {
      x_start += count_x[i] * particle_size[0];
    }
    for (int i = 0; i < local_block_coord[1]; i++) {
      y_start += count_y[i] * particle_size[1];
    }

    double x_end = x_start + count_x[local_block_coord[0]] * particle_size[0];
    double y_end = y_start + count_y[local_block_coord[1]] * particle_size[1];

    local_bounding_box[0][0] = x_start;
    local_bounding_box[0][1] = y_start;
    local_bounding_box[1][0] = x_end;
    local_bounding_box[1][1] = y_end;
  }
  if (dimension == 3) {
  }
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