#ifndef _DOMAIN_DECOMPOSITION_H_
#define _DOMAIN_DECOMPOSITION_H_

#include <cmath>
#include <iostream>
#include <tgmath.h>
#include <vector>

#include "vec3.h"

void ProcessSplit(int &nX, int &nY, int &nI, int &nJ, const int nProc,
                  const int ID);

void ProcessSplit(int &nX, int &nY, int &nZ, int &nI, int &nJ, int &nK,
                  const int nProc, const int ID);

int BoundingBoxSplit(vec3 &boundingBoxSize, triple<int> &boundingBoxCount,
                     std::vector<vec3> &boundingBox, vec3 &particleSize,
                     std::vector<vec3> &domainBoundingBox,
                     triple<int> &domainCount, std::vector<vec3> &domain,
                     int nX, int nY, int nI, int nJ, double minDis,
                     int maxLevel);

int BoundingBoxSplit(vec3 &boundingBoxSize, triple<int> &boundingBoxCount,
                     std::vector<vec3> &boundingBox, vec3 &particleSize,
                     std::vector<vec3> &domainBoundingBox,
                     triple<int> &domainCount, std::vector<vec3> &domain,
                     int nX, int nY, int nZ, int nI, int nJ, int nK,
                     double minDis);

#endif