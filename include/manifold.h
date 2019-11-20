#pragma once

#include <cmath>
#include <iostream>

#include "vec3.h"

using namespace std;

#define PI 3.1415926

inline vec3 Manifold(double xScalar, double y) {
  double r = 1.0;
  double theta = xScalar;
  double phi = y;
  xScalar = r * sin(phi) * cos(theta);
  y = r * sin(phi) * sin(theta);
  double z = r * cos(phi);
  return vec3(xScalar, y, z);
}

inline vec3 ManifoldNorm(double xScalar, double y) {
  double z = 1;
  vec3 normal =
      vec3(-2 * xScalar * (pow(y, 2) - 1), -2 * y * (pow(xScalar, 2) - 1), z);
  normal = normal * (1.0 / normal.mag());

  return normal;
}