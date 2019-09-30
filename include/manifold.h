#pragma once

#include <cmath>
#include <iostream>

#include "vec3.h"

using namespace std;

inline double Manifold(double x, double y) {
  double z = x * x - 1;
  return z;
}

inline vec3 ManifoldNorm(double x, double y) {
  double z = 1;
  vec3 normal = vec3(-2 * x * (pow(y, 2) - 1), -2 * y * (pow(x, 2) - 1), z);
  normal = normal * (1.0 / normal.mag());

  return normal;
}