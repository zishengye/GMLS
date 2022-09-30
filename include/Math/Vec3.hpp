#ifndef _Math_Vec3_Hpp_
#define _Math_Vec3_Hpp_

#include <cmath>
#include <fstream>
#include <vector>

#include "Core/Typedef.hpp"

namespace Math {
class Vec3 {
private:
  Scalar data_[3];

public:
  Vec3() {
    data_[0] = 0;
    data_[1] = 0;
    data_[2] = 0;
  }

  Vec3(const Scalar first, const Scalar second, const Scalar third) {
    data_[0] = first;
    data_[1] = second;
    data_[2] = third;
  }

  Vec3(const Vec3 &t) {
    data_[0] = t.data_[0];
    data_[1] = t.data_[1];
    data_[2] = t.data_[2];
  }

  Scalar operator[](const LocalIndex i) { return data_[i]; }

  Scalar operator[](const LocalIndex i) const { return data_[i]; }

  Vec3 &operator+=(const Vec3 &y) {
    data_[0] += y[0];
    data_[1] += y[1];
    data_[2] += y[2];

    return *this;
  }

  Vec3 &operator-=(const Vec3 &y) {
    data_[0] -= y[0];
    data_[1] -= y[1];
    data_[2] -= y[2];

    return *this;
  }

  Vec3 &operator=(const Vec3 &y) {
    data_[0] = y[0];
    data_[1] = y[1];
    data_[2] = y[2];

    return *this;
  }

  Vec3 &operator*=(const Scalar a) {
    data_[0] *= a;
    data_[1] *= a;
    data_[2] *= a;

    return *this;
  }

  Vec3 operator-(const Vec3 &y) {
    return Vec3(data_[0] - y[0], data_[1] - y[1], data_[2] - y[2]);
  }

  Vec3 operator*(const Scalar a) {
    return Vec3(a * data_[0], a * data_[1], a * data_[2]);
  }

  Scalar Mag() const {
    return sqrt(data_[0] * data_[0] + data_[1] * data_[1] +
                data_[2] * data_[2]);
  }

  Scalar Dot(const Vec3 &y) {
    return y[0] * data_[0] + y[1] * data_[1] + y[2] * data_[2];
  }
};

inline Vec3 operator+(const Vec3 &x, const Vec3 &y) {
  return Vec3(x[0] + y[0], x[1] + y[1], x[2] + y[2]);
}
} // namespace Math

#endif