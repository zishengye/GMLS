#ifndef _Dynamics_Quaternion_Hpp_
#define _Dynamics_Quaternion_Hpp_

#include <cmath>
#include <iostream>

#include "Core/Typedef.hpp"
#include "Math/Vec3.hpp"

namespace Dynamics {
class Quaternion {
private:
  Scalar data_[4];

public:
  // default constructor
  Quaternion() {
    data_[0] = 0.0;
    data_[1] = 0.0;
    data_[2] = 0.0;
    data_[3] = 0.0;
  }

  Quaternion(const Quaternion &q) {
    data_[0] = q.data_[0];
    data_[1] = q.data_[1];
    data_[2] = q.data_[2];
    data_[3] = q.data_[3];
  }

  // constructor from a vector, means the scalar part would be zero
  Quaternion(const Math::Vec3 &vec) {
    data_[0] = 0.0;
    data_[1] = vec[0];
    data_[2] = vec[1];
    data_[3] = vec[2];
  }

  // constructor from a rotation axis and rotation angle theta
  Quaternion(const Math::Vec3 &omega, const Scalar theta) {
    Scalar norm = omega.Mag();
    Scalar normReciprocal = 1.0 / norm;
    Scalar h = theta * norm;
    data_[0] = cos(h / 2.0);
    data_[1] = sin(h / 2.0) * omega[0] * normReciprocal;
    data_[2] = sin(h / 2.0) * omega[1] * normReciprocal;
    data_[3] = sin(h / 2.0) * omega[2] * normReciprocal;
  }

  // constructor from euler angle, theta1, 2, 3, sequence - x, y, z
  Quaternion(const Scalar roll, const Scalar pitch, const Scalar yaw) {
    Scalar cy = cos(yaw * 0.5);
    Scalar sy = sin(yaw * 0.5);
    Scalar cp = cos(pitch * 0.5);
    Scalar sp = sin(pitch * 0.5);
    Scalar cr = cos(roll * 0.5);
    Scalar sr = sin(roll * 0.5);

    data_[0] = cr * cp * cy + sr * sp * sy;
    data_[1] = sr * cp * cy - cr * sp * sy;
    data_[2] = cr * sp * cy + sr * cp * sy;
    data_[3] = cr * cp * sy - sr * sp * cy;
  }

  Quaternion product(const Quaternion &q) {
    Quaternion res;

    res.data_[0] = data_[0] * q.data_[0] - data_[1] * q.data_[1] -
                   data_[2] * q.data_[2] - data_[3] * q.data_[3];
    res.data_[1] = data_[0] * q.data_[1] + data_[1] * q.data_[0] +
                   data_[2] * q.data_[3] - data_[3] * q.data_[2];
    res.data_[2] = data_[0] * q.data_[2] - data_[1] * q.data_[3] +
                   data_[2] * q.data_[0] + data_[3] * q.data_[1];
    res.data_[3] = data_[0] * q.data_[3] + data_[1] * q.data_[2] -
                   data_[2] * q.data_[1] + data_[3] * q.data_[0];

    return res;
  }

  Scalar Q0() const { return data_[0]; }
  Scalar Q1() const { return data_[1]; }
  Scalar Q2() const { return data_[2]; }
  Scalar Q3() const { return data_[3]; }

  void SetQ0(const Scalar q0) { data_[0] = q0; }
  void SetQ1(const Scalar q1) { data_[1] = q1; }
  void SetQ2(const Scalar q2) { data_[2] = q2; }
  void SetQ3(const Scalar q3) { data_[3] = q3; }

  void Cross(const Quaternion &qa, const Quaternion &qb) {
    Scalar w = qa.data_[0] * qb.data_[0] - qa.data_[1] * qb.data_[1] -
               qa.data_[2] * qb.data_[2] - qa.data_[3] * qb.data_[3];
    Scalar x = qa.data_[0] * qb.data_[1] + qa.data_[1] * qb.data_[0] -
               qa.data_[3] * qb.data_[2] + qa.data_[2] * qb.data_[3];
    Scalar y = qa.data_[0] * qb.data_[2] + qa.data_[2] * qb.data_[0] +
               qa.data_[3] * qb.data_[1] - qa.data_[1] * qb.data_[3];
    Scalar z = qa.data_[0] * qb.data_[3] + qa.data_[3] * qb.data_[0] -
               qa.data_[2] * qb.data_[1] + qa.data_[1] * qb.data_[2];
    data_[0] = w;
    data_[1] = x;
    data_[2] = y;
    data_[3] = z;
  }

  void ToEulerAngle(Scalar &roll, Scalar &pitch, Scalar &yaw) {
    const Scalar sinRCosP = 2.0 * (data_[0] * data_[1] + data_[2] * data_[3]);
    const Scalar cosRCosP =
        1.0 - 2.0 * (data_[1] * data_[1] + data_[2] * data_[2]);
    roll = std::atan2(sinRCosP, cosRCosP);

    // pitch (y-axis rotation)
    const Scalar sinP = 2.0 * (data_[0] * data_[2] - data_[3] * data_[1]);
    if (std::abs(sinP) >= 1.0)
      pitch = std::copysign(M_PI / 2.0, sinP); // use 90 degrees if out of range
    else
      pitch = std::asin(sinP);

    // yaw (z-axis rotation)
    Scalar sinYCosP = 2.0 * (data_[0] * data_[3] + data_[1] * data_[2]);
    Scalar cosYCosP = 1.0 - 2.0 * (data_[2] * data_[2] + data_[3] * data_[3]);
    yaw = std::atan2(sinYCosP, cosYCosP);
  }

  Quaternion &operator=(const Quaternion &q) {
    data_[0] = q.data_[0];
    data_[1] = q.data_[1];
    data_[2] = q.data_[2];
    data_[3] = q.data_[3];

    return *this;
  }

  void Conjugate() {
    Quaternion q = *this;
    for (int i = 1; i < 4; i++) {
      q.data_[i] = -q.data_[i];
    }
  }

  Math::Vec3 Rotate(const Math::Vec3 &vec) {
    Scalar e0e0 = data_[0] * data_[0];
    Scalar e1e1 = data_[1] * data_[1];
    Scalar e2e2 = data_[2] * data_[2];
    Scalar e3e3 = data_[3] * data_[3];
    Scalar e0e1 = data_[0] * data_[1];
    Scalar e0e2 = data_[0] * data_[2];
    Scalar e0e3 = data_[0] * data_[3];
    Scalar e1e2 = data_[1] * data_[2];
    Scalar e1e3 = data_[1] * data_[3];
    Scalar e2e3 = data_[2] * data_[3];
    return Math::Vec3(
        ((e0e0 + e1e1) * 2.0 - 1.0) * vec[0] + ((e1e2 - e0e3) * 2.0) * vec[1] +
            ((e1e3 + e0e2) * 2.0) * vec[2],
        ((e1e2 + e0e3) * 2.0) * vec[0] + ((e0e0 + e2e2) * 2.0 - 1.0) * vec[1] +
            ((e2e3 - e0e1) * 2) * vec[2],
        ((e1e3 - e0e2) * 2.0) * vec[0] + ((e2e3 + e0e1) * 2.0) * vec[1] +
            ((e0e0 + e3e3) * 2.0 - 1.0) * vec[2]);
  }

  Math::Vec3 RotateBack(const Math::Vec3 &vec) {
    Scalar e0e0 = +data_[0] * data_[0];
    Scalar e1e1 = +data_[1] * data_[1];
    Scalar e2e2 = +data_[2] * data_[2];
    Scalar e3e3 = +data_[3] * data_[3];
    Scalar e0e1 = -data_[0] * data_[1];
    Scalar e0e2 = -data_[0] * data_[2];
    Scalar e0e3 = -data_[0] * data_[3];
    Scalar e1e2 = +data_[1] * data_[2];
    Scalar e1e3 = +data_[1] * data_[3];
    Scalar e2e3 = +data_[2] * data_[3];
    return Math::Vec3(
        ((e0e0 + e1e1) * 2.0 - 1.0) * vec[0] + ((e1e2 - e0e3) * 2.0) * vec[1] +
            ((e1e3 + e0e2) * 2.0) * vec[2],
        ((e1e2 + e0e3) * 2.0) * vec[0] + ((e0e0 + e2e2) * 2.0 - 1.0) * vec[1] +
            ((e2e3 - e0e1) * 2.0) * vec[2],
        ((e1e3 - e0e2) * 2.0) * vec[0] + ((e2e3 + e0e1) * 2.0) * vec[1] +
            ((e0e0 + e3e3) * 2.0 - 1.0) * vec[2]);
  }

  void RotateByAngle(Math::Vec3 &omega, Quaternion &q) {
    Scalar theta = omega.Mag();
    Quaternion deltaQ(omega * (1.0 / theta), theta);
    this->Cross(q, deltaQ);
  }
};
} // namespace Dynamics

#endif