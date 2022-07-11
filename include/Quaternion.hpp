#ifndef _Quaternion_Hpp_
#define _Quaternion_Hpp_

#include <Vec3.hpp>
#include <iostream>

class quaternion {
private:
  double data_[4];

public:
  // default constructor
  quaternion() {
    data_[0] = 0.0;
    data_[1] = 0.0;
    data_[2] = 0.0;
    data_[3] = 0.0;
  }

  quaternion(const quaternion &q) {
    data_[0] = q.data_[0];
    data_[1] = q.data_[1];
    data_[2] = q.data_[2];
    data_[3] = q.data_[3];
  }

  // constructor from a vector, means the scalar part would be zero
  quaternion(Vec3 vec) {
    data_[0] = 0.0;
    data_[1] = vec[0];
    data_[2] = vec[1];
    data_[3] = vec[2];
  }

  // constructor from a rotation axis and rotation angle theta
  quaternion(Vec3 omega, const double theta) {
    double norm = omega.mag();
    omega = omega * (1.0 / norm);
    double h = theta * norm;
    data_[0] = cos(h / 2.0);
    data_[1] = sin(h / 2.0) * omega[0];
    data_[2] = sin(h / 2.0) * omega[1];
    data_[3] = sin(h / 2.0) * omega[2];
  }

  // constructor from euler angle, theta1, 2, 3, sequence - x, y, z
  quaternion(double roll, double pitch, double yaw) {
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);

    data_[0] = cr * cp * cy + sr * sp * sy;
    data_[1] = sr * cp * cy - cr * sp * sy;
    data_[2] = cr * sp * cy + sr * cp * sy;
    data_[3] = cr * cp * sy - sr * sp * cy;
  }

  quaternion product(quaternion &q) {
    quaternion res;

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

  const double Q0() const { return data_[0]; }
  const double Q1() const { return data_[1]; }
  const double Q2() const { return data_[2]; }
  const double Q3() const { return data_[3]; }

  void SetQ0(const double q0) { data_[0] = q0; }
  void SetQ1(const double q1) { data_[1] = q1; }
  void SetQ2(const double q2) { data_[2] = q2; }
  void SetQ3(const double q3) { data_[3] = q3; }

  void Cross(const quaternion &qa, const quaternion &qb) {
    double w = qa.data_[0] * qb.data_[0] - qa.data_[1] * qb.data_[1] -
               qa.data_[2] * qb.data_[2] - qa.data_[3] * qb.data_[3];
    double x = qa.data_[0] * qb.data_[1] + qa.data_[1] * qb.data_[0] -
               qa.data_[3] * qb.data_[2] + qa.data_[2] * qb.data_[3];
    double y = qa.data_[0] * qb.data_[2] + qa.data_[2] * qb.data_[0] +
               qa.data_[3] * qb.data_[1] - qa.data_[1] * qb.data_[3];
    double z = qa.data_[0] * qb.data_[3] + qa.data_[3] * qb.data_[0] -
               qa.data_[2] * qb.data_[1] + qa.data_[1] * qb.data_[2];
    data_[0] = w;
    data_[1] = x;
    data_[2] = y;
    data_[3] = z;
  }

  void ToEulerAngle(double &roll, double &pitch, double &yaw) {
    double sinr_cosp = 2.0 * (data_[0] * data_[1] + data_[2] * data_[3]);
    double cosr_cosp = 1.0 - 2.0 * (data_[1] * data_[1] + data_[2] * data_[2]);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2.0 * (data_[0] * data_[2] - data_[3] * data_[1]);
    if (std::abs(sinp) >= 1.0)
      pitch = std::copysign(M_PI / 2.0, sinp); // use 90 degrees if out of range
    else
      pitch = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2.0 * (data_[0] * data_[3] + data_[1] * data_[2]);
    double cosy_cosp = 1.0 - 2.0 * (data_[2] * data_[2] + data_[3] * data_[3]);
    yaw = std::atan2(siny_cosp, cosy_cosp);
  }

  quaternion &operator=(const quaternion &q) {
    data_[0] = q.data_[0];
    data_[1] = q.data_[1];
    data_[2] = q.data_[2];
    data_[3] = q.data_[3];

    return *this;
  }

  void Conjugate() {
    quaternion q = *this;
    for (int i = 1; i < 4; i++) {
      q.data_[i] = -q.data_[i];
    }
  }

  Vec3 Rotate(const Vec3 &vec) {
    double e0e0 = data_[0] * data_[0];
    double e1e1 = data_[1] * data_[1];
    double e2e2 = data_[2] * data_[2];
    double e3e3 = data_[3] * data_[3];
    double e0e1 = data_[0] * data_[1];
    double e0e2 = data_[0] * data_[2];
    double e0e3 = data_[0] * data_[3];
    double e1e2 = data_[1] * data_[2];
    double e1e3 = data_[1] * data_[3];
    double e2e3 = data_[2] * data_[3];
    return Vec3(
        ((e0e0 + e1e1) * 2.0 - 1.0) * vec[0] + ((e1e2 - e0e3) * 2.0) * vec[1] +
            ((e1e3 + e0e2) * 2.0) * vec[2],
        ((e1e2 + e0e3) * 2.0) * vec[0] + ((e0e0 + e2e2) * 2.0 - 1.0) * vec[1] +
            ((e2e3 - e0e1) * 2) * vec[2],
        ((e1e3 - e0e2) * 2.0) * vec[0] + ((e2e3 + e0e1) * 2.0) * vec[1] +
            ((e0e0 + e3e3) * 2.0 - 1.0) * vec[2]);
  }

  Vec3 RotateBack(const Vec3 &vec) {
    double e0e0 = +data_[0] * data_[0];
    double e1e1 = +data_[1] * data_[1];
    double e2e2 = +data_[2] * data_[2];
    double e3e3 = +data_[3] * data_[3];
    double e0e1 = -data_[0] * data_[1];
    double e0e2 = -data_[0] * data_[2];
    double e0e3 = -data_[0] * data_[3];
    double e1e2 = +data_[1] * data_[2];
    double e1e3 = +data_[1] * data_[3];
    double e2e3 = +data_[2] * data_[3];
    return Vec3(
        ((e0e0 + e1e1) * 2.0 - 1.0) * vec[0] + ((e1e2 - e0e3) * 2.0) * vec[1] +
            ((e1e3 + e0e2) * 2.0) * vec[2],
        ((e1e2 + e0e3) * 2.0) * vec[0] + ((e0e0 + e2e2) * 2.0 - 1.0) * vec[1] +
            ((e2e3 - e0e1) * 2.0) * vec[2],
        ((e1e3 - e0e2) * 2.0) * vec[0] + ((e2e3 + e0e1) * 2.0) * vec[1] +
            ((e0e0 + e3e3) * 2.0 - 1.0) * vec[2]);
  }

  void RotateByAngle(Vec3 &omega, quaternion &q) {
    double theta = omega.mag();
    quaternion deltaQ(omega * (1.0 / theta), theta);
    this->Cross(q, deltaQ);
  }
};

#endif