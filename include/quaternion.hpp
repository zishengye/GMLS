#ifndef _QUATERNION_HPP_
#define _QUATERNION_HPP_

#include <vec3.hpp>

class quaternion {
private:
  double m_data[4];

public:
  // default constructor
  quaternion() {}

  // constructor from a vector, means the scalar part would be zero
  quaternion(vec3 vec) {
    m_data[0] = 0.0;
    m_data[1] = vec[0];
    m_data[2] = vec[1];
    m_data[3] = vec[2];
  }

  // constructor from a rotation axis and rotation angle theta
  quaternion(vec3 omega, const double theta) {
    m_data[0] = cos(theta / 2.0);
    m_data[1] = sin(theta / 2.0) * omega[0];
    m_data[2] = sin(theta / 2.0) * omega[1];
    m_data[3] = sin(theta / 2.0) * omega[2];
  }

  // constructor from euler angle, theta1, 2, 3, sequence - x, y, z
  quaternion(double roll, double pitch, double yaw) {
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);

    m_data[0] = cr * cp * cy + sr * sp * sy;
    m_data[1] = sr * cp * cy - cr * sp * sy;
    m_data[2] = cr * sp * cy + sr * cp * sy;
    m_data[3] = cr * cp * sy - sr * sp * cy;
  }

  quaternion product(quaternion &q) {
    quaternion res;

    res.m_data[0] = m_data[0] * q.m_data[0] - m_data[1] * q.m_data[1] -
                    m_data[2] * q.m_data[2] - m_data[3] * q.m_data[3];
    res.m_data[1] = m_data[0] * q.m_data[1] + m_data[1] * q.m_data[0] +
                    m_data[2] * q.m_data[3] - m_data[3] * q.m_data[2];
    res.m_data[2] = m_data[0] * q.m_data[2] - m_data[1] * q.m_data[3] +
                    m_data[2] * q.m_data[0] + m_data[3] * q.m_data[1];
    res.m_data[3] = m_data[0] * q.m_data[3] + m_data[1] * q.m_data[2] -
                    m_data[2] * q.m_data[1] + m_data[3] * q.m_data[0];

    return res;
  }

  const double q0() const { return m_data[0]; }
  const double q1() const { return m_data[1]; }
  const double q2() const { return m_data[2]; }
  const double q3() const { return m_data[3]; }

  void set_q0(const double q0) { m_data[0] = q0; }
  void set_q1(const double q1) { m_data[1] = q1; }
  void set_q2(const double q2) { m_data[2] = q2; }
  void set_q3(const double q3) { m_data[3] = q3; }

  void Cross(const quaternion &qa, const quaternion &qb) {
    double w = qa.m_data[0] * qb.m_data[0] - qa.m_data[1] * qb.m_data[1] -
               qa.m_data[2] * qb.m_data[2] - qa.m_data[3] * qb.m_data[3];
    double x = qa.m_data[0] * qb.m_data[1] + qa.m_data[1] * qb.m_data[0] -
               qa.m_data[3] * qb.m_data[2] + qa.m_data[2] * qb.m_data[3];
    double y = qa.m_data[0] * qb.m_data[2] + qa.m_data[2] * qb.m_data[0] +
               qa.m_data[3] * qb.m_data[1] - qa.m_data[1] * qb.m_data[3];
    double z = qa.m_data[0] * qb.m_data[3] + qa.m_data[3] * qb.m_data[0] -
               qa.m_data[2] * qb.m_data[1] + qa.m_data[1] * qb.m_data[2];
    m_data[0] = w;
    m_data[1] = x;
    m_data[2] = y;
    m_data[3] = z;
  }

  void to_euler_angles(double &roll, double &pitch, double &yaw) {
    double sinr_cosp = 2.0 * (m_data[0] * m_data[1] + m_data[2] * m_data[3]);
    double cosr_cosp =
        1.0 - 2.0 * (m_data[1] * m_data[1] + m_data[2] * m_data[2]);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2.0 * (m_data[0] * m_data[2] - m_data[3] * m_data[1]);
    if (std::abs(sinp) >= 1.0)
      pitch = std::copysign(M_PI / 2.0, sinp); // use 90 degrees if out of range
    else
      pitch = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2.0 * (m_data[0] * m_data[3] + m_data[1] * m_data[2]);
    double cosy_cosp =
        1.0 - 2.0 * (m_data[2] * m_data[2] + m_data[3] * m_data[3]);
    yaw = std::atan2(siny_cosp, cosy_cosp);
  }

  quaternion &operator=(quaternion &q) {
    for (int i = 0; i < 4; i++) {
      m_data[i] = q.m_data[i];
    }
  }

  void conjugate() {
    quaternion q = *this;
    for (int i = 1; i < 4; i++) {
      q.m_data[i] = -q.m_data[i];
    }
  }

  vec3 rotate(vec3 &vec) {
    double e0e0 = m_data[0] * m_data[0];
    double e1e1 = m_data[1] * m_data[1];
    double e2e2 = m_data[2] * m_data[2];
    double e3e3 = m_data[3] * m_data[3];
    double e0e1 = m_data[0] * m_data[1];
    double e0e2 = m_data[0] * m_data[2];
    double e0e3 = m_data[0] * m_data[3];
    double e1e2 = m_data[1] * m_data[2];
    double e1e3 = m_data[1] * m_data[3];
    double e2e3 = m_data[2] * m_data[3];
    return vec3(
        ((e0e0 + e1e1) * 2.0 - 1.0) * vec[0] + ((e1e2 - e0e3) * 2.0) * vec[1] +
            ((e1e3 + e0e2) * 2.0) * vec[2],
        ((e1e2 + e0e3) * 2.0) * vec[0] + ((e0e0 + e2e2) * 2.0 - 1.0) * vec[1] +
            ((e2e3 - e0e1) * 2) * vec[2],
        ((e1e3 - e0e2) * 2.0) * vec[0] + ((e2e3 + e0e1) * 2.0) * vec[1] +
            ((e0e0 + e3e3) * 2.0 - 1.0) * vec[2]);
  }

  vec3 rotate_back(vec3 &vec) {
    double e0e0 = +m_data[0] * m_data[0];
    double e1e1 = +m_data[1] * m_data[1];
    double e2e2 = +m_data[2] * m_data[2];
    double e3e3 = +m_data[3] * m_data[3];
    double e0e1 = -m_data[0] * m_data[1];
    double e0e2 = -m_data[0] * m_data[2];
    double e0e3 = -m_data[0] * m_data[3];
    double e1e2 = +m_data[1] * m_data[2];
    double e1e3 = +m_data[1] * m_data[3];
    double e2e3 = +m_data[2] * m_data[3];
    return vec3(
        ((e0e0 + e1e1) * 2.0 - 1.0) * vec[0] + ((e1e2 - e0e3) * 2.0) * vec[1] +
            ((e1e3 + e0e2) * 2.0) * vec[2],
        ((e1e2 + e0e3) * 2.0) * vec[0] + ((e0e0 + e2e2) * 2.0 - 1.0) * vec[1] +
            ((e2e3 - e0e1) * 2.0) * vec[2],
        ((e1e3 - e0e2) * 2.0) * vec[0] + ((e2e3 + e0e1) * 2.0) * vec[1] +
            ((e0e0 + e3e3) * 2.0 - 1.0) * vec[2]);
  }

  void rotate_by_Wabs(vec3 &omega, quaternion &q) {}
};

#endif