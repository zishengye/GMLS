#ifndef _Vec_Hpp_
#define _Vec_Hpp_

#include <cmath>
#include <fstream>
#include <vector>

template <class T> class Triple {
  T data[3];

public:
  Triple() {
    data[0] = 0;
    data[1] = 0;
    data[2] = 0;
  }

  Triple(T first, T second, T third) {
    data[0] = first;
    data[1] = second;
    data[2] = third;
  }

  Triple(const Triple<T> &t) {
    data[0] = t.data[0];
    data[1] = t.data[1];
    data[2] = t.data[2];
  }

  T &operator[](int i) { return data[i]; }

  const T operator[](const int i) const { return data[i]; }

  void operator+=(const Triple<T> y) {
    data[0] += y[0];
    data[1] += y[1];
    data[2] += y[2];
  }

  void operator-=(const Triple<T> y) {
    data[0] -= y[0];
    data[1] -= y[1];
    data[2] -= y[2];
  }

  void operator=(const Triple<T> y) {
    data[0] = y[0];
    data[1] = y[1];
    data[2] = y[2];
  }

  void operator*=(const double a) {
    data[0] *= a;
    data[1] *= a;
    data[2] *= a;
  }

  bool operator>(Triple<T> y) {
    return ((data[0] > y[0]) || (data[1] > y[1]) || (data[2] > y[2]));
  }

  bool operator<(Triple<T> y) {
    return ((data[0] < y[0]) || (data[1] < y[1]) || (data[2] < y[2]));
  }

  Triple<T> operator-(Triple<T> y) const {
    return Triple<T>(data[0] - y[0], data[1] - y[1], data[2] - y[2]);
  }

  Triple<T> operator*(double a) {
    return Triple<T>(a * data[0], a * data[1], a * data[2]);
  }

  double mag() const {
    return sqrt(data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
  }

  double cdot(Triple<T> y) {
    return y[0] * data[0] + y[1] * data[1] + y[2] * data[2];
  }
};

template <class T> Triple<T> operator+(Triple<T> xScalar, Triple<T> y) {
  return Triple<T>(xScalar[0] + y[0], xScalar[1] + y[1], xScalar[2] + y[2]);
}

template <class T> double maxMag(std::vector<Triple<T>> maxOf) {
  double maxM = 0.0;
  for (unsigned int i = 0; i < maxOf.size(); i++) {
    maxM = MAX(maxM, maxOf[i].mag());
  }
  return maxM;
}

typedef Triple<double> Vec3;

#endif