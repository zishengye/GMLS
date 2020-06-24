#pragma once

#include <fstream>
#include <vector>

template <class T> class triple {
  T data[3];

public:
  triple() {
    data[0] = 0;
    data[1] = 0;
    data[2] = 0;
  }

  triple(T first, T second, T third) {
    data[0] = first;
    data[1] = second;
    data[2] = third;
  }

  T &operator[](int i) { return data[i]; }

  const T operator[](const int i) const { return data[i]; }

  void operator+=(triple<T> y) {
    data[0] += y[0];
    data[1] += y[1];
    data[2] += y[2];
  }

  void operator-=(triple<T> y) {
    data[0] -= y[0];
    data[1] -= y[1];
    data[2] -= y[2];
  }

  void operator=(triple<T> y) {
    data[0] = y[0];
    data[1] = y[1];
    data[2] = y[2];
  }

  void operator*=(double a) {
    data[0] *= a;
    data[1] *= a;
    data[2] *= a;
  }

  bool operator>(triple<T> y) {
    return ((data[0] > y[0]) || (data[1] > y[1]) || (data[2] > y[2]));
  }

  bool operator<(triple<T> y) {
    return ((data[0] < y[0]) || (data[1] < y[1]) || (data[2] < y[2]));
  }

  const triple<T> operator-(triple<T> y) const {
    return triple<T>(data[0] - y[0], data[1] - y[1], data[2] - y[2]);
  }

  triple<T> operator*(double a) {
    return triple<T>(a * data[0], a * data[1], a * data[2]);
  }

  double mag() {
    return sqrt(data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
  }

  double cdot(triple<T> y) {
    return y[0] * data[0] + y[1] * data[1] + y[2] * data[2];
  }
};

template <class T> triple<T> operator+(triple<T> xScalar, triple<T> y) {
  return triple<T>(xScalar[0] + y[0], xScalar[1] + y[1], xScalar[2] + y[2]);
}

template <class T> double maxmag(std::vector<triple<T>> maxof) {
  double maxm = 0.0;
  for (int i = 0; i < maxof.size(); i++) {
    maxm = MAX(maxm, maxof[i].mag());
  }
  return maxm;
}

typedef triple<double> vec3;