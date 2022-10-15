#ifndef _Discretization_PolyBasis_Hpp_
#define _Discretization_PolyBasis_Hpp_

#include <Compadre_GMLS.hpp>

#include <vector>

namespace Discretization {
inline double
CalScalar(const int dimension, const double x, const double y, const double z,
          const int polyOrder, const double h,
          Kokkos::View<double *, Kokkos::DefaultHostExecutionSpace> coeff) {
  double scalar = 0.0;
  const double factorial[] = {
      1,     1,      2,       6,        24,        120,        720,        5040,
      40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200};
  if (dimension == 3) {
    std::vector<double> xOverH(polyOrder + 1), yOverH(polyOrder + 1),
        zOverH(polyOrder + 1);
    xOverH[0] = 1;
    yOverH[0] = 1;
    zOverH[0] = 1;
    for (int i = 1; i <= polyOrder; ++i) {
      xOverH[i] = xOverH[i - 1] * (x / h);
      yOverH[i] = yOverH[i - 1] * (y / h);
      zOverH[i] = zOverH[i - 1] * (z / h);
    }

    int alphaX, alphaY, alphaZ;
    double alphaF;
    int i = 0, s = 0;
    for (int n = 0; n <= polyOrder; n++) {
      for (alphaZ = 0; alphaZ <= n; alphaZ++) {
        s = n - alphaZ;
        for (alphaY = 0; alphaY <= s; alphaY++) {
          alphaX = s - alphaY;
          alphaF = factorial[alphaX] * factorial[alphaY] * factorial[alphaZ];
          scalar += xOverH[alphaX] * yOverH[alphaY] * zOverH[alphaZ] / alphaF *
                    coeff(i);
          i++;
        }
      }
    }
  } else if (dimension == 2) {
    std::vector<double> xOverH(polyOrder + 1), yOverH(polyOrder + 1);
    xOverH[0] = 1;
    yOverH[0] = 1;
    for (int i = 1; i <= polyOrder; ++i) {
      xOverH[i] = xOverH[i - 1] * (x / h);
      yOverH[i] = yOverH[i - 1] * (y / h);
    }

    int alphaX, alphaY;
    double alphaF;
    int i = 0;
    for (int n = 0; n <= polyOrder; n++) {
      for (alphaY = 0; alphaY <= n; alphaY++) {
        alphaX = n - alphaY;
        alphaF = factorial[alphaX] * factorial[alphaY];
        scalar += xOverH[alphaX] * yOverH[alphaY] / alphaF * coeff(i);
        i++;
      }
    }
  } else {
    std::vector<double> xOverH(polyOrder + 1);
    xOverH[0] = 1;
    for (int i = 1; i <= polyOrder; ++i) {
      xOverH[i] = xOverH[i - 1] * (x / h);
    }
    int alphaX;
    double alphaF;
    int i = 0;
    for (int n = 0; n <= polyOrder; n++) {
      alphaX = n;
      alphaF = factorial[alphaX];
      scalar += xOverH[alphaX] / alphaF * coeff(i);
      i++;
    }
  }

  return scalar;
}

inline double
CalScalarGrad(const int outputAxes, const int dimension, const double x,
              const double y, const double z, const int polyOrder,
              const double h,
              Kokkos::View<double *, Kokkos::DefaultHostExecutionSpace> coeff) {
  double grad = 0.0;
  const double factorial[] = {
      1,     1,      2,       6,        24,        120,        720,        5040,
      40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200};
  if (dimension == 3) {
    std::vector<double> xOverH(polyOrder + 1), yOverH(polyOrder + 1),
        zOverH(polyOrder + 1);
    xOverH[0] = 1;
    yOverH[0] = 1;
    zOverH[0] = 1;
    for (int i = 1; i <= polyOrder; ++i) {
      xOverH[i] = xOverH[i - 1] * (x / h);
      yOverH[i] = yOverH[i - 1] * (y / h);
      zOverH[i] = zOverH[i - 1] * (z / h);
    }

    int alphaX, alphaY, alphaZ;
    double alphaF;
    int i = 0, s = 0;
    for (int n = 0; n <= polyOrder; n++) {
      for (alphaZ = 0; alphaZ <= n; alphaZ++) {
        s = n - alphaZ;
        for (alphaY = 0; alphaY <= s; alphaY++) {
          alphaX = s - alphaY;

          int varPow[3] = {alphaX, alphaY, alphaZ};
          varPow[outputAxes]--;

          if (varPow[0] < 0 || varPow[1] < 0 || varPow[2] < 0) {
          } else {
            alphaF = factorial[varPow[0]] * factorial[varPow[1]] *
                     factorial[varPow[2]];
            grad += 1. / h * xOverH[varPow[0]] * yOverH[varPow[1]] *
                    zOverH[varPow[2]] / alphaF * coeff(i);
          }
          i++;
        }
      }
    }
  } else if (dimension == 2) {
    std::vector<double> xOverH(polyOrder + 1), yOverH(polyOrder + 1);
    xOverH[0] = 1;
    yOverH[0] = 1;
    for (int i = 1; i <= polyOrder; ++i) {
      xOverH[i] = xOverH[i - 1] * (x / h);
      yOverH[i] = yOverH[i - 1] * (y / h);
    }

    int alphaX, alphaY;
    double alphaF;
    int i = 0;
    for (int n = 0; n <= polyOrder; n++) {
      for (alphaY = 0; alphaY <= n; alphaY++) {
        alphaX = n - alphaY;

        int varPow[2] = {alphaX, alphaY};
        varPow[outputAxes]--;

        if (varPow[0] < 0 || varPow[1] < 0) {
        } else {
          alphaF = factorial[varPow[0]] * factorial[varPow[1]];
          grad += 1. / h * xOverH[varPow[0]] * yOverH[varPow[1]] / alphaF *
                  coeff(i);
        }
        i++;
      }
    }
  } else {
    std::vector<double> xOverH(polyOrder + 1);
    xOverH[0] = 1;
    for (int i = 1; i <= polyOrder; ++i) {
      xOverH[i] = xOverH[i - 1] * (x / h);
    }
    int alphaX;
    double alphaF;
    int i = 0;
    for (int n = 0; n <= polyOrder; n++) {
      alphaX = n;

      int varPow[1] = {alphaX};
      varPow[outputAxes]--;

      if (varPow[0] < 0) {
      } else {
        alphaF = factorial[varPow[0]];
        grad += 1. / h * xOverH[varPow[0]] / alphaF * coeff(i);
      }
      i++;
    }
  }

  return grad;
}
} // namespace Discretization

#endif