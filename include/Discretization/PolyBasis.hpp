#ifndef _Discretization_PolyBasis_Hpp_
#define _Discretization_PolyBasis_Hpp_

#include <Compadre_GMLS.hpp>

#include <vector>

namespace Discretization {
inline void PrepareScratchSpace(const int dimension, const double x,
                                const double y, const double z, const double h,
                                const int polyOrder, double *scratchSpace) {
  if (dimension == 2) {
    double *xOverH = scratchSpace;
    double *yOverH = scratchSpace + polyOrder + 1;

    xOverH[0] = 1;
    yOverH[0] = 1;

    for (int p = 1; p <= polyOrder; ++p) {
      xOverH[p] = xOverH[p - 1] * (x / h);
      yOverH[p] = yOverH[p - 1] * (y / h);
    }
  }

  if (dimension == 3) {
    double *xOverH = scratchSpace;
    double *yOverH = scratchSpace + polyOrder + 1;
    double *zOverH = scratchSpace + 2 * (polyOrder + 1);

    xOverH[0] = 1;
    yOverH[0] = 1;
    zOverH[0] = 1;

    for (int p = 1; p <= polyOrder; ++p) {
      xOverH[p] = xOverH[p - 1] * (x / h);
      yOverH[p] = yOverH[p - 1] * (y / h);
      zOverH[p] = zOverH[p - 1] * (z / h);
    }
  }
}

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

inline int GetSize(const int degree, const int dimension) {
  if (dimension == 3)
    return (degree + 1) * (degree + 2) * (degree + 3) / 6;
  else if (dimension == 2)
    return (degree + 1) * (degree + 2) / 2;
  else
    return degree + 1;
}

inline double CalDivFreeGrad(
    const int outputAxes1, const int outputAxes2, const int dimension,
    const int polyOrder, const double h, double *scratchSpace,
    Kokkos::View<double **, Kokkos::DefaultHostExecutionSpace> &coeff,
    const unsigned int coeffIndex) {
  double grad = 0.0;
  double weightOfNewValue = 1.0;

  int startingOrder = 0;

  const double factorial[] = {
      1,     1,      2,       6,        24,        120,        720,        5040,
      40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200};

  if (dimension == 3) {
    double *xOverH = scratchSpace;
    double *yOverH = scratchSpace + polyOrder + 1;
    double *zOverH = scratchSpace + 2 * (polyOrder + 1);

    int i = 0;
    for (int d = 0; d < dimension; ++d) {
      if ((d + 1) == dimension) {
        if (outputAxes1 == d) {
          // use 2D partial derivative of scalar basis definition
          // (in 2D) \sum_{n=0}^{n=P} \sum_{k=0}^{k=n} (x/h)^(n-k)*(y/h)^k /
          // ((n-k)!k!)
          int alphaX, alphaY;
          double alphaF;
          for (int n = startingOrder; n <= polyOrder; n++) {
            for (alphaY = 0; alphaY <= n; alphaY++) {
              alphaX = n - alphaY;

              int varPow[3] = {alphaX, alphaY, 0};
              varPow[outputAxes2]--;

              if (varPow[0] < 0 || varPow[1] < 0 || varPow[2] < 0) {
                grad += coeff(coeffIndex, i) * weightOfNewValue * 0.0;
              } else {
                alphaF = factorial[varPow[0]] * factorial[varPow[1]];
                grad += coeff(coeffIndex, i) * weightOfNewValue * 1. / h *
                        xOverH[varPow[0]] * yOverH[varPow[1]] / alphaF;
              }
              i++;
            }
          }
        } else {
          for (int j = 0; j < GetSize(polyOrder, dimension - 1); ++j) {
            grad += coeff(coeffIndex, i) * weightOfNewValue * 0;
            i++;
          }
        }
      } else {
        if (outputAxes1 == d) {
          // use 3D partial derivative of scalar basis definition
          // (in 3D) \sum_{p=0}^{p=P} \sum_{k1+k2+k3=n}
          // (x/h)^k1*(y/h)^k2*(z/h)^k3 / (k1!k2!k3!)
          int alphaX, alphaY, alphaZ;
          double alphaF;
          int s = 0;
          for (int n = startingOrder; n <= polyOrder; n++) {
            for (alphaZ = 0; alphaZ <= n; alphaZ++) {
              s = n - alphaZ;
              for (alphaY = 0; alphaY <= s; alphaY++) {
                alphaX = s - alphaY;

                int varPow[3] = {alphaX, alphaY, alphaZ};
                varPow[outputAxes2]--;

                if (varPow[0] < 0 || varPow[1] < 0 || varPow[2] < 0) {
                  grad += coeff(coeffIndex, i) * weightOfNewValue * 0.0;
                } else {
                  alphaF = factorial[varPow[0]] * factorial[varPow[1]] *
                           factorial[varPow[2]];
                  grad += coeff(coeffIndex, i) * weightOfNewValue * 1. / h *
                          xOverH[varPow[0]] * yOverH[varPow[1]] *
                          zOverH[varPow[2]] / alphaF;
                }
                i++;
              }
            }
          }
        } else if (outputAxes1 == (d + 1) % 3) {
          for (int j = 0; j < GetSize(polyOrder, dimension); ++j) {
            grad += coeff(coeffIndex, i) * weightOfNewValue * 0;
            i++;
          }
        } else {
          // (in 3D)
          int alphaX, alphaY, alphaZ;
          double alphaF;
          int s = 0;
          for (int n = startingOrder; n <= polyOrder; n++) {
            for (alphaZ = 0; alphaZ <= n; alphaZ++) {
              s = n - alphaZ;
              for (alphaY = 0; alphaY <= s; alphaY++) {
                alphaX = s - alphaY;

                int varPow[3] = {alphaX, alphaY, alphaZ};
                varPow[d]--;

                if (varPow[0] < 0 || varPow[1] < 0 || varPow[2] < 0) {
                  grad += coeff(coeffIndex, i) * weightOfNewValue * 0.0;
                } else {
                  varPow[outputAxes1]++;
                  varPow[outputAxes2]--;
                  if (varPow[0] < 0 || varPow[1] < 0 || varPow[2] < 0) {
                    grad += coeff(coeffIndex, i) * weightOfNewValue * 0.0;
                  } else {
                    alphaF = factorial[varPow[0]] * factorial[varPow[1]] *
                             factorial[varPow[2]];
                    grad += coeff(coeffIndex, i) * weightOfNewValue * -1.0 / h *
                            xOverH[varPow[0]] * yOverH[varPow[1]] *
                            zOverH[varPow[2]] / alphaF;
                  }
                }
                i++;
              }
            }
          }
        }
      }
    }
  } else if (dimension == 2) {
    double *xOverH = scratchSpace;
    double *yOverH = scratchSpace + polyOrder + 1;

    int i = 0;
    for (int d = 0; d < dimension; ++d) {
      if ((d + 1) == dimension) {
        if (outputAxes1 == d) {
          // use 1D partial derivative of scalar basis definition
          int alphaX;
          double alphaF;
          for (int n = startingOrder; n <= polyOrder; n++) {
            alphaX = n;

            int varPow[2] = {alphaX, 0};
            varPow[outputAxes2]--;

            if (varPow[0] < 0 || varPow[1] < 0) {
              grad += coeff(coeffIndex, i) * weightOfNewValue * 0.0;
            } else {
              alphaF = factorial[varPow[0]];
              grad += coeff(coeffIndex, i) * weightOfNewValue * 1. / h *
                      xOverH[varPow[0]] / alphaF;
            }
            i++;
          }
        } else {
          for (int j = 0; j < GetSize(polyOrder, dimension - 1); ++j) {
            grad += coeff(coeffIndex, i) * weightOfNewValue * 0;
            i++;
          }
        }
      } else {
        if (outputAxes1 == d) {
          // use 2D partial derivative of scalar basis definition
          int alphaX, alphaY;
          double alphaF;
          for (int n = startingOrder; n <= polyOrder; n++) {
            for (alphaY = 0; alphaY <= n; alphaY++) {
              alphaX = n - alphaY;

              int varPow[2] = {alphaX, alphaY};
              varPow[outputAxes2]--;

              if (varPow[0] < 0 || varPow[1] < 0) {
                grad += coeff(coeffIndex, i) * weightOfNewValue * 0.0;
              } else {
                alphaF = factorial[varPow[0]] * factorial[varPow[1]];
                grad += coeff(coeffIndex, i) * weightOfNewValue * 1. / h *
                        xOverH[varPow[0]] * yOverH[varPow[1]] / alphaF;
              }
              i++;
            }
          }
        } else {
          // (in 2D)
          int alphaX, alphaY;
          double alphaF;
          for (int n = startingOrder; n <= polyOrder; n++) {
            for (alphaY = 0; alphaY <= n; alphaY++) {
              alphaX = n - alphaY;

              int varPow[2] = {alphaX, alphaY};
              varPow[d]--;

              if (varPow[0] < 0 || varPow[1] < 0) {
                grad += coeff(coeffIndex, i) * weightOfNewValue * 0.0;
              } else {
                varPow[outputAxes1]++;
                varPow[outputAxes2]--;
                if (varPow[0] < 0 || varPow[1] < 0) {
                  grad += coeff(coeffIndex, i) * weightOfNewValue * 0.0;
                } else {
                  alphaF = factorial[varPow[0]] * factorial[varPow[1]];
                  grad += coeff(coeffIndex, i) * weightOfNewValue * -1.0 / h *
                          xOverH[varPow[0]] * yOverH[varPow[1]] / alphaF;
                }
              }
              i++;
            }
          }
        }
      }
    }
  }

  return grad;
}
} // namespace Discretization

#endif