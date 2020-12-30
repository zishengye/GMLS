#ifndef _DIVERGENCEFREE_HPP_
#define _DIVERGENCEFREE_HPP_

#include <cmath>
#include <vector>

#include "vec3.hpp"

inline double calDivFreeBasisGrad(const int output_axes1,
                                  const int output_axes2, const double dx,
                                  const double dy, const double dz,
                                  const int polynomial_order,
                                  const double epsilon,
                                  const std::vector<double> &coeff) {
  double grad = 0.0;
  std::vector<double> epsilon_inverse(polynomial_order);

  for (int i = 0; i < polynomial_order; i++) {
    epsilon_inverse[i] = std::pow(epsilon, -(i + 1));
  }

  switch (output_axes1) {
  case 0:
    switch (output_axes2) {
    case 0: {
      // partial u partial x
      grad += coeff[9] * epsilon_inverse[0];
      grad += coeff[10] * epsilon_inverse[0];
      if (polynomial_order > 1) {
        grad += 2.0 * coeff[20] * dx * epsilon_inverse[1];
        grad += coeff[21] * dy * epsilon_inverse[1];
        grad += coeff[22] * dz * epsilon_inverse[1];
        grad += -2.0 * coeff[23] * dy * epsilon_inverse[1];
        grad += -2.0 * coeff[25] * dz * epsilon_inverse[1];
      }
      if (polynomial_order > 2) {
        grad += 3.0 * coeff[38] * dx * dx * epsilon_inverse[2];
        grad += 3.0 * coeff[39] * dx * dx * epsilon_inverse[2];
        grad += 2.0 * coeff[40] * dx * dy * epsilon_inverse[2];
        grad += 2.0 * coeff[41] * dx * dy * epsilon_inverse[2];
        grad += 2.0 * coeff[42] * dx * dz * epsilon_inverse[2];
        grad += 2.0 * coeff[43] * dx * dz * epsilon_inverse[2];
        grad += coeff[44] * dy * dy * epsilon_inverse[2];
        grad += coeff[45] * dy * dy * epsilon_inverse[2];
        grad += coeff[46] * dy * dz * epsilon_inverse[2];
        grad += coeff[47] * dy * dz * epsilon_inverse[2];
        grad += coeff[48] * dz * dz * epsilon_inverse[2];
        grad += coeff[49] * dz * dz * epsilon_inverse[2];
      }
      if (polynomial_order > 3) {
        grad += 4.0 * coeff[65] * dx * dx * dx * epsilon_inverse[3];
        grad += 4.0 * coeff[66] * dx * dx * dx * epsilon_inverse[3];
        grad += 3.0 * coeff[67] * dx * dx * dy * epsilon_inverse[3];
        grad += 3.0 * coeff[68] * dx * dx * dy * epsilon_inverse[3];
        grad += 3.0 * coeff[69] * dx * dx * dz * epsilon_inverse[3];
        grad += 3.0 * coeff[70] * dx * dx * dz * epsilon_inverse[3];
        grad += 2.0 * coeff[71] * dx * dy * dy * epsilon_inverse[3];
        grad += 2.0 * coeff[72] * dx * dy * dy * epsilon_inverse[3];
        grad += 2.0 * coeff[73] * dx * dy * dz * epsilon_inverse[3];
        grad += 2.0 * coeff[74] * dx * dy * dz * epsilon_inverse[3];
        grad += 2.0 * coeff[75] * dx * dz * dz * epsilon_inverse[3];
        grad += 2.0 * coeff[76] * dx * dz * dz * epsilon_inverse[3];
        grad += coeff[77] * dy * dy * dy * epsilon_inverse[3];
        grad += coeff[78] * dy * dy * dy * epsilon_inverse[3];
        grad += coeff[79] * dy * dy * dz * epsilon_inverse[3];
        grad += coeff[80] * dy * dy * dz * epsilon_inverse[3];
        grad += coeff[81] * dy * dz * dz * epsilon_inverse[3];
        grad += coeff[82] * dy * dz * dz * epsilon_inverse[3];
        grad += coeff[83] * dz * dz * dz * epsilon_inverse[3];
        grad += coeff[84] * dz * dz * dz * epsilon_inverse[3];
      }
    }

    break;

    case 1: {
      // partial u partial y
      grad += coeff[3] * epsilon_inverse[0];
      if (polynomial_order > 1) {
        grad += 2.0 * coeff[11] * dy * epsilon_inverse[1];
        grad += coeff[13] * dz * epsilon_inverse[1];
        grad += coeff[21] * dx * epsilon_inverse[1];
        grad += -2.0 * coeff[23] * dx * epsilon_inverse[1];
      }
      if (polynomial_order > 2) {
        grad += 3.0 * coeff[26] * dy * dy * epsilon_inverse[2];
        grad += 2.0 * coeff[27] * dy * dz * epsilon_inverse[2];
        grad += coeff[28] * dz * dz * epsilon_inverse[2];
        grad += coeff[40] * dx * dx * epsilon_inverse[2];
        grad += coeff[41] * dx * dx * epsilon_inverse[2];
        grad += 2.0 * coeff[44] * dx * dy * epsilon_inverse[2];
        grad += 2.0 * coeff[45] * dx * dy * epsilon_inverse[2];
        grad += coeff[46] * dx * dz * epsilon_inverse[2];
        grad += coeff[47] * dx * dz * epsilon_inverse[2];
      }
      if (polynomial_order > 3) {
        grad += 4.0 * coeff[50] * dy * dy * dy * epsilon_inverse[3];
        grad += 3.0 * coeff[51] * dy * dy * dz * epsilon_inverse[3];
        grad += 2.0 * coeff[52] * dy * dz * dz * epsilon_inverse[3];
        grad += coeff[53] * dz * dz * dz * epsilon_inverse[3];
        grad += coeff[67] * dx * dx * dx * epsilon_inverse[3];
        grad += coeff[68] * dx * dx * dx * epsilon_inverse[3];
        grad += 2.0 * coeff[71] * dx * dx * dy * epsilon_inverse[3];
        grad += 2.0 * coeff[72] * dx * dx * dy * epsilon_inverse[3];
        grad += coeff[73] * dx * dx * dz * epsilon_inverse[3];
        grad += coeff[74] * dx * dx * dz * epsilon_inverse[3];
        grad += 3.0 * coeff[77] * dx * dy * dy * epsilon_inverse[3];
        grad += 3.0 * coeff[78] * dx * dy * dy * epsilon_inverse[3];
        grad += 2.0 * coeff[79] * dx * dy * dz * epsilon_inverse[3];
        grad += 2.0 * coeff[80] * dx * dy * dz * epsilon_inverse[3];
        grad += coeff[81] * dx * dz * dz * epsilon_inverse[3];
        grad += coeff[82] * dx * dz * dz * epsilon_inverse[3];
      }
    }

    break;

    case 2: {
      // partial u partial z
      grad += coeff[4] * epsilon_inverse[0];
      if (polynomial_order > 1) {
        grad += 2.0 * coeff[12] * dz * epsilon_inverse[1];
        grad += coeff[13] * dy * epsilon_inverse[1];
        grad += coeff[22] * dx * epsilon_inverse[1];
        grad += -2.0 * coeff[25] * dx * epsilon_inverse[1];
      }
      if (polynomial_order > 2) {
      }
      if (polynomial_order > 3) {
      }
    }

    break;
    }

    break;

  case 1:
    switch (output_axes2) {
    case 0: {
      // partial v partial x
      grad += coeff[5] * epsilon_inverse[0];
      if (polynomial_order > 1) {
        grad += 2.0 * coeff[14] * dx * epsilon_inverse[1];
        grad += coeff[16] * dz * epsilon_inverse[1];
        grad += -2.0 * coeff[20] * dy * epsilon_inverse[1];
        grad += coeff[24] * dy * epsilon_inverse[1];
      }
      if (polynomial_order > 2) {
      }
      if (polynomial_order > 3) {
      }
    }

    break;

    case 1: {
      // partial v partial y
      grad += -coeff[9] * epsilon_inverse[0];
      if (polynomial_order > 1) {
        grad += -2.0 * coeff[20] * dx * epsilon_inverse[1];
        grad += -1.0 * coeff[22] * dz * epsilon_inverse[1];
        grad += 2.0 * coeff[23] * dy * epsilon_inverse[1];
        grad += coeff[24] * dx * epsilon_inverse[1];
      }
      if (polynomial_order > 2) {
      }
      if (polynomial_order > 3) {
      }
    }

    break;

    case 2: {
      // partial v partial z
      grad += coeff[6] * epsilon_inverse[0];
      if (polynomial_order > 1) {
        grad += 2.0 * coeff[15] * dz * epsilon_inverse[1];
        grad += coeff[16] * dx * epsilon_inverse[1];
        grad += -1.0 * coeff[22] * dy * epsilon_inverse[1];
      }
      if (polynomial_order > 2) {
      }
      if (polynomial_order > 3) {
      }
    }

    break;
    }

    break;

  case 2:
    switch (output_axes2) {
    case 0: {
      // partial w partial x
      grad += coeff[7] * epsilon_inverse[0];
      if (polynomial_order > 1) {
        grad += 2.0 * coeff[17] * dx * epsilon_inverse[1];
        grad += coeff[19] * dy * epsilon_inverse[1];
        grad += -1.0 * coeff[24] * dz * epsilon_inverse[1];
      }
      if (polynomial_order > 2) {
      }
      if (polynomial_order > 3) {
      }
    }

    break;

    case 1: {
      // partial w partial y
      grad += coeff[8] * epsilon_inverse[0];
      if (polynomial_order > 1) {
        grad += 2.0 * coeff[18] * dy * epsilon_inverse[1];
        grad += coeff[19] * dx * epsilon_inverse[1];
        grad += -1.0 * coeff[21] * dz * epsilon_inverse[1];
      }
      if (polynomial_order > 2) {
      }
      if (polynomial_order > 3) {
      }
    }

    break;

    case 2: {
      // partial w partial z
      grad += -1.0 * coeff[10] * epsilon_inverse[0];
      if (polynomial_order > 1) {
        grad += -1.0 * coeff[21] * dy * epsilon_inverse[1];
        grad += -1.0 * coeff[24] * dx * epsilon_inverse[1];
        grad += 2.0 * coeff[25] * dz * epsilon_inverse[1];
      }
      if (polynomial_order > 2) {
      }
      if (polynomial_order > 3) {
      }
    }

    break;
    }
    break;
  }

  return grad;
}

inline double calDivFreeBasisGrad(const int output_axes1,
                                  const int output_axes2, const double dx,
                                  const double dy, const int polynomial_order,
                                  const double epsilon,
                                  const std::vector<double> &coeff) {
  double grad = 0.0;
  std::vector<double> epsilon_inverse;
  epsilon_inverse.resize(polynomial_order);

  for (int i = 0; i < polynomial_order; i++) {
    epsilon_inverse[i] = std::pow(epsilon, -(i + 1));
  }

  switch (output_axes1) {
  case 0:
    switch (output_axes2) {
    case 0:
      grad += coeff[4] * epsilon_inverse[0];
      if (polynomial_order > 1) {
        grad += 2.0 * coeff[7] * dx * epsilon_inverse[1];
        grad += -2.0 * coeff[8] * dy * epsilon_inverse[1];
      }
      if (polynomial_order > 2) {
        grad += 3.0 * coeff[11] * dx * dx * epsilon_inverse[2];
        grad += 2.0 * coeff[12] * dx * dy * epsilon_inverse[2];
        grad += -3.0 * coeff[13] * dy * dy * epsilon_inverse[2];
      }
      if (polynomial_order > 3) {
        grad += 4.0 * coeff[16] * dx * dx * dx * epsilon_inverse[3];
        grad += 6.0 * coeff[17] * dx * dx * dy * epsilon_inverse[3];
        grad += -6.0 * coeff[18] * dx * dy * dy * epsilon_inverse[3];
        grad += -4.0 * coeff[19] * dy * dy * dy * epsilon_inverse[3];
      }
      break;

    case 1:
      grad += coeff[2] * epsilon_inverse[0];
      if (polynomial_order > 1) {
        grad += 2.0 * coeff[5] * dy * epsilon_inverse[1];
        grad += -2.0 * coeff[8] * dx * epsilon_inverse[1];
      }
      if (polynomial_order > 2) {
        grad += 3.0 * coeff[9] * dy * dy * epsilon_inverse[2];
        grad += coeff[12] * dx * dx * epsilon_inverse[2];
        grad += -6.0 * coeff[13] * dx * dy * epsilon_inverse[2];
      }
      if (polynomial_order > 3) {
        grad += 4.0 * coeff[14] * dy * dy * dy * epsilon_inverse[3];
        grad += 2.0 * coeff[17] * dx * dx * dx * epsilon_inverse[3];
        grad += -6.0 * coeff[18] * dx * dx * dy * epsilon_inverse[3];
        grad += -12.0 * coeff[19] * dx * dy * dy * epsilon_inverse[3];
      }
      break;
    }
    break;

  case 1:
    switch (output_axes2) {
    case 0:
      grad += coeff[3] * epsilon_inverse[0];
      if (polynomial_order > 1) {
        grad += 2.0 * coeff[6] * dx * epsilon_inverse[1];
        grad += -2.0 * coeff[7] * dy * epsilon_inverse[1];
      }
      if (polynomial_order > 2) {
        grad += 3.0 * coeff[10] * dx * dx * epsilon_inverse[2];
        grad += -6.0 * coeff[11] * dx * dy * epsilon_inverse[2];
        grad += -1.0 * coeff[12] * dy * dy * epsilon_inverse[2];
      }
      if (polynomial_order > 3) {
        grad += 4.0 * coeff[15] * dx * dx * dx * epsilon_inverse[3];
        grad += -12.0 * coeff[16] * dx * dx * dy * epsilon_inverse[3];
        grad += -6.0 * coeff[17] * dx * dy * dy * epsilon_inverse[3];
        grad += 2.0 * coeff[18] * dy * dy * dy * epsilon_inverse[3];
      }
      break;

    case 1:
      grad += -coeff[4] * epsilon_inverse[0];
      if (polynomial_order > 1) {
        grad += -2.0 * coeff[7] * dx * epsilon_inverse[1];
        grad += 2.0 * coeff[8] * dy * epsilon_inverse[1];
      }
      if (polynomial_order > 2) {
        grad += -3.0 * coeff[11] * dx * dx * epsilon_inverse[2];
        grad += -2.0 * coeff[12] * dx * dy * epsilon_inverse[2];
        grad += 3.0 * coeff[13] * dy * dy * epsilon_inverse[2];
      }
      if (polynomial_order > 3) {
        grad += -4.0 * coeff[16] * dx * dx * dx * epsilon_inverse[3];
        grad += -6.0 * coeff[17] * dx * dx * dy * epsilon_inverse[3];
        grad += 6.0 * coeff[18] * dx * dy * dy * epsilon_inverse[3];
        grad += 4.0 * coeff[19] * dy * dy * dy * epsilon_inverse[3];
      }
      break;
    }
    break;
  }

  return grad;
}

inline double calStaggeredScalarGrad(const int output_axes, const int dimension,
                                     vec3 dX, const int poly_order,
                                     const double epsilon,
                                     const std::vector<double> &coeff) {

  const double factorial[15] = {
      1,     1,      2,       6,        24,        120,        720,        5040,
      40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200};

  double grad = 0.0;

  int alphax, alphay, alphaz;
  double alphaf;
  double cutoff_p = epsilon;
  int i = 0;

  for (int n = 0; n <= poly_order; n++) {
    if (dimension == 3) {
      for (alphaz = 0; alphaz <= n; alphaz++) {

        int s = n - alphaz;
        for (alphay = 0; alphay <= s; alphay++) {
          alphax = s - alphay;

          int x_pow = (output_axes == 0) ? alphax - 1 : alphax;
          int y_pow = (output_axes == 1) ? alphay - 1 : alphay;
          int z_pow = (output_axes == 2) ? alphaz - 1 : alphaz;

          if (x_pow < 0 || y_pow < 0 || z_pow < 0) {
            grad += 0;
          } else {
            alphaf = factorial[x_pow] * factorial[y_pow] * factorial[z_pow];
            grad += 0.5 / cutoff_p * std::pow(dX[0] / cutoff_p, x_pow) *
                    std::pow(dX[1] / cutoff_p, y_pow) *
                    std::pow(dX[2] / cutoff_p, z_pow) / alphaf * coeff[i];
          }
          i++;
        }
      }
    } else if (dimension == 2) {
      for (alphay = 0; alphay <= n; alphay++) {
        alphax = n - alphay;

        int x_pow = (output_axes == 0) ? alphax - 1 : alphax;
        int y_pow = (output_axes == 1) ? alphay - 1 : alphay;

        if (x_pow < 0 || y_pow < 0) {
          grad += 0;
        } else {
          alphaf = factorial[x_pow] * factorial[y_pow];
          grad += 0.5 / cutoff_p * std::pow(dX[0] / cutoff_p, x_pow) *
                  std::pow(dX[1] / cutoff_p, y_pow) / alphaf * coeff[i];
        }
        i++;
      }
    } else { // dimension == 1
      alphax = n;

      int x_pow = (output_axes == 0) ? alphax - 1 : alphax;
      if (x_pow < 0) {
        grad += 0;
      } else {
        alphaf = factorial[x_pow];
        grad += 0.5 / cutoff_p * std::pow(dX[0] / cutoff_p, x_pow) / alphaf *
                coeff[i];
      }
      i++;
    }
  }

  return grad;
}

#endif