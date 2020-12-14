#ifndef _DIVERGENCEFREE_HPP_
#define _DIVERGENCEFREE_HPP_

#include <cmath>
#include <vector>

inline double calDivFreeBasisGrad(const int outputAxes1, const int outputAxes2,
                                  const double dx, const double dy,
                                  const double dz, const int polynomialOrder,
                                  const double epsilon,
                                  const std::vector<double> &coeff) {
  double grad = 0.0;
  std::vector<double> epsilonInverse(polynomialOrder);

  for (int i = 0; i < polynomialOrder; i++) {
    epsilonInverse[i] = std::pow(epsilon, -(i + 1));
  }

  switch (outputAxes1) {
  case 0:
    switch (outputAxes2) {
    case 0: {
      // partial u partial x
      grad += coeff[9] * epsilonInverse[0];
      grad += coeff[10] * epsilonInverse[0];
      if (polynomialOrder > 1) {
        grad += 2.0 * coeff[20] * dx * epsilonInverse[1];
        grad += coeff[21] * dy * epsilonInverse[1];
        grad += coeff[22] * dz * epsilonInverse[1];
        grad += -2.0 * coeff[23] * dy * epsilonInverse[1];
        grad += -2.0 * coeff[25] * dz * epsilonInverse[1];
      }
      if (polynomialOrder > 2) {
        grad += 3.0 * coeff[38] * dx * dx * epsilonInverse[2];
        grad += 3.0 * coeff[39] * dx * dx * epsilonInverse[2];
        grad += 2.0 * coeff[40] * dx * dy * epsilonInverse[2];
        grad += 2.0 * coeff[41] * dx * dy * epsilonInverse[2];
        grad += 2.0 * coeff[42] * dx * dz * epsilonInverse[2];
        grad += 2.0 * coeff[43] * dx * dz * epsilonInverse[2];
        grad += coeff[44] * dy * dy * epsilonInverse[2];
        grad += coeff[45] * dy * dy * epsilonInverse[2];
        grad += coeff[46] * dy * dz * epsilonInverse[2];
        grad += coeff[47] * dy * dz * epsilonInverse[2];
        grad += coeff[48] * dz * dz * epsilonInverse[2];
        grad += coeff[49] * dz * dz * epsilonInverse[2];
      }
      if (polynomialOrder > 3) {
        grad += 4.0 * coeff[65] * dx * dx * dx * epsilonInverse[3];
        grad += 4.0 * coeff[66] * dx * dx * dx * epsilonInverse[3];
        grad += 3.0 * coeff[67] * dx * dx * dy * epsilonInverse[3];
        grad += 3.0 * coeff[68] * dx * dx * dy * epsilonInverse[3];
        grad += 3.0 * coeff[69] * dx * dx * dz * epsilonInverse[3];
        grad += 3.0 * coeff[70] * dx * dx * dz * epsilonInverse[3];
        grad += 2.0 * coeff[71] * dx * dy * dy * epsilonInverse[3];
        grad += 2.0 * coeff[72] * dx * dy * dy * epsilonInverse[3];
        grad += 2.0 * coeff[73] * dx * dy * dz * epsilonInverse[3];
        grad += 2.0 * coeff[74] * dx * dy * dz * epsilonInverse[3];
        grad += 2.0 * coeff[75] * dx * dz * dz * epsilonInverse[3];
        grad += 2.0 * coeff[76] * dx * dz * dz * epsilonInverse[3];
        grad += coeff[77] * dy * dy * dy * epsilonInverse[3];
        grad += coeff[78] * dy * dy * dy * epsilonInverse[3];
        grad += coeff[79] * dy * dy * dz * epsilonInverse[3];
        grad += coeff[80] * dy * dy * dz * epsilonInverse[3];
        grad += coeff[81] * dy * dz * dz * epsilonInverse[3];
        grad += coeff[82] * dy * dz * dz * epsilonInverse[3];
        grad += coeff[83] * dz * dz * dz * epsilonInverse[3];
        grad += coeff[84] * dz * dz * dz * epsilonInverse[3];
      }
    }

    break;

    case 1: {
      // partial u partial y
      grad += coeff[3] * epsilonInverse[0];
      if (polynomialOrder > 1) {
        grad += 2.0 * coeff[11] * dy * epsilonInverse[1];
        grad += coeff[13] * dz * epsilonInverse[1];
        grad += coeff[21] * dx * epsilonInverse[1];
        grad += -2.0 * coeff[23] * dx * epsilonInverse[1];
      }
      if (polynomialOrder > 2) {
        grad += 3.0 * coeff[26] * dy * dy * epsilonInverse[2];
        grad += 2.0 * coeff[27] * dy * dz * epsilonInverse[2];
        grad += coeff[28] * dz * dz * epsilonInverse[2];
        grad += coeff[40] * dx * dx * epsilonInverse[2];
        grad += coeff[41] * dx * dx * epsilonInverse[2];
        grad += 2.0 * coeff[44] * dx * dy * epsilonInverse[2];
        grad += 2.0 * coeff[45] * dx * dy * epsilonInverse[2];
        grad += coeff[46] * dx * dz * epsilonInverse[2];
        grad += coeff[47] * dx * dz * epsilonInverse[2];
      }
      if (polynomialOrder > 3) {
        grad += 4.0 * coeff[50] * dy * dy * dy * epsilonInverse[3];
        grad += 3.0 * coeff[51] * dy * dy * dz * epsilonInverse[3];
        grad += 2.0 * coeff[52] * dy * dz * dz * epsilonInverse[3];
        grad += coeff[53] * dz * dz * dz * epsilonInverse[3];
        grad += coeff[67] * dx * dx * dx * epsilonInverse[3];
        grad += coeff[68] * dx * dx * dx * epsilonInverse[3];
        grad += 2.0 * coeff[71] * dx * dx * dy * epsilonInverse[3];
        grad += 2.0 * coeff[72] * dx * dx * dy * epsilonInverse[3];
        grad += coeff[73] * dx * dx * dz * epsilonInverse[3];
        grad += coeff[74] * dx * dx * dz * epsilonInverse[3];
        grad += 3.0 * coeff[77] * dx * dy * dy * epsilonInverse[3];
        grad += 3.0 * coeff[78] * dx * dy * dy * epsilonInverse[3];
        grad += 2.0 * coeff[79] * dx * dy * dz * epsilonInverse[3];
        grad += 2.0 * coeff[80] * dx * dy * dz * epsilonInverse[3];
        grad += coeff[81] * dx * dz * dz * epsilonInverse[3];
        grad += coeff[82] * dx * dz * dz * epsilonInverse[3];
      }
    }

    break;

    case 2: {
      // partial u partial z
      grad += coeff[4] * epsilonInverse[0];
      if (polynomialOrder > 1) {
        grad += 2.0 * coeff[12] * dz * epsilonInverse[1];
        grad += coeff[13] * dy * epsilonInverse[1];
        grad += coeff[22] * dx * epsilonInverse[1];
        grad += -2.0 * coeff[25] * dx * epsilonInverse[1];
      }
      if (polynomialOrder > 2) {
      }
      if (polynomialOrder > 3) {
      }
    }

    break;
    }

    break;

  case 1:
    switch (outputAxes2) {
    case 0: {
      // partial v partial x
      grad += coeff[5] * epsilonInverse[0];
      if (polynomialOrder > 1) {
        grad += 2.0 * coeff[14] * dx * epsilonInverse[1];
        grad += coeff[16] * dz * epsilonInverse[1];
        grad += -2.0 * coeff[20] * dy * epsilonInverse[1];
        grad += coeff[24] * dy * epsilonInverse[1];
      }
      if (polynomialOrder > 2) {
      }
      if (polynomialOrder > 3) {
      }
    }

    break;

    case 1: {
      // partial v partial y
      grad += -coeff[9] * epsilonInverse[0];
      if (polynomialOrder > 1) {
        grad += -2.0 * coeff[20] * dx * epsilonInverse[1];
        grad += -1.0 * coeff[22] * dz * epsilonInverse[1];
        grad += 2.0 * coeff[23] * dy * epsilonInverse[1];
        grad += coeff[24] * dx * epsilonInverse[1];
      }
      if (polynomialOrder > 2) {
      }
      if (polynomialOrder > 3) {
      }
    }

    break;

    case 2: {
      // partial v partial z
      grad += coeff[6] * epsilonInverse[0];
      if (polynomialOrder > 1) {
        grad += 2.0 * coeff[15] * dz * epsilonInverse[1];
        grad += coeff[16] * dx * epsilonInverse[1];
        grad += -1.0 * coeff[22] * dy * epsilonInverse[1];
      }
      if (polynomialOrder > 2) {
      }
      if (polynomialOrder > 3) {
      }
    }

    break;
    }

    break;

  case 2:
    switch (outputAxes2) {
    case 0: {
      // partial w partial x
      grad += coeff[7] * epsilonInverse[0];
      if (polynomialOrder > 1) {
        grad += 2.0 * coeff[17] * dx * epsilonInverse[1];
        grad += coeff[19] * dy * epsilonInverse[1];
        grad += -1.0 * coeff[24] * dz * epsilonInverse[1];
      }
      if (polynomialOrder > 2) {
      }
      if (polynomialOrder > 3) {
      }
    }

    break;

    case 1: {
      // partial w partial y
      grad += coeff[8] * epsilonInverse[0];
      if (polynomialOrder > 1) {
        grad += 2.0 * coeff[18] * dy * epsilonInverse[1];
        grad += coeff[19] * dx * epsilonInverse[1];
        grad += -1.0 * coeff[21] * dz * epsilonInverse[1];
      }
      if (polynomialOrder > 2) {
      }
      if (polynomialOrder > 3) {
      }
    }

    break;

    case 2: {
      // partial w partial z
      grad += -1.0 * coeff[10] * epsilonInverse[0];
      if (polynomialOrder > 1) {
        grad += -1.0 * coeff[21] * dy * epsilonInverse[1];
        grad += -1.0 * coeff[24] * dx * epsilonInverse[1];
        grad += 2.0 * coeff[25] * dz * epsilonInverse[1];
      }
      if (polynomialOrder > 2) {
      }
      if (polynomialOrder > 3) {
      }
    }

    break;
    }
    break;
  }

  return grad;
}

inline double calDivFreeBasisGrad(const int outputAxes1, const int outputAxes2,
                                  const double dx, const double dy,
                                  const int polynomialOrder,
                                  const double epsilon,
                                  const std::vector<double> &coeff) {
  double grad = 0.0;
  std::vector<double> epsilonInverse(polynomialOrder);

  for (int i = 0; i < polynomialOrder; i++) {
    epsilonInverse[i] = std::pow(epsilon, -(i + 1));
  }

  switch (outputAxes1) {
  case 0:
    switch (outputAxes2) {
    case 0:
      grad += coeff[4] * epsilonInverse[0];
      if (polynomialOrder > 1) {
        grad += 2.0 * coeff[7] * dx * epsilonInverse[1];
        grad += -2.0 * coeff[8] * dy * epsilonInverse[1];
      }
      if (polynomialOrder > 2) {
        grad += 3.0 * coeff[11] * dx * dx * epsilonInverse[2];
        grad += 2.0 * coeff[12] * dx * dy * epsilonInverse[2];
        grad += -3.0 * coeff[13] * dy * dy * epsilonInverse[2];
      }
      if (polynomialOrder > 3) {
        grad += 4.0 * coeff[16] * dx * dx * dx * epsilonInverse[3];
        grad += 6.0 * coeff[17] * dx * dx * dy * epsilonInverse[3];
        grad += -6.0 * coeff[18] * dx * dy * dy * epsilonInverse[3];
        grad += -4.0 * coeff[19] * dy * dy * dy * epsilonInverse[3];
      }
      break;

    case 1:
      grad += coeff[2] * epsilonInverse[0];
      if (polynomialOrder > 1) {
        grad += 2.0 * coeff[5] * dy * epsilonInverse[1];
        grad += -2.0 * coeff[8] * dx * epsilonInverse[1];
      }
      if (polynomialOrder > 2) {
        grad += 3.0 * coeff[9] * dy * dy * epsilonInverse[2];
        grad += coeff[12] * dx * dx * epsilonInverse[2];
        grad += -6.0 * coeff[13] * dx * dy * epsilonInverse[2];
      }
      if (polynomialOrder > 3) {
        grad += 4.0 * coeff[14] * dy * dy * dy * epsilonInverse[3];
        grad += 2.0 * coeff[17] * dx * dx * dx * epsilonInverse[3];
        grad += -6.0 * coeff[18] * dx * dx * dy * epsilonInverse[3];
        grad += -12.0 * coeff[19] * dx * dy * dy * epsilonInverse[3];
      }
      break;
    }
    break;

  case 1:
    switch (outputAxes2) {
    case 0:
      grad += coeff[3] * epsilonInverse[0];
      if (polynomialOrder > 1) {
        grad += 2.0 * coeff[6] * dx * epsilonInverse[1];
        grad += -2.0 * coeff[7] * dy * epsilonInverse[1];
      }
      if (polynomialOrder > 2) {
        grad += 3.0 * coeff[10] * dx * dx * epsilonInverse[2];
        grad += -6.0 * coeff[11] * dx * dy * epsilonInverse[2];
        grad += -1.0 * coeff[12] * dy * dy * epsilonInverse[2];
      }
      if (polynomialOrder > 3) {
        grad += 4.0 * coeff[15] * dx * dx * dx * epsilonInverse[3];
        grad += -12.0 * coeff[16] * dx * dx * dy * epsilonInverse[3];
        grad += -6.0 * coeff[17] * dx * dy * dy * epsilonInverse[3];
        grad += 2.0 * coeff[18] * dy * dy * dy * epsilonInverse[3];
      }
      break;

    case 1:
      grad += -coeff[4] * epsilonInverse[0];
      if (polynomialOrder > 1) {
        grad += -2.0 * coeff[7] * dx * epsilonInverse[1];
        grad += 2.0 * coeff[8] * dy * epsilonInverse[1];
      }
      if (polynomialOrder > 2) {
        grad += -3.0 * coeff[11] * dx * dx * epsilonInverse[2];
        grad += -2.0 * coeff[12] * dx * dy * epsilonInverse[2];
        grad += 3.0 * coeff[13] * dy * dy * epsilonInverse[2];
      }
      if (polynomialOrder > 3) {
        grad += -4.0 * coeff[16] * dx * dx * dx * epsilonInverse[3];
        grad += -6.0 * coeff[17] * dx * dx * dy * epsilonInverse[3];
        grad += 6.0 * coeff[18] * dx * dy * dy * epsilonInverse[3];
        grad += 4.0 * coeff[19] * dy * dy * dy * epsilonInverse[3];
      }
      break;
    }
    break;
  }

  return grad;
}

inline double calStaggeredScalarGrad(const int outputAxes, const double dx,
                                     const double dy, const int polynomialOrder,
                                     const double epsilon,
                                     const std::vector<double> &coeff) {
  double grad = 0.0;

  std::vector<double> epsilonInverse(polynomialOrder);

  for (int i = 0; i < polynomialOrder; i++) {
    epsilonInverse[i] = std::pow(epsilon, -(i + 1));
  }

  return grad;
}

#endif