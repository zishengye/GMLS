#pragma once

#include <cmath>
#include <vector>

inline double calDivFreeBasisGrad(const int outputAxes1, const int outputAxes2,
                                  const double dx, const double dy,
                                  const double dz, const int polynomialOrder,
                                  const double epsilon,
                                  const std::vector<double> &coeff) {
  switch (outputAxes1) {
    case 0:
      break;

    case 1:
      break;

    case 2:
      break;
  }
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