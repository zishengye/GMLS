#ifndef _DIVERGENCEFREE_HPP_
#define _DIVERGENCEFREE_HPP_

#include <cmath>
#include <vector>

#include "vec3.hpp"

inline int get_size(const int degree, const int dimension) {
  if (dimension == 3)
    return (degree + 1) * (degree + 2) * (degree + 3) / 6;
  else if (dimension == 2)
    return (degree + 1) * (degree + 2) / 2;
  else
    return degree + 1;
}

inline double cal_div_free_grad(const int output_axes1, const int output_axes2,
                                const int dimension, vec3 dX,
                                const int poly_order, const double h,
                                const std::vector<double> &coeff) {
  double grad = 0.0;
  double weight_of_original_value = 0.0;
  double weight_of_new_value = 1.0;

  int starting_order = 0;

  const double factorial[] = {
      1,     1,      2,       6,        24,        120,        720,        5040,
      40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200};

  if (dimension == 3) {
    double x, y, z;
    x = dX[0];
    y = dX[1];
    z = dX[2];

    std::vector<double> x_over_h_to_i, y_over_h_to_i, z_over_h_to_i;

    x_over_h_to_i.resize(poly_order + 1);
    y_over_h_to_i.resize(poly_order + 1);
    z_over_h_to_i.resize(poly_order + 1);

    x_over_h_to_i[0] = 1;
    y_over_h_to_i[0] = 1;
    z_over_h_to_i[0] = 1;

    for (int i = 1; i <= poly_order; ++i) {
      x_over_h_to_i[i] = x_over_h_to_i[i - 1] * (x / h);
      y_over_h_to_i[i] = y_over_h_to_i[i - 1] * (y / h);
      z_over_h_to_i[i] = z_over_h_to_i[i - 1] * (z / h);
    }
    int i = 0;
    for (int d = 0; d < dimension; ++d) {
      if ((d + 1) == dimension) {
        if (output_axes1 == d) {
          // use 2D partial derivative of scalar basis definition
          // (in 2D) \sum_{n=0}^{n=P} \sum_{k=0}^{k=n} (x/h)^(n-k)*(y/h)^k /
          // ((n-k)!k!)
          int alphax, alphay;
          double alphaf;
          for (int n = starting_order; n <= poly_order; n++) {
            for (alphay = 0; alphay <= n; alphay++) {
              alphax = n - alphay;

              int var_pow[3] = {alphax, alphay, 0};
              var_pow[output_axes2]--;

              if (var_pow[0] < 0 || var_pow[1] < 0 || var_pow[2] < 0) {
                grad += coeff[i] * weight_of_new_value * 0.0;
              } else {
                alphaf = factorial[var_pow[0]] * factorial[var_pow[1]];
                grad += coeff[i] * weight_of_new_value * 1. / h *
                        x_over_h_to_i[var_pow[0]] * y_over_h_to_i[var_pow[1]] /
                        alphaf;
              }
              i++;
            }
          }
        } else {
          for (int j = 0; j < get_size(poly_order, dimension - 1); ++j) {
            grad += coeff[i] * weight_of_new_value * 0;
            i++;
          }
        }
      } else {
        if (output_axes1 == d) {
          // use 3D partial derivative of scalar basis definition
          // (in 3D) \sum_{p=0}^{p=P} \sum_{k1+k2+k3=n}
          // (x/h)^k1*(y/h)^k2*(z/h)^k3 / (k1!k2!k3!)
          int alphax, alphay, alphaz;
          double alphaf;
          int s = 0;
          for (int n = starting_order; n <= poly_order; n++) {
            for (alphaz = 0; alphaz <= n; alphaz++) {
              s = n - alphaz;
              for (alphay = 0; alphay <= s; alphay++) {
                alphax = s - alphay;

                int var_pow[3] = {alphax, alphay, alphaz};
                var_pow[output_axes2]--;

                if (var_pow[0] < 0 || var_pow[1] < 0 || var_pow[2] < 0) {
                  grad += coeff[i] * weight_of_new_value * 0.0;
                } else {
                  alphaf = factorial[var_pow[0]] * factorial[var_pow[1]] *
                           factorial[var_pow[2]];
                  grad += coeff[i] * weight_of_new_value * 1. / h *
                          x_over_h_to_i[var_pow[0]] *
                          y_over_h_to_i[var_pow[1]] *
                          z_over_h_to_i[var_pow[2]] / alphaf;
                }
                i++;
              }
            }
          }
        } else if (output_axes1 == (d + 1) % 3) {
          for (int j = 0; j < get_size(poly_order, dimension); ++j) {
            grad += coeff[i] * weight_of_new_value * 0;
            i++;
          }
        } else {
          // (in 3D)
          int alphax, alphay, alphaz;
          double alphaf;
          int s = 0;
          for (int n = starting_order; n <= poly_order; n++) {
            for (alphaz = 0; alphaz <= n; alphaz++) {
              s = n - alphaz;
              for (alphay = 0; alphay <= s; alphay++) {
                alphax = s - alphay;

                int var_pow[3] = {alphax, alphay, alphaz};
                var_pow[d]--;

                if (var_pow[0] < 0 || var_pow[1] < 0 || var_pow[2] < 0) {
                  grad += coeff[i] * weight_of_new_value * 0.0;
                } else {
                  var_pow[output_axes1]++;
                  var_pow[output_axes2]--;
                  if (var_pow[0] < 0 || var_pow[1] < 0 || var_pow[2] < 0) {
                    grad += coeff[i] * weight_of_new_value * 0.0;
                  } else {
                    alphaf = factorial[var_pow[0]] * factorial[var_pow[1]] *
                             factorial[var_pow[2]];
                    grad += coeff[i] * weight_of_new_value * -1.0 / h *
                            x_over_h_to_i[var_pow[0]] *
                            y_over_h_to_i[var_pow[1]] *
                            z_over_h_to_i[var_pow[2]] / alphaf;
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
    double x, y;
    x = dX[0];
    y = dX[1];

    std::vector<double> x_over_h_to_i, y_over_h_to_i;

    x_over_h_to_i.resize(poly_order + 1);
    y_over_h_to_i.resize(poly_order + 1);

    x_over_h_to_i[0] = 1;
    y_over_h_to_i[0] = 1;

    for (int i = 1; i <= poly_order; ++i) {
      x_over_h_to_i[i] = x_over_h_to_i[i - 1] * (x / h);
      y_over_h_to_i[i] = y_over_h_to_i[i - 1] * (y / h);
    }
    int i = 0;
    for (int d = 0; d < dimension; ++d) {
      if ((d + 1) == dimension) {
        if (output_axes1 == d) {
          // use 1D partial derivative of scalar basis definition
          int alphax;
          double alphaf;
          for (int n = starting_order; n <= poly_order; n++) {
            alphax = n;

            int var_pow[2] = {alphax, 0};
            var_pow[output_axes2]--;

            if (var_pow[0] < 0 || var_pow[1] < 0) {
              grad += coeff[i] * weight_of_new_value * 0.0;
            } else {
              alphaf = factorial[var_pow[0]];
              grad += coeff[i] * weight_of_new_value * 1. / h *
                      x_over_h_to_i[var_pow[0]] / alphaf;
            }
            i++;
          }
        } else {
          for (int j = 0; j < get_size(poly_order, dimension - 1); ++j) {
            grad += coeff[i] * weight_of_new_value * 0;
            i++;
          }
        }
      } else {
        if (output_axes1 == d) {
          // use 2D partial derivative of scalar basis definition
          int alphax, alphay;
          double alphaf;
          for (int n = starting_order; n <= poly_order; n++) {
            for (alphay = 0; alphay <= n; alphay++) {
              alphax = n - alphay;

              int var_pow[2] = {alphax, alphay};
              var_pow[output_axes2]--;

              if (var_pow[0] < 0 || var_pow[1] < 0) {
                grad += coeff[i] * weight_of_new_value * 0.0;
              } else {
                alphaf = factorial[var_pow[0]] * factorial[var_pow[1]];
                grad += coeff[i] * weight_of_new_value * 1. / h *
                        x_over_h_to_i[var_pow[0]] * y_over_h_to_i[var_pow[1]] /
                        alphaf;
              }
              i++;
            }
          }
        } else {
          // (in 2D)
          int alphax, alphay;
          double alphaf;
          for (int n = starting_order; n <= poly_order; n++) {
            for (alphay = 0; alphay <= n; alphay++) {
              alphax = n - alphay;

              int var_pow[2] = {alphax, alphay};
              var_pow[d]--;

              if (var_pow[0] < 0 || var_pow[1] < 0) {
                grad += coeff[i] * weight_of_new_value * 0.0;
              } else {
                var_pow[output_axes1]++;
                var_pow[output_axes2]--;
                if (var_pow[0] < 0 || var_pow[1] < 0) {
                  grad += coeff[i] * weight_of_new_value * 0.0;
                } else {
                  alphaf = factorial[var_pow[0]] * factorial[var_pow[1]];
                  grad += coeff[i] * weight_of_new_value * -1.0 / h *
                          x_over_h_to_i[var_pow[0]] *
                          y_over_h_to_i[var_pow[1]] / alphaf;
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