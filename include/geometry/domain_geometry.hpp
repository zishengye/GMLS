#ifndef _GEOMETRY_DOMAIN_GEOMETRY_HPP_
#define _GEOMETRY_DOMAIN_GEOMETRY_HPP_

#include <vector>

#include <Kokkos_Core.hpp>

class domain_geometry {
private:
  std::vector<double> size_;

public:
  domain_geometry();
  domain_geometry(domain_geometry &geo);

  ~domain_geometry();

  bool is_interior(double x, double y, double z);
  bool is_boundary(double x, double y, double z);
  double to_boundary(double x, double y, double z);

  void set_type();
  void set_size(std::vector<double> &size);
};

#endif