#ifndef _GEOMETRY_DOMAIN_GEOMETRY_HPP_
#define _GEOMETRY_DOMAIN_GEOMETRY_HPP_

#include <iostream>
#include <vector>

#include "parallel.hpp"
#include "typedef.hpp"

enum SimpleDomainShape { Box, Cylinder, Sphere };

class DomainGeometry {
private:
  std::vector<scalar_type> size_;
  int dimension_;
  SimpleDomainShape shape_;

public:
  DomainGeometry();
  DomainGeometry(const DomainGeometry &geo);

  ~DomainGeometry();

  bool IsInterior(double x, double y, double z);
  void IsInterior(Kokkos::View<double **> coords, Kokkos::View<bool *> results);
  bool IsBoundary(double x, double y, double z);
  void IsBoundary(Kokkos::View<double **> coords, Kokkos::View<bool *> results);
  double MeasureToBoundary(double x, double y, double z);
  void MeasusreToBoundary(Kokkos::View<double **> coords,
                          Kokkos::View<double *> results);

  void SetType(SimpleDomainShape shape);
  void SetDimension(int dimension);
  void SetGeometryFile(std::string filename);
  void SetSize(const std::vector<scalar_type> &size);
};

#endif