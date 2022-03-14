#include "domain_geometry.hpp"

DomainGeometry::DomainGeometry() {}

DomainGeometry::DomainGeometry(const DomainGeometry &geo) {
  dimension_ = geo.dimension_;
  size_ = geo.size_;
}

DomainGeometry::~DomainGeometry() {}

bool DomainGeometry::IsInterior(double x, double y, double z) {
  if (dimension_ == 2) {
    if (shape_ == Box) {
      if (x > -size_[0] / 2.0 && x < size_[0] / 2.0 && y > -size_[1] / 2.0 &&
          y < size_[1] / 2.0)
        return true;
      else
        return false;
    } else if (shape_ == Cylinder) {
      double r = sqrt(x * x + y * y);
      if (r < size_[0])
        return true;
      else
        return false;
    } else {
      return false;
    }
  }

  if (dimension_ == 3) {
    if (shape_ == Box) {
      if (x > -size_[0] / 2.0 && x < size_[0] / 2.0 && y > -size_[1] / 2.0 &&
          y < size_[1] / 2.0 && z > -size_[2] / 2.0 && z < size_[2] / 2.0)
        return true;
      else
        return false;
    } else if (shape_ == Cylinder) {
      double r = sqrt(x * x + y * y);
      if (r < size_[0] && z > -size_[1] / 2.0 && z < size_[1] / 2.0)
        return true;
      else
        return false;
    } else {
      return false;
    }
  }
}

void DomainGeometry::IsInterior(Kokkos::View<double **> coords,
                                Kokkos::View<bool *> results) {}

void DomainGeometry::SetType(SimpleDomainShape shape) { shape_ = shape; }

void DomainGeometry::SetDimension(int dimension) { dimension_ = dimension; }

void DomainGeometry::SetSize(const std::vector<scalar_type> &size) {
  size_ = size;
}