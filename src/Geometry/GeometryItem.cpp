#include "Geometry/GeometryItem.hpp"

Geometry::BoundingBox::BoundingBox(const int dimension)
    : dimension_(dimension) {}

Geometry::BoundingBox::BoundingBox(const int dimension,
                                   const BoundingBoxEntry &boundingBox)
    : dimension_(dimension), boundingBox_(boundingBox) {}

Void Geometry::BoundingBox::ResizeBoundingBox(
    const BoundingBoxEntry &boundingBox) {
  boundingBox_ = boundingBox;
}

Boolean Geometry::BoundingBox::IsIntersect(const BoundingBox &boundingBox) {
  if (dimension_ == 1) {
  }
  if (dimension_ == 2) {
  }
  if (dimension_ == 3) {
  }

  return true;
}