#ifndef _Geometry_GeometryItem_Hpp_
#define _Geometry_GeometryItem_Hpp_

#include "Math/Vec3.hpp"

namespace Geometry {
struct BoundingBoxEntry {
  Math::Vec3 boundingBoxLow_, boundingBoxHigh_;
};

struct BoundingSphereEntry {
  Math::Vec3 center_;
  double radius_;
};

class BoundingBox {
protected:
  int dimension_;
  BoundingBoxEntry boundingBox_;

public:
  BoundingBox(const int dimension);
  BoundingBox(const int dimension, const BoundingBoxEntry &boundingBox);

  Void ResizeBoundingBox(const BoundingBoxEntry &boundingBox);

  inline Boolean IsIntersect(const BoundingBox &boundingBox);
};

/*! \note Base class of all geometry entries in the solver.
 *! \date Dec 8, 2022
 *! \author Zisheng Ye <zisheng_ye@outlook.com>
 *
 * The hierarchical bounding boxes are the basic element for the interaction
 * between any geometry items in the solver for a high performance.  The base
 * class helps the maintenance of the boxes.
 */
class GeometryItem {};
} // namespace Geometry

#endif