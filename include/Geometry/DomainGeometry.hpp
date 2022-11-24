#ifndef _Geometry_DomainGeometry_Hpp_
#define _Geometry_DomainGeometry_Hpp_

#include <Kokkos_Core_fwd.hpp>
#include <iostream>
#include <vector>

#include "Core/Parallel.hpp"
#include "Core/Typedef.hpp"

namespace Geometry {
enum SupportedDomainShape { UndefinedDomain, Box, Cylinder, Sphere };

class DomainGeometry {
private:
  std::vector<Scalar> size_;
  Size dimension_;
  SupportedDomainShape shape_;

  int mpiSize_, mpiRank_;

public:
  DomainGeometry();
  DomainGeometry(const DomainGeometry &geo);

  ~DomainGeometry();

  Boolean IsInterior(Scalar x, Scalar y, Scalar z);
  Void IsInterior(HostRealMatrix coords, HostBooleanVector results);
  Boolean IsBoundary(Scalar x, Scalar y, Scalar z);
  Void IsBoundary(HostRealMatrix coords, HostBooleanVector results);
  Scalar MeasureToBoundary(Scalar x, Scalar y, Scalar z);
  Void MeasureToBoundary(HostRealMatrix coords, HostRealVector results);

  Void SetType(const SupportedDomainShape shape);
  Void SetDimension(const Size dimension);
  Void SetGeometryFile(const String filename);
  Void SetSize(const std::vector<Scalar> &size);

  SupportedDomainShape GetType();
  Size GetDimension();
  Scalar GetSize(const LocalIndex sizeIndex);

  LocalIndex EstimateNodeNum(const Scalar spacing);
  Void AssignUniformNode(HostRealMatrix nodeCoords, HostRealMatrix nodeNormal,
                         HostRealVector nodeSize, HostIndexVector nodeType,
                         const Scalar spacing);
};
} // namespace Geometry

#endif