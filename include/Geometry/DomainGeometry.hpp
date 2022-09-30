#ifndef _Geometry_DomainGeometry_Hpp_
#define _Geometry_DomainGeometry_Hpp_

#include <iostream>
#include <vector>

#include "Core/Parallel.hpp"
#include "Core/Typedef.hpp"

namespace Geometry {
enum SupportedDomainShape { UndefinedDomain, Box, Cylinder, Sphere };

class DomainGeometry {
private:
  std::vector<Scalar> size_;
  int dimension_;
  SupportedDomainShape shape_;

  int mpiSize_, mpiRank_;

public:
  DomainGeometry();
  DomainGeometry(const DomainGeometry &geo);

  ~DomainGeometry();

  bool IsInterior(Scalar x, Scalar y, Scalar z);
  void IsInterior(Kokkos::View<Scalar **> coords, Kokkos::View<bool *> results);
  bool IsBoundary(Scalar x, Scalar y, Scalar z);
  void IsBoundary(Kokkos::View<Scalar **> coords, Kokkos::View<bool *> results);
  Scalar MeasureToBoundary(Scalar x, Scalar y, Scalar z);
  void MeasureToBoundary(Kokkos::View<Scalar **> coords,
                         Kokkos::View<Scalar *> results);

  void SetType(const SupportedDomainShape shape);
  void SetDimension(const int dimension);
  void SetGeometryFile(const std::string filename);
  void SetSize(const std::vector<Scalar> &size);

  SupportedDomainShape GetType();
  int GetDimension();
  Scalar GetSize(const int size_index);

  LocalIndex EstimateNodeNum(const Scalar spacing);
  void AssignUniformNode(Kokkos::View<Scalar **> nodeCoords,
                         Kokkos::View<Scalar **> nodeNormal,
                         Kokkos::View<Scalar *> nodeSize,
                         Kokkos::View<std::size_t *> nodeType,
                         const Scalar spacing);
};
} // namespace Geometry

#endif