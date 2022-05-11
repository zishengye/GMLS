#ifndef _DomainGeometry_Hpp_
#define _DomainGeometry_Hpp_

#include <iostream>
#include <vector>

#include "Parallel.hpp"
#include "Typedef.hpp"

enum SimpleDomainShape { UndefinedDomain, Box, Cylinder, Sphere };

class DomainGeometry {
private:
  std::vector<Scalar> size_;
  int dimension_;
  SimpleDomainShape shape_;

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

  void SetType(const SimpleDomainShape shape);
  void SetDimension(const int dimension);
  void SetGeometryFile(const std::string filename);
  void SetSize(const std::vector<Scalar> &size);

  SimpleDomainShape GetType();
  int GetDimension();
  Scalar GetSize(const int size_index);

  LocalIndex EstimateNodeNum(const Scalar spacing);
  void AssignUniformNode(Kokkos::View<Scalar **> nodeCoords,
                         Kokkos::View<Scalar **> nodeNormal,
                         Kokkos::View<Scalar *> nodeSize,
                         Kokkos::View<std::size_t *> nodeType,
                         const Scalar spacing);
};

#endif