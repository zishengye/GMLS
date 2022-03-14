#ifndef _PARTICLE_MANAGER_PARTICLE_MANAGER_HPP_
#define _PARTICLE_MANAGER_PARTICLE_MANAGER_HPP_

#include <memory>
#include <vector>

#include <Kokkos_Core.hpp>

#include "domain_geometry.hpp"
#include "parallel.hpp"
#include "typedef.hpp"

enum CoordType {
  CartesianCoordinates,
  SphericalCoordinates,
  CylindricalCoordinates
};

class ParticleSet {
private:
  CoordType coordType_;

public:
  ParticleSet(CoordType coordType = CartesianCoordinates);
};

class ParticleManager {
private:
  std::shared_ptr<DomainGeometry> geoPtr;

public:
  void SetDomainGeometry(DomainGeometry &geo);
};

class HierarchicalParticleManager {
public:
  HierarchicalParticleManager();

  void Init();
  void Clear();
};

#endif