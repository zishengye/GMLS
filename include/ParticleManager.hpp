#ifndef _PARTICLE_MANAGER_PARTICLE_MANAGER_HPP_
#define _PARTICLE_MANAGER_PARTICLE_MANAGER_HPP_

#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "DomainGeometry.hpp"
#include "Ghost.hpp"
#include "Parallel.hpp"
#include "Partition.hpp"
#include "Typedef.hpp"

template <typename T> void SwapEnd(T &var) {
  char *varArray = reinterpret_cast<char *>(&var);
  for (long i = 0; i < static_cast<long>(sizeof(var) / 2); i++)
    std::swap(varArray[sizeof(var) - 1 - i], varArray[i]);
}

enum CoordType {
  CartesianCoordinates,
  SphericalCoordinates,
  CylindricalCoordinates
};

class ParticleSet {
protected:
  CoordType coordType_;
  int dimension_;

  HostRealMatrix hostParticleCoords_;
  HostRealMatrix hostParticleNormal_;
  HostRealVector hostParticleSize_;
  HostIntVector hostParticleType_;
  HostIndexVector hostParticleIndex_;

  DeviceRealMatrix deviceCoords_;
  DeviceIntVector deviceParticleType_;
  DeviceIndexVector deviceParticleIndex_;

  int mpiRank_, mpiSize_;

  Partition partition_;

public:
  ParticleSet(CoordType coordType = CartesianCoordinates);

  void SetDimension(const int dimension);

  HostRealMatrix &GetParticleCoords();
  HostRealMatrix &GetParticleNormal();
  HostRealVector &GetParticleSize();
  HostIntVector &GetParticleType();
  HostIndexVector &GetParticleIndex();

  DeviceRealMatrix &PrepareDeviceCoords();
  DeviceIntVector &PrepareDeviceParticleType();
  DeviceIndexVector &PrepareDeviceParticleIndex();

  void GetBackCoords();
  void GetBackParticleType();
  void GetBackParticleIndex();

  const LocalIndex GetLocalParticleNum();
  const GlobalIndex GetGlobalParticleNum();

  const LocalIndex GetLocalGhostParticleNum();

  Scalar GetParticleCoords(const int index, const int dimension);
  int GetParticleType(const int index);
  std::size_t GetParticleIndex(const int index);

  void Resize(const int newLocalSize);
  void Balance();
  void BuildGhost();

  void Output(std::string outputFileName = "output.vtk", bool isBinary = true);
};

class ParticleManager {
protected:
  DomainGeometry geometry_;
  ParticleSet particleSet_;
  Scalar spacing_;
  int mpiSize_, mpiRank_;

public:
  ParticleManager();

  void SetDimension(const int dimension);
  void SetDomainType(const SimpleDomainShape shape);
  void SetSize(const std::vector<Scalar> &size);
  void SetSpacing(const Scalar spacing);

  const int GetDimension();

  virtual void Init();
  virtual void Clear();

  const LocalIndex GetLocalParticleNum();
  const GlobalIndex GetGlobalParticleNum();

  HostRealMatrix &GetParticleCoords();
  HostRealMatrix &GetParticleNormal();
  HostRealVector &GetParticleSize();
  HostIntVector &GetParticleType();
  HostIndexVector &GetParticleIndex();

  void Output(std::string outputFileName = "output.vtk", bool isBinary = true);
};

class HierarchicalParticleManager : public ParticleManager {
protected:
  std::vector<ParticleSet> hierarchicalParticleSet_;

public:
  HierarchicalParticleManager();

  virtual void Init();
  virtual void Clear();
};

#endif