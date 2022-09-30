#ifndef _Geometry_ParticleGeometry_Hpp_
#define _Geometry_ParticleGeometry_Hpp_

#include <memory>
#include <string>
#include <vector>

#include "Core/Typedef.hpp"
#include "Geometry/DomainGeometry.hpp"
#include "Geometry/Ghost.hpp"
#include "Geometry/Partition.hpp"

template <typename T> void SwapEnd(T &var) {
  char *varArray = reinterpret_cast<char *>(&var);
  for (long i = 0; i < static_cast<long>(sizeof(var) / 2); i++)
    std::swap(varArray[sizeof(var) - 1 - i], varArray[i]);
}

namespace Geometry {
enum CoordType {
  CartesianCoordinates,
  SphericalCoordinates,
  CylindricalCoordinates
};

class ParticleSet {
protected:
  CoordType coordType_;
  Size dimension_;

  HostRealMatrix hostParticleCoords_;
  HostRealMatrix hostParticleNormal_;
  HostRealVector hostParticleSize_;
  HostIndexVector hostParticleType_;
  HostIndexVector hostParticleIndex_;

  DeviceRealMatrix deviceCoords_;
  DeviceIntVector deviceParticleType_;
  DeviceIndexVector deviceParticleIndex_;

  LocalIndex mpiRank_, mpiSize_;

  Partition partition_;

public:
  ParticleSet(CoordType coordType = CartesianCoordinates);

  Void SetDimension(const Size dimension);

  HostRealMatrix &GetParticleCoords();
  HostRealMatrix &GetParticleNormal();
  HostRealVector &GetParticleSize();
  HostIndexVector &GetParticleType();
  HostIndexVector &GetParticleIndex();

  DeviceRealMatrix &PrepareDeviceCoords();
  DeviceIntVector &PrepareDeviceParticleType();
  DeviceIndexVector &PrepareDeviceParticleIndex();

  Void GetBackCoords();
  Void GetBackParticleType();
  Void GetBackParticleIndex();

  LocalIndex GetLocalParticleNum();
  GlobalIndex GetGlobalParticleNum();

  LocalIndex GetLocalGhostParticleNum();

  Scalar GetParticleCoords(const LocalIndex index, const Size dimension);
  LocalIndex GetParticleType(const LocalIndex index);
  std::size_t GetParticleIndex(const LocalIndex index);

  Void Resize(const LocalIndex newLocalSize);
  Void Balance();

  Void Output(String outputFileName = "output.vtk", Boolean isBinary = true);

  Partition &GetPartition();
};

class EulerianParticleManager {
protected:
  std::shared_ptr<DomainGeometry> geometryPtr_;
  std::shared_ptr<ParticleSet> particleSetPtr_;
  Scalar spacing_;
  LocalIndex mpiSize_, mpiRank_;
  Boolean isPeriodicBCs_;

  Void BalanceAndIndexInternal();

public:
  EulerianParticleManager();

  Void SetDimension(const Size dimension);
  Void SetDomainType(const SupportedDomainShape shape);
  Void SetSize(const std::vector<Scalar> &size);
  Void SetSpacing(const Scalar spacing);
  Void SetPeriodicBCs();

  Size GetDimension();

  virtual Void Init();
  virtual Void Clear();

  virtual LocalIndex GetLocalParticleNum();
  virtual GlobalIndex GetGlobalParticleNum();

  HostRealMatrix &GetParticleCoords();
  HostRealMatrix &GetParticleNormal();
  HostRealVector &GetParticleSize();
  HostIndexVector &GetParticleType();
  HostIndexVector &GetParticleIndex();

  virtual Void Output(const String outputFileName = "output.vtk",
                      const Boolean isBinary = true);
};

class HierarchicalEulerianParticleManager : public EulerianParticleManager {
protected:
  std::vector<std::shared_ptr<ParticleSet>> hierarchicalParticleSetPtr_;

  LocalIndex currentRefinementLevel_;

  HostIndexVector hostParticleRefinementLevel_;

  virtual Void RefineInternal(const HostIndexVector &splitTag);

public:
  HierarchicalEulerianParticleManager();

  HostIndexVector &GetParticleRefinementLevel();

  virtual Void Init();
  virtual Void Clear();

  Void Refine(const HostIndexVector &splitTag);

  virtual LocalIndex GetLocalParticleNum();
  virtual GlobalIndex GetGlobalParticleNum();

  HostRealMatrix &GetParticleCoordsByLevel(const LocalIndex level);
  HostRealMatrix &GetParticleNormalByLevel(const LocalIndex level);
  HostRealVector &GetParticleSizeByLevel(const LocalIndex level);
  HostIndexVector &GetParticleTypeByLevel(const LocalIndex level);
  HostIndexVector &GetParticleIndexByLevel(const LocalIndex level);

  virtual Void Output(const String outputFileName = "output.vtk",
                      const Boolean isBinary = true);
};

class FieldRigidBodyInteractionEulerianParticleManager
    : public HierarchicalEulerianParticleManager {};
} // namespace Geometry

#endif