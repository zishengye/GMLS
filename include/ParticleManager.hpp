#ifndef _ParticleManager_Hpp_
#define _ParticleManager_Hpp_

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
  HostIndexVector hostParticleType_;
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
  HostIndexVector &GetParticleType();
  HostIndexVector &GetParticleIndex();

  DeviceRealMatrix &PrepareDeviceCoords();
  DeviceIntVector &PrepareDeviceParticleType();
  DeviceIndexVector &PrepareDeviceParticleIndex();

  void GetBackCoords();
  void GetBackParticleType();
  void GetBackParticleIndex();

  LocalIndex GetLocalParticleNum();
  GlobalIndex GetGlobalParticleNum();

  LocalIndex GetLocalGhostParticleNum();

  Scalar GetParticleCoords(const int index, const int dimension);
  int GetParticleType(const int index);
  std::size_t GetParticleIndex(const int index);

  void Resize(const int newLocalSize);
  void Balance();

  void Output(std::string outputFileName = "output.vtk", bool isBinary = true);

  Partition &GetPartition();
};

class ParticleManager {
protected:
  std::shared_ptr<DomainGeometry> geometryPtr_;
  std::shared_ptr<ParticleSet> particleSetPtr_;
  Scalar spacing_;
  int mpiSize_, mpiRank_;
  bool isPeriodicBCs_;

  void BalanceAndIndexInternal();

public:
  ParticleManager();

  void SetDimension(const int dimension);
  void SetDomainType(const SimpleDomainShape shape);
  void SetSize(const std::vector<Scalar> &size);
  void SetSpacing(const Scalar spacing);
  void SetPeriodicBCs();

  int GetDimension();

  virtual void Init();
  virtual void Clear();

  virtual LocalIndex GetLocalParticleNum();
  virtual GlobalIndex GetGlobalParticleNum();

  HostRealMatrix &GetParticleCoords();
  HostRealMatrix &GetParticleNormal();
  HostRealVector &GetParticleSize();
  HostIndexVector &GetParticleType();
  HostIndexVector &GetParticleIndex();

  virtual void Output(const std::string outputFileName = "output.vtk",
                      const bool isBinary = true);
};

class HierarchicalParticleManager : public ParticleManager {
protected:
  std::vector<std::shared_ptr<ParticleSet>> hierarchicalParticleSetPtr_;

  int currentRefinementLevel_;

  HostIndexVector hostParticleRefinementLevel_;

  virtual void RefineInternal(const HostIndexVector &splitTag);

public:
  HierarchicalParticleManager();

  HostIndexVector &GetParticleRefinementLevel();

  virtual void Init();
  virtual void Clear();

  void Refine(const HostIndexVector &splitTag);

  virtual LocalIndex GetLocalParticleNum();
  virtual GlobalIndex GetGlobalParticleNum();

  HostRealMatrix &GetParticleCoordsByLevel(const int level);
  HostRealMatrix &GetParticleNormalByLevel(const int level);
  HostRealVector &GetParticleSizeByLevel(const int level);
  HostIndexVector &GetParticleTypeByLevel(const int level);
  HostIndexVector &GetParticleIndexByLevel(const int level);

  virtual void Output(const std::string outputFileName = "output.vtk",
                      const bool isBinary = true);
};

#endif