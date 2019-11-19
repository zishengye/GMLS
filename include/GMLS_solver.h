#pragma once

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include <Compadre_Config.h>
#include <Compadre_Evaluator.hpp>
#include <Compadre_GMLS.hpp>
#include <Compadre_PointCloudSearch.hpp>

#include <petscksp.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>

#include "vec3.h"

struct ParticleInfo {
  // geometry info
  std::vector<vec3> X;
  std::vector<vec3> X_origin;
  std::vector<int> particleType;
  std::vector<vec3> particleSize;
  std::vector<vec3> normal;
  std::vector<int> globalIndex;
  std::vector<double> vol;
  int localParticleNum;
  int globalParticleNum;
  std::vector<int> particleOffset;
  std::vector<double> d;

  // physical info
  std::vector<double> pressure;
  std::vector<double> us;
  std::vector<double> us_old;
  std::vector<vec3> flux;

  // GMLS info
  Compadre::GMLS *scalarBasis;
  Compadre::GMLS *vectorBasis;
};

struct ColloidInfo {
  std::vector<vec3> Ci_X;
  std::vector<vec3> Ci_Theta;
  std::vector<vec3> Ci_V;
  std::vector<vec3> Ci_Omega;
  std::vector<vec3> Ci_F;
  std::vector<vec3> Ci_Torque;
  std::vector<double> Ci_R;
  std::vector<int> type;
};

struct EquationInfo {
  std::vector<vec3> F;
  std::vector<vec3> gradP;
  std::vector<double> dP;
  std::vector<double> rhs;
  std::vector<double> x;
};

struct neighborListInfo {
  std::vector<vec3> coord;
  std::vector<int> index;

  void clear() {
    coord.clear();
    index.clear();
  }
};

template <typename T>
int SearchCommand(int argc, char **argv, const std::string &commandName,
                  T &res);

class GMLS_Solver {
private:
  // MPI setting
  int __myID;
  int __MPISize;

  // solver control parameter
  std::string __equationType;
  std::string __timeIntegrationMethod;
  std::string __schemeType;
  int __polynomialOrder;
  int __writeData;

  bool __successInitialized;

  int __dim;

  double __finalTime;
  double __dt;

  double __recoveryGradUerrorTolerance;

  // solver step info
  double __recoveryGradUerror;
  double __gradUTotal;
  double __volTotal;

  int __currentTimeIngrationStep;
  int __currentAdaptiveStep;
  double __currentTime;
  double __currentTimePeriod;

  int __coordinateSystem;
  // 1 cartesian coordinate system
  // 2 cylindrical coordinate system
  // 3 spherical coordinate system

  // particle info
  double __cutoffDistance;

  vec3 __particleSize0;

  // colloid info
  ColloidInfo __colloid;

  // domain info
  void SetBoundingBox();
  void SetBoundingBoxBoundary();
  void SetDomainBoundary();

  void SetBoundingBoxManifold();
  void SetBoundingBoxBoundaryManifold();
  void SetDomainBoundaryManifold();

  void InitFluidParticle();
  void InitWallParticle();

  void InitFluidParticleManifold();
  void InitWallParticleManifold();

  // manifold info
  int __manifoldOrder;

  template <typename IncX, typename IncY, typename ConX, typename ConY>
  void InitWallFaceParticle(vec3 &startPos, vec3 &endPos, IncX incX, IncY incY,
                            ConX conX, ConY conY, int &globalIndex, double vol,
                            vec3 &normal) {
    vec3 pos = startPos;
    vec3 oldPos = startPos;
    while (conY(pos, endPos)) {
      while (conX(pos, endPos)) {
        InsertParticle(pos, 1, __particleSize0, normal, globalIndex++, vol);
        incX(pos);
      }
      pos = oldPos;
      incY(pos);
      oldPos = pos;
    }
  }

  void InsertParticle(vec3 &X, int particleType, vec3 &particleSize,
                      vec3 &normal, int globalIndex, double vol) {
    __particle.X.push_back(X);
    __particle.particleType.push_back(particleType);
    __particle.particleSize.push_back(particleSize);
    __particle.normal.push_back(normal);
    __particle.globalIndex.push_back(globalIndex);
    __particle.vol.push_back(vol);
    __particle.d.push_back(particleSize[0]);
  }

  triple<int> __domainCount;
  triple<int> __boundingBoxCount;

  vec3 __domainBoundingBox[2];
  vec3 __boundingBoxSize;

  std::vector<vec3> __domain;
  std::vector<vec3> __boundingBox;

  std::vector<int> __boundingBoxBoundaryType;
  std::vector<int> __domainBoundaryType;

  int __nX, __nY, __nZ;
  // process block size
  int __nI, __nJ, __nK;
  // process block coordinate

  // fluid info
  double __eta;

  ParticleInfo __particle;
  ParticleInfo __gap;
  std::vector<bool> __neighborFlag;
  std::vector<neighborListInfo> __neighborSendParticle;
  neighborListInfo __backgroundParticle;

  bool IsInGap(vec3 &x);

  // equation info
  EquationInfo __eq;

  // domain decomposition
  void InitDomainDecomposition();
  void InitDomainDecompositionManifold();

  // neighborlist
  void InitNeighborList();
  void BuildNeighborList();

  void InitNeighborListManifold();
  void BuildNeighborListManifold();

  MPI_Win __neighborWinCount;
  MPI_Win __neighborWinIndex;
  MPI_Win __neighborWinOffset;
  MPI_Win __neighborWinParticleCoord;
  MPI_Win __neighborWinParticleIndex;

  template <typename Cond>
  bool PutParticleInNeighborList(int neighborBlockNum, Cond cond) {
    return __neighborFlag[neighborBlockNum] && cond();
  }

  // adaptive refinement
  void SplitMerge();

  // solving functions
  void InitColloid();

  void InitUniformParticleField();
  void InitUniformParticleManifoldField();

  void EmposeBoundaryCondition();

  void InitialCondition();

  // equation type
  void PoissonEquation();
  void PoissonEquationManifold();
  void DiffusionEquation();
  void DiffusionEquationManifold();
  void StokesEquation();
  void NavierStokesEquation();

  // function pointer
  void (GMLS_Solver::*__equationSolver)(void);
  void (GMLS_Solver::*__particleUniformInitializer)(void);

  // time integration scheme
  void ForwardEulerIntegration();

  // operator
  void ClearMemory();

  template <typename Func> void SerialOperation(Func operation) {
    for (int i = 0; i < __MPISize; i++) {
      if (i == __myID) {
        operation();
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  template <typename Func> void MasterOperation(int master, Func operation) {
    if (master == __myID) {
      operation();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  template <typename Func>
  void CollectiveWrite(std::string filename, Func operation) {}

  void WriteData();

public:
  GMLS_Solver(int argc, char **argv);

  void TimeIntegration();

  bool IsSuccessInit() { return __successInitialized; }
};