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

#include "info.h"
#include "search_command.h"
#include "vec3.h"

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

  GeneralInfo __background;
  GeneralInfo __field;
  GeneralInfo __eq;

  // rigid body info
  GeneralInfo __rigidBody;

  // gmls info
  gmlsInfo __gmls;

  // domain info
  void SetBoundingBox();
  void SetBoundingBoxBoundary();
  void SetDomainBoundary();

  void SetBoundingBoxManifold();
  void SetBoundingBoxBoundaryManifold();
  void SetDomainBoundaryManifold();

  void InitFieldParticle();
  void InitFieldBoundaryParticle();

  void InitFieldParticleManifold();
  void InitFieldBoundaryParticleManifold();

  void InitRigidBody();

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
        InsertParticle(pos, 1, __particleSize0, normal, globalIndex, vol);
        incX(pos);
      }
      pos = oldPos;
      incY(pos);
      oldPos = pos;
    }
  }

  void InsertParticle(vec3 &X, int particleType, vec3 &particleSize,
                      vec3 &normal, int &globalIndex, double vol,
                      bool rigidBodyParticle = false,
                      size_t rigidBodyIndex = -1) {
    static std::vector<vec3> &_coord = __field.vector.GetHandle("coord");
    static std::vector<vec3> &_normal = __field.vector.GetHandle("normal");
    static std::vector<vec3> &_particleSize = __field.vector.GetHandle("size");
    static std::vector<int> &_globalIndex =
        __field.index.GetHandle("global index");
    static std::vector<int> &_particleType =
        __field.index.GetHandle("particle type");
    static std::vector<int> &_attachedRigidBodyIndex =
        __field.index.GetHandle("attached rigid body index");

    if (rigidBodyParticle || IsInRigidBody(X) == -2) {
      _coord.push_back(X);
      _particleType.push_back(particleType);
      _particleSize.push_back(particleSize);
      _normal.push_back(normal);
      _globalIndex.push_back(globalIndex++);
      _attachedRigidBodyIndex.push_back(rigidBodyIndex);
    }
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

  // fluid parameter
  double __eta;

  bool IsInGap(vec3 &xScalar);

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

  GeneralInfo __neighbor;

  template <typename Cond>
  bool PutParticleInNeighborList(int neighborBlockNum, Cond cond) {
    static std::vector<int> &neighborFlag =
        __neighbor.index.GetHandle("neighbor flag");
    return neighborFlag[neighborBlockNum] && cond();
  }

  // solving functions
  void InitParticle();
  void ClearParticle();
  void InitUniformParticleField();
  void InitUniformParticleManifoldField();

  void EmposeBoundaryCondition();

  void InitialCondition();

  // particle adjustment
  bool NeedRefinement();

  // adaptive refinement
  void SplitMergeVectorField();
  void SplitMergeScalarField();

  int __adaptive_step;

  // rigid body supporting functions
  int IsInRigidBody(vec3 &pos);

  void InitRigidBodySurfaceParticle();

  // equation type
  void PoissonEquation();
  void PoissonEquationManifold();
  void DiffusionEquation();
  void DiffusionEquationManifold();
  void StokesEquation();
  void StokesEquationInitialization();
  void NavierStokesEquation();

  // function pointer
  void (GMLS_Solver::*__equationSolver)(void);
  void (GMLS_Solver::*__equationSolverInitialization)(void);
  void (GMLS_Solver::*__particleUniformInitializer)(void);
  void (GMLS_Solver::*__splitMerger)(void);

  // time integration scheme
  void ForwardEulerIntegration();

  // operator
  template <typename Func>
  void SerialOperation(Func operation) {
    for (int i = 0; i < __MPISize; i++) {
      if (i == __myID) {
        operation();
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  template <typename Func>
  void MasterOperation(int master, Func operation) {
    if (master == __myID) {
      operation();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  template <typename Func>
  void CollectiveWrite(std::string filename, Func operation) {}

  void WriteDataTimeStep();
  void WriteDataAdaptiveStep();

 public:
  GMLS_Solver(int argc, char **argv);

  void TimeIntegration();

  bool IsSuccessInit() { return __successInitialized; }
};