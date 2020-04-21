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
#include "sparse_matrix.h"
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
  int __weightFuncOrder;
  int __writeData;
  int __batchSize;
  int __adaptiveRefinement;
  double __adaptiveRefinementTolerance;
  int __adaptive_step;
  std::string __adaptive_base_field;

  bool __successInitialized;

  int __dim;

  double __finalTime;
  double __dt;

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

  GeneralInfo __gap;

  // rigid body info
  GeneralInfo __rigidBody;

  std::string __rigidBodyInputFileName;
  bool __rigidBodyInclusion;

  // gmls info
  gmlsInfo __gmls;

  // domain info
  void SetBoundingBox();
  void SetBoundingBoxBoundary();
  void SetDomainBoundary();

  void SetBoundingBoxManifold();
  void SetBoundingBoxBoundaryManifold();
  void SetDomainBoundaryManifold();

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
        InsertParticle(pos, 1, __particleSize0, normal, globalIndex, 0, vol);
        incX(pos);
      }
      pos = oldPos;
      incY(pos);
      oldPos = pos;
    }
  }

  void InitWallFaceParticle(vec3 &X, int particleType, vec3 &particleSize,
                            vec3 &normal, int &globalIndex, int adaptive_level,
                            double vol, bool rigidBodyParticle = false,
                            int rigidBodyIndex = -1,
                            vec3 pCoord = vec3(0.0, 0.0, 0.0)) {
    static std::vector<vec3> &_coord = __field.vector.GetHandle("coord");
    static std::vector<vec3> &_normal = __field.vector.GetHandle("normal");
    static std::vector<vec3> &_particleSize = __field.vector.GetHandle("size");
    static std::vector<vec3> &_pCoord =
        __field.vector.GetHandle("parameter coordinate");
    static auto &_volume = __field.scalar.GetHandle("volume");
    static std::vector<int> &_globalIndex =
        __field.index.GetHandle("global index");
    static auto &_adaptive_level = __field.index.GetHandle("adaptive level");
    static std::vector<int> &_particleType =
        __field.index.GetHandle("particle type");
    static std::vector<int> &_attachedRigidBodyIndex =
        __field.index.GetHandle("attached rigid body index");

    _coord.push_back(X);
    _particleType.push_back(particleType);
    _particleSize.push_back(particleSize);
    _normal.push_back(normal);
    _globalIndex.push_back(globalIndex++);
    _adaptive_level.push_back(adaptive_level);
    _attachedRigidBodyIndex.push_back(rigidBodyIndex);
    _pCoord.push_back(pCoord);
    _volume.push_back(vol);
  }

  void InsertParticle(const vec3 &X, int particleType, const vec3 &particleSize,
                      const vec3 &normal, int &globalIndex, int adaptive_level,
                      double vol, bool rigidBodyParticle = false,
                      int rigidBodyIndex = -1,
                      vec3 pCoord = vec3(0.0, 0.0, 0.0)) {
    static auto &_coord = __field.vector.GetHandle("coord");
    static auto &_normal = __field.vector.GetHandle("normal");
    static auto &_particleSize = __field.vector.GetHandle("size");
    static auto &_pCoord = __field.vector.GetHandle("parameter coordinate");
    static auto &_volume = __field.scalar.GetHandle("volume");
    static auto &_globalIndex = __field.index.GetHandle("global index");
    static auto &_adaptive_level = __field.index.GetHandle("adaptive level");
    static auto &_particleType = __field.index.GetHandle("particle type");
    static auto &_attachedRigidBodyIndex =
        __field.index.GetHandle("attached rigid body index");

    static auto &_gapCoord = __gap.vector.GetHandle("coord");
    static auto &_gapNormal = __gap.vector.GetHandle("normal");
    static auto &_gapParticleSize = __gap.vector.GetHandle("size");
    static auto &_gapParticleType = __gap.index.GetHandle("particle type");
    static auto &_gap_particle_adaptive_level =
        __gap.index.GetHandle("adaptive level");

    int idx = IsInRigidBody(X, particleSize[0]);

    if (rigidBodyParticle || idx == -2) {
      _coord.push_back(X);
      _particleType.push_back(particleType);
      _particleSize.push_back(particleSize);
      _normal.push_back(normal);
      _volume.push_back(vol);
      _globalIndex.push_back(globalIndex++);
      _adaptive_level.push_back(adaptive_level);
      _attachedRigidBodyIndex.push_back(rigidBodyIndex);
      _pCoord.push_back(pCoord);
    } else if (idx > -1) {
      _gapCoord.push_back(X);
      _gapNormal.push_back(normal);
      _gapParticleSize.push_back(particleSize);
      _gapParticleType.push_back(particleType);
      _gap_particle_adaptive_level.push_back(adaptive_level);
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

  void DataSwapAmongNeighbor(std::vector<int> &sendData,
                             std::vector<int> &recvData);
  void DataSwapAmongNeighbor(std::vector<double> &sendData,
                             std::vector<double> &recvData);
  void DataSwapAmongNeighbor(std::vector<vec3> &sendData,
                             std::vector<vec3> &recvData);
  void DataSwapAmongNeighbor(std::vector<std::vector<double>> &sendData,
                             std::vector<std::vector<double>> &recvData,
                             const int unitLength);

  MPI_Win __neighborWinCount;
  MPI_Win __neighborWinIndex;
  MPI_Win __neighborWinOffset;
  MPI_Win __neighborWinParticleSwap;

  GeneralInfo __neighbor;

  std::vector<std::vector<int>> __neighborSendParticleIndex;

  std::vector<std::vector<int>> __neighborLists;
  std::vector<double> __epsilon;

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

  void InitFieldParticle();
  void InitFieldBoundaryParticle();

  void InitFieldParticleManifold();
  void InitFieldBoundaryParticleManifold();

  void ParticleIndex();

  // particle adjustment
  bool NeedRefinement();

  // adaptive refinement
  void SplitParticle(std::vector<int> &splitTag);

  void SplitFieldParticle(std::vector<int> &splitTag);
  void SplitFieldBoundaryParticle(std::vector<int> &splitTag);
  void SplitRigidBodySurfaceParticle(std::vector<int> &splitTag);
  void SplitGapParticle(std::vector<int> &splitTag);

  // rigid body supporting functions
  int IsInRigidBody(const vec3 &pos, double h);

  void InitRigidBodySurfaceParticle();

  // equation type
  void PoissonEquation();
  void PoissonEquationManifold();
  void DiffusionEquation();
  void DiffusionEquationManifold();
  void StokesEquation();
  void StokesEquationInitialization();
  void NavierStokesEquation();

  // equation multigrid solving
  void BuildInterpolationAndRelaxationMatrices(PetscSparseMatrix &I,
                                               PetscSparseMatrix &R,
                                               int num_rigid_body,
                                               int dimension);
  void InitialGuessFromPreviousAdaptiveStep(PetscSparseMatrix &I,
                                            std::vector<double> &initial_guess);

  // function pointer
  void (GMLS_Solver::*__equationSolver)(void);
  void (GMLS_Solver::*__equationSolverInitialization)(void);
  void (GMLS_Solver::*__particleUniformInitializer)(void);
  void (GMLS_Solver::*__splitMerger)(void);

  // time integration scheme
  void ForwardEulerIntegration();
  void RungeKuttaIntegration();

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
  void WriteDataAdaptiveGeometry();

 public:
  GMLS_Solver(int argc, char **argv);

  void TimeIntegration();

  bool IsSuccessInit() { return __successInitialized; }
};