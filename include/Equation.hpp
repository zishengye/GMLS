#ifndef _Equation_Hpp_
#define _Equation_Hpp_

#include <functional>

#include "MultilevelPreconditioning.hpp"
#include "ParticleManager.hpp"
#include "PetscKsp.hpp"
#include "PetscMatrixBase.hpp"
#include "Typedef.hpp"

#include <Compadre_PointCloudSearch.hpp>

class BoundaryCondition {};

class Field {};

enum RefinementMethod { UniformRefinement, AdaptiveRefinement };

class Equation {
protected:
  double globalError_, globalNormalizedError_, errorTolerance_, markRatio_;
  unsigned int maxRefinementIteration_, refinementIteration_;
  RefinementMethod refinementMethod_;

  HostRealVector error_;

  HostIndexMatrix neighborLists_;
  HostRealVector epsilon_;

  HostIndexVector splitTag_;

  std::vector<std::shared_ptr<PetscMatrixBase>> linearSystemsPtr_;
  std::shared_ptr<MultilevelPreconditioning> preconditionerPtr_;
  PetscVector b_;
  PetscVector x_;

  PetscKsp ksp_;

  HostRealMatrix hostGhostParticleCoords_;
  HostIndexVector hostGhostParticleType_;
  HostIndexVector hostGhostParticleIndex_;

  void AddLinearSystem(std::shared_ptr<PetscMatrixBase> mat);
  virtual void InitLinearSystem();
  virtual void ConstructLinearSystem();
  virtual void ConstructRhs();

  virtual void DiscretizeEquation();
  virtual void InitPreconditioner();
  virtual void SolveEquation();
  virtual void CalculateError();
  virtual void Mark();

  virtual void BuildGhost();
  virtual void Output();

  virtual void ConstructNeighborLists(const unsigned int satisfiedNumNeighbor);

  HierarchicalParticleManager particleMgr_;

  int mpiRank_, mpiSize_;

  unsigned int outputLevel_, polyOrder_;

  Ghost ghost_;
  double ghostMultiplier_;

public:
  Equation();

  void SetPolyOrder(const unsigned int polyOrder);
  void SetDimension(const unsigned int dimension);
  void SetDomainType(const SimpleDomainShape shape);
  void SetDomainSize(const std::vector<Scalar> &size);
  void SetInitialDiscretizationResolution(const double spacing);
  void SetErrorTolerance(const double errorTolerance);
  void SetRefinementMethod(const RefinementMethod refinementMethod);
  void SetMaxRefinementIteration(const unsigned int maxRefinementIteration);
  void SetOutputLevel(const unsigned int outputLevel);
  void SetGhostMultiplier(const double multiplier);
  void SetRefinementMarkRatio(const double ratio = 0.8);

  virtual void Init();
  virtual void Update();
};

#endif