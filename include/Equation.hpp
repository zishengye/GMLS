#ifndef _EQUATION_HPP_
#define _EQUATION_HPP_

#include "ParticleManager.hpp"
#include "PetscMatrix.hpp"
#include "Typedef.hpp"

#include <Compadre_PointCloudSearch.hpp>

class BoundaryCondition {};

class Field {};

enum RefinementMethod { UniformRefinement, AdaptiveRefinement };

class Equation {
protected:
  double errorTolerance_;
  int maxRefinementIteration_, refinementIteration_;
  RefinementMethod refinementMethod_;

  HostRealVector error_;

  HostIntMatrix neighborLists_;
  HostRealVector epsilon_;

  std::vector<PetscMatrix> linearSystems_;
  std::vector<PetscVector> b_;
  std::vector<PetscVector> x_;

  virtual void InitLinearSystem();
  virtual void ConstructLinearSystem();
  virtual void ConstructRhs();

  virtual void DiscretizeEquation();
  virtual void InitPreconditioner();
  virtual void SolveEquation();
  virtual void CalculateError();
  virtual void Refine();

  virtual void Output();

  virtual void ConstructNeighborLists(const int satisfiedNumNeighbor);

  HierarchicalParticleManager particleMgr_;

  int mpiRank_, mpiSize_;

  int outputLevel_, polyOrder_;

public:
  Equation();

  void SetPolyOrder(const int polyOrder);
  void SetDimension(const int dimension);
  void SetDomainType(const SimpleDomainShape shape);
  void SetDomainSize(const std::vector<Scalar> &size);
  void SetInitialDiscretizationResolution(const double spacing);
  void SetErrorTolerance(const double errorTolerance);
  void SetRefinementMethod(const RefinementMethod refinementMethod);
  void SetMaxRefinementIteration(const int maxRefinementIteration);

  virtual void Init();
  virtual void Update();
  const double GetError();
  const int GetRefinementIteration();
};

#endif