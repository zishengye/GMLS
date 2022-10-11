#ifndef _Equation_Equation_Hpp_
#define _Equation_Equation_Hpp_

#include <memory>
#include <vector>

#include <Compadre_PointCloudSearch.hpp>

#include "Core/Typedef.hpp"
#include "Equation/MultilevelPreconditioner.hpp"
#include "Geometry/DomainGeometry.hpp"
#include "Geometry/Ghost.hpp"
#include "Geometry/ParticleGeometry.hpp"
#include "LinearAlgebra/LinearAlgebra.hpp"
#include "LinearAlgebra/LinearSolver.hpp"

namespace Equation {
enum RefinementMethod { UniformRefinement, AdaptiveRefinement };

class Equation {
public:
  typedef typename Geometry::HierarchicalEulerianParticleManager
      DefaultParticleManager;

  typedef typename LinearAlgebra::Vector<DefaultLinearAlgebraBackend>
      DefaultVector;
  typedef typename LinearAlgebra::Matrix<DefaultLinearAlgebraBackend>
      DefaultMatrix;
  typedef typename LinearAlgebra::LinearSolver<DefaultLinearAlgebraBackend>
      DefaultLinearSolver;

protected:
  double globalError_, globalNormalizedError_, errorTolerance_, markRatio_;
  Size maxRefinementIteration_, refinementIteration_;
  RefinementMethod refinementMethod_;

  HostRealVector error_;

  HostIndexMatrix neighborLists_;
  HostRealVector epsilon_;

  HostIndexVector splitTag_;

  std::vector<std::shared_ptr<DefaultMatrix>> linearSystemsPtr_;
  std::shared_ptr<MultilevelPreconditioner> preconditionerPtr_;
  DefaultVector b_;
  DefaultVector x_;

  DefaultLinearSolver solver_;

  LinearAlgebra::LinearSolverDescriptor<DefaultLinearAlgebraBackend>
      descriptor_;

  HostRealMatrix hostGhostParticleCoords_;
  HostIndexVector hostGhostParticleType_;
  HostIndexVector hostGhostParticleIndex_;

  Void AddLinearSystem(std::shared_ptr<DefaultMatrix> mat);
  virtual Void InitLinearSystem();
  virtual Void ConstructLinearSystem();
  virtual Void ConstructRhs();

  virtual Void DiscretizeEquation();
  virtual Void InitPreconditioner();
  virtual Void SolveEquation();
  virtual Void CalculateError();
  virtual Void Mark();

  virtual Void BuildGhost();
  virtual Void Output();

  virtual Void ConstructNeighborLists(const Size satisfiedNumNeighbor);

  DefaultParticleManager particleMgr_;

  int mpiRank_, mpiSize_;

  Size outputLevel_, polyOrder_;

  Geometry::Ghost ghost_;
  Scalar ghostMultiplier_;

public:
  Equation();

  Void SetPolyOrder(const Size polyOrder);
  Void SetDimension(const Size dimension);
  Void SetDomainType(const Geometry::SupportedDomainShape shape);
  Void SetDomainSize(const std::vector<Scalar> &size);
  Void SetInitialDiscretizationResolution(const Scalar spacing);
  Void SetErrorTolerance(const Scalar errorTolerance);
  Void SetRefinementMethod(const RefinementMethod refinementMethod);
  Void SetMaxRefinementIteration(const Size maxRefinementIteration);
  Void SetOutputLevel(const Size outputLevel);
  Void SetGhostMultiplier(const Scalar multiplier);
  Void SetRefinementMarkRatio(const Scalar ratio = 0.8);

  virtual Void Init();
  virtual Void Update();
};
} // namespace Equation

#endif