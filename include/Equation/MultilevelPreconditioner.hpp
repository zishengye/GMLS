#ifndef _Equation_MultilevelPreconditioner_Hpp_
#define _Equation_MultilevelPreconditioner_Hpp_

#include <memory>
#include <vector>

#include "Core/Typedef.hpp"
#include "Geometry/DomainGeometry.hpp"
#include "Geometry/Ghost.hpp"
#include "Geometry/ParticleGeometry.hpp"
#include "LinearAlgebra/LinearAlgebra.hpp"

namespace Equation {
class MultilevelPreconditioner {
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
  int mpiRank_, mpiSize_;

  std::vector<std::shared_ptr<DefaultMatrix>> linearSystemsPtr_;
  std::vector<std::shared_ptr<DefaultMatrix>> adjointLinearSystemsPtr_;
  std::vector<std::shared_ptr<DefaultMatrix>> interpolationPtr_;
  std::vector<std::shared_ptr<DefaultMatrix>> restrictionPtr_;

  std::vector<DefaultVector> auxiliaryVectorXPtr_;
  std::vector<DefaultVector> auxiliaryVectorRPtr_;
  std::vector<DefaultVector> auxiliaryVectorBPtr_;

  std::vector<std::shared_ptr<DefaultLinearSolver>> preSmootherPtr_,
      postSmootherPtr_;
  std::vector<DefaultLinearSolver> adjointSmootherPtr_;

  std::vector<double> fieldRelaxationDuration_;

  Geometry::Ghost interpolationGhost_, restrictionGhost_;

public:
  MultilevelPreconditioner();

  ~MultilevelPreconditioner();

  Void ClearTimer();
  double GetFieldRelaxationTimer(const unsigned int level);

  virtual Void ApplyPreconditioningIteration(DefaultVector &x,
                                             DefaultVector &y);
  Void ApplyAdjointPreconditioningIteration(DefaultVector &x, DefaultVector &y);

  DefaultMatrix &GetInterpolation(const Size level);
  DefaultMatrix &GetRestriction(const Size level);
  DefaultLinearSolver &GetPreSmoother(const Size level);
  DefaultLinearSolver &GetPostSmoother(const Size level);
  DefaultLinearSolver &GetAdjointSmoother(const Size level);

  Void AddLinearSystem(std::shared_ptr<DefaultMatrix> &mat);
  Void AddAdjointLinearSystem(std::shared_ptr<DefaultMatrix> &mat);
  Void PrepareVectors(const Size localSize);
  virtual Void ConstructInterpolation(DefaultParticleManager &particleMgr);
  virtual Void ConstructRestriction(DefaultParticleManager &particleMgr);
  virtual Void ConstructSmoother();
  virtual Void ConstructAdjointSmoother();
};
} // namespace Equation

#endif