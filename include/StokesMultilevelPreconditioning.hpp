#ifndef _STOKES_MULTILEVEL_HPP_
#define _STOKES_MULTILEVEL_HPP_

#include <memory>
#include <mpi.h>
#include <vector>

#include "ParticleGeometry.hpp"
#include "PetscWrapper.hpp"
#include "StokesMatrix.hpp"
#include "petscsystypes.h"

PetscErrorCode StokesMultilevelIterationWrapper(PC pc, Vec x, Vec y);

class StokesMultilevelPreconditioning {
private:
  std::vector<std::shared_ptr<StokesMatrix>>
      linearSystemList_; // coefficient matrix list
  std::vector<std::shared_ptr<PetscNestedMatrix>>
      interpolationList_; // interpolation matrix list
  std::vector<std::shared_ptr<PetscNestedMatrix>>
      restrictionList_; // restriction matrix list

  // vector list
  std::vector<std::shared_ptr<PetscVector>> xList_;
  std::vector<std::shared_ptr<PetscVector>> yList_;
  std::vector<std::shared_ptr<PetscVector>> bList_;
  std::vector<std::shared_ptr<PetscVector>> rList_;

  std::vector<std::shared_ptr<PetscVector>> xFieldList_;
  std::vector<std::shared_ptr<PetscVector>> yFieldList_;
  std::vector<std::shared_ptr<PetscVector>> bFieldList_;
  std::vector<std::shared_ptr<PetscVector>> rFieldList_;

  std::vector<double> fieldRelaxationDuration_;
  std::vector<double> neighborRelaxationDuration_;

  // relaxation list
  std::vector<std::shared_ptr<PetscKsp>> fieldRelaxationList_;
  std::vector<std::shared_ptr<PetscKsp>> neighborRelaxationList_;
  std::vector<std::shared_ptr<PetscKsp>> wholeRelaxationList_;

  std::vector<unsigned int> numLocalParticleList_;
  std::vector<unsigned int> numGlobalParticleList_;

  int mpiRank_, mpiSize_;

  int dimension_, numRigidBody_;

  int currentRefinementLevel_;

  std::shared_ptr<ParticleGeometry> geoMgr_;

public:
  StokesMultilevelPreconditioning() : currentRefinementLevel_(-1) {}

  ~StokesMultilevelPreconditioning() { Clear(); }

  void Init(const int dimension, std::shared_ptr<ParticleGeometry> geoMgr) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);

    dimension_ = dimension;

    geoMgr_ = geoMgr;
  }

  void Reset() { Clear(); }

  inline void SetNumRigidBody(const int numRigidBody) {
    numRigidBody_ = numRigidBody;
  }

  void AddNewLevel(const unsigned int numLocalParticle) {
    linearSystemList_.push_back(std::make_shared<StokesMatrix>(dimension_));
    if (currentRefinementLevel_ >= 0) {
      interpolationList_.push_back(std::make_shared<PetscNestedMatrix>(2, 2));
      restrictionList_.push_back(std::make_shared<PetscNestedMatrix>(2, 2));
    }

    currentRefinementLevel_++;

    unsigned int numGlobalParticle;
    MPI_Allreduce(&numLocalParticle, &numGlobalParticle, 1, MPI_UNSIGNED,
                  MPI_SUM, MPI_COMM_WORLD);

    numLocalParticleList_.push_back(numLocalParticle);
    numGlobalParticleList_.push_back(numGlobalParticle);
  }

  void RemoveConstant(const unsigned int refinementLevel, Vec x);

  void Clear();

  void InitialGuess(std::vector<double> &initial_guess,
                    std::vector<Vec3> &velocity, std::vector<double> &pressure);
  void BuildInterpolationRestrictionOperators(const int numRigidBody,
                                              const int dimension);

  int Solve(std::vector<double> &rhs1, std::vector<double> &x1,
            std::vector<double> &rhs2, std::vector<double> &x2);

  std::shared_ptr<StokesMatrix> GetLinearSystem(int num_level) {
    return linearSystemList_[num_level];
  }

  PetscErrorCode MultilevelIteration(Vec x, Vec y);
};

#endif