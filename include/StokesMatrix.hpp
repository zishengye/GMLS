#ifndef _PetscStokesMatrix_Hpp_
#define _PetscStokesMatrix_Hpp_

#include "PetscNestedMatrix.hpp"
#include "PetscNestedVec.hpp"
#include "petscsystypes.h"

PetscErrorCode FieldMatrixMultWrapper(Mat mat, Vec x, Vec y);

class StokesMatrix : public PetscNestedMatrix {
private:
  void InitInternal();

protected:
  unsigned int dimension_, numLocalParticle_, numGlobalParticle_;
  unsigned int fieldDof_, velocityDof_;
  unsigned int rigidBodyDof_, numRigidBody_, numLocalRigidBody_;
  unsigned int rigidBodyStartIndex_, rigidBodyEndIndex_;
  unsigned int translationDof_, rotationDof_;

  IS isgNeighbor_;
  Mat neighborSchurMat_, neighborMat_, neighborWholeMat_;
  Mat fieldFieldShellMat_;

  std::vector<Mat> nestedNeighborMat_;
  std::vector<Mat> nestedNeighborWholeMat_;

  std::shared_ptr<PetscNestedVec> x_, b_;

public:
  StokesMatrix();
  StokesMatrix(const unsigned int dimension);
  StokesMatrix(const unsigned long numLocalParticle,
               const unsigned int dimension);
  StokesMatrix(const unsigned long numLocalParticle,
               const unsigned int numRigidBody, const unsigned int dimension);
  ~StokesMatrix();

  void Resize(const unsigned long numLocalParticle,
              const unsigned int numRigidBody);

  void SetGraph(const std::vector<int> &localIndex,
                const std::vector<int> &globalIndex,
                const std::vector<int> &particleType,
                const std::vector<int> &attachedRigidBody,
                const Kokkos::View<int **, Kokkos::HostSpace> &neighborLists);

  unsigned long Assemble();

  void IncrementFieldField(const PetscInt row,
                           const std::vector<PetscInt> &index,
                           const std::vector<PetscReal> &value);
  void IncrementFieldRigidBody(const PetscInt row, const PetscInt index,
                               const PetscInt value);
  void IncrementRigidBodyField(const PetscInt row, const PetscInt index,
                               const PetscInt value);
  void IncrementRigidBodyRigidBody(const PetscInt row, const PetscInt index,
                                   const PetscInt value);

  Mat &GetNeighborNeighborMatrix() { return neighborMat_; }

  Mat &GetNeighborWholeMatrix() { return neighborWholeMat_; }

  Mat &GetNeighborNeighborSubMatrix(const PetscInt row, const PetscInt col) {
    return nestedNeighborMat_[row * 2 + col];
  }

  Mat &GetNeighborSchurMatrix() { return neighborSchurMat_; }

  std::shared_ptr<PetscNestedVec> &GetNeighborX() { return x_; }

  std::shared_ptr<PetscNestedVec> &GetNeighborB() { return b_; }

  IS &GetNeighborIS() { return isgNeighbor_; }

  PetscErrorCode FieldMatrixMult(Vec x, Vec y);
};

#endif