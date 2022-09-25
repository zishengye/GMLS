#ifndef _PetscStokesMatrix_Hpp_
#define _PetscStokesMatrix_Hpp_

#include "PetscNestedMatrix.hpp"
#include "petscsystypes.h"

PetscErrorCode FieldMatrixMultWrapper(Mat mat, Vec x, Vec y);
PetscErrorCode FieldMatrixSORWrapper(Mat mat, Vec b, PetscReal omega,
                                     MatSORType flag, PetscReal shift,
                                     PetscInt its, PetscInt lits, Vec x);

class StokesMatrix : public PetscNestedMatrix {
private:
  void InitInternal();

protected:
  unsigned int dimension_, numLocalParticle_, numGlobalParticle_;
  unsigned int fieldDof_, velocityDof_;
  unsigned int rigidBodyDof_, numRigidBody_, numLocalRigidBody_;
  unsigned int rigidBodyStartIndex_, rigidBodyEndIndex_;
  unsigned int translationDof_, rotationDof_;

  IS isgNeighbor_, isgColloid_, isgField_;
  Mat neighborSchurMat_, neighborMat_, schurMat_;

  std::vector<Mat> nestedNeighborMat_;
  std::vector<PetscInt> idxNeighbor_;
  std::vector<PetscInt> idxNonNeighborPressure_, idxNeighborPressure_;

  std::shared_ptr<PetscVector> x_, b_;
  std::vector<std::vector<PetscInt>> rigidBodyFieldIndexMap_;
  std::vector<std::vector<PetscReal>> rigidBodyFieldValueMap_;

public:
  StokesMatrix();
  StokesMatrix(const unsigned int dimension);
  StokesMatrix(const unsigned long numLocalParticle,
               const unsigned int dimension);
  StokesMatrix(const unsigned long numLocalParticle,
               const unsigned int numRigidBody, const unsigned int dimension);
  ~StokesMatrix();

  void SetSize(const unsigned long numLocalParticle,
               const unsigned int numRigidBody);

  void SetGraph(const std::vector<int> &localIndex,
                const std::vector<int> &globalIndex,
                const std::vector<int> &particleType,
                const std::vector<int> &attachedRigidBody,
                const Kokkos::View<int **, Kokkos::HostSpace> &neighborLists);

  unsigned long Assemble();

  void IncrementVelocityVelocity(const PetscInt row,
                                 const std::vector<PetscInt> &index,
                                 const std::vector<PetscReal> &value);
  void IncrementVelocityPressure(const PetscInt row,
                                 const std::vector<PetscInt> &index,
                                 const std::vector<PetscReal> &value);
  void IncrementVelocityRigidBody(const PetscInt row, const PetscInt index,
                                  const PetscReal value);

  void IncrementPressureVelocity(const PetscInt row,
                                 const std::vector<PetscInt> &index,
                                 const std::vector<PetscReal> &value);
  void IncrementPressurePressure(const PetscInt row,
                                 const std::vector<PetscInt> &index,
                                 const std::vector<PetscReal> &value);
  void IncrementPressureRigidBody(const PetscInt row, const PetscInt index,
                                  const PetscReal value);

  void IncrementRigidBodyVelocity(const PetscInt row, const PetscInt index,
                                  const PetscReal value);
  void IncrementRigidBodyPressure(const PetscInt row, const PetscInt index,
                                  const PetscReal value);
  void IncrementRigidBodyRigidBody(const PetscInt row, const PetscInt index,
                                   const PetscReal value);

  Mat &GetNeighborNeighborMatrix() { return neighborMat_; }

  Mat &GetNeighborNeighborSubMatrix(const PetscInt row, const PetscInt col) {
    return nestedNeighborMat_[row * 2 + col];
  }

  Mat &GetNeighborSchurMatrix() { return neighborSchurMat_; }

  Mat &GetSchurMatrix() { return schurMat_; }

  Mat &GetFieldFieldShellMatrix() { return nestedMat_[0]; }

  std::shared_ptr<PetscVector> &GetNeighborX() { return x_; }

  std::shared_ptr<PetscVector> &GetNeighborB() { return b_; }

  IS &GetNeighborIS() { return isgNeighbor_; }

  IS &GetFieldIS() { return isgField_; }

  IS &GetColloidIS() { return isgColloid_; }

  PetscErrorCode FieldMatrixMult(Vec x, Vec y);

  PetscErrorCode FieldMatrixSOR(Vec b, PetscReal omega, MatSORType flag,
                                PetscReal shift, PetscInt its, PetscInt lits,
                                Vec x);

  void ConstantVec(Vec v);

  void ConstantVecNonNeighborPressure(Vec v);

  void ForwardField(Vec x, Vec y);

  void BackwardField(Vec x, Vec y);

  void ForwardNeighbor(Vec x, Vec y);

  void BackwardNeighbor(Vec x, Vec y);
};

#endif