#ifndef _PetscStokesMatrix_Hpp_
#define _PetscStokesMatrix_Hpp_

#include "PetscNestedMatrix.hpp"

class StokesMatrix : public PetscNestedMatrix {
private:
  void InitInternal();

protected:
  unsigned int dimension_, numLocalParticle_;
  unsigned int fieldDof_, velocityDof_;
  unsigned int rigidBodyDof_, numRigidBody_, numLocalRigidBody_;
  unsigned int rigidBodyStartIndex_, rigidBodyEndIndex_;
  unsigned int translationDof_, rotationDof_;

public:
  StokesMatrix();
  StokesMatrix(const unsigned int dimension);
  StokesMatrix(const unsigned long numLocalParticle,
               const unsigned int dimension);
  StokesMatrix(const unsigned long numLocalParticle,
               const unsigned int numRigidBody, const unsigned int dimension);

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
};

#endif