#ifndef _PetscNestedVec_Hpp_
#define _PetscNestedVec_Hpp_

#include "PetscVector.hpp"

#include <vector>

class PetscNestedVec : public PetscVector {
protected:
  std::vector<Vec> nestedVec_;

public:
  PetscNestedVec(const PetscInt size);
  ~PetscNestedVec();

  void Create(const PetscInt index, const std::vector<double> &vec);
  void Create(const PetscInt index, const PetscInt size);
  void Create();

  void Copy(const PetscInt index, std::vector<double> &vec);

  void Duplicate(PetscNestedVec &vec);

  Vec &GetSubVector(const PetscInt index);
};

#endif