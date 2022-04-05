#ifndef _PETSCVECTOR_HPP_
#define _PETSCVECTOR_HPP_

#include <vector>

#include <petscksp.h>

#include "Typedef.hpp"

class PetscVector {
protected:
  Vec vec_;

public:
  PetscVector();

  ~PetscVector();

  void Create(std::vector<double> &vec);
  void Create(PetscVector &vec);
  void Create(HostRealVector &vec);

  void Copy(std::vector<double> &vec);
  void Copy(HostRealVector &vec);

  void Clear();

  Vec &GetReference();
  Vec *GetPointer();
};

#endif