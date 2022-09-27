#ifndef _PetscVector_Hpp_
#define _PetscVector_Hpp_

#include <vector>

#include "Typedef.hpp"

namespace LinearAlgebra {
template <class LinearAlgebraBackend> class Vector {
protected:
  Vec vec_;

public:
  PetscVector();
  PetscVector(const PetscInt localSize);

  ~PetscVector();

  void Create(std::vector<double> &vec);
  void Create(PetscVector &vec);
  void Create(HostRealVector &vec);
  void Create(const PetscInt size);

  void Copy(std::vector<double> &vec);
  void Copy(HostRealVector &vec);

  void Clear();

  Vec &GetReference();
  Vec *GetPointer();
};
} // namespace LinearAlgebra

#endif