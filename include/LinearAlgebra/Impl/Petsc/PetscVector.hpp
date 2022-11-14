#ifndef _LinearAlgebra_Impl_Petsc_PetscVector_Hpp_
#define _LinearAlgebra_Impl_Petsc_PetscVector_Hpp_

#include <vector>

#include <petscksp.h>

#include "Core/Typedef.hpp"
#include "LinearAlgebra/Impl/Petsc/Petsc.hpp"

namespace LinearAlgebra {
namespace Impl {
class PetscVector {
protected:
  std::shared_ptr<Vec> vecPtr_;

  Scalar *ptr_;

  PetscInt localSize_;

public:
  PetscVector();
  PetscVector(const PetscInt localSize);

  ~PetscVector();

  PetscInt GetLocalSize();

  Void Create(const std::vector<Scalar> &vec);
  Void Create(const Vec &vec);
  Void Create(const PetscVector &vec);
  Void Create(const HostRealVector &vec);

  Void Copy(std::vector<Scalar> &vec);
  Void Copy(Vec &vec);
  Void Copy(PetscVector &vec);
  Void Copy(HostRealVector &vec);

  Void Clear();

  Void Scale(const Scalar scalar);

  Scalar &operator()(const LocalIndex index);
  Void operator=(const PetscVector &vec);

  Void operator+=(const PetscVector &vec);
  Void operator-=(const PetscVector &vec);
  Void operator*=(const PetscReal scalar);

  friend class PetscKsp;
  friend class PetscMatrix;
  friend class PetscBlockMatrix;
};
} // namespace Impl
} // namespace LinearAlgebra

#endif