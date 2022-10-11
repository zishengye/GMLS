#ifndef _LinearAlgebra_Vector_Hpp_
#define _LinearAlgebra_Vector_Hpp_

#include "Core/Typedef.hpp"
#include "LinearAlgebra/LinearAlgebraDec.hpp"

/*! \brief Vector interface.
 *! \date Sept 30, 2022
 *! \author Zisheng Ye <zisheng_ye@outlook.com>
 *
 * The main functionality of this layer is to provide a same to all
 * implementation, easy-to-read layout of vector operations required by the
 * system.  This layer supports lazy evaluation.
 */

namespace LinearAlgebra {
template <class LinearAlgebraBackend> class Vector {
protected:
  typename LinearAlgebraBackend::VectorBase vec_;

  typedef typename LinearAlgebraBackend::DefaultInteger Integer;
  typedef typename LinearAlgebraBackend::DefaultScalar Scalar;

public:
  Vector();
  Vector(const Integer localSize);

  ~Vector();

  virtual Integer GetLocalSize();

  virtual Void Create(const std::vector<Scalar> &vec);
  virtual Void Create(const typename LinearAlgebraBackend::VectorBase &vec);
  virtual Void Create(const Vector<LinearAlgebraBackend> &vec);
  virtual Void Create(const HostRealVector &vec);

  virtual Void Copy(std::vector<Scalar> &vec);
  virtual Void Copy(typename LinearAlgebraBackend::VectorBase &vec);
  virtual Void Copy(HostRealVector &vec);

  virtual Void Clear();

  virtual Vector &Scale(const Scalar scalar);

  virtual Scalar &operator()(const LocalIndex index);

  virtual Vector &operator=(Vector<LinearAlgebraBackend> &vec);
  virtual Vector &operator=(VectorEntry<LinearAlgebraBackend> vecEntry);

  virtual Vector &operator+=(const Vector<LinearAlgebraBackend> &vec);
  virtual Vector &operator-=(const Vector<LinearAlgebraBackend> &vec);
  virtual Vector &operator*=(const Scalar scalar);

  typename LinearAlgebraBackend::VectorBase &GetVector();

  friend class Matrix<LinearAlgebraBackend>;
};

} // namespace LinearAlgebra

#include "LinearAlgebra/Impl/Vector.hpp"

#endif