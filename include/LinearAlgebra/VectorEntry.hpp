#ifndef _LinearAlgebra_VectorEntry_Hpp_
#define _LinearAlgebra_VectorEntry_Hpp_

#include <memory>

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
template <class LinearAlgebraBackend> class VectorEntry {
protected:
  typedef typename LinearAlgebraBackend::DefaultInteger Integer;
  typedef typename LinearAlgebraBackend::DefaultScalar Scalar;

  std::shared_ptr<Matrix<LinearAlgebraBackend>> matOperandPtr_;
  std::shared_ptr<Vector<LinearAlgebraBackend>> leftVectorOperand_;
  std::shared_ptr<Vector<LinearAlgebraBackend>> rightVectorOperand_;
  std::shared_ptr<VectorEntry<LinearAlgebraBackend>> leftVectorEntryOperand_;
  std::shared_ptr<VectorEntry<LinearAlgebraBackend>> rightVectorEntryOperand_;

public:
  VectorEntry();

  ~VectorEntry();

  inline Void InsertLeftMatrixOperand(Matrix<LinearAlgebraBackend> &mat);
  inline Void InsertLeftVectorOperand(Vector<LinearAlgebraBackend> &vec);
  inline Void InsertRightVectorOperand(Vector<LinearAlgebraBackend> &vec);

  inline Boolean ExistLeftMatrixOperand();

  inline Matrix<LinearAlgebraBackend> &GetLeftMatrixOperand();
  inline Vector<LinearAlgebraBackend> &GetLeftVectorOperand();
  inline Vector<LinearAlgebraBackend> &GetRightVectorOperand();
};

} // namespace LinearAlgebra

#include "LinearAlgebra/Impl/VectorEntry.hpp"

#endif