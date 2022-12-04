#ifndef _LinearAlgebra_Matrix_Hpp_
#define _LinearAlgebra_Matrix_Hpp_

/*! \brief Abstract layer of matrix used in linear algebra.
 *! \author Zisheng Ye <zisheng_ye@outlook.com>
 */

#include "Core/Typedef.hpp"
#include "LinearAlgebra/Vector.hpp"

#include <memory>

namespace LinearAlgebra {
template <class LinearAlgebraBackend> class Matrix {
protected:
  std::shared_ptr<typename LinearAlgebraBackend::MatrixBase> mat_;

  typedef typename LinearAlgebraBackend::DefaultInteger Integer;
  typedef typename LinearAlgebraBackend::DefaultScalar Scalar;

public:
  Matrix();
  ~Matrix();

  virtual Void Clear();

  virtual Void Resize(const Integer m, const Integer n,
                      const Integer blockSize = 1);

  virtual Void Transpose(Matrix<LinearAlgebraBackend> &mat);

  virtual Integer GetLocalColSize() const;
  virtual Integer GetLocalRowSize() const;

  virtual Integer GetGlobalColSize() const;
  virtual Integer GetGlobalRowSize() const;

  inline virtual Void SetColIndex(const Integer row,
                                  const std::vector<Integer> &colIndex);
  inline virtual Void Increment(const Integer row, const Integer col,
                                const Scalar value);
  inline virtual Void Increment(const Integer row,
                                const std::vector<Integer> colIndex,
                                const std::vector<Scalar> &value);

  virtual Void GraphAssemble();
  virtual Void Assemble();

  VectorEntry<LinearAlgebraBackend>
  operator*(Vector<LinearAlgebraBackend> &vec);

  virtual Void MatrixVectorMultiplication(Vector<LinearAlgebraBackend> &vec1,
                                          Vector<LinearAlgebraBackend> &vec2);

  std::shared_ptr<typename LinearAlgebraBackend::MatrixBase> GetMatrix();
};
} // namespace LinearAlgebra

#include "LinearAlgebra/Impl/Matrix.hpp"

#endif