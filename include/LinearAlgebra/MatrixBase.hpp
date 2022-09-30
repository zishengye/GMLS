#ifndef _LinearAlgebra_MatrixBase_Hpp_
#define _LinearAlgebra_MatrixBase_Hpp_

/*!
 *
 */

#include "Core/Typedef.hpp"

namespace LinearAlgebra {
template <class LinearAlgebraBackend> class MatrixBase {
protected:
  typename LinearAlgebraBackend::Matrix mat_;

  typedef typename LinearAlgebraBackend::DefaultInteger Integer;
  typedef typename LinearAlgebraBackend::DefaultScalar Scalar;

  Integer localRowSize_, localColSize_;
  Integer globalRowSize_, globalColSize_;
  Integer colRangeLow_, colRangeHigh_;
  Integer blockSize_, blockStorage_;

public:
  MatrixBase();
  ~MatrixBase();

  Void Clear();

  virtual Integer GetRowSize();
  virtual Integer GetColSize();

  virtual Void Resize(const Integer m, const Integer n,
                      const Integer blockSize = 1);

  virtual Void SetColIndex(const Integer row,
                           const std::vector<Integer> &colIndex);
  virtual Void Increment(const Integer row, const Integer col,
                         const Scalar value);
  virtual Void Increment(const Integer row, const std::vector<Integer> colIndex,
                         const std::vector<Scalar> &value);

  virtual Void GraphAssemble();
  virtual Void Assemble();
};
} // namespace LinearAlgebra

#endif