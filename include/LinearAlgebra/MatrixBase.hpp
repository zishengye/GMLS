#ifndef _LinearAlgebra_MatrixBase_Hpp_
#define _LinearAlgebra_MatrixBase_Hpp_

/*!
 *
 */

namespace LinearAlgebra {
template <class LinearAlgebraBackend> class MatrixBase {
protected:
  typename LinearAlgebraBackend::Matrix mat_;

  typedef typename LinearAlgebraBackend::DefaultInteger Integer;

  Integer localRowSize_, localColSize_;
  Integer globalRowSize_, globalColSize_;
  Integer colRangeLow_, colRangeHigh_;
  Integer blockSize_, blockStorage_;

public:
  MatrixBase();
  ~MatrixBase();

  void Clear();

  virtual int GetRowSize();
  virtual int GetColSize();

  virtual void Resize(const Integer m, const Integer n,
                      const Integer blockSize = 1);

  virtual unsigned long GraphAssemble();
  virtual unsigned long Assemble();
};
} // namespace LinearAlgebra

#endif