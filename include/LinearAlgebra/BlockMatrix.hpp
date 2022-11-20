#ifndef _LinearAlgebra_BlockMatrix_Hpp_
#define _LinearAlgebra_BlockMatrix_Hpp_

#include "Core/Typedef.hpp"
#include "LinearAlgebra/Matrix.hpp"
#include "LinearAlgebra/Vector.hpp"

#include <memory>
#include <vector>

namespace LinearAlgebra {
template <class LinearAlgebraBackend>
class BlockMatrix : public Matrix<LinearAlgebraBackend> {
public:
  typedef typename LinearAlgebra::Vector<LinearAlgebraBackend> DefaultVector;

protected:
  typedef typename LinearAlgebraBackend::DefaultInteger Integer;
  typedef typename LinearAlgebraBackend::DefaultScalar Scalar;

  unsigned int blockRow_, blockCol_;

  std::vector<std::shared_ptr<Matrix<LinearAlgebraBackend>>> subMatrix_;

  std::shared_ptr<typename LinearAlgebraBackend::BlockMatrixBase> blockMat_;

  std::vector<unsigned long> localLhsVectorOffset_;
  std::vector<unsigned long> localRhsVectorOffset_;

public:
  BlockMatrix();
  ~BlockMatrix();

  virtual Void Clear();
  virtual Void Resize(const Integer blockM, const Integer blockN,
                      const Integer blockSize = 1);

  virtual Integer GetLocalColSize() const;
  virtual Integer GetLocalRowSize() const;

  virtual Integer GetGlobalColSize() const;
  virtual Integer GetGlobalRowSize() const;

  std::shared_ptr<Matrix<LinearAlgebraBackend>>
  GetSubMat(const unsigned int blockI, const unsigned int blockJ);

  Void SetSubMat(const unsigned int blockI, const unsigned int blockJ,
                 std::shared_ptr<Matrix<LinearAlgebraBackend>> mat);

  virtual Void Assemble();

  virtual Void PrepareSchurComplementPreconditioner();
  virtual Void ApplySchurComplementPreconditioningIteration(DefaultVector &x,
                                                            DefaultVector &y);
};
} // namespace LinearAlgebra

#include "LinearAlgebra/Impl/BlockMatrix.hpp"

#endif