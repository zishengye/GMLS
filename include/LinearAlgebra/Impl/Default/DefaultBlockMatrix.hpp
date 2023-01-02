#ifndef _LinearAlgebra_Impl_Default_DefaultBlockMatrix_Hpp_
#define _LinearAlgebra_Impl_Default_DefaultBlockMatrix_Hpp_

#include <memory>
#include <vector>

#include "Core/Typedef.hpp"
#include "LinearAlgebra/BlockMatrix.hpp"
#include "LinearAlgebra/Impl/Default/Default.hpp"
#include "LinearAlgebra/Impl/Default/DefaultBackend.hpp"
#include "LinearAlgebra/Impl/Default/DefaultLinearSolver.hpp"
#include "LinearAlgebra/Impl/Default/DefaultMatrix.hpp"
#include "LinearAlgebra/Impl/Default/DefaultVector.hpp"

namespace LinearAlgebra {
namespace Impl {
class DefaultBlockMatrix : public DefaultMatrix {
protected:
  std::vector<std::shared_ptr<DefaultMatrix>> subMat_;

  std::vector<std::size_t> localLhsVectorOffset_;
  std::vector<std::size_t> localRhsVectorOffset_;

  std::vector<DefaultVector> lhsVector_;
  std::vector<DefaultVector> rhsVector_;

  LocalIndex blockM_, blockN_;

  DefaultLinearSolver a00LinearSolver_, a11LinearSolver_;

  Scalar a00Timer_, a11Timer_;

public:
  DefaultBlockMatrix();

  ~DefaultBlockMatrix();

  Void ClearTimer();

  Scalar GetA00Timer();
  Scalar GetA11Timer();

  virtual Void Resize(const GlobalIndex blockM, const GlobalIndex blockN,
                      const LocalIndex blockSize = 1);

  std::shared_ptr<DefaultMatrix> GetSubMat(const LocalIndex blockI,
                                           const LocalIndex blockJ);
  Void SetSubMat(const LocalIndex blockI, const LocalIndex blockJ,
                 std::shared_ptr<DefaultMatrix> mat);

  Void SetCallbackPointer(
      LinearAlgebra::BlockMatrix<LinearAlgebra::Impl::DefaultBackend> *ptr);

  virtual Void Assemble();

  virtual Void MatrixVectorMultiplication(DefaultVector &vec1,
                                          DefaultVector &vec2);
  virtual Void MatrixVectorMultiplicationAddition(DefaultVector &vec1,
                                                  DefaultVector &vec2);

  virtual Void PrepareSchurComplementPreconditioner();
  virtual Void
  ApplySchurComplementPreconditioningIteration(DefaultVector &vec1,
                                               DefaultVector &vec2);

  friend class DefaultLinearSolver;
};
} // namespace Impl
} // namespace LinearAlgebra

#endif