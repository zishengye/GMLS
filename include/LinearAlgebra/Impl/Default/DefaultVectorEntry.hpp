#ifndef _LinearAlgebra_Impl_Default_DefaultVectorEntry_Hpp_
#define _LinearAlgebra_Impl_Default_DefaultVectorEntry_Hpp_

#include "Core/Typedef.hpp"

#include <Kokkos_Core.hpp>
#include <memory>

#include "LinearAlgebra/Impl/Default/Default.hpp"

namespace LinearAlgebra {
namespace Impl {
class DefaultVectorEntry {
protected:
  std::shared_ptr<DefaultMatrix> matOperandPtr_;
  std::shared_ptr<DefaultVector> leftVectorOperandPtr_;
  std::shared_ptr<DefaultVector> rightVectorOperandPtr_;
  std::shared_ptr<DefaultVectorEntry> leftVectorEntryOperandPtr_;
  std::shared_ptr<DefaultVectorEntry> rightVectorEntryOperandPtr_;

  char operatorName_;

  Scalar scalar_;

  Boolean isRightOperandScalar_;

public:
  DefaultVectorEntry();

  ~DefaultVectorEntry();

  Void InsertLeftMatrixOperand(const std::shared_ptr<DefaultMatrix> &matPtr);
  Void InsertLeftVectorOperand(const std::shared_ptr<DefaultVector> &vecPtr);
  Void InsertRightVectorOperand(const std::shared_ptr<DefaultVector> &vecPtr);
  Void InsertRightScalarOperand(const Scalar scalar);
  Void InsertOperator(const char operatorName);

  Boolean ExistLeftMatrixOperand() const;
  Boolean ExistRightScalarOperand() const;

  const std::shared_ptr<DefaultMatrix> &GetLeftMatrixOperand() const;
  const std::shared_ptr<DefaultVector> &GetLeftVectorOperand() const;
  const std::shared_ptr<DefaultVector> &GetRightVectorOperand() const;
  Scalar GetRightScalarOperand() const;
  char GetOperatorName() const;
};
} // namespace Impl
} // namespace LinearAlgebra

#endif