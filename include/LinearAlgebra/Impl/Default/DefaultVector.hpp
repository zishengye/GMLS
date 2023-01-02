#ifndef _LinearAlgebra_Impl_Default_DefaultVector_Hpp_
#define _LinearAlgebra_Impl_Default_DefaultVector_Hpp_

#include "Core/Typedef.hpp"

#include <Kokkos_Core.hpp>
#include <memory>

#include "LinearAlgebra/Impl/Default/Default.hpp"
#include "LinearAlgebra/Impl/Default/DefaultVectorEntry.hpp"
#include "LinearAlgebra/Impl/Default/MapGraph.hpp"

namespace LinearAlgebra {
namespace Impl {
class DefaultVector {
  /*
   * The data creation usually takes place on host side, but the computation
   * takes place on device size. Therefore,
   */
protected:
  int mpiRank_, mpiSize_;

  Boolean isDeviceAvailable_;
  Boolean isHostDeviceSynchronized_; // is host data updated but has not been
                                     // synchronized on device
  Boolean isDeviceHostSynchronized_; // is device data update but has not been
                                     // synchronized on host

  LocalIndex localSize_;
  GlobalIndex globalSize_;

  std::shared_ptr<DeviceRealVector::HostMirror> hostDataPtr_;
  std::shared_ptr<DeviceRealVector> deviceDataPtr_;

  Void ResizeInternal(const LocalIndex localSize);

  Void HostDeviceSynchronization();
  Void DeviceHostSynchronization();

public:
  DefaultVector();
  DefaultVector(const LocalIndex localSize);

  ~DefaultVector();

  LocalIndex GetLocalSize() const;
  GlobalIndex GetGlobalSize() const;

  Void Resize(const LocalIndex localSize);

  Void Create(const std::vector<Scalar> &vec);
  Void Create(const HostRealVector &vec);
  Void Create(const DefaultVector &vec);

  Void Copy(std::vector<Scalar> &vec);
  Void Copy(HostRealVector &vec);
  Void Copy(DefaultVector &vec);

  Void Clear();

  // operation

  // norm
  Scalar Norm1();
  Scalar Norm2();
  Scalar NormInf();

  Void Normalize();

  // unary operation
  Void Scale(const Scalar factor);
  Void Set(const Scalar value);
  Void Zeros();
  Void Ones();

  Scalar &operator()(const LocalIndex index);
  const Scalar &operator()(const LocalIndex index) const;

  Void operator=(const DefaultVector &vec);
  Void operator=(const DefaultVectorEntry &vecEntry);

  Void operator+=(const DefaultVector &vec);
  Void operator+=(const DefaultVectorEntry &vecEntry);
  Void operator-=(const DefaultVector &vec);
  Void operator-=(const DefaultVectorEntry &vecEntry);

  Void operator*=(const Scalar scalar);

  DefaultVectorEntry operator*(const Scalar scalar);
  DefaultVectorEntry operator/(const Scalar scalar);

  Scalar Dot(const DefaultVector &vec);

  DeviceRealVector &GetDeviceVector();

  Void OrthogonalizeToConstant(const LocalIndex start, const LocalIndex end);
};

DefaultVectorEntry operator*(const Scalar scalar, DefaultVector &vec);
} // namespace Impl
} // namespace LinearAlgebra

#endif