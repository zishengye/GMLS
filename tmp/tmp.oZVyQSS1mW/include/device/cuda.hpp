#ifndef _HYDROGEN_DEVICE_CUDA_HPP_
#define _HYDROGEN_DEVICE_CUDA_HPP_

#include "device.hpp"

namespace hydrogen {
class CudaDeviceSpace : public DeviceSpace {
 protected:
 public:
  CudaDeviceSpace();
  CudaDeviceSpace(CudaDeviceSpace &space);

  virtual void printSummary(std::ostream &output);

  virtual void *malloc(std::size_t size, Boolean isAligned = true);
  virtual void free(void *ptr);
};
}  // namespace hydrogen

#endif  // HYDROGEN_CUDA_HPP
