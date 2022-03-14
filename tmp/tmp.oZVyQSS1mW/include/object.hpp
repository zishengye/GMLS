#ifndef _HYDROGEN_OBJECT_HPP_
#define _HYDROGEN_OBJECT_HPP_

#include "execution.hpp"
#include "type.hpp"

namespace hydrogen {
template <class ExecutionSpace_>
class Object {
 private:
  Boolean hostDeviceSynchronization_;
  Integer hostMemoryMode_;
  Integer deviceMemoryMode_;
  Integer lastMemoryAction_;

 protected:
  typedef typename ExecutionSpace_::HostSpace HostSpace_;
  typedef typename ExecutionSpace_::DeviceSpace DeviceSpace_;

  virtual void synchronizeInternal();
  virtual void synchronizeToDeviceInternal();
  virtual void synchronizeFromDeviceInternal();

  virtual void releaseMemory();
  virtual void requestMemory(std::size_t memorySize);

  virtual void releaseHostMemory();
  virtual void requestHostMemory(std::size_t memorySize);

  virtual void releaseDeviceMemory();
  virtual void requestDeviceMemory(std::size_t memorySize);

  virtual void *getPointer();
  virtual void releasePointer();

  virtual void registerActionOnHost();
  virtual void registerActionOnDevice();

 public:
  typedef ExecutionSpace_ ExecutionSpace;
  Object();

  ~Object();

  virtual void fence();

  // Update data according to last operation working location.
  virtual void update();
  // Explicitly choose the place to update data.
  virtual void updateFromDevice();
  virtual void updateToDevice();
};
}  // namespace hydrogen

#include "impl/object.hpp"

#endif