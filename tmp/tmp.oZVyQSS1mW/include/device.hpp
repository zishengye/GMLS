#ifndef _HYDROGEN_DEVICE_HPP_
#define _HYDROGEN_DEVICE_HPP_

#include "memory/memory.hpp"

#include "type.hpp"

#include <proton.hpp>

#include <iostream>

namespace hydrogen {
struct DeviceSpaceParameter {};

static DeviceSpaceParameter defaultDeviceSpaceParameter;

struct SequentialDevice {};

class DeviceSpace {
 protected:
  static UnsignedInteger deviceSpaceInstanceNum_;

  DeviceSpaceParameter parameter_;

  void initialize();
  void initialize(DeviceSpaceParameter &parameter);
  void finalize();
  void exceptionFinalize();

  void *ptr_;
  std::size_t memorySize_;

 public:
  DeviceSpace();
  DeviceSpace(DeviceSpaceParameter &parameter, Boolean replace = false);
  DeviceSpace(DeviceSpace &space);

  virtual void printSummary(std::ostream &output);

  virtual void *malloc(std::size_t size, Boolean isAligned = true) = 0;
  virtual void free() = 0;

  ~DeviceSpace();

  template <typename Functor>
  void forEach(int start, int end, int stride, Functor functor);
};
}  // namespace hydrogen

#endif