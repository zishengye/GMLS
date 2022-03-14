#ifndef _HYDROGEN_OPERATION_HPP_
#define _HYDROGEN_OPERATION_HPP_

#include "device.hpp"
#include "host.hpp"

namespace hydrogen {
template <class T>
struct isExecutionSpace {
  bool value() {
    return false;
  }
};

template <class HostSpace_, class DeviceSpace_>
class ExecutionSpace {
 public:
  typedef HostSpace_ HostSpace;
  typedef DeviceSpace_ DeviceSpace;
};
}  // namespace hydrogen

#endif