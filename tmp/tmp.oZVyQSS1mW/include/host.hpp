#ifndef _HYDROGEN_HOST_HPP_
#define _HYDROGEN_HOST_HPP_

#include "file/file.hpp"
#include "memory/memory.hpp"

#include "type.hpp"

#include <proton.hpp>

namespace hydrogen {
struct HostSpaceParameter {
  UnsignedInteger hostThreadPerCore_;

  HostSpaceParameter();
  HostSpaceParameter(HostSpaceParameter &param);
};

static HostSpaceParameter defaultHostSpaceParameter;

// host execution space
class HostSpace {
 protected:
  static UnsignedInteger hostSpaceInstanceNum_;

  HostSpaceParameter parameter_;

  void initialize();
  void initialize(HostSpaceParameter &parameter);
  void finalize();
  void exceptionFinalize();

  void *ptr_;
  std::size_t memorySize_;

 public:
  HostSpace();
  HostSpace(HostSpaceParameter &parameter, Boolean replace = false);
  HostSpace(HostSpace &space);

  void printSummary();

  void *allocate(std::size_t size);
  void deallocate(void *const ptr);

  ~HostSpace();

  template <typename Functor>
  void forEach(int start, int end, int stride, Functor functor);
};

// host execution policy
class HostPolicy {
 private:
  static UnsignedInteger HostPolicyInstanceNum_;

 protected:
  void initialize();
  void finalize();
  void exceptionFinalize();

 public:
  HostPolicy() = delete;
  HostPolicy(HostPolicy &policy);

  ~HostPolicy();
};
}  // namespace hydrogen

#include "impl/host.hpp"

#endif  // _HYDROGEN_HOST_HPP_