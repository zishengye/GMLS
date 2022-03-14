#ifndef _HYDROGEN_IMPL_HOST_HPP_
#define _HYDROGEN_IMPL_HOST_HPP_

#include "type.hpp"

namespace hydrogen {
UnsignedInteger HostSpace::hostSpaceInstanceNum_ = 0;

HostSpaceParameter::HostSpaceParameter() {
  hostThreadPerCore_ = 1;

#ifdef ENABLE_OPENMP
  hostThreadPerCore_ = omp_get_num_threads();
#endif
}

void HostSpace::initialize() {
}

void HostSpace::initialize(HostSpaceParameter &parameter) {
  parameter_ = parameter;

  initialize();
}

void HostSpace::finalize() {
}

void HostSpace::exceptionFinalize() {
}

HostSpace::HostSpace() {
  initialize();
}

HostSpace::HostSpace(HostSpaceParameter &parameter, Boolean replace) {
  if (replace)
    initialize(parameter);
  else
    initialize();
}

HostSpace::~HostSpace() {
  finalize();
}

UnsignedInteger HostPolicy::HostPolicyInstanceNum_ = 0;

HostPolicy::HostPolicy(HostPolicy &policy) {
  HostPolicyInstanceNum_++;
}

HostPolicy::~HostPolicy() {
  HostPolicyInstanceNum_--;
}
}  // namespace hydrogen

#endif