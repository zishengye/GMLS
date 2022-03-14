#ifndef _HYDROGEN_IMPL_OBJECT_HPP_
#define _HYDROGEN_IMPL_OBJECT_HPP_

namespace hydrogen {
template <class ExecutionSpace_>
void Object<ExecutionSpace_>::synchronizeInternal() {
}

template <class ExecutionSpace_>
void Object<ExecutionSpace_>::synchronizeToDeviceInternal() {
}

template <class ExecutionSpace_>
void Object<ExecutionSpace_>::synchronizeFromDeviceInternal() {
}

template <class ExecutionSpace_>
void Object<ExecutionSpace_>::releaseMemory() {
  // Always release memory on device first
  releaseDeviceMemory();
  releaseHostMemory();
}

template <class ExecutionSpace_>
void Object<ExecutionSpace_>::releaseHostMemory() {
}

template <class ExecutionSpace_>
void Object<ExecutionSpace_>::releaseDeviceMemory() {
}

template <class ExecutionSpace_>
void Object<ExecutionSpace_>::registerActionOnHost() {
}

template <class ExecutionSpace_>
void Object<ExecutionSpace_>::registerActionOnDevice() {
}

template <class ExecutionSpace_>
Object<ExecutionSpace_>::Object()
    : hostDeviceSynchronization_(false), lastMemoryAction_(0) {
}

template <class ExecutionSpace_>
Object<ExecutionSpace_>::~Object() {
  releaseMemory();
}

template <class ExecutionSpace_>
void Object<ExecutionSpace_>::fence() {
}

template <class ExecutionSpace_>
void Object<ExecutionSpace_>::update() {
  if (lastMemoryAction_ == 0) {
    // Last action is on host
    updateToDevice();
  } else {
    // Last action is on device
    updateFromDevice();
  }
}

template <class ExecutionSpace_>
void Object<ExecutionSpace_>::updateFromDevice() {
  if (!hostDeviceSynchronization_) {
    hostDeviceSynchronization_ = true;
  }

  synchronizeInternal();
  synchronizeFromDeviceInternal();
}

template <class ExecutionSpace_>
void Object<ExecutionSpace_>::updateToDevice() {
  if (!hostDeviceSynchronization_) {
    hostDeviceSynchronization_ = true;
  }

  synchronizeInternal();
  synchronizeToDeviceInternal();
}
}  // namespace hydrogen

#endif