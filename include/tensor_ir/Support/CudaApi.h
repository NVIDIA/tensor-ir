// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_SUPPORT_CUDA_API_H
#define TENSOR_IR_SUPPORT_CUDA_API_H

#include "tensor_ir/Support/Status.h"

#include "cuda.h"
#include <cstddef>
#include <string>

namespace mlir::nv_tensor_ir::cuda {

namespace driver {

/// Return the CUDA driver version encoded as major * 1000 + minor * 10.
StatusOr<int> getVersion();

/// Load CUDA device code through the CUDA Library API.
StatusOr<CUlibrary> loadLibrary(const void *deviceCode);

/// Resolve a kernel by name from a loaded CUDA library.
StatusOr<CUkernel> getKernel(CUlibrary library, const char *name);

/// Unload a CUDA library.
Status unloadLibrary(CUlibrary library);

/// Launch a kernel through the extended CUDA launch API.
Status launchKernel(const CUlaunchConfig &config, CUkernel kernel,
                    void **kernelParams);

} // namespace driver

namespace runtime {

/// CUDA device properties currently consumed by TensorIR.
struct DeviceProperties {
  std::string name;
  int major = 0;
  int minor = 0;
  int multiprocessorCount = 0;
  std::size_t sharedMemoryPerMultiprocessor = 0;
  int l2CacheSize = 0;
};

/// Opaque CUDA Runtime event handle.
class Event {
public:
  Event(void *handle) : handle(handle) {}

  void *getNativeHandle() const { return handle; }

  operator bool() const { return handle != nullptr; }
  bool operator==(const Event &other) const { return handle == other.handle; }
  bool operator!=(const Event &other) const { return handle != other.handle; }

private:
  void *handle = nullptr;
};

/// Initialize the CUDA Runtime for the current process.
Status initialize();

/// Return the number of CUDA devices visible to the process.
StatusOr<int> getDeviceCount();

/// Return the current CUDA device ordinal.
StatusOr<int> getDevice();

/// Return the compute capability of a CUDA device.
StatusOr<DeviceProperties> getDeviceProperties(int device);

/// Allocate CUDA device memory.
StatusOr<void *> allocate(std::size_t sizeInBytes);

/// Release CUDA device memory.
Status free(void *ptr);

/// Fill CUDA device memory with a byte value.
Status memset(void *ptr, int value, std::size_t sizeInBytes);

/// Copy host memory to CUDA device memory.
Status copyHostToDevice(void *dst, const void *src, std::size_t sizeInBytes);

/// Copy CUDA device memory to host memory.
Status copyDeviceToHost(void *dst, const void *src, std::size_t sizeInBytes);

/// Wait for preceding work on the current CUDA device to complete.
Status synchronizeDevice();

/// Create a CUDA event.
StatusOr<Event> createEvent();

/// Destroy a CUDA event.
Status destroyEvent(Event event);

/// Record a CUDA event on the default stream.
Status recordEvent(Event event);

/// Wait for a CUDA event to complete.
Status synchronizeEvent(Event event);

/// Return elapsed milliseconds between two CUDA events.
StatusOr<float> getElapsedTime(Event start, Event end);

} // namespace runtime
} // namespace mlir::nv_tensor_ir::cuda

#endif // TENSOR_IR_SUPPORT_CUDA_API_H
