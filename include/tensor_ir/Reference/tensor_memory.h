// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "tensor_ir/Runtime/Types.h"
#include "tensor_ir/Support/Status.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace mlir::nv_tensor_ir::reference {

class SimplifiedTensor;

/**
 * @brief TensorMemory - Manages host memory allocation for tensors
 *
 * This class owns one CPU buffer used for interpreter inputs and reference
 * results. The buffer is allocated on construction and released on destruction.
 */
class TensorMemory {
public:
  /**
   * @brief Construct and zero-initialize a host buffer
   * @param sizeInBytes Total number of bytes to allocate
   */
  explicit TensorMemory(size_t sizeInBytes);

  /**
   * @brief Release the owned host buffer
   */
  ~TensorMemory();

  TensorMemory(const TensorMemory &) = delete;
  TensorMemory &operator=(const TensorMemory &) = delete;

  /// Return whether construction successfully allocated the host buffer.
  bool isAllocated() const { return hostMemory != nullptr; }

  /// Get the host buffer, or nullptr if allocation failed.
  void *hostPtr() { return hostMemory; }
  const void *hostPtr() const { return hostMemory; }

private:
  void *hostMemory;
};

/// RAII owner of a single CUDA device allocation.
///
/// Move transfers ownership and leaves the source empty (get() == nullptr,
/// size() == 0). Copying is deleted to prevent a double free. A
/// default-constructed buffer is empty and owns nothing.
class DeviceBuffer {
public:
  /// Allocate `sizeInBytes` of zero-initialized device memory.
  static StatusOr<DeviceBuffer> create(size_t sizeInBytes);

  /// Construct an empty buffer that owns nothing.
  DeviceBuffer() = default;
  ~DeviceBuffer();

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  DeviceBuffer(DeviceBuffer &&other) noexcept;
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept;

  /// Copy `size()` bytes from host memory into this device allocation.
  Status copyFrom(const void *host);
  /// Copy `size()` bytes from this device allocation into host memory.
  Status copyTo(void *host) const;

  /// Device pointer owned by this buffer, or nullptr when empty.
  void *get() const { return ptr; }
  /// Size in bytes of the current allocation, or 0 when empty.
  size_t size() const { return sizeInBytes; }

private:
  DeviceBuffer(void *ptr, size_t sizeInBytes)
      : ptr(ptr), sizeInBytes(sizeInBytes) {}

  void *ptr = nullptr;
  size_t sizeInBytes = 0;
};

/// Launch-scoped owner for all device buffers and tensor arguments needed by
/// one kernel run.
class DeviceRun {
public:
  static StatusOr<DeviceRun>
  create(llvm::ArrayRef<std::shared_ptr<SimplifiedTensor>> inputs,
         llvm::ArrayRef<std::shared_ptr<SimplifiedTensor>> outputs);

  llvm::ArrayRef<tensor_ir::rt::Any> args() const { return argsStorage; }

  Status copyOutputsTo(std::vector<std::vector<std::byte>> &results) const;
  Status printInputs() const;
  Status printOutputs() const;

private:
  std::vector<DeviceBuffer> inputBuffers;
  std::vector<DeviceBuffer> outputBuffers;
  std::vector<std::shared_ptr<SimplifiedTensor>> inputs;
  std::vector<std::shared_ptr<SimplifiedTensor>> outputs;
  std::vector<tensor_ir::rt::Any> argsStorage;
};

} // namespace mlir::nv_tensor_ir::reference
