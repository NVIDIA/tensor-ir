// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "tensor_ir/Support/Status.h"

#include "llvm/ADT/ArrayRef.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>
namespace mlir::nv_tensor_ir::reference {

class TensorMemory;

// Supported data types (simplified version)
enum class DataType {
  BOOL,
  FLOAT32,
  FLOAT16,
  BFLOAT16,
  F8E4M3FN,
  F8E5M2,
  INT32,
  UINT32,
  INT8,
  UINT8,
  DOUBLE
};

/// Returns whether \p dtype represents an integer or boolean value.
bool isIntegerDataType(DataType dtype);

/// Returns whether \p dtype represents an unsigned integer value.
bool isUnsignedIntegerDataType(DataType dtype);

/// Returns whether \p dtype represents a floating-point value.
bool isFloatDataType(DataType dtype);

const char *getEnumName(DataType dtype);

// Tensor descriptor
struct TensorDesc {
  std::vector<int64_t> dims{};    // Dimensions
  std::vector<int64_t> strides{}; // Strides
  DataType dtype{DataType::FLOAT32};
  int64_t totalElements{0}; // Total number of elements

  TensorDesc() = default;

  void computeStrides(); // Automatically compute strides

  /// Return the byte size of the complete strided storage footprint, or
  /// std::nullopt when the descriptor is invalid or the size overflows.
  std::optional<size_t> getSizeInBytes() const;

  /// Validate that the strided storage footprint is representable.
  Status
  validateStridedOffsetsWithinAllocation(const std::string &tensorName) const;

  void print() const;
};

// Utility functions
size_t getDataTypeSize(DataType dtype);

/**
 * @brief SimplifiedTensor - Tensor representation with metadata and memory
 * management
 *
 * This class manages tensor metadata (shape, strides, data type) and delegates
 * memory management to the TensorMemory class through a smart pointer.
 *
 * Memory lifecycle:
 * 1. init() - Initialize tensor descriptor
 * 2. allocate() - Create TensorMemory and allocate actual memory
 * 3. Use memory pointers for computation
 * 4. Automatic cleanup when SimplifiedTensor is destroyed
 */
class SimplifiedTensor {
public:
  SimplifiedTensor();
  ~SimplifiedTensor();

  // Disable copy, enable move
  SimplifiedTensor(const SimplifiedTensor &) = delete;
  SimplifiedTensor &operator=(const SimplifiedTensor &) = delete;
  SimplifiedTensor(SimplifiedTensor &&) noexcept;
  SimplifiedTensor &operator=(SimplifiedTensor &&) noexcept;

  // === Initialization ===
  bool init(const std::string &name, llvm::ArrayRef<int64_t> dims,
            DataType dtype);

  /// Initialize with explicit strides (skips computeStrides).
  bool init(const std::string &name, llvm::ArrayRef<int64_t> dims,
            llvm::ArrayRef<int64_t> strides, DataType dtype);

  // === Memory Management ===
  /**
   * @brief Allocate memory for this tensor
   * Creates TensorMemory object and allocates all required memory buffers
   */
  bool allocate();

  /**
   * @brief Check if memory has been allocated
   */
  bool isAllocated() const;

  /// Validate metadata and memory required by reference-node computations.
  Status validate(const std::string &role) const;

  // === Data Access ===
  void *hostPtr();
  const void *hostPtr() const;

  /// Convert a logical row-major linear index into this tensor's strided
  /// storage offset.
  int64_t getStorageIndex(int64_t linearIndex) const;

  /// Convert a logical multi-dimensional index into this tensor's strided
  /// storage offset.
  int64_t getStorageIndex(llvm::ArrayRef<int64_t> indices) const;

  template <typename T>
  T &hostElement(int64_t linearIndex) {
    void *ptr = hostPtr();
    assert(ptr && "host memory must be allocated");
    return static_cast<T *>(ptr)[getStorageIndex(linearIndex)];
  }

  template <typename T>
  const T &hostElement(int64_t linearIndex) const {
    const void *ptr = hostPtr();
    assert(ptr && "host memory must be allocated");
    return static_cast<const T *>(ptr)[getStorageIndex(linearIndex)];
  }

  /// Access an element by its logical multi-dimensional coordinate.
  template <typename T>
  T &hostElement(llvm::ArrayRef<int64_t> coord) {
    void *ptr = hostPtr();
    assert(ptr && "host memory must be allocated");
    return static_cast<T *>(ptr)[getStorageIndex(coord)];
  }

  template <typename T>
  const T &hostElement(llvm::ArrayRef<int64_t> coord) const {
    const void *ptr = hostPtr();
    assert(ptr && "host memory must be allocated");
    return static_cast<const T *>(ptr)[getStorageIndex(coord)];
  }

  // === Data Fill ===
  /// Fill logical host elements with \p value converted to the tensor dtype.
  /// Returns false if storage is unavailable, the dtype is unsupported, or an
  /// integer dtype cannot represent \p value.
  bool fillConstant(double value);
  /// Fill host storage with values drawn from \p gen in [minVal, maxVal].
  /// Reuses the caller's generator so multi-tensor fills stay deterministic.
  bool fillRandom(float minVal, float maxVal, std::mt19937 &gen);

  // === Data Printing ===
  void printHostData() const;

  /// Print tensor data from a caller-provided host-accessible buffer.
  void printData(const void *dataPtr, const char *label, const char *ptrName,
                 const void *ptrValue) const;

  // === Property Access ===
  const std::string &getName() const { return name; }
  const TensorDesc &getDesc() const { return desc; }
  DataType getDataType() const { return desc.dtype; }
  llvm::ArrayRef<int64_t> getDims() const { return desc.dims; }
  int64_t getTotalElements() const { return desc.totalElements; }
  std::optional<size_t> getSizeInBytes() const { return desc.getSizeInBytes(); }

  // === Utility Methods ===
  void print() const; // Print tensor information

  /// Compare integer and boolean elements exactly, and floating-point elements
  /// using \p rtol and \p atol.
  bool checkRefAgainst(const void *gpuResultHost, double rtol = 1e-5,
                       double atol = 1e-8, bool verbose = false) const;

private:
  // Tensor metadata
  std::string name;
  TensorDesc desc;

  // Memory management (delegated to TensorMemory via smart pointer)
  std::unique_ptr<TensorMemory> memory;
};

// === Factory Functions ===
std::shared_ptr<SimplifiedTensor> createTensor(const std::string &name,
                                               llvm::ArrayRef<int64_t> dims,
                                               DataType dtype);

/// Factory with explicit strides.
std::shared_ptr<SimplifiedTensor> createTensor(const std::string &name,
                                               llvm::ArrayRef<int64_t> dims,
                                               llvm::ArrayRef<int64_t> strides,
                                               DataType dtype);

} // namespace mlir::nv_tensor_ir::reference
