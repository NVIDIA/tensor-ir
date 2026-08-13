// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Common types for TensorIR Runtime

#ifndef TENSOR_IR_RUNTIME_TYPES_H
#define TENSOR_IR_RUNTIME_TYPES_H

#include "tensor_ir/Support/Status.h"

#include <cstddef>
#include <cstdint>

// Forward declare CUDA types to avoid header dependency
using cudaStream_t = struct CUstream_st *;

namespace tensor_ir::rt {

using mlir::nv_tensor_ir::Status;
using mlir::nv_tensor_ir::StatusOr;

/// CUDA stream handle
using Stream = cudaStream_t;

/// Device workspace for temporary storage
struct Workspace {
  void *ptr = nullptr;
  size_t size = 0;
};

//===----------------------------------------------------------------------===//
// PackedArgs - Abstraction for variadic arguments
//
// This is a minimal interface that can be backed by:
//   - TVM FFI PackedArgs (zero-allocation)
//   - A simple fallback implementation
//===----------------------------------------------------------------------===//

enum class TypeIndex : int32_t {
  kNone = 0,
  kInt64 = 1,
  kFloat64 = 2,
  kPointer = 3,
  kTensor = 4,
  // ... add more as needed
};

struct TensorView {
  void *data;
  int32_t ndim;
  int64_t *shape;
  int64_t *stride;
};

struct Any {
  TypeIndex typeIndex; // Type tag
  union {
    int64_t i64;
    double f64;
    void *ptr; // For DLTensor*, handles, etc.
    TensorView tensorView;
    // ... other types
  };
  Any() : typeIndex(TypeIndex::kNone), i64(0) {}
  Any(int64_t i64) : typeIndex(TypeIndex::kInt64), i64(i64) {}
  Any(double f64) : typeIndex(TypeIndex::kFloat64), f64(f64) {}
  Any(void *ptr) : typeIndex(TypeIndex::kPointer), ptr(ptr) {}
  Any(const TensorView &tensorView)
      : typeIndex(TypeIndex::kTensor), tensorView(tensorView) {}

  /// Type-safe cast with runtime validation.
  template <typename T>
  StatusOr<T> cast() const {
    if constexpr (std::is_same_v<T, int64_t>) {
      if (typeIndex != TypeIndex::kInt64) {
        return Status::InvalidArgument("Type mismatch: expected int64");
      }
      return i64;
    } else if constexpr (std::is_same_v<T, double>) {
      if (typeIndex != TypeIndex::kFloat64) {
        return Status::InvalidArgument("Type mismatch: expected float64");
      }
      return f64;
    } else if constexpr (std::is_same_v<T, void *>) {
      if (typeIndex != TypeIndex::kPointer) {
        return Status::InvalidArgument("Type mismatch: expected pointer");
      }
      return ptr;
    } else if constexpr (std::is_same_v<T, TensorView>) {
      if (typeIndex != TypeIndex::kTensor) {
        return Status::InvalidArgument("Type mismatch: expected TensorView");
      }
      return tensorView;
    }
    return Status::NotSupported("Unsupported type for Any::cast<T>");
  }
};

/// PackedArgs is just a pointer + count - No heap allocation per call.
class PackedArgs {
  const Any *args_;
  int32_t num_args_;

public:
  PackedArgs(const Any *args, int32_t n) : args_(args), num_args_(n) {}

  /// Returns the number of arguments
  int32_t size() const { return num_args_; }

  /// Type-safe access
  template <typename T>
  StatusOr<T> At(int i) const {
    if (i < 0 || i >= num_args_) {
      return Status::InvalidArgument("PackedArgs index out of bounds");
    }
    return args_[i].cast<T>();
  }

  StatusOr<Any> operator[](int i) const {
    if (i < 0 || i >= num_args_) {
      return Status::InvalidArgument("PackedArgs index out of bounds");
    }
    return args_[i];
  }
};

} // namespace tensor_ir::rt

#endif // TENSOR_IR_RUNTIME_TYPES_H
