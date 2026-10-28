// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Common types for TensorIR Runtime

#ifndef TENSOR_IR_RUNTIME_TYPES_H
#define TENSOR_IR_RUNTIME_TYPES_H

#include "tensor_ir/Artifact/KernelMetadata.h"
#include "tensor_ir/Support/Status.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

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
  kPointer = 1,
  kTensor = 2,
  kScalar = 3,
  // ... add more as needed
};

struct TensorView {
  void *data;
  int32_t ndim;
  int64_t *shape;
  int64_t *stride;
};

/// A by-value scalar kernel argument: element type plus the raw bit pattern,
/// left-aligned in the low-order bytes of `bits`.
struct ScalarValue {
  ElementType type;
  int64_t bits;
};

struct Any {
  TypeIndex typeIndex; // Type tag
  union {
    void *ptr; // For DLTensor*, handles, etc.
    TensorView tensorView;
    ScalarValue scalar;
    // ... other types
  };
  Any() : typeIndex(TypeIndex::kNone), ptr(nullptr) {}
  Any(void *ptr) : typeIndex(TypeIndex::kPointer), ptr(ptr) {}
  Any(const TensorView &tensorView)
      : typeIndex(TypeIndex::kTensor), tensorView(tensorView) {}

  /// Builds a scalar argument from a runtime element type and a pointer to
  /// the value's bytes, copying only the type's native width. Pass the
  /// address of an already-packed `int64_t` bit pattern or of a
  /// compile-time-typed value (float, int32_t, ...) interchangeably.
  /// `valuePtr` must be non-null and point to at least `type`'s byte width.
  static Any makeScalarAny(ElementType type, const void *valuePtr) {
    int64_t bits = 0;
    std::memcpy(&bits, valuePtr,
                static_cast<size_t>(getElementByteWidth(type)));
    return Any(type, bits);
  }

  /// Type-safe cast with runtime validation.
  template <typename T>
  StatusOr<T> cast() const {
    if constexpr (std::is_same_v<T, void *>) {
      if (typeIndex != TypeIndex::kPointer) {
        return Status::InvalidArgument("Type mismatch: expected pointer");
      }
      return ptr;
    } else if constexpr (std::is_same_v<T, TensorView>) {
      if (typeIndex != TypeIndex::kTensor) {
        return Status::InvalidArgument("Type mismatch: expected TensorView");
      }
      return tensorView;
    } else if constexpr (std::is_same_v<T, ScalarValue>) {
      if (typeIndex != TypeIndex::kScalar) {
        return Status::InvalidArgument("Type mismatch: expected scalar");
      }
      return scalar;
    }
    return Status::NotSupported("Unsupported type for Any::cast<T>");
  }

private:
  Any(ElementType type, int64_t bits)
      : typeIndex(TypeIndex::kScalar), scalar{type, bits} {}
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

  /// Returns a pointer into the backing storage for argument i, or nullptr
  /// if out of bounds. The pointer aliases PackedArgs' own storage, not a
  /// copy, so it stays valid for as long as the backing storage does.
  const Any *operator[](int i) const {
    if (i < 0 || i >= num_args_) {
      return nullptr;
    }
    return &args_[i];
  }
};

} // namespace tensor_ir::rt

#endif // TENSOR_IR_RUNTIME_TYPES_H
