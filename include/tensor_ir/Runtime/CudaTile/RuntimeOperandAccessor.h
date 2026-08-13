// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// RuntimeOperandAccessor — thin adapter for PackedArgs / TensorView
//
// Implements the OperandAccessor concept expected by the templated
// KernelArgPacker / GridSizeComputer:
//
//   int32_t  size()               const;
//   void*    pointer(int32_t i)   const;
//   int64_t  shape(int32_t i, int32_t d)   const;
//   int64_t  stride(int32_t i, int32_t d)  const;
//   bool     hasShape(int32_t i)  const;
//   bool     isDense(int32_t i)   const;
//   bool     isScalar(int32_t i)  const;
//   int32_t  scalarSizeInBytes(int32_t i) const;
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_IR_RUNTIME_CUDATILE_RUNTIME_OPERAND_ACCESSOR_H_
#define TENSOR_IR_RUNTIME_CUDATILE_RUNTIME_OPERAND_ACCESSOR_H_

#include "tensor_ir/Runtime/Types.h"

namespace tensor_ir::rt {

/// Lightweight, non-owning adapter over PackedArgs for kernel launch helpers.
///
/// Each element in PackedArgs is expected to be a TensorView
/// (TypeIndex::kTensor) or a pointer (TypeIndex::kPointer).  The accessor
/// exposes shape/stride info from TensorView entries and data pointers from
/// both kinds.
struct RuntimeOperandAccessor {
  const PackedArgs &args;

  int32_t size() const { return args.size(); }

  /// Returns the data pointer for operand i.
  /// For kTensor: returns TensorView::data.
  /// For kPointer: returns the raw pointer.
  /// Returns nullptr for unsupported kinds.
  void *pointer(int32_t i) const {
    auto arg = args[i];
    if (!arg.ok()) {
      return nullptr;
    }
    switch (arg->typeIndex) {
    case TypeIndex::kTensor:
      return arg->tensorView.data;
    case TypeIndex::kPointer:
      return arg->ptr;
    default:
      return nullptr;
    }
  }

  int64_t shape(int32_t i, int32_t d) const {
    auto arg = args[i];
    if (!arg.ok() || arg->typeIndex != TypeIndex::kTensor) {
      return 0;
    }
    return arg->tensorView.shape[d];
  }

  int64_t stride(int32_t i, int32_t d) const {
    auto arg = args[i];
    if (!arg.ok() || arg->typeIndex != TypeIndex::kTensor) {
      return 0;
    }
    return arg->tensorView.stride[d];
  }

  /// Returns true if operand i is a tensor with shape information available.
  bool hasShape(int32_t i) const {
    auto arg = args[i];
    if (!arg.ok()) {
      return false;
    }
    return arg->typeIndex == TypeIndex::kTensor &&
           arg->tensorView.shape != nullptr;
  }

  /// Returns true if operand i is a tensor (dense).
  /// In the public runtime all tensors are dense (strided) views.
  bool isDense(int32_t i) const {
    auto arg = args[i];
    if (!arg.ok()) {
      return false;
    }
    return arg->typeIndex == TypeIndex::kTensor;
  }

  /// True when operand i is a scalar (by-value) kernel argument.
  /// The public runtime does not currently plumb scalar operands through
  /// PackedArgs, so this always returns false.  Teaching it to recognise
  /// scalars requires extending `PackedArgs` / `Any` with a kScalar tag
  /// carrying the value bytes plus a width.
  bool isScalar(int32_t /*i*/) const { return false; }

  /// Width in bytes of operand i when it is a scalar.  Unused while
  /// isScalar() is always false.
  int32_t scalarSizeInBytes(int32_t /*i*/) const { return 0; }
};

} // namespace tensor_ir::rt

#endif // TENSOR_IR_RUNTIME_CUDATILE_RUNTIME_OPERAND_ACCESSOR_H_
