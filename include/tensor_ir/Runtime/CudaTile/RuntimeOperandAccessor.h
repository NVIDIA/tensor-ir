// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// RuntimeOperandAccessor — thin adapter for PackedArgs / TensorView
//
// Implements the OperandAccessor concept expected by the templated
// ArgumentPacker / GridSizeComputer:
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
/// (TypeIndex::kTensor), a pointer (TypeIndex::kPointer), or a by-value
/// scalar (TypeIndex::kScalar).  The accessor exposes shape/stride info from
/// TensorView entries, data pointers from all three kinds, and the scalar's
/// own ElementType-derived width for kScalar.
struct RuntimeOperandAccessor {
  const PackedArgs &args;

  int32_t size() const { return args.size(); }

  /// Returns the data pointer for operand i.
  /// For kTensor: returns TensorView::data.
  /// For kPointer: returns the raw pointer.
  /// For kScalar (by-value scalars): returns the address of the scalar's
  /// bits within the backing PackedArgs array, so that callers (e.g.
  /// KernelArgPacker) can memcpy the value bytes.
  /// Returns nullptr for unsupported kinds.
  ///
  /// TODO: pointer() returns the value itself for kTensor/kPointer but an
  /// address to dereference for kScalar -- callers must separately know the
  /// kind to interpret it (see packOperandIntoInt64Slot's `scalar` flag).
  /// Replacing it with one method, e.g. packingBytes(i) -> {address, length},
  /// would let packing just copy `length` bytes from `address` without
  /// needing to know the operand's kind at all.
  const void *pointer(int32_t i) const {
    const Any *arg = args[i];
    if (!arg) {
      return nullptr;
    }
    switch (arg->typeIndex) {
    case TypeIndex::kTensor:
      return arg->tensorView.data;
    case TypeIndex::kPointer:
      return arg->ptr;
    case TypeIndex::kScalar:
      return &arg->scalar.bits;
    default:
      return nullptr;
    }
  }

  int64_t shape(int32_t i, int32_t d) const {
    const Any *arg = args[i];
    if (!arg || arg->typeIndex != TypeIndex::kTensor) {
      return 0;
    }
    return arg->tensorView.shape[d];
  }

  int64_t stride(int32_t i, int32_t d) const {
    const Any *arg = args[i];
    if (!arg || arg->typeIndex != TypeIndex::kTensor) {
      return 0;
    }
    return arg->tensorView.stride[d];
  }

  /// Returns true if operand i is a tensor with shape information available.
  bool hasShape(int32_t i) const {
    const Any *arg = args[i];
    if (!arg) {
      return false;
    }
    return arg->typeIndex == TypeIndex::kTensor &&
           arg->tensorView.shape != nullptr;
  }

  /// Returns true if operand i is a tensor (dense).
  /// In the public runtime all tensors are dense (strided) views.
  bool isDense(int32_t i) const {
    const Any *arg = args[i];
    if (!arg) {
      return false;
    }
    return arg->typeIndex == TypeIndex::kTensor;
  }

  /// True when operand i is a scalar (by-value) kernel argument, i.e. an
  /// Any tagged kScalar.
  bool isScalar(int32_t i) const {
    const Any *arg = args[i];
    return arg && arg->typeIndex == TypeIndex::kScalar;
  }

  /// Width in bytes of operand i when it is a scalar, derived from the
  /// scalar's own ElementType. Returns 0 for non-scalar operands.
  int32_t scalarSizeInBytes(int32_t i) const {
    const Any *arg = args[i];
    if (!arg || arg->typeIndex != TypeIndex::kScalar) {
      return 0;
    }
    return getElementByteWidth(arg->scalar.type);
  }
};

} // namespace tensor_ir::rt

#endif // TENSOR_IR_RUNTIME_CUDATILE_RUNTIME_OPERAND_ACCESSOR_H_
