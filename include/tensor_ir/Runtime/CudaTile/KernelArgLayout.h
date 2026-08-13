// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// KernelArgLayout — runtime kernel argument ABI metadata
//
// TensorArgDesc + KernelArgLayout describe how the flat kernel argument
// list is laid out. The compiler produces this metadata and the runtime uses
// it to pack arguments and compute grid dimensions dynamically.
//===----------------------------------------------------------------------===//

#ifndef TENSOR_IR_RUNTIME_CUDATILE_KERNEL_ARG_LAYOUT_H_
#define TENSOR_IR_RUNTIME_CUDATILE_KERNEL_ARG_LAYOUT_H_

#include "llvm/ADT/SmallVector.h"

namespace tensor_ir::rt {

/// Describes how one operand maps to flat kernel arguments.
///
/// For tensor operands the kernel signature is expanded as:
///   [ptr] [dyn_size_0, dyn_size_1, ...] [dyn_stride_0, dyn_stride_1, ...]
///
/// For scalar (non-tensor) operands the kernel signature is a single
/// by-value argument.  `isScalar` is set to true and `scalarSizeInBytes`
/// records the argument width so the runtime arg-packer can copy the
/// value bytes instead of a pointer.
///
/// staticShape/staticStrides use kDynamic for runtime-determined dims.
/// This matches mlir::ShapedType::kDynamic so that values extracted from
/// TensorType::getShape() can be stored directly without conversion.
struct TensorArgDesc {
  static constexpr int64_t kDynamic = std::numeric_limits<int64_t>::min();

  int32_t rank = 0;

  /// Static shape — concrete value for known dims, kDynamic for dynamic.
  llvm::SmallVector<int64_t> staticShape;

  /// Static strides — concrete value for known dims, kDynamic for dynamic.
  /// Empty means strides are computed from sizes (no stride args in kernel).
  llvm::SmallVector<int64_t> staticStrides;

  int32_t numDynSizes = 0;
  int32_t numDynStrides = 0;
  bool hasExplicitStrides = false;

  /// True when this operand is a scalar (non-tensor) kernel argument that
  /// is passed by value, not via a pointer + size/stride tuple.
  bool isScalar = false;

  /// Width in bytes of the scalar value (only meaningful when isScalar).
  /// For tensor operands this field is 0.
  int32_t scalarSizeInBytes = 0;

  /// Total flat kernel args consumed by this operand.  A scalar operand
  /// contributes exactly one by-value argument; a tensor operand
  /// contributes 1 (ptr) + dynamic sizes + dynamic strides.
  int32_t totalArgs() const {
    return isScalar ? 1 : (1 + numDynSizes + numDynStrides);
  }
};

/// Complete layout of the kernel's flat argument list.
struct KernelArgLayout {
  llvm::SmallVector<TensorArgDesc> tensorDescs;
  llvm::SmallVector<int32_t> tileSizes;
  int32_t numInputs = 0;
  int32_t totalKernelArgs = 0;

  /// Index of the tensor whose runtime shape is used to compute grid dims.
  ///
  /// The compile-time path (TileAnalyzer::calculateGridSizeForGraph) uses
  /// the first input tensor's shape by default.  For ops where the grid
  /// should cover output dimensions instead (e.g. matmul: tile over M,N
  /// rather than M,K; concat: cover the full concatenated extent), set this
  /// to outputTensorStartIdx().
  int32_t gridShapeTensorIdx = 0;

  /// When true, the kernel signature includes args for ALL tensor sizes and
  /// strides (uniform layout).  When false, only dynamic dims produce args.
  bool uniformSignature = false;

  bool hasDynamicShapes() const {
    for (const auto &d : tensorDescs) {
      if (d.numDynSizes > 0 || d.numDynStrides > 0) {
        return true;
      }
    }
    return false;
  }

  int32_t outputTensorStartIdx() const { return numInputs; }

  int32_t getArgOffset(int32_t tensorIdx) const {
    int32_t off = 0;
    for (int32_t i = 0; i < tensorIdx; ++i) {
      off += tensorDescs[i].totalArgs();
    }
    return off;
  }
};

} // namespace tensor_ir::rt

#endif // TENSOR_IR_RUNTIME_CUDATILE_KERNEL_ARG_LAYOUT_H_
