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

#include "tensor_ir/Conversion/TensorToCudaTile/Options.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <limits>

namespace tensor_ir::rt {

/// Element type of a tensor or scalar kernel argument.
enum class ElementType : uint8_t {
  Bool = 0,
  F16,
  BF16,
  F32,
  F64,
  F8E4M3FN,
  F8E5M2,
  SI8,
  SI16,
  SI32,
  SI64,
  UI8,
  UI16,
  UI32,
  UI64,
  // The largest defined value -- the enum is dense, so [Bool,
  // kMaxElementType] is exactly its valid range.
  kMaxElementType = UI64,
};

/// Wraps an ElementType with a byteWidth() accessor computed on demand.
class ElementTypeInfo {
public:
  explicit ElementTypeInfo(ElementType type) : type_(type) {}

  ElementType type() const { return type_; }

  int32_t byteWidth() const {
    switch (type_) {
    case ElementType::Bool:
    case ElementType::SI8:
    case ElementType::UI8:
    case ElementType::F8E4M3FN:
    case ElementType::F8E5M2:
      return 1;
    case ElementType::F16:
    case ElementType::BF16:
    case ElementType::SI16:
    case ElementType::UI16:
      return 2;
    case ElementType::F32:
    case ElementType::SI32:
    case ElementType::UI32:
      return 4;
    case ElementType::F64:
    case ElementType::SI64:
    case ElementType::UI64:
      return 8;
    }
    llvm_unreachable("unhandled ElementType");
  }

  bool operator==(const ElementTypeInfo &other) const {
    return type_ == other.type_;
  }

private:
  ElementType type_;
};

/// Describes how one operand maps to flat kernel arguments.
///
/// For tensor operands the kernel signature is expanded as:
///   [ptr] [dyn_size_0, dyn_size_1, ...] [dyn_stride_0, dyn_stride_1, ...]
///
/// For scalar (non-tensor) operands the kernel signature is a single
/// by-value argument.  `isScalar` is set to true; the runtime arg-packer
/// memcpy's `elementInfo.byteWidth()` bytes instead of copying a pointer.
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

  /// Element type of the tensor, or the value type of the scalar.
  ElementTypeInfo elementInfo;

  /// Total flat kernel args consumed by this operand.  A scalar operand
  /// contributes exactly one by-value argument; a tensor operand
  /// contributes 1 (ptr) + dynamic sizes + dynamic strides.
  int32_t totalArgs() const {
    return isScalar ? 1 : (1 + numDynSizes + numDynStrides);
  }

  bool operator==(const TensorArgDesc &other) const {
    return rank == other.rank && staticShape == other.staticShape &&
           staticStrides == other.staticStrides &&
           numDynSizes == other.numDynSizes &&
           numDynStrides == other.numDynStrides &&
           hasExplicitStrides == other.hasExplicitStrides &&
           isScalar == other.isScalar && elementInfo == other.elementInfo;
  }
};

/// Count elements equal to kDynamic in an int64_t array.
inline int32_t countDynamicDims(llvm::ArrayRef<int64_t> vals) {
  return llvm::count_if(vals,
                        [](int64_t v) { return v == TensorArgDesc::kDynamic; });
}

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

  /// Normalized iteration-space shape used for runtime grid computation.
  ///
  /// An empty vector preserves the legacy behavior of using the selected
  /// tensor's shape directly. Dynamic dimensions use TensorArgDesc::kDynamic.
  llvm::SmallVector<int64_t> gridShape;

  /// Maps each normalized grid dimension to a dimension of
  /// gridShapeTensorIdx. Static grid dimensions use -1.
  llvm::SmallVector<int32_t> gridShapeDimMapping;

  /// When true, the kernel signature includes args for ALL tensor sizes and
  /// strides (uniform layout).  When false, only dynamic dims produce args.
  bool uniformSignature = false;

  /// Optional runtime-grid persistence metadata. This is only consumed by
  /// TileBasedGridComputer when the generated kernel IR also uses a static
  /// persistence loop.
  mlir::nv_tensor_ir::PersistenceMode persistence =
      mlir::nv_tensor_ir::PersistenceMode::None;
  int32_t smCount = 0;
  int32_t occupancy = 1;

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

  bool operator==(const KernelArgLayout &other) const {
    return tensorDescs == other.tensorDescs && tileSizes == other.tileSizes &&
           numInputs == other.numInputs &&
           totalKernelArgs == other.totalKernelArgs &&
           gridShapeTensorIdx == other.gridShapeTensorIdx &&
           gridShape == other.gridShape &&
           gridShapeDimMapping == other.gridShapeDimMapping &&
           uniformSignature == other.uniformSignature &&
           persistence == other.persistence && smCount == other.smCount &&
           occupancy == other.occupancy;
  }
};

} // namespace tensor_ir::rt

#endif // TENSOR_IR_RUNTIME_CUDATILE_KERNEL_ARG_LAYOUT_H_
