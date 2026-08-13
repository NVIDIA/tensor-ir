// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_INCLUDE_UTILS_H_
#define TENSOR_IR_INCLUDE_UTILS_H_

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Support/Macros.h"
#include "tensor_ir/Support/TCutegen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

#include "llvm/Support/raw_ostream.h"

#include <array>
#include <string>

namespace mlir::nv_tensor_ir {

//===----------------------------------------------------------------------===//
// Error handling macros
//===----------------------------------------------------------------------===//

// Optional-msg arity dispatch for MLIR_ASSIGN_OR_RETURN: empty __VA_ARGS__ is a
// C++20 extension and warns under -pedantic with C++17. The trailing /*end*/
// sentinel keeps `...` non-empty in the selector call.
#define TIR_GET_MACRO_3(_1, _2, _3, NAME, ...) NAME

#define MLIR_RETURN_IF_ERROR(expr)                                             \
  do {                                                                         \
    const auto &status = (expr);                                               \
    if (failed(status))                                                        \
      return failure();                                                        \
  } while (false)

#if defined(_MSVC_TRADITIONAL) && _MSVC_TRADITIONAL
// The traditional MSVC preprocessor keeps forwarded __VA_ARGS__ bundled as a
// single macro argument, so the arity selector below does not expand. Use the
// comma-elision behavior supported by that preprocessor instead.
#define MLIR_ASSIGN_OR_RETURN(lhs, expr, ...)                                  \
  MLIR_ASSIGN_OR_RETURN_MSVC_IMPL(TIR_CONCAT(_status_or_, __COUNTER__), lhs,   \
                                  expr, ##__VA_ARGS__)

#define MLIR_ASSIGN_OR_RETURN_MSVC_IMPL(status_or, lhs, expr, ...)             \
  auto status_or = (expr);                                                     \
  do {                                                                         \
    if (failed(status_or)) {                                                   \
      __VA_ARGS__;                                                             \
      return failure();                                                        \
    }                                                                          \
  } while (false);                                                             \
  lhs = std::move(*status_or)
#else
#define MLIR_ASSIGN_OR_RETURN(...)                                             \
  TIR_GET_MACRO_3(__VA_ARGS__, MLIR_ASSIGN_OR_RETURN_MSG,                      \
                  MLIR_ASSIGN_OR_RETURN_NOMSG,                                 \
                  /*end*/)                                                     \
  (__VA_ARGS__)

#define MLIR_ASSIGN_OR_RETURN_NOMSG(lhs, expr)                                 \
  MLIR_ASSIGN_OR_RETURN_NOMSG_IMPL(TIR_CONCAT(_status_or_, __COUNTER__), lhs,  \
                                   expr)

#define MLIR_ASSIGN_OR_RETURN_NOMSG_IMPL(status_or, lhs, expr)                 \
  auto status_or = (expr);                                                     \
  MLIR_RETURN_IF_ERROR(status_or);                                             \
  lhs = std::move(*status_or)

#define MLIR_ASSIGN_OR_RETURN_MSG(lhs, expr, msg)                              \
  MLIR_ASSIGN_OR_RETURN_MSG_IMPL(TIR_CONCAT(_status_or_, __COUNTER__), lhs,    \
                                 expr, msg)

#define MLIR_ASSIGN_OR_RETURN_MSG_IMPL(status_or, lhs, expr, msg)              \
  auto status_or = (expr);                                                     \
  do {                                                                         \
    if (failed(status_or)) {                                                   \
      msg;                                                                     \
      return failure();                                                        \
    }                                                                          \
  } while (false);                                                             \
  lhs = std::move(*status_or)
#endif

// Get the alignment of the tensor which should be graph input or output
FailureOr<int64_t> getAlignmentFromGraph(Value tensor);

// Get the stride of the tensor which should be graph input or output
FailureOr<tcutegen::Stride> getStrideFromGraph(Value tensor);

/// Returns true if the graph has no operations in the region other than the
/// `results` op. The graph may have inputs and outputs.
bool isEmptyRegionGraphOp(nv_tensor_ir::GraphOp graphOp);

/// Returns true if any input or output tensor in the graph has dynamic shape.
bool hasDynamicInputOrOutputTensor(nv_tensor_ir::GraphOp graphOp);

/// Get layout attribute for an operation or block argument.
Attribute getLayoutSourceAttr(Value value);

/// Get block argument index offsets for a dynamic layout.
/// By convention, a tensor block argument is followed by the dynamic
/// dimension size arguments and then dynamic stride arguments.
SmallVector<int32_t> getDynamicValueMapping(const tcutegen::Layout &layout);

/// Describes the layout of a tensor for codegen: per-dimension sizes and
/// strides (each dimension may be a known constant or a dynamic value),
/// together with memory alignment, an access cost hint, and whether TMA
/// may be used. This data may be obtained from a `GraphOp` signature.
struct TensorDescriptor {
  /// Per-dimension size or stride: either a compile-time constant or a dynamic
  /// size/stride carried by an SSA value.
  /// If `staticValue` is `kDynamic`, `dynamicValue` should be set.
  struct StaticOrDynamic {
    int64_t staticValue;
    Value dynamicValue;
    /// Known power-of-two divisibility for dynamic descriptor values. TensorIR
    /// currently uses this for dynamic strides parsed from `?{div=N}` in the
    /// stride layout string. A value of 1 means no useful dynamic divisibility
    /// fact is known.
    int64_t divisibility = 1;
  };

  /// Tensor pointer (when known).
  Value pointer = nullptr;
  /// Tensor dimension sizes (static or dynamic), may be empty for scalars.
  SmallVector<StaticOrDynamic> sizes;
  /// Tensor dimension strides (static or dynamic), may be empty.
  SmallVector<StaticOrDynamic> strides;
  /// Memory alignment in bytes (0 means use auto-detection).
  int64_t alignment = 0;
  /// Memory access cost/latency hint (-1 means use default).
  int32_t cost = -1;
  /// Hint whether to use TMA (true) or not (false).
  bool allowTma = true;
};

/// Returns a vector of `TensorDescriptor`s for each tensor in the graph
/// signature (inputs or outputs).
FailureOr<SmallVector<TensorDescriptor>>
getTensorDescriptors(ArrayRef<Type> types, ArrayAttr attributes);

FailureOr<llvm::hash_code> getModuleHash(ModuleOp moduleOp);

//===----------------------------------------------------------------------===//
// Shape / tcutegen helpers
//===----------------------------------------------------------------------===//

/// @brief Type alias for dimension sizes
using DimSize = int64_t;

template <typename T>
FailureOr<llvm::SmallVector<int64_t>> toSmallVector(T attribute) {
  if (!tcutegen::is_static(attribute) ||
      (tcutegen::depth(attribute) != 0 && tcutegen::depth(attribute) != 1)) {
    return failure();
  }

  int64_t rank = tcutegen::rank(attribute);
  llvm::SmallVector<int64_t> result;
  result.reserve(rank);

  for (int64_t i = 0; i < rank; ++i) {
    result.push_back(attribute[i].as_int());
  }

  return result;
}

int64_t getDefaultAlignment(nv_tensor_ir::TensorType tensorType);

/**
 * @brief Maps the grid size to the 3D grid size needed for kernel launches.
 * @param grid_size The grid size
 * @param loc Location used to emit diagnostics on failure
 * @return The grid size mapped to 3D, or failure (with a diagnostic emitted at
 * @p loc) if the grid is empty or its flattened size exceeds the representable
 * 32-bit limit.
 */
FailureOr<std::array<int32_t, 3>>
mapGridSizeTo3D(llvm::ArrayRef<DimSize> grid_size, Location loc);

//===----------------------------------------------------------------------===//
// Type conversions
//===----------------------------------------------------------------------===//

FailureOr<arith::CmpIPredicate>
convertToArithIPredicate(mlir::nv_tensor_ir::Comparator comparator);

FailureOr<arith::CmpFPredicate>
convertToArithFPredicate(mlir::nv_tensor_ir::Comparator comparator);
Type toSignless(Type ty);

//===----------------------------------------------------------------------===//
// String formatting
//===----------------------------------------------------------------------===//

/// A variant of llvm::join that supports non-string element types via
/// llvm::raw_string_ostream, including LLVM/MLIR types with operator<<.
template <typename IteratorT>
inline std::string joinToStr(IteratorT begin, IteratorT end,
                             llvm::StringRef sep) {
  std::string s;
  llvm::raw_string_ostream os(s);
  if (begin == end) {
    return s;
  }
  os << *begin;
  while (++begin != end) {
    os << sep;
    os << *begin;
  }
  return os.str();
}

template <typename Range>
inline std::string joinToStr(Range &&R, llvm::StringRef sep) {
  return joinToStr(R.begin(), R.end(), sep);
}

/// Converts a vector-like container to a "[a, b, c]" string.
template <typename T>
std::string vectorToString(const T &vec) {
  return "[" + joinToStr(vec, ", ") + "]";
}

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_INCLUDE_UTILS_H_
