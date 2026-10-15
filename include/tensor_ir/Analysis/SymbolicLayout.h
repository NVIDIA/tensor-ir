// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// `SymbolicLayout` models multidimensional affine index mappings using
/// `PolyExpr` extents and strides, converting coordinate tuples into linear
/// offsets:
///
///     layout(c) = sum_d apply(dim_d, c_d)
///
/// Dimensions adhere to a row-major convention (last axis varies fastest),
/// conforming to TensorIR `reshape` semantics. Each dimension consists of one
/// or more strided components ("leaves"). Grouping preserves tensor rank when
/// splits introduce multi-stride runs and isolates contracted axes in reduction
/// and contraction operations.
///
/// This structure generalizes the legacy `tcutegen::Layout` algebra to support
/// symbolic polynomials. Standard algebraic primitives (coalescing, composition,
/// broadcasting, transposition, slicing) evaluate dimension identities via
/// polynomial arithmetic, enabling dynamic shapes throughout normalization.

#ifndef TENSOR_IR_ANALYSIS_SYMBOLICLAYOUT_H
#define TENSOR_IR_ANALYSIS_SYMBOLICLAYOUT_H

#include "tensor_ir/Analysis/PolyExprs.h"
#include "tensor_ir/Support/Status.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>

namespace mlir::nv_tensor_ir {

/// An elementary strided segment: spans `size` coordinates advancing the index
/// by `stride` per unit.
struct LayoutLeaf {
  PolyExpr size;
  PolyExpr stride;

  LayoutLeaf() = default;
  LayoutLeaf(PolyExpr size, PolyExpr stride) : size(size), stride(stride) {}

  bool operator==(const LayoutLeaf &other) const;
};

/// A logical coordinate axis composed of one or more strided leaves.
struct LayoutDim {
  llvm::SmallVector<LayoutLeaf, 1> leaves;

  LayoutDim() = default;
  LayoutDim(PolyExpr size, PolyExpr stride) {
    leaves.emplace_back(size, stride);
  }
  static LayoutDim getGroup(llvm::ArrayRef<LayoutLeaf> leaves);

  /// Dimension extent, defined as the product of constituent leaf sizes.
  PolyExpr getSize() const;

  /// Stride of the fastest leaf, corresponding to the dimension stride when
  /// ungrouped.
  PolyExpr getStride() const { return leaves.back().stride; }

  bool isUnit() const;

  bool operator==(const LayoutDim &other) const {
    return leaves == other.leaves;
  }
};

/// An N-dimensional strided tensor layout. Fallible construction and operations
/// return a status rather than an invalid layout. The context owns all expressions
/// and must outlive the layout. All expressions must belong to that context;
/// algebraic comparisons and arithmetic account for its symbol definitions.
/// Composition and iteration-space refinement require positive extents, with
/// shape symbols assumed positive.
class SymbolicLayout {
public:
  /// Constructs a layout from dimensions, checking expressions and their context.
  static StatusOr<SymbolicLayout> get(PolyExprContext &ctx,
                                      llvm::ArrayRef<LayoutDim> dims);

  /// Constructs a contiguous row-major layout spanning `shape`.
  static StatusOr<SymbolicLayout> getCompact(PolyExprContext &ctx,
                                             llvm::ArrayRef<PolyExpr> shape);

  /// Constructs a layout with explicit sizes and strides along each dimension.
  static StatusOr<SymbolicLayout> get(PolyExprContext &ctx,
                                      llvm::ArrayRef<PolyExpr> shape,
                                      llvm::ArrayRef<PolyExpr> strides);

  PolyExprContext &getContext() const { return *context; }
  size_t getRank() const { return dims.size(); }
  llvm::ArrayRef<LayoutDim> getDims() const { return dims; }
  const LayoutDim &getDim(size_t index) const { return dims[index]; }

  /// Extents along each logical axis.
  llvm::SmallVector<PolyExpr> getShape() const;

  /// Total coordinate count across all dimensions.
  PolyExpr getSize() const;

  /// Expands all grouped dimensions into individual top-level dimensions.
  SymbolicLayout flatten() const;

  /// Fuses contiguous adjacent axes and discards unit dimensions starting from
  /// the fastest-varying axis. Merges dimension `i` into `i+1` when
  /// `stride(i) == size(i+1) * stride(i+1)`.
  ///
  /// Produces a flat layout, collapsing pure unit shapes to `(1):(0)`.
  /// Leaves remain separate if their merged extent cannot be represented.
  SymbolicLayout coalesce() const;

  /// Projects this layout onto `targetShape` (equivalent to
  /// `reshape(value, targetShape)`). Fails if the reshape cannot be formed
  /// purely through re-indexing without strided data movement.
  ///
  /// Analogous to CuTe composition under row-major ordering: processes axes
  /// from fastest to slowest, bisecting larger segments and bundling multiple
  /// runs into grouped dimensions.
  StatusOr<SymbolicLayout> compose(llvm::ArrayRef<PolyExpr> targetShape) const;

  /// Broadcasts unit-length axes to `newShape` with zero strides. Non-unit
  /// dimensions must match the destination extent identically.
  StatusOr<SymbolicLayout> broadcastTo(llvm::ArrayRef<PolyExpr> newShape) const;

  /// Reorders dimensions according to `permutation`, setting output axis `i`
  /// to input axis `permutation[i]` (matching `nv_tensor_ir.transpose`).
  /// Requires a permutation containing each axis exactly once.
  SymbolicLayout transpose(llvm::ArrayRef<int64_t> permutation) const;

  /// Restricts coordinate ranges using `[start, limit)` and `stride` vectors,
  /// accumulating linear index displacements into `offset`.
  StatusOr<SymbolicLayout> slice(llvm::ArrayRef<int64_t> starts,
                                 llvm::ArrayRef<int64_t> limits,
                                 llvm::ArrayRef<int64_t> strides,
                                 PolyExpr &offset) const;

  /// Subsets dimensions in range `[begin, end)`.
  /// Requires begin <= end <= getRank().
  SymbolicLayout take(size_t begin, size_t end) const;

  bool operator==(const SymbolicLayout &other) const {
    return context == other.context && dims == other.dims;
  }
  bool operator!=(const SymbolicLayout &other) const {
    return !(*this == other);
  }

  /// Formats layout using `(shape):(stride)` notation consistent with legacy
  /// `tcutegen` strings, emitting `?` for non-constant dimensions.
  std::string toString() const;

  /// Prints the layout with expanded symbolic polynomial expressions.
  void print(llvm::raw_ostream &os) const;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const SymbolicLayout &layout) {
    layout.print(os);
    return os;
  }

  /// Emits layout representation with expanded symbolic polynomial expressions.
  std::string toDebugString() const;

private:
  explicit SymbolicLayout(PolyExprContext &ctx,
                          llvm::ArrayRef<LayoutDim> dims = {})
      : context(&ctx), dims(dims.begin(), dims.end()) {}

  PolyExprContext *context;
  llvm::SmallVector<LayoutDim> dims;
};

/// Maps linear index into coordinates over `shape` under row-major ordering.
/// Requires all dimensions to be compile-time constants.
std::optional<llvm::SmallVector<int64_t>>
delinearize(int64_t index, llvm::ArrayRef<PolyExpr> shape);

/// Finds index `i` where suffix product `shape[i:]` matches `suffixProduct`,
/// returning `std::nullopt` if no exact partition exists.
std::optional<size_t>
findDimensionWithSuffixProduct(llvm::ArrayRef<PolyExpr> shape,
                               PolyExpr suffixProduct);

/// Computes the coarsest shared factorization across multiple shapes of equal
/// volume without reordering axes.
///
/// Example: `(128,4)` and `(4,128)` resolve to `(4,32,4)`.
///
/// Aligns dimensions starting from the fastest axis. Refinement requires
/// divisibility: `s0` and `2*s0` refine to `(2,s0)`, whereas independent
/// symbols `s0` and `s1` cannot be reconciled.
std::optional<llvm::SmallVector<PolyExpr>>
getCommonIterationSpace(PolyExprContext &ctx,
                        llvm::ArrayRef<llvm::SmallVector<PolyExpr>> input);

/// Convenience binary helper for `getCommonIterationSpace`.
std::optional<llvm::SmallVector<PolyExpr>>
getCommonIterationSpace(PolyExprContext &ctx, llvm::ArrayRef<PolyExpr> lhs,
                        llvm::ArrayRef<PolyExpr> rhs);

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_ANALYSIS_SYMBOLICLAYOUT_H
