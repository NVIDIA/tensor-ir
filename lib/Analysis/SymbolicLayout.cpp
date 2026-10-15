// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Analysis/SymbolicLayout.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>

#define DEBUG_TYPE "symbolic-layout"

using namespace mlir;
using namespace mlir::nv_tensor_ir;

static bool equals(PolyExpr lhs, PolyExpr rhs) {
  return lhs && rhs && lhs.getContext().areEqual(lhs, rhs).value_or(false);
}

static PolyExpr product(PolyExprContext &context,
                        llvm::ArrayRef<PolyExpr> values) {
  PolyExpr result = PolyExpr::getConstant(context, 1);
  for (PolyExpr value : values) {
    result *= value;
  }
  return result;
}

// Shape symbols denote positive extents here. Reject zero and expressions whose
// positivity cannot be established under that precondition.
static bool isPositiveExtent(PolyExprContext &context, PolyExpr expr) {
  expr = context.normalize(expr);
  return expr && !expr.isZero() && llvm::all_of(expr, [](PolyExpr::Term term) {
           return term.coefficient > 0;
         });
}

static bool validShape(PolyExprContext &context,
                       llvm::ArrayRef<PolyExpr> shape) {
  return llvm::all_of(
      shape, [&](PolyExpr expr) { return isPositiveExtent(context, expr); });
}

/// Computes the linear index offset for a single dimension coordinate. Grouped
/// dimensions decompose coordinates across leaf sizes in row-major order.
static PolyExpr applyDim(const LayoutDim &dim, int64_t coordinate) {
  auto *context = &dim.leaves.front().size.getContext();
  if (dim.leaves.size() == 1) {
    return dim.leaves.front().stride *
           PolyExpr::getConstant(*context, coordinate);
  }
  PolyExpr offset = PolyExpr::getConstant(*context, 0);
  int64_t remaining = coordinate;
  for (const LayoutLeaf &leaf : llvm::reverse(dim.leaves)) {
    std::optional<int64_t> size = leaf.size.normalize().getConstantInt64();
    if (!size || *size <= 0) {
      return {};
    }
    offset += leaf.stride * PolyExpr::getConstant(*context, remaining % *size);
    remaining /= *size;
  }
  return offset;
}

static void printExpr(llvm::raw_ostream &os, PolyExpr expr, bool symbolic) {
  if (symbolic) {
    expr.print(os);
    return;
  }
  expr = expr.normalize();
  if (expr && expr.isConstant()) {
    os << *expr.getConstantValue();
    return;
  }
  os << '?';
}

static void printLayout(llvm::raw_ostream &os, llvm::ArrayRef<LayoutDim> dims,
                        bool symbolic, bool strides) {
  // Omits spaces around separators to conform to legacy layout-propagation string syntax.
  os << '(';
  llvm::interleave(
      dims, os,
      [&](const LayoutDim &dim) {
        if (dim.leaves.size() > 1) {
          os << '(';
        }
        llvm::interleave(
            dim.leaves, os,
            [&](const LayoutLeaf &leaf) {
              printExpr(os, strides ? leaf.stride : leaf.size, symbolic);
            },
            ",");
        if (dim.leaves.size() > 1) {
          os << ')';
        }
      },
      ",");
  os << ')';
}

//===----------------------------------------------------------------------===//
// LayoutDim
//===----------------------------------------------------------------------===//

bool LayoutLeaf::operator==(const LayoutLeaf &other) const {
  return equals(size, other.size) && equals(stride, other.stride);
}

bool LayoutDim::isUnit() const { return getSize().isOne(); }

LayoutDim LayoutDim::getGroup(llvm::ArrayRef<LayoutLeaf> leaves) {
  LayoutDim dim;
  dim.leaves.assign(leaves.begin(), leaves.end());
  return dim;
}

PolyExpr LayoutDim::getSize() const {
  if (leaves.empty() || !leaves.front().size) {
    return {};
  }
  auto *context = &leaves.front().size.getContext();
  PolyExpr size = PolyExpr::getConstant(*context, 1);
  for (const LayoutLeaf &leaf : leaves) {
    size *= leaf.size;
  }
  return size;
}

//===----------------------------------------------------------------------===//
// SymbolicLayout construction
//===----------------------------------------------------------------------===//

StatusOr<SymbolicLayout> SymbolicLayout::get(PolyExprContext &ctx,
                                             llvm::ArrayRef<LayoutDim> dims) {
  for (const LayoutDim &dim : dims) {
    if (dim.leaves.empty()) {
      return Status::InvalidArgument("layout dimensions must contain leaves");
    }
    for (const LayoutLeaf &leaf : dim.leaves) {
      if (!ctx.normalize(leaf.size) || !ctx.normalize(leaf.stride)) {
        return Status::InvalidArgument(
            "layout expressions must belong to its context");
      }
    }
    if (!dim.getSize()) {
      return Status::NotSupported("cannot compute dimension extent");
    }
  }
  return SymbolicLayout(ctx, dims);
}

StatusOr<SymbolicLayout>
SymbolicLayout::getCompact(PolyExprContext &ctx,
                           llvm::ArrayRef<PolyExpr> shape) {
  auto *context = &ctx;
  llvm::SmallVector<PolyExpr> strides(shape.size());
  PolyExpr running = PolyExpr::getConstant(*context, 1);
  for (size_t index = shape.size(); index-- > 0;) {
    strides[index] = running;
    running *= shape[index];
  }
  if (!running) {
    return Status::InvalidArgument(
        "compact shape contains invalid expressions");
  }
  return get(*context, shape, strides);
}

StatusOr<SymbolicLayout> SymbolicLayout::get(PolyExprContext &ctx,
                                             llvm::ArrayRef<PolyExpr> shape,
                                             llvm::ArrayRef<PolyExpr> strides) {
  auto *context = &ctx;
  if (shape.size() != strides.size()) {
    return Status::InvalidArgument("shape and stride ranks must match");
  }
  llvm::SmallVector<LayoutDim> layoutDims;
  for (auto [size, stride] : llvm::zip_equal(shape, strides)) {
    if (!size || !stride) {
      return Status::InvalidArgument(
          "layout sizes and strides must be non-null");
    }
    layoutDims.emplace_back(size, stride);
  }
  return get(*context, layoutDims);
}

llvm::SmallVector<PolyExpr> SymbolicLayout::getShape() const {
  llvm::SmallVector<PolyExpr> shape;
  shape.reserve(dims.size());
  for (const LayoutDim &dim : dims) {
    shape.push_back(dim.getSize());
  }
  return shape;
}

PolyExpr SymbolicLayout::getSize() const {
  PolyExpr size = PolyExpr::getConstant(*context, 1);
  for (const LayoutDim &dim : dims) {
    size *= dim.getSize();
  }
  return size;
}

SymbolicLayout SymbolicLayout::take(size_t begin, size_t end) const {
  assert(begin <= end && end <= dims.size() &&
         "dimension range is out of bounds");
  return SymbolicLayout(
      *context, llvm::ArrayRef<LayoutDim>(dims).slice(begin, end - begin));
}

//===----------------------------------------------------------------------===//
// flatten / coalesce
//===----------------------------------------------------------------------===//

SymbolicLayout SymbolicLayout::flatten() const {
  SymbolicLayout result(*context);
  for (const LayoutDim &dim : dims) {
    for (const LayoutLeaf &leaf : dim.leaves) {
      result.dims.emplace_back(leaf.size, leaf.stride);
    }
  }
  return result;
}

SymbolicLayout SymbolicLayout::coalesce() const {
  llvm::SmallVector<LayoutLeaf> leaves;
  for (const LayoutDim &dim : dims) {
    llvm::append_range(leaves, dim.leaves);
  }
  if (leaves.empty()) {
    return SymbolicLayout(*context, llvm::ArrayRef<LayoutDim>{LayoutDim(
                                        PolyExpr::getConstant(*context, 1),
                                        PolyExpr::getConstant(*context, 0))});
  }

  // Process dimensions from fastest to slowest. A dimension absorbs its adjacent
  // faster neighbor when its stride exactly equals the full span of that neighbor.
  for (size_t index = leaves.size() - 1; index > 0; --index) {
    if (leaves[index - 1].size.normalize().isOne()) {
      leaves.erase(leaves.begin() + index - 1);
      continue;
    }
    const LayoutLeaf &left = leaves[index - 1];
    const LayoutLeaf &right = leaves[index];
    if (!equals(left.stride, right.size * right.stride)) {
      continue;
    }
    LayoutLeaf merged(left.size * right.size, right.stride);
    if (!merged.size) {
      continue;
    }
    leaves[index - 1] = merged;
    leaves.erase(leaves.begin() + index);
  }

  if (llvm::all_of(leaves, [](const LayoutLeaf &leaf) {
        return leaf.size.normalize().isOne();
      })) {
    return SymbolicLayout(*context, llvm::ArrayRef<LayoutDim>{LayoutDim(
                                        PolyExpr::getConstant(*context, 1),
                                        PolyExpr::getConstant(*context, 0))});
  }
  if (leaves.back().size.normalize().isOne()) {
    leaves.pop_back();
  }

  llvm::SmallVector<LayoutDim> resultDims;
  for (const LayoutLeaf &leaf : leaves) {
    resultDims.emplace_back(leaf.size, leaf.stride);
  }
  return SymbolicLayout(*context, resultDims);
}

//===----------------------------------------------------------------------===//
// compose
//===----------------------------------------------------------------------===//

StatusOr<SymbolicLayout>
SymbolicLayout::compose(llvm::ArrayRef<PolyExpr> targetShape) const {
  if (!validShape(*context, targetShape) || !validShape(*context, getShape())) {
    return Status::InvalidArgument(
        "composition requires provably positive extents");
  }
  SymbolicLayout coalesced = coalesce();

  // Match source and target dimensions starting from the fastest axis, inverting
  // results upon completion. `group` gathers constituent source runs for each
  // target axis.
  llvm::SmallVector<LayoutDim> reversedResult;
  llvm::SmallVector<LayoutLeaf> group;
  auto flushGroup = [&]() {
    if (group.size() == 1) {
      reversedResult.emplace_back(group.front().size, group.front().stride);
    } else if (group.size() > 1) {
      std::reverse(group.begin(), group.end());
      reversedResult.push_back(LayoutDim::getGroup(group));
    }
    group.clear();
  };

  size_t sourceRemaining = coalesced.getRank();
  size_t targetRemaining = targetShape.size();
  std::optional<LayoutLeaf> sourcePart;
  PolyExpr targetPart;

  while ((sourcePart || sourceRemaining > 0) &&
         (targetPart || targetRemaining > 0)) {
    if (!sourcePart) {
      sourcePart = coalesced.getDim(--sourceRemaining).leaves.front();
    }
    if (!targetPart) {
      flushGroup();
      targetPart = targetShape[--targetRemaining];
    }

    PolyExpr sourceSize = sourcePart->size;
    PolyExpr targetSize = targetPart;

    // Consume the entire source segment when its size does not exceed the target axis.
    if (PolyExpr rest = targetSize / sourceSize) {
      group.push_back(*sourcePart);
      sourcePart = std::nullopt;
      targetPart = rest.isOne() ? nullptr : rest;
      continue;
    }

    // Partition the source run when it spans past the target axis, retaining
    // the residue for slower dimensions.
    if (PolyExpr rest = sourceSize / targetSize) {
      PolyExpr stride = sourcePart->stride;
      group.emplace_back(targetSize, targetSize.normalize().isOne()
                                         ? PolyExpr::getConstant(*context, 0)
                                         : stride);
      sourcePart = LayoutLeaf(rest, stride * targetSize);
      targetPart = nullptr;
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "compose: incompatible extents\n");
    return Status::NotSupported(
        "cannot prove divisibility of composition extents");
  }

  // Unmatched remaining axes on either side must have unit extent; non-unit
  // residues indicate mismatched total volumes or ordering.
  bool sourceHasLeading = false;
  while (sourceRemaining > 0) {
    sourceHasLeading |= !coalesced.getDim(--sourceRemaining).isUnit();
  }
  bool targetHasLeading = false;
  int64_t leadingUnits = 0;
  while (targetRemaining > 0) {
    bool isUnit = targetShape[--targetRemaining].normalize().isOne();
    targetHasLeading |= !isUnit;
    leadingUnits += isUnit;
  }
  if (sourcePart || sourceHasLeading || targetPart || targetHasLeading) {
    LLVM_DEBUG(llvm::dbgs() << "compose: leftover extents\n");
    return Status::InvalidArgument("composition has unmatched extents");
  }

  flushGroup();
  while (leadingUnits-- > 0) {
    reversedResult.emplace_back(PolyExpr::getConstant(*context, 1),
                                PolyExpr::getConstant(*context, 0));
  }
  std::reverse(reversedResult.begin(), reversedResult.end());
  return get(*context, reversedResult);
}

//===----------------------------------------------------------------------===//
// broadcast / transpose / slice
//===----------------------------------------------------------------------===//

StatusOr<SymbolicLayout>
SymbolicLayout::broadcastTo(llvm::ArrayRef<PolyExpr> newShape) const {
  if (!validShape(*context, newShape) || newShape.size() != dims.size()) {
    return Status::InvalidArgument(
        "broadcast requires matching ranks and positive extents");
  }
  llvm::SmallVector<LayoutDim> resultDims;
  for (auto [dim, newSize] : llvm::zip_equal(dims, newShape)) {
    PolyExpr size = dim.getSize();
    if (equals(size, newSize)) {
      resultDims.push_back(dim);
      continue;
    }
    if (!size.isOne()) {
      return Status::InvalidArgument("cannot broadcast a non-unit dimension");
    }
    resultDims.emplace_back(newSize, PolyExpr::getConstant(*context, 0));
  }
  return get(*context, resultDims);
}

SymbolicLayout
SymbolicLayout::transpose(llvm::ArrayRef<int64_t> permutation) const {
  assert(permutation.size() == dims.size() &&
         "permutation rank must match layout rank");
#ifndef NDEBUG
  llvm::SmallVector<bool> seen(dims.size(), false);
  for (int64_t position : permutation) {
    assert(position >= 0 && static_cast<size_t>(position) < dims.size() &&
           !seen[position] &&
           "permutation must contain each axis exactly once");
    seen[position] = true;
  }
#endif
  llvm::SmallVector<LayoutDim> resultDims;
  resultDims.reserve(dims.size());
  for (int64_t position : permutation) {
    resultDims.push_back(dims[position]);
  }
  return SymbolicLayout(*context, resultDims);
}

StatusOr<SymbolicLayout> SymbolicLayout::slice(llvm::ArrayRef<int64_t> starts,
                                               llvm::ArrayRef<int64_t> limits,
                                               llvm::ArrayRef<int64_t> strides,
                                               PolyExpr &offset) const {
  size_t rank = dims.size();
  if (starts.size() != rank || limits.size() != rank ||
      strides.size() != rank || !context->normalize(offset)) {
    return Status::InvalidArgument(
        "slice requires matching ranks and an offset in the layout context");
  }

  llvm::SmallVector<LayoutDim> resultDims;
  PolyExpr sliceOffset = PolyExpr::getConstant(*context, 0);
  for (size_t index = 0; index < rank; ++index) {
    const LayoutDim &dim = dims[index];
    int64_t start = starts[index];
    int64_t end = limits[index];
    int64_t step = strides[index];
    if (start < 0 || end <= start || step <= 0) {
      return Status::InvalidArgument(
          "slice bounds and strides must be positive and ordered");
    }
    // Static slicing over dynamic extents is valid because in-bounds limits
    // constitute a runtime invariant for SliceOp.
    if (std::optional<int64_t> extent = dim.getSize().getConstantInt64()) {
      if (end > *extent) {
        return Status::InvalidArgument("slice exceeds dimension extent");
      }
    }
    int64_t count = 1 + (end - start - 1) / step;

    // For an ungrouped dimension, the slice produces another strided run.
    if (dim.leaves.size() == 1) {
      const LayoutLeaf &leaf = dim.leaves.front();
      PolyExpr newStride =
          (leaf.stride * PolyExpr::getConstant(*context, step));
      resultDims.emplace_back(PolyExpr::getConstant(*context, count),
                              count > 1 ? newStride
                                        : PolyExpr::getConstant(*context, 0));
      sliceOffset = (sliceOffset +
                     (leaf.stride * PolyExpr::getConstant(*context, start)));
      continue;
    }

    // Grouped dimension: the slice has to be re-expressed against the leaves
    // it spans. Only patterns that stay strided are representable.
    // For grouped dimensions, the slice must be projected across constituent
    // leaves; only transformations preserving regular strided access can be modeled.
    std::optional<llvm::SmallVector<int64_t>> coordinate = delinearize(
        start,
        llvm::to_vector(llvm::map_range(
            dim.leaves, [](const LayoutLeaf &leaf) { return leaf.size; })));
    if (!coordinate) {
      return Status::NotSupported("cannot delinearize grouped slice start");
    }
    sliceOffset += applyDim(dim, start);

    llvm::SmallVector<LayoutLeaf> subLeaves;
    for (size_t leafIndex = dim.leaves.size(); leafIndex-- > 0;) {
      const LayoutLeaf &leaf = dim.leaves[leafIndex];
      std::optional<int64_t> limit = leaf.size.normalize().getConstantInt64();
      if (!limit) {
        return Status::NotSupported(
            "grouped slicing requires constant leaf extents");
      }
      int64_t leafStart = (*coordinate)[leafIndex];

      // Case 1: all extracted coordinates reside entirely within this leaf.
      //   layout (8,2):(1,8), slice [2:7:2] -> ..x.x.x.
      if (count - 1 <= (*limit - 1 - leafStart) / step) {
        subLeaves.emplace_back(
            PolyExpr::getConstant(*context, count),
            count > 1 ? (leaf.stride * PolyExpr::getConstant(*context, step))
                      : PolyExpr::getConstant(*context, 0));
        break;
      }

      // Case 2: the step evenly divides the leaf period, contributing an
      // identical coordinate count per cycle.
      //   layout (8,2):(1,8), slice [1:16:2] -> .x.x.x.x / .x.x.x.x
      int64_t fits = *limit / step;
      if (*limit % step == 0 && leafStart < step && fits > 1 &&
          count % fits == 0) {
        subLeaves.emplace_back(
            PolyExpr::getConstant(*context, fits),
            (leaf.stride * PolyExpr::getConstant(*context, step)));
        count /= fits;
        step = 1;
        continue;
      }

      // Case 3: the stride step skips across whole cycles of this leaf.
      //   layout (8,3):(1,8), slice [4:21:16]
      if (step % *limit == 0) {
        step /= *limit;
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "slice: unsupported strided pattern\n");
      return Status::NotSupported(
          "slice pattern cannot be represented by strided leaves");
    }
    if (subLeaves.empty()) {
      return Status::NotSupported("slice produces no strided leaves");
    }
    std::reverse(subLeaves.begin(), subLeaves.end());
    if (subLeaves.size() == 1) {
      resultDims.emplace_back(subLeaves.front().size, subLeaves.front().stride);
    } else {
      resultDims.push_back(LayoutDim::getGroup(subLeaves));
    }
  }

  PolyExpr newOffset = (offset + sliceOffset);
  if (!newOffset) {
    return Status::NotSupported("cannot compute slice offset");
  }
  TIR_ASSIGN_OR_RETURN(auto result, get(*context, resultDims));
  offset = newOffset;
  return result;
}

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

std::string SymbolicLayout::toString() const {
  std::string result;
  llvm::raw_string_ostream os(result);
  printLayout(os, dims, /*symbolic=*/false, /*strides=*/false);
  os << ':';
  printLayout(os, dims, /*symbolic=*/false, /*strides=*/true);
  return os.str();
}

void SymbolicLayout::print(llvm::raw_ostream &os) const {
  printLayout(os, dims, /*symbolic=*/true, /*strides=*/false);
  os << ':';
  printLayout(os, dims, /*symbolic=*/true, /*strides=*/true);
}

std::string SymbolicLayout::toDebugString() const {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << *this;
  return os.str();
}

//===----------------------------------------------------------------------===//
// Free functions
//===----------------------------------------------------------------------===//

std::optional<llvm::SmallVector<int64_t>>
nv_tensor_ir::delinearize(int64_t index, llvm::ArrayRef<PolyExpr> shape) {
  if (index < 0) {
    return std::nullopt;
  }
  llvm::SmallVector<int64_t> coordinate(shape.size());
  int64_t remaining = index;
  for (size_t position = shape.size(); position-- > 0;) {
    std::optional<int64_t> extent =
        shape[position].normalize().getConstantInt64();
    if (!extent || *extent <= 0) {
      return std::nullopt;
    }
    coordinate[position] = remaining % *extent;
    remaining /= *extent;
  }
  return coordinate;
}

std::optional<size_t>
nv_tensor_ir::findDimensionWithSuffixProduct(llvm::ArrayRef<PolyExpr> shape,
                                             PolyExpr suffixProduct) {
  if (!suffixProduct) {
    return std::nullopt;
  }
  auto *context = &suffixProduct.getContext();
  PolyExpr accumulated = PolyExpr::getConstant(*context, 1);
  for (size_t index = shape.size(); index > 0; --index) {
    if (equals(suffixProduct, accumulated)) {
      return index;
    }
    accumulated *= shape[index - 1];
  }
  if (equals(suffixProduct, accumulated)) {
    return 0;
  }
  return std::nullopt;
}

std::optional<llvm::SmallVector<PolyExpr>>
nv_tensor_ir::getCommonIterationSpace(
    PolyExprContext &ctx, llvm::ArrayRef<llvm::SmallVector<PolyExpr>> input) {
  auto *context = &ctx;
  if (input.empty()) {
    return llvm::SmallVector<PolyExpr>{};
  }
  llvm::SmallVector<llvm::SmallVector<PolyExpr>> shapes(input.begin(),
                                                        input.end());
  PolyExpr volume = product(*context, shapes.front());
  for (const auto &shape : shapes) {
    if (!validShape(*context, shape) ||
        !equals(volume, product(*context, shape))) {
      return std::nullopt;
    }
  }
  llvm::SmallVector<PolyExpr> result;

  while (!shapes.front().empty()) {
    // Select the finest trailing axis that evenly divides all competing trailing
    // extents. For integers this corresponds to the minimum; for symbolic
    // extents, divisibility is the necessary and sufficient condition.
    PolyExpr divisor;
    for (const auto &shape : shapes) {
      if (shape.empty()) {
        continue;
      }
      PolyExpr candidate = context->normalize(shape.back());
      bool dividesAll = llvm::all_of(shapes, [&](const auto &other) {
        return other.empty() || (other.back() / candidate) != nullptr;
      });
      if (dividesAll) {
        divisor = candidate;
        break;
      }
    }
    if (!divisor || !isPositiveExtent(*context, divisor)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "common iteration space: incompatible trailing extents\n");
      return std::nullopt;
    }
    result.push_back(divisor);

    for (auto &shape : shapes) {
      if (shape.empty()) {
        continue;
      }
      PolyExpr quotient = (shape.back() / divisor);
      if (!quotient) {
        return std::nullopt;
      }
      if (quotient.isOne()) {
        shape.pop_back();
      } else {
        shape.back() = quotient;
      }
    }
  }

  // Any unmatched leading dimensions across input shapes must evaluate to unit extent.
  size_t unitDims = 0;
  for (const auto &shape : shapes) {
    PolyExpr size = product(*context, shape);
    if (!size || !size.isOne()) {
      return std::nullopt;
    }
    unitDims = std::max(unitDims, shape.size());
  }
  result.append(unitDims, PolyExpr::getConstant(*context, 1));
  std::reverse(result.begin(), result.end());
  return result;
}

std::optional<llvm::SmallVector<PolyExpr>>
nv_tensor_ir::getCommonIterationSpace(PolyExprContext &ctx,
                                      llvm::ArrayRef<PolyExpr> lhs,
                                      llvm::ArrayRef<PolyExpr> rhs) {
  llvm::SmallVector<llvm::SmallVector<PolyExpr>> shapes;
  shapes.emplace_back(lhs.begin(), lhs.end());
  shapes.emplace_back(rhs.begin(), rhs.end());
  return getCommonIterationSpace(ctx, shapes);
}
