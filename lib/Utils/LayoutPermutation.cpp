// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Utils/LayoutPermutation.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::nv_tensor_ir;
namespace tcg = mlir::nv_tensor_ir::tcutegen;

/// Describe each top-level axis, preserving grouped shape/stride structure.
static SmallVector<Attribute> getViewKeys(MLIRContext *context,
                                          const tcg::Layout &layout) {
  SmallVector<Attribute> keys;
  for (size_t i = 0; i < tcg::rank(layout); ++i) {
    keys.push_back(StringAttr::get(context, tcg::get(layout, i).toString()));
  }
  return keys;
}

static FailureOr<SmallVector<Attribute>>
getDimensionKeys(LayoutSourceAttrInterface source);

/// Composite and concatenated layouts transpose all children together. Axes
/// must agree in every child, in source order, to be interchangeable.
static FailureOr<SmallVector<Attribute>>
getSourceKeys(MLIRContext *context,
              ArrayRef<LayoutSourceAttrInterface> sources) {
  SmallVector<SmallVector<Attribute>> children;
  for (LayoutSourceAttrInterface child : sources) {
    auto keys = getDimensionKeys(child);
    if (failed(keys) ||
        (!children.empty() && keys->size() != children.front().size())) {
      return failure();
    }
    children.push_back(std::move(*keys));
  }
  if (children.empty()) {
    return failure();
  }
  SmallVector<Attribute> keys;
  for (size_t i = 0; i < children.front().size(); ++i) {
    SmallVector<Attribute> parts;
    for (const auto &child : children) {
      parts.push_back(child[i]);
    }
    keys.push_back(ArrayAttr::get(context, parts));
  }
  return keys;
}

/// Keys describe only data that moves with an axis under transpose. Invariant
/// metadata (buffer identities, offsets, contraction sources, etc.) is checked
/// by the final full-attribute comparison in findLayoutPermutation.
static FailureOr<SmallVector<Attribute>>
getDimensionKeys(LayoutSourceAttrInterface source) {
  MLIRContext *context = source.getContext();
  if (auto tensor = dyn_cast<TensorSourceAttr>(source)) {
    auto layout = tensor.getCuteLayout();
    SmallVector<Attribute> keys = getViewKeys(context, layout);
    SmallVector<int32_t> sizes(keys.size(), -1), strides(keys.size(), -1);
    ArrayRef<int32_t> mapping = tensor.getDynamicValueMapping();
    size_t next = 0;
    // This is the descriptor order used by TensorSourceAttr::transpose:
    // runtime-backed sizes first, then runtime strides. Dynamic broadcast
    // extents have no runtime size in the source descriptor.
    for (size_t i = 0; i < keys.size(); ++i) {
      auto shape = tcg::get(layout.shape(), i);
      auto stride = tcg::get(layout.stride(), i);
      if (!tcg::is_static(shape) &&
          (!tcg::is_static(stride) || stride.as_int() != 0)) {
        sizes[i] = mapping[next++];
      }
    }
    for (size_t i = 0; i < keys.size(); ++i) {
      if (!tcg::is_static(tcg::get(layout.stride(), i))) {
        strides[i] = mapping[next++];
      }
      auto dynamic = DenseI32ArrayAttr::get(context, {sizes[i], strides[i]});
      keys[i] = ArrayAttr::get(context, {keys[i], dynamic});
    }
    return keys;
  }
  if (auto composite = dyn_cast<CompositeSourceAttr>(source)) {
    return getSourceKeys(context, composite.getSources());
  }
  if (auto concat = dyn_cast<ConcatSourceAttr>(source)) {
    auto keys = getSourceKeys(context, concat.getSources());
    if (failed(keys)) {
      return failure();
    }
    for (auto [i, key] : llvm::enumerate(*keys)) {
      bool isConcatAxis = static_cast<int64_t>(i) == concat.getDimension();
      key =
          ArrayAttr::get(context, {key, BoolAttr::get(context, isConcatAxis)});
    }
    return keys;
  }
  if (auto reduction = dyn_cast<ReductionSourceAttr>(source)) {
    auto keys = getViewKeys(context, reduction.getCuteLayout());
    keys.pop_back(); // The contracting axis does not participate in transpose.
    return keys;
  }
  if (auto matmul = dyn_cast<MatmulSourceAttr>(source)) {
    auto keys = getViewKeys(context, matmul.getCuteLayout());
    keys.pop_back(); // The contracting axis does not participate in transpose.
    return keys;
  }
  if (auto scan = dyn_cast<ScanSliceSourceAttr>(source)) {
    // ScanSliceSourceAttr only supports the identity permutation.
    SmallVector<Attribute> keys;
    size_t rank = scan.getShape().size();
    for (size_t i = 0; i < rank; ++i) {
      keys.push_back(DenseI64ArrayAttr::get(context, {int64_t(i)}));
    }
    return keys;
  }
  return failure();
}

FailureOr<SmallVector<int64_t>> nv_tensor_ir::findLayoutPermutation(
    LayoutSourceAttrInterface source, LayoutSourceAttrInterface target,
    llvm::function_ref<bool(unsigned, unsigned)> matchDimensions) {
  auto sourceKeys = getDimensionKeys(source);
  auto targetKeys = getDimensionKeys(target);
  if (failed(sourceKeys) || failed(targetKeys) ||
      sourceKeys->size() != targetKeys->size()) {
    return failure();
  }
  size_t rank = sourceKeys->size();
  SmallVector<int64_t> permutation;
  SmallVector<bool> used(rank);
  for (size_t targetDim = 0; targetDim < rank; ++targetDim) {
    size_t sourceDim = 0;
    for (; sourceDim < rank; ++sourceDim) {
      if (!used[sourceDim] &&
          (*sourceKeys)[sourceDim] == (*targetKeys)[targetDim] &&
          (!matchDimensions || matchDimensions(sourceDim, targetDim))) {
        break;
      }
    }
    if (sourceDim == rank) {
      return failure();
    }
    used[sourceDim] = true;
    permutation.push_back(sourceDim);
  }
  if (source.transpose(permutation) != target) {
    return failure();
  }
  return permutation;
}
