// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Dialect/TensorIRAttrs.h"
#include "tensor_ir/Support/TCutegen.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/Debug.h"

#include <algorithm>
#include <functional>
#include <numeric>

#define DEBUG_TYPE "tensor-layout-attrs"

// ODS-generated definitions.
#include "tensor_ir/Dialect/TensorAttrInterfaces.cpp.inc"

namespace mlir::nv_tensor_ir {

namespace tcg = mlir::nv_tensor_ir::tcutegen;

namespace {

constexpr int64_t kDynamic = mlir::ShapedType::kDynamic;

bool isValidTensorElementType(Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type)) {
    if (intTy.getWidth() == 1) {
      return true;
    }
    if (intTy.isSignless()) {
      return false;
    }
    switch (intTy.getWidth()) {
    case 8:
    case 16:
    case 32:
    case 64:
      return true;
    default:
      return false;
    }
  }
  return isa<FloatType>(type);
}

/// Calculate product of dimension sizes, yields one for an empty range.
/// Returns `kDynamic` if any sizes are non-positive.
int64_t product(ArrayRef<int64_t> values) {
  if (llvm::any_of(values, [](int64_t v) { return v <= 0; })) {
    return kDynamic;
  }
  return std::accumulate(values.begin(), values.end(), int64_t(1),
                         std::multiplies<int64_t>());
}

/// Calculate number of dynamic sizes in a tcutegen layout.
size_t countDynamicSizes(const tcg::Layout &layout) {
  std::function<size_t(ArrayRef<tcg::detail::RecursiveDimValue>)> countDynamic =
      [&](ArrayRef<tcg::detail::RecursiveDimValue> values) {
        size_t count = 0;
        for (const auto &value : values) {
          if (!value.isLeaf()) {
            count += countDynamic(value.getValues());
          } else if (!value.isStatic()) {
            count++;
          }
        }
        return count;
      };
  return countDynamic(layout.shape().getValues()) +
         countDynamic(layout.stride().getValues());
}

/// Build an iteration space that is compatible with every input shape.
/// Example: (128,4) and (4,128) have a common iteration space of (4,32,4).
llvm::FailureOr<SmallVector<int64_t>>
getCommonIterationSpace(SmallVector<SmallVector<int64_t>> shapes) {
  SmallVector<int64_t> result;
  while (!shapes[0].empty()) {
    // Select smallest shape size at each iteration.
    // Negative size denotes a dynamic dimension.
    int64_t div = shapes[0].back();
    for (const auto &shape : shapes) {
      if (!shape.empty()) {
        div = std::min(div, shape.back());
      }
    }
    result.push_back(div);
    bool expectDynamic = div <= 0;

    // Update the remaining parts vector.
    for (auto &shape : shapes) {
      if (shape.empty()) {
        continue;
      }
      // If the current dimension is dynamic, expect all shapes to be dynamic.
      // Otherwise, the dimension must be divisible by the current size.
      int64_t size = shape.back();
      bool isDynamic = size <= 0;
      if (expectDynamic ? !isDynamic : isDynamic || size % div != 0) {
        return llvm::failure();
      }
      // Reduce the dimension size or remove it.
      if (size != div) {
        shape.back() /= div;
      } else {
        shape.pop_back();
      }
    }
  }

  // Add remaining unit dimensions (if present).
  int unitDims = 0;
  for (const auto &shape : shapes) {
    if (product(shape) != 1) {
      return llvm::failure();
    }
    unitDims = std::max(unitDims, static_cast<int>(shape.size()));
  }
  while (unitDims--) {
    result.push_back(1);
  }

  // The resulting shape which was built in reverse order, fix it.
  std::reverse(result.begin(), result.end());
  return result;
}

// Apply a function to each underlying source and return as a vector.
llvm::FailureOr<SmallVector<LayoutSourceAttrInterface>>
mapSources(ArrayRef<LayoutSourceAttrInterface> sources,
           const std::function<LayoutSourceAttrInterface(
               const LayoutSourceAttrInterface &)> &fn) {
  SmallVector<LayoutSourceAttrInterface> newSources;
  newSources.reserve(sources.size());
  for (const auto &src : sources) {
    LayoutSourceAttrInterface newSrc = fn(src);
    if (!newSrc) {
      return llvm::failure();
    }
    newSources.push_back(newSrc);
  }
  return newSources;
}

// Find the boundary whose trailing dimensions have the requested product.
// Return the rank for an empty trailing range.
llvm::FailureOr<size_t> findDimensionWithSuffixProduct(ArrayRef<int64_t> shape,
                                                       int64_t product) {
  int64_t accum = 1;
  for (size_t i = shape.size(); i > 0; --i) {
    if (product == accum) {
      return i;
    }
    accum *= shape[i - 1];
  }
  if (product == accum) {
    return 0;
  }
  return llvm::failure();
}

// Compose a layout with a target shape in logical order. Coalesced source and
// target modes are consumed from fastest to slowest, then the target modes are
// restored to their original order.
tcg::Layout composition(const tcg::Layout &layout, const tcg::Shape &target) {
  tcg::Layout coalesced = tcg::coalesce(layout);
  if (tcg::has_error(coalesced)) {
    return coalesced;
  }

  std::vector<tcg::Layout> result;
  std::vector<tcg::Layout> mode;
  auto appendMode = [&]() {
    if (mode.size() == 1) {
      result.push_back(mode[0]);
    } else if (mode.size() > 1) {
      std::reverse(mode.begin(), mode.end());
      result.push_back(tcg::make_layout(mode));
    }
  };

  size_t layoutRemaining = tcg::rank(coalesced);
  size_t targetRemaining = tcg::rank(target);
  std::optional<tcg::Layout> layoutPart;
  std::optional<tcg::Shape> targetPart;

  while ((layoutPart.has_value() || layoutRemaining > 0) &&
         (targetPart.has_value() || targetRemaining > 0)) {
    if (!layoutPart.has_value()) {
      layoutPart = tcg::get(coalesced, --layoutRemaining);
    }
    if (!targetPart.has_value()) {
      appendMode();
      mode.clear();
      targetPart = *std::next(target.begin(), --targetRemaining);
    }

    if (tcg::is_static(*targetPart)) {
      if (!tcg::is_static(layoutPart->shape())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Composition failed: current layout is dynamic\n");
        return tcg::Layout(tcg::CgErrorT{});
      }

      int64_t layoutSize = tcg::static_size(layoutPart->shape());
      int64_t targetSize = tcg::static_size(*targetPart);
      if (layoutSize <= targetSize) {
        if (targetSize % layoutSize != 0) {
          LLVM_DEBUG(llvm::dbgs() << "Composition failed: target size is not "
                                     "divisible by layout size\n");
          return tcg::Layout(tcg::CgErrorT{});
        }
        mode.push_back(*layoutPart);
        layoutPart = std::nullopt;
        targetPart =
            layoutSize != targetSize
                ? std::make_optional(tcg::Shape(targetSize / layoutSize))
                : std::nullopt;
      } else {
        if (layoutSize % targetSize != 0) {
          LLVM_DEBUG(llvm::dbgs() << "Composition failed: layout size is not "
                                     "divisible by target size\n");
          return tcg::Layout(tcg::CgErrorT{});
        }
        if (!tcg::is_static(layoutPart->stride())) {
          LLVM_DEBUG(llvm::dbgs() << "Composition failed: layout stride is "
                                     "dynamic\n");
          return tcg::Layout(tcg::CgErrorT{});
        }
        int64_t stride = tcg::static_size(layoutPart->stride());
        mode.emplace_back(tcg::Shape(targetSize),
                          tcg::Stride(targetSize > 1 ? stride : 0));
        layoutPart = tcg::Layout(tcg::Shape(layoutSize / targetSize),
                                 tcg::Stride(stride * targetSize));
        targetPart = std::nullopt;
      }
    } else {
      if (tcg::is_static(layoutPart->shape())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Composition failed: current layout is static\n");
        return tcg::Layout(tcg::CgErrorT{});
      }
      mode.push_back(*layoutPart);
      layoutPart = std::nullopt;
      targetPart = std::nullopt;
    }
  }

  bool layoutHasLeadingDimensions = false;
  while (layoutRemaining > 0) {
    const tcg::Layout part = tcg::get(coalesced, --layoutRemaining);
    bool hasUnitSize =
        tcg::is_static(part.shape()) && tcg::static_size(part.shape()) == 1;
    layoutHasLeadingDimensions |= !hasUnitSize;
  }

  bool targetHasLeadingDimensions = false;
  int unitDimensionCount = 0;
  while (targetRemaining > 0) {
    const tcg::Shape part = *std::next(target.begin(), --targetRemaining);
    bool hasUnitSize = tcg::is_static(part) && tcg::static_size(part) == 1;
    targetHasLeadingDimensions |= !hasUnitSize;
    unitDimensionCount += hasUnitSize;
  }

  if (layoutPart.has_value() || layoutHasLeadingDimensions ||
      targetPart.has_value() || targetHasLeadingDimensions) {
    LLVM_DEBUG(llvm::dbgs()
               << "Composition failed: mismatching layout and target sizes\n");
    return tcg::Layout(tcg::CgErrorT{});
  }

  appendMode();
  while (unitDimensionCount--) {
    result.emplace_back(tcg::Shape(1), tcg::Stride(0));
  }
  std::reverse(result.begin(), result.end());
  return tcg::make_layout(result);
}

} // namespace

bool isTensorType(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || !isValidTensorElementType(tensorType.getElementType())) {
    return false;
  }
  Attribute encoding = tensorType.getEncoding();
  if (!encoding) {
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TensorSourceAttr
//===----------------------------------------------------------------------===//

LogicalResult
TensorSourceAttr::verify(::llvm::function_ref<InFlightDiagnostic()> emitError,
                         int32_t tensorId, int64_t offset,
                         llvm::StringRef layout,
                         ArrayRef<int32_t> dynamicValueMapping) {
  auto cgLayout = tcg::from_string<tcg::Layout>(layout);
  if (!cgLayout.has_value()) {
    return emitError() << "Invalid layout: " << layout;
  }

  if (tcg::depth(*cgLayout) > 2) {
    return emitError() << "Unsupported layout: " << layout;
  }

  size_t dynamicSizes = countDynamicSizes(*cgLayout);
  if (dynamicSizes != dynamicValueMapping.size()) {
    return emitError() << "Layout has " << dynamicSizes
                       << " dynamic size(s) but " << dynamicValueMapping.size()
                       << " dynamic offset(s) provided";
  }

  if (dynamicSizes > 0) {
    SmallVector<bool> mark(dynamicSizes, false);
    for (int32_t offset : dynamicValueMapping) {
      if (size_t(offset) < dynamicSizes) {
        mark[offset] = true;
      }
    }
    bool isPermutation = llvm::all_of(mark, [](bool m) { return m; });
    if (!isPermutation) {
      return emitError() << "Dynamic offsets must be in the range [0.."
                         << dynamicSizes << ") and unique";
    }
  }

  return success();
}

tcg::Layout TensorSourceAttr::getCuteLayout() const {
  auto result = tcg::from_string<tcg::Layout>(getLayout());
  if (!result) {
    return tcg::Layout(tcg::CgErrorT{});
  }
  return *result;
}

SmallVector<int64_t> TensorSourceAttr::getShape() const {
  auto layout = getCuteLayout();
  const auto &shape = layout.shape();
  size_t rank = tcg::rank(shape);
  SmallVector<int64_t> result;
  result.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    auto dim = tcg::get(shape, i);
    result.push_back(tcg::is_static(dim) ? tcg::static_size(dim) : kDynamic);
  }
  return result;
}

LayoutSourceAttrInterface
TensorSourceAttr::reshape(ArrayRef<int64_t> newShape) const {
  tcg::Shape targetShape;
  for (int64_t dim : newShape) {
    if (dim > 0) {
      targetShape.append(dim);
    } else {
      targetShape.appendDynamic();
    }
  }

  tcg::Layout result = composition(getCuteLayout(), targetShape);
  if (tcg::has_error(result)) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot reshape: composition failed\n");
    return nullptr;
  }

  return TensorSourceAttr::get(getContext(), getTensorId(), getOffset(),
                               result.toString(), getDynamicValueMapping());
}

LayoutSourceAttrInterface
TensorSourceAttr::broadcast(ArrayRef<int64_t> newShape) const {
  // Check that new shape is valid.
  SmallVector<int64_t> currentShape = getShape();
  bool isValid =
      currentShape.size() == newShape.size() &&
      llvm::all_of(llvm::zip_equal(currentShape, newShape), [](auto pair) {
        auto [oldDim, newDim] = pair;
        return oldDim == newDim || (oldDim == 1 && newDim > 0);
      });
  if (!isValid) {
    return nullptr;
  }

  // Build resulting layout.
  auto layout = getCuteLayout();
  std::vector<tcg::Layout> parts;
  for (size_t i = 0; i < newShape.size(); ++i) {
    parts.push_back(currentShape[i] == 1 ? tcg::Layout(newShape[i], 0)
                                         : tcg::get(layout, i));
  }

  auto newLayout = tcg::make_layout(parts);
  return TensorSourceAttr::get(getContext(), getTensorId(), getOffset(),
                               newLayout.toString(), getDynamicValueMapping());
}

LayoutSourceAttrInterface
TensorSourceAttr::transpose(ArrayRef<int64_t> permutation) const {
  // Check that permutation is valid.
  auto layout = getCuteLayout();
  size_t rank = tcg::rank(layout);

  SmallVector<bool> mark(rank);
  for (int64_t dim : permutation) {
    if (static_cast<size_t>(dim) < rank) {
      mark[dim] = true;
    }
  }
  if (permutation.size() != rank ||
      llvm::any_of(mark, [](bool m) { return !m; })) {
    return nullptr;
  }

  // Transpose the dynamic offsets, if necessary.
  SmallVector<int32_t> dynamicValueMapping(getDynamicValueMapping().begin(),
                                           getDynamicValueMapping().end());
  if (!dynamicValueMapping.empty()) {
    SmallVector<int32_t> dynamicResult(rank * 2, -1);
    auto src = dynamicValueMapping.begin();
    auto dst = dynamicResult.begin();

    // Rearrange dynamic dimensions.
    SmallVector<int32_t> dynamicDims(rank, -1);
    for (size_t i = 0; i < rank; ++i) {
      if (!tcg::is_static(tcg::get(layout.shape(), i))) {
        assert(src != dynamicValueMapping.end() &&
               "Incorrect dynamic value mapping");
        dynamicDims[i] = *src++;
      }
    }
    for (size_t idx : permutation) {
      *dst++ = dynamicDims[idx];
    }

    // Rearrange dynamic strides.
    SmallVector<int32_t> dynamicStrides(rank, -1);
    for (size_t i = 0; i < rank; ++i) {
      if (!tcg::is_static(tcg::get(layout.stride(), i))) {
        assert(src != dynamicValueMapping.end() &&
               "Incorrect dynamic value mapping");
        dynamicStrides[i] = *src++;
      }
    }
    for (size_t idx : permutation) {
      *dst++ = dynamicStrides[idx];
    }

    // Build new dynamic offsets.
    dynamicValueMapping.clear();
    llvm::copy_if(dynamicResult, std::back_inserter(dynamicValueMapping),
                  [](int32_t offset) { return offset >= 0; });
  }

  // Build resulting layout.
  std::vector<tcg::Layout> parts;
  parts.reserve(rank);
  for (size_t idx : permutation) {
    parts.push_back(tcg::get(layout, idx));
  }

  auto newLayout = tcg::make_layout(parts);
  return TensorSourceAttr::get(getContext(), getTensorId(), getOffset(),
                               newLayout.toString(), dynamicValueMapping);
}

LayoutSourceAttrInterface
TensorSourceAttr::slice(ArrayRef<int64_t> starts, ArrayRef<int64_t> limits,
                        ArrayRef<int64_t> strides) const {
  // Check that arguments are valid.
  auto layout = getCuteLayout();
  size_t rank = tcg::rank(layout);

  if (starts.size() != rank || limits.size() != rank ||
      strides.size() != rank) {
    return nullptr;
  }
  for (size_t i = 0; i < rank; ++i) {
    auto part = tcg::get(layout, i);
    if (!tcg::is_static(part)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Cannot slice: dynamic dimension (" << i << ")\n");
      return nullptr;
    }
    if (starts[i] < 0 || limits[i] <= starts[i] || strides[i] <= 0 ||
        limits[i] > tcg::static_size(part.shape())) {
      return nullptr;
    }
  }

  // Build new layout with slice applied to each dimension.
  std::vector<tcg::Layout> parts;
  parts.reserve(rank);
  int64_t sliceOffset = 0;

  for (size_t i = 0; i < rank; ++i) {
    int64_t start = starts[i];
    int64_t end = limits[i];
    int64_t step = strides[i];
    int64_t count = (end - start + step - 1) / step;
    auto dimLayout = tcg::get(layout, i);

    // Simple case: single layout dimension.
    // Example: (16):(2) [4,8,2] -> (2):(4) offset 8
    if (tcg::depth(dimLayout) == 0) {
      int64_t newSize = count;
      int64_t newStride = step * dimLayout.stride().as_int();
      parts.emplace_back(newSize, newSize > 1 ? newStride : 0);
      sliceOffset += dimLayout(tcg::Coord(start)).as_int();
      continue;
    }

    // Complex case: multiple layout dimensions.
    // Decompose the slice start index into layout coordinates.
    auto coord = tcg::idx2crd(start, dimLayout.shape());
    sliceOffset += dimLayout(coord).as_int();

    // Iteratively descend into each layout dimension and build the resulting
    // layout components. The resulting rank could be smaller than the original.
    std::vector<tcg::Layout> subParts;
    for (size_t j = tcg::rank(coord); j-- > 0;) {
      // Extracted layout must be flat (parent layout depth is 2).
      auto subLayout = tcg::get(dimLayout, j);
      int64_t limit = subLayout.shape().as_int();
      int64_t stride = subLayout.stride().as_int();
      int64_t start = tcg::get(coord, j).as_int();

      // Case 1: all elements fit into the current layout dimension.
      // Example: ..x.x.x.    layout: (8,2):(1,8)
      //          ........    slice:  [2:7:2]
      if (start + (count - 1) * step < limit) {
        subParts.emplace_back(count, count > 1 ? step * stride : 0);
        break;
      }

      // Case 2: multiple elements fill the current layout dimension.
      // Example: .x.x.x.x    layout: (8,2):(1,8)
      //          .x.x.x.x    slice:  [1:16:2]
      int64_t fits = limit / step;
      if (limit % step == 0 && // next line has the same start position
          start < step &&      // no extra padding on the left side
          count % fits == 0 && // each line has the same number of elements
          fits > 1             // has multiple elements in the line
      ) {
        subParts.emplace_back(fits, step * stride);
        count /= fits;
        step = 1;
        continue;
      }

      // Case 3: one element per multiple rows.
      // Example: ....x...    layout: (8,3):(1,8)
      //          ........    slice:  [4:21:16]
      //          ....x...
      if (step % limit == 0) {
        step /= limit;
        continue;
      }

      // No other slicing patterns are supported.
      LLVM_DEBUG(llvm::dbgs() << "Cannot slice: unsupported pattern\n");
      return nullptr;
    }

    // Add the dimension layout to the resulting layout.
    std::reverse(subParts.begin(), subParts.end());
    parts.push_back(subParts.size() > 1 ? tcg::make_layout(subParts)
                                        : std::move(subParts[0]));
  }

  auto newLayout = tcg::make_layout(parts);
  return TensorSourceAttr::get(getContext(), getTensorId(),
                               getOffset() + sliceOffset, newLayout.toString(),
                               /*dynamicValueMapping=*/{});
}

LayoutSourceAttrInterface TensorSourceAttr::normalize() const {
  // Coalesce the CuTe layout.
  auto layout = getCuteLayout();
  auto flat = tcg::coalesce(layout);

  // Ensure we have a proper layout (not a scalar).
  if (tcg::depth(flat) == 0) {
    flat = tcg::make_layout(std::vector{flat});
  }

  LLVM_DEBUG(llvm::dbgs() << "Normalized " << getLayout() << " to "
                          << flat.toString() << "\n");

  // Verify that the number of dynamic sizes is the same after normalization.
  if (countDynamicSizes(flat) != getDynamicValueMapping().size()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Cannot normalize: number of dynamic sizes changed\n");
    return nullptr;
  }

  return TensorSourceAttr::get(getContext(), getTensorId(), getOffset(),
                               flat.toString(), getDynamicValueMapping());
}

//===----------------------------------------------------------------------===//
// CompositeSourceAttr
//===----------------------------------------------------------------------===//

LogicalResult CompositeSourceAttr::verify(
    ::llvm::function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<LayoutSourceAttrInterface> sources) {
  if (sources.empty()) {
    return emitError() << "Composite source has no children";
  }

  auto firstSource = sources[0];
  SmallVector<int64_t> firstShape = firstSource.getShape();
  for (size_t i = 1; i < sources.size(); ++i) {
    if (sources[i].getShape() != firstShape) {
      return emitError() << "Underlying sources must have the same shape";
    }
  }

  return success();
}

SmallVector<int64_t> CompositeSourceAttr::getShape() const {
  return getSource(0).getShape();
}

LayoutSourceAttrInterface
CompositeSourceAttr::reshape(ArrayRef<int64_t> newShape) const {
  auto newSources =
      mapSources(getSources(), [&](LayoutSourceAttrInterface child) {
        return child.reshape(newShape);
      });
  return succeeded(newSources)
             ? CompositeSourceAttr::get(getContext(), *newSources)
             : nullptr;
}

LayoutSourceAttrInterface
CompositeSourceAttr::broadcast(ArrayRef<int64_t> newShape) const {
  auto newSources =
      mapSources(getSources(), [&](const LayoutSourceAttrInterface &src) {
        return src.broadcast(newShape);
      });
  return succeeded(newSources)
             ? CompositeSourceAttr::get(getContext(), *newSources)
             : nullptr;
}

LayoutSourceAttrInterface
CompositeSourceAttr::transpose(ArrayRef<int64_t> permutation) const {
  auto newSources =
      mapSources(getSources(), [&](const LayoutSourceAttrInterface &src) {
        return src.transpose(permutation);
      });
  return succeeded(newSources)
             ? CompositeSourceAttr::get(getContext(), *newSources)
             : nullptr;
}

LayoutSourceAttrInterface
CompositeSourceAttr::slice(ArrayRef<int64_t> starts, ArrayRef<int64_t> limits,
                           ArrayRef<int64_t> strides) const {
  auto newSources =
      mapSources(getSources(), [&](const LayoutSourceAttrInterface &src) {
        return src.slice(starts, limits, strides);
      });
  return succeeded(newSources)
             ? CompositeSourceAttr::get(getContext(), *newSources)
             : nullptr;
}

LayoutSourceAttrInterface CompositeSourceAttr::normalize() const {
  // Normalize underlying sources.
  auto normalizedSources =
      mapSources(getSources(), [](const LayoutSourceAttrInterface &src) {
        return src.normalize();
      });
  if (!succeeded(normalizedSources)) {
    return nullptr;
  }

  // Collect the normalized shapes.
  SmallVector<SmallVector<int64_t>> shapes;
  shapes.reserve(normalizedSources->size());
  for (LayoutSourceAttrInterface src : *normalizedSources) {
    shapes.push_back(src.getShape());
  }

  // Infer a common iteration space for all inputs.
  auto iterSpace = getCommonIterationSpace(std::move(shapes));
  if (llvm::failed(iterSpace)) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot normalize composite source\n");
    return nullptr;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Common iteration space: (";
    llvm::interleaveComma(*iterSpace, llvm::dbgs());
    llvm::dbgs() << ")\n";
  });

  // Reshape the sources to the common shape.
  for (LayoutSourceAttrInterface &src : *normalizedSources) {
    if (src.getShape() != *iterSpace) {
      src = src.reshape(*iterSpace);
      if (!src) {
        return nullptr;
      }
    }
  }

  return CompositeSourceAttr::get(getContext(), *normalizedSources);
}

//===----------------------------------------------------------------------===//
// ConcatSourceAttr
//===----------------------------------------------------------------------===//

LogicalResult
ConcatSourceAttr::verify(::llvm::function_ref<InFlightDiagnostic()> emitError,
                         int64_t dimension,
                         ArrayRef<LayoutSourceAttrInterface> sources,
                         ArrayRef<int32_t> argumentIndex) {
  if (sources.empty()) {
    return emitError() << "Concat source has no children";
  }

  if (llvm::any_of(sources, [](const LayoutSourceAttrInterface &src) {
        return product(src.getShape()) == kDynamic;
      })) {
    return emitError() << "Dynamic shapes for concatenation are not supported";
  }

  auto firstSource = sources[0];
  SmallVector<int64_t> testShape = firstSource.getShape();
  if (static_cast<size_t>(dimension) >= testShape.size()) {
    return emitError() << "Concatenation dimension is invalid";
  }

  for (size_t i = 1; i < sources.size(); ++i) {
    SmallVector<int64_t> shape = sources[i].getShape();
    if (static_cast<size_t>(dimension) < shape.size()) {
      testShape[dimension] = shape[dimension];
    }
    if (shape != testShape) {
      return emitError() << "Concatenated sources must have the same shape, "
                            "except for the concatenation dimension";
    }
  }

  if (!argumentIndex.empty() && argumentIndex.size() != sources.size()) {
    return emitError() << "Argument index must have the same size as sources";
  }

  return success();
}

SmallVector<int64_t> ConcatSourceAttr::getShape() const {
  SmallVector<int64_t> result = getSource(0).getShape();
  for (size_t i = 1; i < size(); ++i) {
    SmallVector<int64_t> shape = getSource(i).getShape();
    result[getDimension()] += shape[getDimension()];
  }
  return result;
}

LayoutSourceAttrInterface
ConcatSourceAttr::reshape(ArrayRef<int64_t> newShape) const {
  // Concat dimension cannot be joined on the right.
  // The split point in the new shape must separate dimensions.
  SmallVector<int64_t> shape = getShape();
  const int64_t *concatDim = shape.begin() + getDimension();
  int64_t sizeLeft = product({shape.begin(), concatDim});
  int64_t sizeConcat = *concatDim;
  int64_t sizeRight = product({concatDim + 1, shape.end()});
  auto newSplit =
      findDimensionWithSuffixProduct(newShape, sizeConcat * sizeRight);

  if (llvm::failed(newSplit) ||
      product(newShape) != sizeLeft * sizeConcat * sizeRight) {
    LLVM_DEBUG(llvm::dbgs()
               << "Cannot reshape: join of concatenated dimension\n");
    return nullptr;
  }

  assert(sizeConcat > 1 && "Concat dimension cannot have unit size");
  size_t newDimension = *newSplit;

  SmallVector<int64_t> partShape(newShape.begin(), newShape.end());
  auto newSources =
      mapSources(getSources(), [&](LayoutSourceAttrInterface child) {
        // An uneven concat split makes the underlying reshape invalid.
        SmallVector<int64_t> childShape = child.getShape();
        partShape[newDimension] = newShape[newDimension] *
                                  childShape[getDimension()] /
                                  shape[getDimension()];
        return child.reshape(partShape);
      });

  return succeeded(newSources)
             ? ConcatSourceAttr::get(getContext(), newDimension, *newSources,
                                     getArgumentIndex())
             : nullptr;
}

LayoutSourceAttrInterface
ConcatSourceAttr::broadcast(ArrayRef<int64_t> newShape) const {
  // Concat dimension cannot be broadcasted.
  SmallVector<int64_t> shape = getShape();
  if (newShape.size() != shape.size() ||
      shape[getDimension()] != newShape[getDimension()]) {
    return nullptr;
  }

  // Apply broadcast operation to each underlying source.
  SmallVector<int64_t> partShape(newShape.begin(), newShape.end());
  auto newSources =
      mapSources(getSources(), [&](const LayoutSourceAttrInterface &src) {
        SmallVector<int64_t> srcShape = src.getShape();
        partShape[getDimension()] = srcShape[getDimension()];
        return src.broadcast(partShape);
      });
  return succeeded(newSources)
             ? ConcatSourceAttr::get(getContext(), getDimension(), *newSources,
                                     getArgumentIndex())
             : nullptr;
}

LayoutSourceAttrInterface
ConcatSourceAttr::transpose(ArrayRef<int64_t> permutation) const {
  // Find new position of the concat dimension.
  int64_t newDimension =
      llvm::find(permutation, getDimension()) - permutation.begin();

  // Apply transpose operation to each underlying source.
  auto newSources =
      mapSources(getSources(), [&](const LayoutSourceAttrInterface &src) {
        return src.transpose(permutation);
      });
  return succeeded(newSources)
             ? ConcatSourceAttr::get(getContext(), newDimension, *newSources,
                                     getArgumentIndex())
             : nullptr;
}

LayoutSourceAttrInterface
ConcatSourceAttr::slice(ArrayRef<int64_t> starts, ArrayRef<int64_t> limits,
                        ArrayRef<int64_t> strides) const {
  // Verify slice of the concat dimension.
  size_t index = static_cast<size_t>(getDimension());
  int64_t start = index < starts.size() ? starts[index] : -1;
  int64_t end = index < limits.size() ? limits[index] : -1;
  int64_t step = index < strides.size() ? strides[index] : -1;

  SmallVector<int64_t> shape = getShape();
  if (start < 0 || end <= start || step <= 0 || end > shape[index]) {
    return nullptr;
  }

  // Build argument index (if implicit).
  SmallVector<int32_t> argIndex(getArgumentIndex());
  if (argIndex.empty()) {
    argIndex.resize(getSources().size());
    std::iota(argIndex.begin(), argIndex.end(), 0);
  }
  SmallVector<int32_t> newIndex;

  // Iterate over the concat dimension and build new sources.
  SmallVector<LayoutSourceAttrInterface> newSources;
  SmallVector<int64_t> partStarts(starts);
  SmallVector<int64_t> partLimits(limits);
  SmallVector<int64_t> partStrides(strides);

  for (const auto &[part, argNo] : llvm::zip_equal(getSources(), argIndex)) {
    SmallVector<int64_t> partShape = part.getShape();

    // Skip sources to the left of the range.
    int64_t size = partShape[index];
    if (start >= size) {
      start -= size;
      end -= size;
      continue;
    }

    // Apply slice operation to the current source.
    partStarts[index] = start;
    partLimits[index] = std::min(end, size);
    partStrides[index] = step;
    auto result = part.slice(partStarts, partLimits, partStrides);
    if (!result) {
      return nullptr;
    }
    newSources.push_back(result);
    newIndex.push_back(argNo);

    // Adjust start and end slice positions.
    // Early stop if the next start position is outside the range.
    int64_t tail = (size - start) % step;
    start = tail ? step - tail : 0;
    if (size + start >= end) {
      break;
    }
    end -= size;
  }

  // Do not set argument index if no sources were removed.
  if (getArgumentIndex().empty() && newIndex.size() == argIndex.size()) {
    newIndex.clear();
  }
  return ConcatSourceAttr::get(getContext(), getDimension(), newSources,
                               newIndex);
}

LayoutSourceAttrInterface ConcatSourceAttr::normalize() const {
  // Normalize underlying sources.
  auto normalizedSources =
      mapSources(getSources(), [&](const LayoutSourceAttrInterface &src) {
        return src.normalize();
      });
  if (!succeeded(normalizedSources)) {
    return nullptr;
  }

  // Collect the source and normalized shapes.
  SmallVector<int64_t> concatShape = getShape();
  SmallVector<SmallVector<int64_t>> sourceShapes;
  SmallVector<SmallVector<int64_t>> normalizedShapes;
  for (size_t i = 0; i < size(); ++i) {
    sourceShapes.push_back(getSource(i).getShape());
    normalizedShapes.push_back((*normalizedSources)[i].getShape());
  }

  // Calculate GCD of the concat dimension sizes.
  // Each concat dimension may be reduced by this factor.
  int64_t gcd = sourceShapes[0][getDimension()];
  for (size_t i = 1; i < size(); ++i) {
    gcd = std::gcd(gcd, sourceShapes[i][getDimension()]);
  }

  // Every shape can be decomposed into (left, size / gcd, gcd * right).
  const int64_t *split = concatShape.begin() + getDimension();
  int64_t sizeLeft = product({concatShape.begin(), split});
  int64_t sizeConcat = *split;
  int64_t sizeRight = product({split + 1, concatShape.end()});
  LLVM_DEBUG(llvm::dbgs() << "Concat profile: [" << sizeLeft << ", "
                          << sizeConcat << "/" << gcd << ", " << gcd << "*"
                          << sizeRight << "]\n");

  // Make sure the concat dimension is separate in each shape.
  for (auto [i, partNormalizedShape] : llvm::enumerate(normalizedShapes)) {

    // Let `s : ℕ := prod(part_shape);`. Facts:
    // - `sizePart | s`
    // - `gcd | sizePart`
    // - `sizeRight | s`
    // - `sizeLeft | s`
    // The following are factorizations of `s`:
    // - `normShape`,
    // - `profile := (sizeLeft, sizePart / gcd, gcd * sizeRight)`
    //
    // We compute the coursest common refinement(`CCR(normShape, profile)`,
    // which is the maximal element in the poset of ordered
    // factorizations of `s` this is a refinement of `normShape` and `profile`.
    // This refinement may not exist (e.g. `normShape`
    // and `profile` may have misaligned cuts), but it is unique when it does
    // exist.
    //
    // Why compute the CCR? Note we can interpret "coarsest" to mean "maximally
    // coalesced". We chose "profile" ensure that the resulting factorization,
    // when it exists, is guaranteed to separate the left/right parts.
    //
    // Note that 'sizePart * sizeRight` is in the suffix product of `profile`
    // and therefore it is in the suffix product of `CCR(normShape, profile)`.
    // This gives us the index of the dimension which divides left/right parts
    // in the CCR.
    const int64_t sizePart = sourceShapes[i][getDimension()];
    SmallVector<int64_t> profile{sizePart / gcd};
    if (sizeLeft > 1) {
      profile.insert(profile.begin(), sizeLeft);
    }
    if (gcd * sizeRight > 1) {
      profile.push_back(gcd * sizeRight);
    }

    // Compute the `CCR(normShape, profile)`.
    FailureOr<SmallVector<int64_t>> result =
        getCommonIterationSpace({partNormalizedShape, profile});
    if (llvm::failed(result)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Cannot normalize concat source [" << i << "]\n");
      return nullptr;
    }

    // Find the concat dimension index. This is guaranteed to succeed -- *result
    // is a refinement of partNormalizedShape.
    FailureOr<unsigned> it =
        findDimensionWithSuffixProduct(*result, sizePart * sizeRight);
    if (failed(it)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "CCR is not a refinement of the part's normalized shape\n");
      return nullptr;
    }
    if (sizePart / gcd == 1) {
      (*it)--;
    }

    // Update the target shape.
    (*result)[*it] = sizeConcat / gcd;
    partNormalizedShape = std::move(*result);
  }

  // Infer a common iteration space for all inputs.
  auto iterSpace = getCommonIterationSpace(std::move(normalizedShapes));
  if (!llvm::succeeded(iterSpace)) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot normalize concat source\n");
    return nullptr;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Common iteration space [concat]: (";
    llvm::interleaveComma(*iterSpace, llvm::dbgs());
    llvm::dbgs() << ")\n";
  });

  // Find the new concat dimension.
  // When only a single source is present, do not move the split point.
  auto newSplit =
      findDimensionWithSuffixProduct(*iterSpace, sizeConcat * sizeRight);
  assert(llvm::succeeded(newSplit));
  int64_t newDimension = *newSplit - (sizeConcat == gcd);

  // Reshape the underlying sources, replacing the concat dimension size.
  SmallVector<int64_t> partShape(*iterSpace);
  int idx = 0;
  for (LayoutSourceAttrInterface &src : *normalizedSources) {
    int64_t partSize = sourceShapes[idx++][getDimension()];
    partShape[newDimension] = partSize / gcd;
    if (src.getShape() != partShape) {
      src = src.reshape(partShape);
      if (!src) {
        return nullptr;
      }
    }
  }

  return ConcatSourceAttr::get(getContext(), newDimension, *normalizedSources,
                               getArgumentIndex());
}

//===----------------------------------------------------------------------===//
// ReductionSourceAttr
//===----------------------------------------------------------------------===//

LogicalResult ReductionSourceAttr::verify(
    ::llvm::function_ref<InFlightDiagnostic()> emitError, llvm::StringRef view,
    LayoutSourceAttrInterface source) {
  auto cgView = tcg::from_string<tcg::Layout>(view);
  if (!cgView.has_value()) {
    return emitError() << "Invalid view: " << view;
  }
  if (!tcg::is_static(*cgView)) {
    return emitError() << "Reduction view must have static shape and stride: "
                       << view;
  }

  if (product(source.getShape()) == kDynamic) {
    return emitError() << "Dynamic shapes for reduction are not supported";
  }

  // Verify that the view is valid for the provided source.
  // It must have the same total size and the strides must be within the bounds.
  auto flat = tcg::flatten(*cgView);
  std::vector<tcg::Layout> parts;
  for (size_t i = 0; i < tcg::rank(flat); ++i) {
    auto part = tcg::get(flat, i);
    if (part.stride().as_int() != 0 && part.shape().as_int() != 1) {
      parts.push_back(part);
    }
  }
  llvm::sort(parts, [](const tcg::Layout &a, const tcg::Layout &b) {
    return a.stride().as_int() < b.stride().as_int();
  });
  SmallVector<int64_t> sourceShape = source.getShape();
  int64_t physicalSize = 1;
  int64_t expectedStride = parts.empty() ? 0 : parts.front().stride().as_int();
  bool isContiguous = !parts.empty() || product(sourceShape) == 1;
  for (const tcg::Layout &part : parts) {
    int64_t partSize = part.shape().as_int();
    isContiguous &= part.stride().as_int() == expectedStride;
    isContiguous &=
        !llvm::MulOverflow(expectedStride, partSize, expectedStride);
    isContiguous &= !llvm::MulOverflow(physicalSize, partSize, physicalSize);
  }
  if (!isContiguous || physicalSize != product(sourceShape)) {
    return emitError() << "Invalid view for the source: " << view;
  }

  return success();
}

tcg::Layout ReductionSourceAttr::getCuteLayout() const {
  auto result = tcg::from_string<tcg::Layout>(getView());
  if (!result) {
    return tcg::Layout(tcg::CgErrorT{});
  }
  return *result;
}

int64_t ReductionSourceAttr::getReductionSize() const {
  return product(getReductionShape());
}

SmallVector<int64_t> ReductionSourceAttr::getReductionShape() const {
  auto layout = getCuteLayout();
  auto last = tcg::get(layout, tcg::rank(layout) - 1);
  SmallVector<int64_t> result;
  for (size_t i = 0, n = tcg::rank(last); i < n; ++i) {
    result.push_back(tcg::static_size(last, i));
  }
  return result;
}

SmallVector<int64_t> ReductionSourceAttr::getShape() const {
  auto layout = getCuteLayout();
  SmallVector<int64_t> result;
  for (size_t i = 0, n = tcg::rank(layout) - 1; i < n; ++i) {
    result.push_back(tcg::static_size(layout, i));
  }
  return result;
}

LayoutSourceAttrInterface
ReductionSourceAttr::reshape(ArrayRef<int64_t> newShape) const {
  auto viewAttr = TensorSourceAttr::get(getContext(), 0, 0, getView(), {});
  SmallVector<int64_t> shape(newShape.begin(), newShape.end());
  shape.push_back(getReductionSize());
  auto newAttr = dyn_cast_or_null<TensorSourceAttr>(viewAttr.reshape(shape));
  return newAttr ? ReductionSourceAttr::get(getContext(),
                                            newAttr.getCuteLayout().toString(),
                                            getSource())
                 : nullptr;
}

LayoutSourceAttrInterface
ReductionSourceAttr::broadcast(ArrayRef<int64_t> newShape) const {
  auto viewAttr = TensorSourceAttr::get(getContext(), 0, 0, getView(), {});
  SmallVector<int64_t> shape(newShape.begin(), newShape.end());
  shape.push_back(getReductionSize());
  auto newAttr = dyn_cast_or_null<TensorSourceAttr>(viewAttr.broadcast(shape));
  return newAttr ? ReductionSourceAttr::get(getContext(),
                                            newAttr.getCuteLayout().toString(),
                                            getSource())
                 : nullptr;
}

LayoutSourceAttrInterface
ReductionSourceAttr::transpose(ArrayRef<int64_t> permutation) const {
  auto viewAttr = TensorSourceAttr::get(getContext(), 0, 0, getView(), {});
  SmallVector<int64_t> temp(permutation.begin(), permutation.end());
  temp.push_back(permutation.size());
  auto newAttr = dyn_cast_or_null<TensorSourceAttr>(viewAttr.transpose(temp));
  return newAttr ? ReductionSourceAttr::get(getContext(),
                                            newAttr.getCuteLayout().toString(),
                                            getSource())
                 : nullptr;
}

LayoutSourceAttrInterface
ReductionSourceAttr::slice(ArrayRef<int64_t> starts, ArrayRef<int64_t> limits,
                           ArrayRef<int64_t> strides) const {
  LLVM_DEBUG(llvm::dbgs() << "Reduction slice is not supported\n");
  return nullptr;
}

LayoutSourceAttrInterface ReductionSourceAttr::normalize() const {
  LLVM_DEBUG(llvm::dbgs() << "Computing iteration space for reduction\n");
  [[maybe_unused]] llvm::StringRef kDebugIndent = "    ";
  LayoutSourceAttrInterface underlying = getSource();
  int64_t reductionSize = getReductionSize();

  // [1] Separate broadcasted and non-broadcasted layout components.
  tcg::Layout flat = tcg::coalesce(getCuteLayout());
  SmallVector<tcg::Layout> broadcasted;
  SmallVector<std::pair<size_t, tcg::Layout>> nonBroadcasted;

  int64_t accum = 1;
  for (size_t i = tcg::rank(flat); i-- > 0;) {
    auto part = tcg::get(flat, i);
    int64_t size = part.shape().as_int();
    if (part.stride().as_int() != 0 || size == 1) {
      accum *= size;
      nonBroadcasted.push_back({0, std::move(part)});
    } else {
      broadcasted.push_back(tcg::Layout(size, accum));
    }
  }
  std::reverse(nonBroadcasted.begin(), nonBroadcasted.end());
  for (size_t i = 0; i < nonBroadcasted.size(); ++i) {
    nonBroadcasted[i].first = i;
  }

  // [2] Reshape the underlying source, if necessary.
  SmallVector<int64_t> newShape;
  llvm::sort(nonBroadcasted, [](auto lhs, auto rhs) {
    return lhs.second.stride().as_int() > rhs.second.stride().as_int();
  });
  for (const auto &part : nonBroadcasted) {
    newShape.push_back(part.second.shape().as_int());
  }

  if (underlying.getShape() != newShape) {
    LLVM_DEBUG({
      llvm::dbgs() << kDebugIndent << "Reshaping underlying source to (";
      llvm::interleaveComma(newShape, llvm::dbgs());
      llvm::dbgs() << ")\n";
    });
    underlying = underlying.reshape(newShape);
    if (!underlying) {
      LLVM_DEBUG(llvm::dbgs() << "Reshape of the underlying source failed\n");
      return nullptr;
    }
  }

  // [3] Transpose the underlying source, if necessary.
  SmallVector<int64_t> permutation(newShape.size(), -1);
  bool needsTranspose = false;
  for (size_t i = 0; i < permutation.size(); ++i) {
    size_t j = nonBroadcasted[i].first;
    permutation[j] = i;
    needsTranspose |= j != i;
  }

  if (needsTranspose) {
    LLVM_DEBUG({
      llvm::dbgs() << kDebugIndent
                   << "Transposing underlying dimensions with {";
      llvm::interleaveComma(permutation, llvm::dbgs());
      llvm::dbgs() << "}\n";
    });
    underlying = underlying.transpose(permutation);
    if (!underlying) {
      LLVM_DEBUG(llvm::dbgs() << "Transpose of the underlying source failed\n");
      return nullptr;
    }
  }

  // [4] Normalize the underlying source.
  underlying = underlying.normalize();
  if (!underlying) {
    LLVM_DEBUG(llvm::dbgs()
               << "Normalization of the underlying source failed\n");
    return nullptr;
  }

  LLVM_DEBUG({
    llvm::dbgs() << kDebugIndent << "Normalized underlying: (";
    llvm::interleaveComma(underlying.getShape(), llvm::dbgs());
    llvm::dbgs() << ")\n";
  });

  // [5] Calculate the target shape for the output.
  SmallVector<int64_t> target;
  SmallVector<int64_t> trailingShape;
  int64_t prev = reductionSize;
  for (const auto &part : broadcasted) {
    int64_t size = part.stride().as_int() / prev;
    if (size > 1) {
      trailingShape.push_back(size);
    }
    prev *= size;
  }
  int64_t rest = accum / prev;
  if (rest > 1 || broadcasted.empty()) {
    target.push_back(rest);
  }
  target.append(trailingShape.rbegin(), trailingShape.rend());
  target.push_back(reductionSize);

  LLVM_DEBUG({
    llvm::dbgs() << kDebugIndent << "Target shape: (";
    llvm::interleaveComma(target, llvm::dbgs());
    llvm::dbgs() << ")\n";
  });

  // [6] Calculate the iteration space and reshape.
  auto iterSpace = getCommonIterationSpace({underlying.getShape(), target});
  if (llvm::failed(iterSpace)) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot infer shape for iteration space\n");
    return nullptr;
  }

  LLVM_DEBUG({
    llvm::dbgs() << kDebugIndent << "Resulting shape: (";
    llvm::interleaveComma(*iterSpace, llvm::dbgs());
    llvm::dbgs() << ")\n";
  });

  if (underlying.getShape() != *iterSpace) {
    underlying = underlying.reshape(*iterSpace);
    if (!underlying) {
      return nullptr;
    }
  }

  // [7] Build the resulting view layout components.
  std::vector<tcg::Layout> parts;
  accum = 1;
  for (int64_t size : llvm::reverse(*iterSpace)) {
    parts.insert(parts.begin(), tcg::Layout(size, accum));
    accum *= size;
  }
  for (const tcg::Layout &part : broadcasted) {
    auto pos =
        findDimensionWithSuffixProduct(*iterSpace, part.stride().as_int());
    assert(llvm::succeeded(pos));
    parts.insert(parts.begin() + *pos, tcg::Layout(part.shape().as_int(), 0));
  }

  // Separate the reduction dimensions.
  std::vector<tcg::Layout> last;
  accum = 1;
  do {
    auto &part = parts.back();
    last.insert(last.begin(), part);
    accum *= part.shape().as_int();
    parts.pop_back();
  } while (accum < reductionSize);
  parts.push_back(tcg::make_layout(last));

  // Print the resulting iteration space.
  LLVM_DEBUG({
    SmallVector<int64_t> tempShape;
    for (size_t i = 0, n = parts.size() - 1; i < n; ++i) {
      tempShape.push_back(parts[i].shape().as_int());
    }
    llvm::dbgs() << kDebugIndent << "Iteration space: (";
    llvm::interleaveComma(tempShape, llvm::dbgs());
    llvm::dbgs() << ")\n";
  });

  // Build the resulting reduction source.
  auto viewLayout = tcg::make_layout(parts);
  return ReductionSourceAttr::get(getContext(), viewLayout.toString(),
                                  underlying);
}

//===----------------------------------------------------------------------===//
// MatmulSourceAttr
//===----------------------------------------------------------------------===//

namespace {

// Return coalesced layout where each sub-layout belongs to a single group
// in the given profile.
tcg::Layout normalizeLayout(const tcg::Layout &layout,
                            ArrayRef<int64_t> profile) {
  auto flat = tcg::coalesce(layout);
  std::vector<tcg::Layout> parts;
  for (size_t i = 0, n = tcg::rank(flat); i < n; ++i) {
    // Extract shape/stride from the layout.
    auto part = tcg::get(flat, i);
    int64_t shape = part.shape().as_int();
    int64_t stride = part.stride().as_int();

    // Keep the broadcasted dimensions.
    if (stride == 0) {
      parts.push_back(part);
      continue;
    }

    // Split the layout into logical groups (defined by the profile). Physical
    // strides still determine which groups a mode spans, while suffix products
    // identify the last-dimension-fastest group boundaries.
    int64_t accum = 1;
    std::vector<tcg::Layout> splitParts;
    for (int64_t size : llvm::reverse(profile)) {
      // Skip groups not matching the layout.
      accum *= size;
      if (stride >= accum) {
        continue;
      }
      if (shape * stride <= accum) {
        splitParts.emplace_back(shape, stride);
        break;
      }
      // More groups are matching the layout.
      assert(accum % stride == 0);
      int64_t n = accum / stride;
      splitParts.emplace_back(n, stride);
      shape /= n;
      stride *= n;
    }
    parts.insert(parts.end(), splitParts.rbegin(), splitParts.rend());
  }
  return tcg::make_layout(parts);
}

// Extract and sort shape sizes for the given stride range.
SmallVector<int64_t> extractSizes(const tcg::Layout &layout, int64_t start,
                                  int64_t size) {
  // Filter layout components matching the range.
  std::vector<tcg::Layout> parts;
  for (size_t i = 0, n = tcg::rank(layout); i < n; ++i) {
    auto part = tcg::get(layout, i);
    int64_t stride = part.stride().as_int();
    if (stride >= start && stride < start * size) {
      parts.push_back(part);
    }
  }
  // Build the shape from the components sorted by stride.
  llvm::sort(parts, [](auto lhs, auto rhs) {
    return lhs.stride().as_int() < rhs.stride().as_int();
  });

  SmallVector<int64_t> result;
  for (size_t i = 0; i < parts.size(); ++i) {
    result.push_back(parts[i].shape().as_int());
  }
  return result;
}

// Normalize underlying source and split into parts.
llvm::FailureOr<SmallVector<int64_t>>
normalizeAndSplit(const LayoutSourceAttrInterface &source,
                  ArrayRef<int64_t> profile) {
  // Remove unit dimensions (helper).
  auto removeUnitDimensions = [](ArrayRef<int64_t> shape) {
    return SmallVector<int64_t>(
        llvm::make_filter_range(shape, [](int64_t n) { return n != 1; }));
  };
  // Normalize and apply profile, ignoring unit dimensions.
  LayoutSourceAttrInterface norm = source.normalize();
  if (!norm) {
    return llvm::failure();
  }
  return getCommonIterationSpace(
      {removeUnitDimensions(norm.getShape()), removeUnitDimensions(profile)});
};

// Build map of strides to dimension index for the output view layout.
std::unordered_map<int64_t, size_t>
buildDimensionMap(const tcg::Layout &layout) {
  std::unordered_map<int64_t, size_t> result;
  auto flat = tcg::flatten(layout);
  for (size_t i = 0, n = tcg::rank(flat); i < n; ++i) {
    int64_t stride = tcg::get(flat, i).stride().as_int();
    result.insert({stride, i});
  }
  return result;
}

} // namespace

LogicalResult
MatmulSourceAttr::verify(::llvm::function_ref<InFlightDiagnostic()> emitError,
                         StringRef view, int64_t b, int64_t m, int64_t n,
                         int64_t k, LayoutSourceAttrInterface lhs,
                         LayoutSourceAttrInterface rhs) {
  auto cgView = tcg::from_string<tcg::Layout>(view);
  if (!cgView.has_value()) {
    return emitError() << "Invalid view: " << view;
  }
  if (!tcg::is_static(*cgView)) {
    return emitError() << "Matmul view must have static shape and stride: "
                       << view;
  }

  SmallVector<int64_t> lhsShape = lhs.getShape();
  if (product(lhsShape) == kDynamic) {
    return emitError() << "Dynamic shapes for matmul are not supported";
  }
  if (product(lhsShape) != b * m * k) {
    return emitError() << "Incorrect underlying LHS layout";
  }

  SmallVector<int64_t> rhsShape = rhs.getShape();
  if (product(rhsShape) == kDynamic) {
    return emitError() << "Dynamic shapes for matmul are not supported";
  }
  if (product(rhsShape) != b * n * k) {
    return emitError() << "Incorrect underlying RHS layout";
  }

  int64_t viewSize = 1;
  auto flat = tcg::flatten(*cgView);
  for (size_t i = 0, n = tcg::rank(flat); i < n; ++i) {
    auto part = tcg::get(flat, i);
    if (part.stride().as_int() != 0) {
      viewSize *= part.shape().as_int();
    }
  }
  if (viewSize != b * m * n * k) {
    return emitError() << "Incorrect matmul view size";
  }

  return success();
}

tcg::Layout MatmulSourceAttr::getCuteLayout() const {
  auto result = tcg::from_string<tcg::Layout>(getView());
  if (!result) {
    return tcg::Layout(tcg::CgErrorT{});
  }
  return *result;
}

SmallVector<int64_t> MatmulSourceAttr::getContractingShape() const {
  auto layout = getCuteLayout();
  auto last = tcg::get(layout, tcg::rank(layout) - 1);
  SmallVector<int64_t> result;
  for (size_t i = 0, n = tcg::rank(last); i < n; ++i) {
    result.push_back(tcg::static_size(last, i));
  }
  return result;
}

llvm::FailureOr<SmallVector<size_t>>
MatmulSourceAttr::getLhsDimensionMap() const {
  // Collect expected strides of LHS dimensions in the output view.
  SmallVector<int64_t> ident;
  SmallVector<int64_t> lhsShape = getLhs().getShape();
  auto lhsM = findDimensionWithSuffixProduct(lhsShape, getM() * getK());
  auto lhsK = findDimensionWithSuffixProduct(lhsShape, getK());
  if (failed(lhsM) || failed(lhsK)) {
    return llvm::failure();
  }
  auto appendStrides = [&](ArrayRef<int64_t> shape, int64_t baseStride) {
    SmallVector<int64_t> strides;
    int64_t stride = baseStride;
    for (int64_t size : llvm::reverse(shape)) {
      strides.push_back(stride);
      stride *= size;
    }
    ident.append(strides.rbegin(), strides.rend());
  };
  appendStrides(ArrayRef(lhsShape).take_front(*lhsM), getM() * getN() * getK());
  appendStrides(ArrayRef(lhsShape).slice(*lhsM, *lhsK - *lhsM),
                getN() * getK());
  appendStrides(ArrayRef(lhsShape).drop_front(*lhsK), 1);

  // Map strides to output dimensions.
  SmallVector<size_t> result;
  auto dimensionMap = buildDimensionMap(getCuteLayout());
  for (int64_t stride : ident) {
    auto it = dimensionMap.find(stride);
    if (it == dimensionMap.end()) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot find output dimension for stride "
                              << stride << " (LHS)\n");
      return llvm::failure();
    }
    result.push_back(it->second);
  }
  return result;
}

llvm::FailureOr<SmallVector<size_t>>
MatmulSourceAttr::getRhsDimensionMap() const {
  // Collect expected strides of RHS dimensions in the output view.
  SmallVector<int64_t> ident;
  SmallVector<int64_t> rhsShape = getRhs().getShape();
  auto rhsK = findDimensionWithSuffixProduct(rhsShape, getK() * getN());
  auto rhsN = findDimensionWithSuffixProduct(rhsShape, getN());
  if (failed(rhsK) || failed(rhsN)) {
    return llvm::failure();
  }
  auto appendStrides = [&](ArrayRef<int64_t> shape, int64_t baseStride) {
    SmallVector<int64_t> strides;
    int64_t stride = baseStride;
    for (int64_t size : llvm::reverse(shape)) {
      strides.push_back(stride);
      stride *= size;
    }
    ident.append(strides.rbegin(), strides.rend());
  };
  appendStrides(ArrayRef(rhsShape).take_front(*rhsK), getM() * getN() * getK());
  appendStrides(ArrayRef(rhsShape).slice(*rhsK, *rhsN - *rhsK), 1);
  appendStrides(ArrayRef(rhsShape).drop_front(*rhsN), getK());

  // Map strides to output dimensions.
  SmallVector<size_t> result;
  auto dimensionMap = buildDimensionMap(getCuteLayout());
  for (int64_t stride : ident) {
    auto it = dimensionMap.find(stride);
    if (it == dimensionMap.end()) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot find output dimension for stride "
                              << stride << " (RHS)\n");
      return llvm::failure();
    }
    result.push_back(it->second);
  }
  return result;
}

SmallVector<int64_t> MatmulSourceAttr::getShape() const {
  auto layout = getCuteLayout();
  SmallVector<int64_t> result;
  for (size_t i = 0, n = tcg::rank(layout) - 1; i < n; ++i) {
    result.push_back(tcg::static_size(layout, i));
  }
  return result;
}

LayoutSourceAttrInterface
MatmulSourceAttr::reshape(ArrayRef<int64_t> newShape) const {
  auto viewAttr = TensorSourceAttr::get(getContext(), 0, 0, getView(), {});
  SmallVector<int64_t> shape(newShape.begin(), newShape.end());
  shape.push_back(getK());
  auto newAttr = dyn_cast_or_null<TensorSourceAttr>(viewAttr.reshape(shape));
  return newAttr
             ? MatmulSourceAttr::get(getContext(),
                                     newAttr.getCuteLayout().toString(), getB(),
                                     getM(), getN(), getK(), getLhs(), getRhs())
             : nullptr;
}

LayoutSourceAttrInterface
MatmulSourceAttr::broadcast(ArrayRef<int64_t> newShape) const {
  auto viewAttr = TensorSourceAttr::get(getContext(), 0, 0, getView(), {});
  SmallVector<int64_t> shape(newShape.begin(), newShape.end());
  shape.push_back(getK());
  auto newAttr = dyn_cast_or_null<TensorSourceAttr>(viewAttr.broadcast(shape));
  return newAttr
             ? MatmulSourceAttr::get(getContext(),
                                     newAttr.getCuteLayout().toString(), getB(),
                                     getM(), getN(), getK(), getLhs(), getRhs())
             : nullptr;
}

LayoutSourceAttrInterface
MatmulSourceAttr::transpose(ArrayRef<int64_t> permutation) const {
  auto viewAttr = TensorSourceAttr::get(getContext(), 0, 0, getView(), {});
  SmallVector<int64_t> temp(permutation.begin(), permutation.end());
  temp.push_back(permutation.size());
  auto newAttr = dyn_cast_or_null<TensorSourceAttr>(viewAttr.transpose(temp));
  return newAttr
             ? MatmulSourceAttr::get(getContext(),
                                     newAttr.getCuteLayout().toString(), getB(),
                                     getM(), getN(), getK(), getLhs(), getRhs())
             : nullptr;
}

LayoutSourceAttrInterface
MatmulSourceAttr::slice(ArrayRef<int64_t> starts, ArrayRef<int64_t> limits,
                        ArrayRef<int64_t> strides) const {
  LLVM_DEBUG(llvm::dbgs() << "Matmul slice is not supported\n");
  return nullptr;
}

LayoutSourceAttrInterface MatmulSourceAttr::normalize() const {
  LLVM_DEBUG(llvm::dbgs() << "Computing iteration space for matmul\n");
  std::string_view tab = "    ";

  // [1] Compute LHS iteration space and get boundaries.
  auto lhsIterSpace = normalizeAndSplit(getLhs(), {getB(), getM(), getK()});
  if (llvm::failed(lhsIterSpace)) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot normalize LHS source\n");
    return nullptr;
  }

  auto lhsM = findDimensionWithSuffixProduct(*lhsIterSpace, getM() * getK());
  auto lhsK = findDimensionWithSuffixProduct(*lhsIterSpace, getK());
  assert(llvm::succeeded(lhsM) && llvm::succeeded(lhsK));

  // [2] Compute RHS iteration space and get boundaries.
  auto rhsIterSpace = normalizeAndSplit(getRhs(), {getB(), getK(), getN()});
  if (llvm::failed(rhsIterSpace)) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot normalize RHS source\n");
    return nullptr;
  }

  auto rhsK = findDimensionWithSuffixProduct(*rhsIterSpace, getK() * getN());
  auto rhsN = findDimensionWithSuffixProduct(*rhsIterSpace, getN());
  assert(llvm::succeeded(rhsK) && llvm::succeeded(rhsN));

  // [3] Normalize output layout.
  auto output =
      normalizeLayout(getCuteLayout(), {getB(), getM(), getN(), getK()});
  LLVM_DEBUG(llvm::dbgs() << tab << "Normalized output layout: "
                          << output.toString() << "\n");

  // [4] Compute iteration space for batch dimensions.
  SmallVector<int64_t> batchShape;
  if (getB() > 1) {
    SmallVector<int64_t> lhsBatch(lhsIterSpace->begin(),
                                  lhsIterSpace->begin() + *lhsM);
    SmallVector<int64_t> rhsBatch(rhsIterSpace->begin(),
                                  rhsIterSpace->begin() + *rhsK);
    SmallVector<int64_t> outBatch =
        extractSizes(output, getM() * getN() * getK(), getB());
    auto batchIterSpace =
        getCommonIterationSpace({lhsBatch, rhsBatch, outBatch});
    if (llvm::failed(batchIterSpace)) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot infer shape for batch dimensions\n");
      return nullptr;
    }
    batchShape = std::move(*batchIterSpace);
    LLVM_DEBUG({
      llvm::dbgs() << tab << "Batch dimensions: (";
      llvm::interleaveComma(batchShape, llvm::dbgs());
      llvm::dbgs() << ")\n";
    });
  }

  // [5] Compute iteration space for LHS noncontracting dimension.
  SmallVector<int64_t> lhsNoncontracting(lhsIterSpace->begin() + *lhsM,
                                         lhsIterSpace->begin() + *lhsK);
  SmallVector<int64_t> outLhs = extractSizes(output, getN() * getK(), getM());
  auto mShape = getCommonIterationSpace({lhsNoncontracting, outLhs});
  if (llvm::failed(mShape)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Cannot infer shape for LHS noncontracting dimension\n");
    return nullptr;
  }
  LLVM_DEBUG({
    llvm::dbgs() << tab << "Noncontracting LHS: (";
    llvm::interleaveComma(*mShape, llvm::dbgs());
    llvm::dbgs() << ")\n";
  });

  // [6] Compute iteration space for RHS noncontracting dimension.
  SmallVector<int64_t> rhsNoncontracting(rhsIterSpace->begin() + *rhsN,
                                         rhsIterSpace->end());
  SmallVector<int64_t> outRhs = extractSizes(output, getK(), getN());
  auto nShape = getCommonIterationSpace({rhsNoncontracting, outRhs});
  if (llvm::failed(nShape)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Cannot infer shape for RHS noncontracting dimension\n");
    return nullptr;
  }
  LLVM_DEBUG({
    llvm::dbgs() << tab << "Noncontracting RHS: (";
    llvm::interleaveComma(*nShape, llvm::dbgs());
    llvm::dbgs() << ")\n";
  });

  // [7] Compute iteration space for contracting dimension.
  SmallVector<int64_t> lhsContracting(lhsIterSpace->begin() + *lhsK,
                                      lhsIterSpace->end());
  SmallVector<int64_t> rhsContracting(rhsIterSpace->begin() + *rhsK,
                                      rhsIterSpace->begin() + *rhsN);
  auto kShape = getCommonIterationSpace({lhsContracting, rhsContracting});
  if (llvm::failed(kShape)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Cannot infer shape for contracting dimension\n");
    return nullptr;
  }
  LLVM_DEBUG({
    llvm::dbgs() << tab << "Contracting dimension: (";
    llvm::interleaveComma(*kShape, llvm::dbgs());
    llvm::dbgs() << ")\n";
  });

  // [8] Reshape LHS underlying source, if necessary.
  SmallVector<int64_t> lhsTarget = batchShape;
  lhsTarget.insert(lhsTarget.end(), mShape->begin(), mShape->end());
  lhsTarget.insert(lhsTarget.end(), kShape->begin(), kShape->end());
  if (lhsTarget.empty()) {
    lhsTarget.push_back(1);
  }

  LayoutSourceAttrInterface lhsResult = getLhs();
  if (lhsResult.getShape() != lhsTarget) {
    lhsResult = lhsResult.reshape(lhsTarget);
    if (!lhsResult) {
      LLVM_DEBUG(llvm::dbgs() << "Reshape of the LHS source failed\n");
      return nullptr;
    }
  }

  // [9] Reshape RHS underlying source, if necessary.
  SmallVector<int64_t> rhsTarget = batchShape;
  rhsTarget.insert(rhsTarget.end(), kShape->begin(), kShape->end());
  rhsTarget.insert(rhsTarget.end(), nShape->begin(), nShape->end());
  if (rhsTarget.empty()) {
    rhsTarget.push_back(1);
  }

  LayoutSourceAttrInterface rhsResult = getRhs();
  if (rhsResult.getShape() != rhsTarget) {
    rhsResult = rhsResult.reshape(rhsTarget);
    if (!rhsResult) {
      LLVM_DEBUG(llvm::dbgs() << "Reshape of the RHS source failed\n");
      return nullptr;
    }
  }

  // [10] Build the resulting view layout.
  SmallVector<int64_t> profile = batchShape;
  profile.insert(profile.end(), mShape->begin(), mShape->end());
  profile.insert(profile.end(), nShape->begin(), nShape->end());
  profile.insert(profile.end(), kShape->begin(), kShape->end());

  auto viewLayout = normalizeLayout(getCuteLayout(), profile);
  LLVM_DEBUG(llvm::dbgs() << tab << "Resulting output layout: "
                          << viewLayout.toString() << "\n");

  // Separate the contracting dimensions.
  if (kShape->size() > 1) {
    size_t rank = tcg::rank(viewLayout);
    size_t split = rank - kShape->size();
    auto contracting = tcg::take(split, rank, viewLayout);
    viewLayout = tcg::take(0, split, viewLayout);
    viewLayout.append(contracting);
  } else if (kShape->empty()) {
    viewLayout.append(tcg::Layout(1, 0));
  }

  // Build the resulting matmul source.
  return MatmulSourceAttr::get(getContext(), viewLayout.toString(), getB(),
                               getM(), getN(), getK(), lhsResult, rhsResult);
}

//===----------------------------------------------------------------------===//
// IterSpaceDimDomainsAttr
//===----------------------------------------------------------------------===//

/// Format: `[<state>, <state>, ...]` where each `<state>` is `undef` or `def`.
/// Example: `[def, undef, def]`

Attribute IterSpaceDimDomainsAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLSquare()) {
    return {};
  }

  SmallVector<DimState> values;
  if (parser.parseCommaSeparatedList([&]() -> ParseResult {
        StringRef keyword;
        if (parser.parseKeyword(&keyword)) {
          return failure();
        }
        auto value = symbolizeDimState(keyword);
        if (!value) {
          parser.emitError(parser.getCurrentLocation(),
                           "unknown dimension state keyword: ")
              << keyword;
          return failure();
        }
        values.push_back(*value);
        return success();
      })) {
    return {};
  }

  if (parser.parseRSquare()) {
    return {};
  }

  return IterSpaceDimDomainsAttr::get(parser.getContext(), values);
}

void IterSpaceDimDomainsAttr::print(AsmPrinter &printer) const {
  printer << "[";
  llvm::interleaveComma(getValue(), printer, [&](DimState state) {
    printer << stringifyDimState(state);
  });
  printer << "]";
}

} // namespace mlir::nv_tensor_ir
