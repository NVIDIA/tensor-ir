// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Dialect/TensorIR.h"

#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::nv_tensor_ir;

static bool hasStaticShape(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.hasStaticShape();
}

static bool isUnsignedIntegerType(IntegerType type) {
  return type.getWidth() == 1 || type.isUnsigned();
}

static bool isLosslessIntegerConvert(IntegerType sourceType,
                                     IntegerType targetType) {
  unsigned sourceWidth = sourceType.getWidth();
  unsigned targetWidth = targetType.getWidth();
  bool sourceIsUnsigned = isUnsignedIntegerType(sourceType);
  bool targetIsUnsigned = isUnsignedIntegerType(targetType);

  if (sourceIsUnsigned) {
    return targetIsUnsigned ? targetWidth >= sourceWidth
                            : targetWidth > sourceWidth;
  }
  return !targetIsUnsigned && targetWidth >= sourceWidth;
}

static bool isLosslessIntegerToFloatConvert(IntegerType sourceType,
                                            FloatType targetType) {
  const llvm::fltSemantics &semantics = targetType.getFloatSemantics();
  unsigned sourceWidth = sourceType.getWidth();
  bool sourceIsUnsigned = isUnsignedIntegerType(sourceType);
  unsigned requiredPrecision = sourceIsUnsigned ? sourceWidth : sourceWidth - 1;
  int requiredMaxExponent = static_cast<int>(sourceWidth - 1);
  return llvm::APFloat::semanticsPrecision(semantics) >= requiredPrecision &&
         llvm::APFloat::semanticsMaxExponent(semantics) >= requiredMaxExponent;
}

static bool isLosslessFloatConvert(FloatType sourceType, FloatType targetType) {
  const llvm::fltSemantics &source = sourceType.getFloatSemantics();
  const llvm::fltSemantics &target = targetType.getFloatSemantics();
  return llvm::APFloat::semanticsPrecision(target) >=
             llvm::APFloat::semanticsPrecision(source) &&
         llvm::APFloat::semanticsMaxExponent(target) >=
             llvm::APFloat::semanticsMaxExponent(source) &&
         llvm::APFloat::semanticsMinExponent(target) <=
             llvm::APFloat::semanticsMinExponent(source);
}

static bool isLosslessConvert(Value source, Value converted) {
  Type sourceType = cast<RankedTensorType>(source.getType()).getElementType();
  Type targetType =
      cast<RankedTensorType>(converted.getType()).getElementType();
  if (sourceType == targetType) {
    return true;
  }

  if (auto sourceInteger = dyn_cast<IntegerType>(sourceType)) {
    if (auto targetInteger = dyn_cast<IntegerType>(targetType)) {
      return isLosslessIntegerConvert(sourceInteger, targetInteger);
    }
    if (auto targetFloat = dyn_cast<FloatType>(targetType)) {
      return isLosslessIntegerToFloatConvert(sourceInteger, targetFloat);
    }
    return false;
  }

  auto sourceFloat = dyn_cast<FloatType>(sourceType);
  auto targetFloat = dyn_cast<FloatType>(targetType);
  return sourceFloat && targetFloat &&
         isLosslessFloatConvert(sourceFloat, targetFloat);
}

static Attribute buildSplatElementsAttr(Value output, Attribute value) {
  auto outputType = cast<RankedTensorType>(output.getType());
  return DenseElementsAttr::get(outputType, cast<TypedAttr>(value));
}

static DenseI64ArrayAttr
composePermutations(PatternRewriter &rewriter,
                    DenseI64ArrayAttr innerPermutation,
                    DenseI64ArrayAttr outerPermutation) {
  SmallVector<int64_t> permutation;
  permutation.reserve(innerPermutation.size());
  for (int64_t dimension : outerPermutation.asArrayRef()) {
    permutation.push_back(innerPermutation.asArrayRef()[dimension]);
  }
  return rewriter.getDenseI64ArrayAttr(permutation);
}

static bool checkedMultiplyAdd(int64_t lhs, int64_t rhs, int64_t addend,
                               int64_t &result) {
  int64_t product;
  return !llvm::MulOverflow(lhs, rhs, product) &&
         !llvm::AddOverflow(product, addend, result);
}

static bool
computeComposedSliceDimension(int64_t innerStart, int64_t innerLimit,
                              int64_t innerStride, int64_t outerStart,
                              int64_t outerLimit, int64_t outerStride,
                              int64_t &start, int64_t &limit, int64_t &stride) {
  if (!checkedMultiplyAdd(outerStart, innerStride, innerStart, start) ||
      llvm::MulOverflow(innerStride, outerStride, stride)) {
    return false;
  }

  int64_t elementCount = 0;
  if (outerLimit > outerStart) {
    elementCount = 1 + (outerLimit - 1 - outerStart) / outerStride;
  }

  if (elementCount == 0) {
    limit = start;
    return start <= innerLimit;
  }

  int64_t lastOffset;
  if (!checkedMultiplyAdd(elementCount - 1, stride, start, lastOffset) ||
      llvm::AddOverflow(lastOffset, int64_t{1}, limit)) {
    return false;
  }
  return limit <= innerLimit;
}

static bool canComposeSlices(Value output) {
  auto outerSlice = output.getDefiningOp<SliceOp>();
  auto innerSlice = outerSlice.getInput().getDefiningOp<SliceOp>();
  assert(innerSlice && "expected a nested slice");
  DenseI64ArrayAttr innerStarts = innerSlice.getStartsAttr();
  DenseI64ArrayAttr innerLimits = innerSlice.getLimitsAttr();
  DenseI64ArrayAttr innerStrides = innerSlice.getStridesAttr();
  DenseI64ArrayAttr outerStarts = outerSlice.getStartsAttr();
  DenseI64ArrayAttr outerLimits = outerSlice.getLimitsAttr();
  DenseI64ArrayAttr outerStrides = outerSlice.getStridesAttr();
  int64_t rank = innerStarts.size();
  if (innerLimits.size() != rank || innerStrides.size() != rank ||
      outerStarts.size() != rank || outerLimits.size() != rank ||
      outerStrides.size() != rank) {
    return false;
  }

  for (auto [innerStart, innerLimit, innerStride, outerStart, outerLimit,
             outerStride] :
       llvm::zip_equal(innerStarts.asArrayRef(), innerLimits.asArrayRef(),
                       innerStrides.asArrayRef(), outerStarts.asArrayRef(),
                       outerLimits.asArrayRef(), outerStrides.asArrayRef())) {
    int64_t start;
    int64_t limit;
    int64_t stride;
    if (!computeComposedSliceDimension(innerStart, innerLimit, innerStride,
                                       outerStart, outerLimit, outerStride,
                                       start, limit, stride)) {
      return false;
    }
  }
  return true;
}

static DenseI64ArrayAttr composeSliceStarts(PatternRewriter &rewriter,
                                            DenseI64ArrayAttr innerStarts,
                                            DenseI64ArrayAttr innerStrides,
                                            DenseI64ArrayAttr outerStarts) {
  SmallVector<int64_t> starts;
  starts.reserve(innerStarts.size());
  for (auto [innerStart, innerStride, outerStart] :
       llvm::zip_equal(innerStarts.asArrayRef(), innerStrides.asArrayRef(),
                       outerStarts.asArrayRef())) {
    int64_t start;
    bool overflow =
        !checkedMultiplyAdd(outerStart, innerStride, innerStart, start);
    assert(!overflow && "slice composition was not checked");
    starts.push_back(start);
  }
  return rewriter.getDenseI64ArrayAttr(starts);
}

static DenseI64ArrayAttr composeSliceLimits(PatternRewriter &rewriter,
                                            DenseI64ArrayAttr innerStarts,
                                            DenseI64ArrayAttr innerStrides,
                                            DenseI64ArrayAttr outerStarts,
                                            DenseI64ArrayAttr outerLimits,
                                            DenseI64ArrayAttr outerStrides,
                                            DenseI64ArrayAttr innerLimits) {
  SmallVector<int64_t> limits;
  limits.reserve(innerStarts.size());
  for (auto [innerStart, innerLimit, innerStride, outerStart, outerLimit,
             outerStride] :
       llvm::zip_equal(innerStarts.asArrayRef(), innerLimits.asArrayRef(),
                       innerStrides.asArrayRef(), outerStarts.asArrayRef(),
                       outerLimits.asArrayRef(), outerStrides.asArrayRef())) {
    int64_t start;
    int64_t limit;
    int64_t stride;
    bool composed = computeComposedSliceDimension(
        innerStart, innerLimit, innerStride, outerStart, outerLimit,
        outerStride, start, limit, stride);
    assert(composed && "slice composition was not checked");
    limits.push_back(limit);
  }
  return rewriter.getDenseI64ArrayAttr(limits);
}

static DenseI64ArrayAttr composeSliceStrides(PatternRewriter &rewriter,
                                             DenseI64ArrayAttr innerStrides,
                                             DenseI64ArrayAttr outerStrides) {
  SmallVector<int64_t> strides;
  strides.reserve(innerStrides.size());
  for (auto [innerStride, outerStride] :
       llvm::zip_equal(innerStrides.asArrayRef(), outerStrides.asArrayRef())) {
    int64_t stride;
    bool overflow = llvm::MulOverflow(innerStride, outerStride, stride);
    assert(!overflow && "slice composition was not checked");
    strides.push_back(stride);
  }
  return rewriter.getDenseI64ArrayAttr(strides);
}

namespace {
#include "TensorOpsCanonicalization.inc"
} // namespace

static int64_t getDimSize(Value value, int64_t dimension) {
  return cast<RankedTensorType>(value.getType()).getDimSize(dimension);
}

/// Returns whether concatenating `inputs` along `dimension` merges to the
/// non-concatenation extents of `outputType`.
static bool hasConcatenatedShape(ValueRange inputs, int64_t dimension,
                                 RankedTensorType outputType) {
  for (int64_t dim = 0, rank = outputType.getRank(); dim < rank; ++dim) {
    if (dim == dimension) {
      continue;
    }
    int64_t merged = ShapedType::kDynamic;
    for (Value input : inputs) {
      FailureOr<int64_t> next =
          mergeEqualDimensions(merged, getDimSize(input, dim));
      if (failed(next)) {
        return false;
      }
      merged = *next;
    }
    if (merged != outputType.getDimSize(dim)) {
      return false;
    }
  }
  return true;
}

/// Returns `value`'s defining slice if it has unit stride and a non-negative
/// extent `limit - start` along `dimension`.
static SliceOp getUnitStrideSlice(Value value, int64_t dimension) {
  auto slice = value.getDefiningOp<SliceOp>();
  if (!slice || slice.getStrides()[dimension] != 1 ||
      slice.getStarts()[dimension] > slice.getLimits()[dimension]) {
    return {};
  }
  return slice;
}

/// Returns whether `rhs` continues `lhs` along `dimension`: both slice the
/// same value, `rhs` starts where `lhs` ends, and the slices are identical
/// along every other dimension.
static bool isAdjacentSlice(SliceOp lhs, SliceOp rhs, int64_t dimension) {
  if (lhs.getInput() != rhs.getInput()) {
    return false;
  }
  for (int64_t dim = 0, rank = lhs.getStarts().size(); dim < rank; ++dim) {
    if (dim == dimension) {
      if (lhs.getLimits()[dim] != rhs.getStarts()[dim]) {
        return false;
      }
      continue;
    }
    if (lhs.getStarts()[dim] != rhs.getStarts()[dim] ||
        lhs.getLimits()[dim] != rhs.getLimits()[dim] ||
        lhs.getStrides()[dim] != rhs.getStrides()[dim]) {
      return false;
    }
  }
  return true;
}

namespace {
/// concatenate(a, concatenate(b, c), d) -> concatenate(a, b, c, d) when both
/// concatenates use the same dimension and the flattened input count is at
/// most 64, bounding operand growth when nested concatenates are shared.
struct FlattenNestedConcatenates : OpRewritePattern<ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs;
    constexpr size_t maxInputs = 64;
    bool flattened = false;
    for (Value input : op.getInputs()) {
      auto inner = input.getDefiningOp<ConcatenateOp>();
      bool canFlatten = inner && inner.getDimension() == op.getDimension();
      size_t addedInputs = canFlatten ? inner.getInputs().size() : 1;
      if (addedInputs > maxInputs - inputs.size()) {
        return rewriter.notifyMatchFailure(op,
                                           "flattened input count exceeds 64");
      }
      if (!canFlatten) {
        inputs.push_back(input);
        continue;
      }
      llvm::append_range(inputs, inner.getInputs());
      flattened = true;
    }
    if (!flattened) {
      return rewriter.notifyMatchFailure(
          op, "no input is a concatenate along the same dimension");
    }
    rewriter.replaceOpWithNewOp<ConcatenateOp>(op, op.getOutput().getType(),
                                               inputs, op.getDimension());
    return success();
  }
};

/// Removes inputs whose extent along the concatenation dimension is a static
/// zero.
struct DropZeroExtentConcatenateInputs : OpRewritePattern<ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    int64_t dimension = op.getDimension();
    SmallVector<Value> inputs =
        llvm::filter_to_vector(op.getInputs(), [&](Value input) {
          return getDimSize(input, dimension) != 0;
        });
    if (inputs.empty()) {
      inputs.push_back(op.getInputs().front());
    }
    if (inputs.size() == op.getInputs().size()) {
      return rewriter.notifyMatchFailure(op, "no zero-extent input to drop");
    }
    // A dropped input may be the only one with a static extent along another
    // dimension; dropping it would make the result type less static.
    if (!hasConcatenatedShape(inputs, dimension, op.getOutput().getType())) {
      return rewriter.notifyMatchFailure(
          op, "dropped inputs determine a static result extent");
    }
    rewriter.replaceOpWithNewOp<ConcatenateOp>(op, op.getOutput().getType(),
                                               inputs, dimension);
    return success();
  }
};

/// concatenate(slice(x, [s0, s1)), slice(x, [s1, s2))) -> slice(x, [s0, s2))
/// for consecutive inputs that are unit-stride slices of the same value along
/// the concatenation dimension and identical slices along every other
/// dimension.
struct MergeAdjacentConcatenatedSlices : OpRewritePattern<ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    int64_t dimension = op.getDimension();
    OperandRange operands = op.getInputs();
    SmallVector<Value> inputs;
    bool merged = false;
    for (size_t begin = 0, size = operands.size(); begin < size;) {
      SliceOp first = getUnitStrideSlice(operands[begin], dimension);
      SliceOp last = first;
      size_t end = begin + 1;
      while (first && end < size) {
        SliceOp next = getUnitStrideSlice(operands[end], dimension);
        if (!next || !isAdjacentSlice(last, next, dimension)) {
          break;
        }
        last = next;
        ++end;
      }
      if (end == begin + 1) {
        inputs.push_back(operands[begin]);
        ++begin;
        continue;
      }

      SmallVector<int64_t> limits(first.getLimits());
      limits[dimension] = last.getLimits()[dimension];
      RankedTensorType firstType = first.getOutput().getType();
      SmallVector<int64_t> shape(firstType.getShape());
      shape[dimension] = limits[dimension] - first.getStarts()[dimension];
      Location loc = rewriter.getFusedLoc({first.getLoc(), last.getLoc()});
      auto slice = SliceOp::create(rewriter, loc, firstType.clone(shape),
                                   first.getInput(), first.getStarts(), limits,
                                   first.getStrides());
      inputs.push_back(slice.getResult());
      merged = true;
      begin = end;
    }
    if (!merged) {
      return rewriter.notifyMatchFailure(op, "no adjacent slices to merge");
    }
    rewriter.replaceOpWithNewOp<ConcatenateOp>(op, op.getOutput().getType(),
                                               inputs, dimension);
    return success();
  }
};

/// slice(concatenate(a, b, c)) -> slice(b) when the slice reads one input, or
/// concatenate(slice(a), slice(b)) when it reads a strict subset of the
/// inputs.
struct SliceConcatenateInputs : OpRewritePattern<SliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SliceOp op,
                                PatternRewriter &rewriter) const override {
    auto concat = op.getInput().getDefiningOp<ConcatenateOp>();
    if (!concat) {
      return rewriter.notifyMatchFailure(op, "input is not a concatenate");
    }
    int64_t dimension = concat.getDimension();
    RankedTensorType outputType = op.getOutput().getType();
    int64_t count = outputType.getDimSize(dimension);
    if (count == 0) {
      return rewriter.notifyMatchFailure(op, "slice is empty");
    }
    int64_t start = op.getStarts()[dimension];
    int64_t stride = op.getStrides()[dimension];
    // The verifier bounds `count` by `limit`, so the last selected index fits.
    int64_t last = start + (count - 1) * stride;

    struct Piece {
      Value input;
      int64_t start;
      int64_t count;
    };
    OperandRange inputs = concat.getInputs();
    SmallVector<Piece> pieces;
    int64_t offset = 0;
    for (auto [index, input] : llvm::enumerate(inputs)) {
      if (offset > last) {
        break;
      }
      // Static slice bounds within the concatenate are a precondition, so the
      // final input holds every selected index at or after its offset even
      // when its extent is dynamic.
      int64_t extent = getDimSize(input, dimension);
      bool isFinal = index + 1 == inputs.size();
      if (ShapedType::isDynamic(extent) && !isFinal) {
        return rewriter.notifyMatchFailure(
            op, "a dynamic input extent precedes the end of the slice");
      }
      int64_t pieceLast = isFinal ? last : std::min(last, offset + extent - 1);
      int64_t lowest = std::max(offset, start);
      int64_t first = start + static_cast<int64_t>(
                                  llvm::divideCeil(lowest - start, stride)) *
                                  stride;
      if (first <= pieceLast) {
        pieces.push_back(
            {input, first - offset, (pieceLast - first) / stride + 1});
      }
      if (!isFinal) {
        offset += extent;
      }
    }
    if (pieces.size() != 1 && pieces.size() == inputs.size()) {
      return rewriter.notifyMatchFailure(op, "slice reads every input");
    }

    SmallVector<Value> slices;
    for (const Piece &piece : pieces) {
      SmallVector<int64_t> starts(op.getStarts());
      SmallVector<int64_t> limits(op.getLimits());
      starts[dimension] = piece.start;
      limits[dimension] = piece.start + (piece.count - 1) * stride + 1;
      SmallVector<int64_t> shape(outputType.getShape());
      shape[dimension] = piece.count;
      auto slice =
          SliceOp::create(rewriter, op.getLoc(), outputType.clone(shape),
                          piece.input, starts, limits, op.getStrides());
      slices.push_back(slice.getResult());
    }
    if (slices.size() == 1) {
      rewriter.replaceOp(op, slices.front());
      return success();
    }
    rewriter.replaceOpWithNewOp<ConcatenateOp>(op, outputType, slices,
                                               dimension);
    return success();
  }
};
} // namespace
void BroadcastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<MergeSequentialBroadcasts>(context);
}

void ConvertOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<MergeSequentialConverts>(context);
}

void SplatOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<SplatConstant>(context);
}

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<MergeSequentialTransposes>(context);
}

void SliceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<MergeSequentialSlices, SliceConcatenateInputs>(context);
}

void ConcatenateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<FlattenNestedConcatenates, DropZeroExtentConcatenateInputs,
              MergeAdjacentConcatenatedSlices>(context);
}

OpFoldResult ConstantOp::fold(FoldAdaptor) { return getValueAttr(); }

OpFoldResult ConvertOp::fold(FoldAdaptor) {
  // Dialect conversion may replace this op's operand before replacing the op
  // itself. Avoid the typed getter here so folding remains valid during that
  // transient state.
  Value input = getOperation()->getOperand(0);
  if (input.getType() == getOperation()->getResult(0).getType()) {
    return input;
  }
  return {};
}

OpFoldResult ReshapeOp::fold(FoldAdaptor) {
  auto inputType = getInput().getType();
  auto outputType = getOutput().getType();
  if (inputType == outputType && inputType.hasStaticShape()) {
    return getInput();
  }

  if (!inputType.hasStaticShape() || !outputType.hasStaticShape()) {
    return {};
  }

  auto innerReshape = getInput().getDefiningOp<ReshapeOp>();
  if (!innerReshape || !innerReshape.getInput().getType().hasStaticShape() ||
      !innerReshape.getOutput().getType().hasStaticShape()) {
    return {};
  }

  getInputMutable().assign(innerReshape.getInput());
  return getOutput();
}

OpFoldResult TransposeOp::fold(FoldAdaptor) {
  auto inputType = getInput().getType();
  auto outputType = getOutput().getType();
  if (inputType != outputType) {
    return {};
  }

  for (auto [index, dimension] : llvm::enumerate(getPermutation())) {
    if (static_cast<int64_t>(index) != dimension) {
      return {};
    }
  }
  return getInput();
}

OpFoldResult SliceOp::fold(FoldAdaptor) {
  auto inputType = getInput().getType();
  auto outputType = getOutput().getType();
  if (inputType != outputType ||
      llvm::any_of(getStarts(), [](int64_t start) { return start != 0; }) ||
      llvm::any_of(getStrides(), [](int64_t stride) { return stride != 1; }) ||
      !llvm::equal(getLimits(), inputType.getShape())) {
    return {};
  }
  return getInput();
}

OpFoldResult ConcatenateOp::fold(FoldAdaptor) {
  if (getInputs().size() != 1 ||
      getInputs().front().getType() != getOutput().getType()) {
    return {};
  }
  return getInputs().front();
}

OpFoldResult ReduceOp::fold(FoldAdaptor) {
  auto inputType = getInput().getType();
  auto outputType = getOutput().getType();
  if (inputType != outputType) {
    return {};
  }
  // A unit reduction is its input only when the mode applies no prologue or
  // epilogue to a single element.
  switch (getReductionMode()) {
  case ReductionMode::add:
  case ReductionMode::mul:
  case ReductionMode::min:
  case ReductionMode::max:
    break;
  default:
    return {};
  }

  for (int32_t dimension : getDimensions()) {
    if (inputType.getDimSize(dimension) != 1) {
      return {};
    }
  }
  return getInput();
}
