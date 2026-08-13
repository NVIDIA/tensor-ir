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

static bool hasLayoutAnnotation(Operation *op) {
  return op->hasAttr(TensorIRDialect::getLayoutAttrName());
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
  results.add<MergeSequentialSlices>(context);
}

OpFoldResult ConstantOp::fold(FoldAdaptor) { return getValueAttr(); }

OpFoldResult ConvertOp::fold(FoldAdaptor) {
  if (getInput().getType() == getOutput().getType()) {
    return getInput();
  }
  return {};
}

OpFoldResult ReshapeOp::fold(FoldAdaptor) {
  auto inputType = getInput().getType();
  auto outputType = getOutput().getType();
  if (hasLayoutAnnotation(getOperation())) {
    return {};
  }

  if (inputType == outputType) {
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
  if (hasLayoutAnnotation(getOperation()) || inputType != outputType) {
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
  if (hasLayoutAnnotation(getOperation()) || inputType != outputType ||
      llvm::any_of(getStarts(), [](int64_t start) { return start != 0; }) ||
      llvm::any_of(getStrides(), [](int64_t stride) { return stride != 1; }) ||
      !llvm::equal(getLimits(), inputType.getShape())) {
    return {};
  }
  return getInput();
}

OpFoldResult ConcatenateOp::fold(FoldAdaptor) {
  if (hasLayoutAnnotation(getOperation()) || getInputs().size() != 1 ||
      getInputs().front().getType() != getOutput().getType()) {
    return {};
  }
  return getInputs().front();
}

OpFoldResult ReduceOp::fold(FoldAdaptor) {
  auto inputType = getInput().getType();
  auto outputType = getOutput().getType();
  if (hasLayoutAnnotation(getOperation()) || inputType != outputType) {
    return {};
  }

  for (int32_t dimension : getDimensions()) {
    if (inputType.getDimSize(dimension) != 1) {
      return {};
    }
  }
  return getInput();
}
