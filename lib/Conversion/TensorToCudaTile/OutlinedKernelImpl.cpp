// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- OutlinedKernelImpl.cpp - Lower staged TensorIR kernels -------------===//
//
// This file is the conversion boundary for the inspectable tiled-program IR.
// It deliberately consumes only explicit function arguments, direct TensorIR
// memory accesses, and standard control-flow/launch operations. In particular,
// it never reads layout-propagation analysis attributes.
//
//===----------------------------------------------------------------------===//

#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTileInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

namespace mlir::nv_tensor_ir::tensor_to_cuda_tile {
namespace {

static SmallVector<Value> flatten(ArrayRef<ValueRange> ranges) {
  SmallVector<Value> values;
  for (ValueRange range : ranges) {
    llvm::append_range(values, range);
  }
  return values;
}

static Type convertElementType(MLIRContext *context, Type type) {
  auto integerType = dyn_cast<IntegerType>(type);
  if (!integerType || integerType.isSignless()) {
    return type;
  }
  return IntegerType::get(context, integerType.getWidth());
}

static cuda_tile::PaddingValue convertPaddingValue(TilePadding padding) {
  switch (padding) {
  case TilePadding::Zero:
    return cuda_tile::PaddingValue::zero;
  case TilePadding::NegZero:
    return cuda_tile::PaddingValue::neg_zero;
  case TilePadding::NaN:
    return cuda_tile::PaddingValue::nan;
  case TilePadding::PosInf:
    return cuda_tile::PaddingValue::pos_inf;
  case TilePadding::NegInf:
    return cuda_tile::PaddingValue::neg_inf;
  }
  llvm_unreachable("unknown TensorIR tile padding value");
}

static cuda_tile::PaddingValueAttr
convertPaddingValue(MLIRContext *context, TilePaddingAttr padding) {
  return padding ? cuda_tile::PaddingValueAttr::get(
                       context, convertPaddingValue(padding.getValue()))
                 : cuda_tile::PaddingValueAttr{};
}

struct BackendTensorViewLayout {
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
};

struct ViewMetadata {
  ArrayRef<int64_t> shape;
  ArrayRef<int64_t> strides;
  int64_t offset;
  int64_t alignment;
};

template <typename OpTy>
static ViewMetadata getViewMetadata(OpTy op) {
  return {op.getStaticSizes(), op.getStaticStrides(),
          op.getStaticOffsets().front(), op.getAlignmentAttr().getInt()};
}

/// CudaTile tensor views require strictly positive constant strides. Represent
/// a zero-stride TensorIR dimension as a size-one backend dimension with an
/// arbitrary non-unit stride. The non-unit stride keeps the collapsed dimension
/// from being mistaken for a contiguous dimension. Tile accesses restore the
/// logical broadcast after loads and select one representative slice before
/// stores.
static BackendTensorViewLayout getBackendLayout(ViewMetadata view) {
  constexpr int64_t kCollapsedBroadcastStride = 2;
  BackendTensorViewLayout layout{SmallVector<int64_t>(view.shape),
                                 SmallVector<int64_t>(view.strides)};
  for (size_t dimension = 0; dimension < layout.shape.size(); ++dimension) {
    if (layout.strides[dimension] != 0) {
      continue;
    }
    layout.shape[dimension] = 1;
    layout.strides[dimension] = kCollapsedBroadcastStride;
  }
  return layout;
}

static bool isBroadcastTileDimension(ViewMetadata view, size_t tileDimension) {
  return view.strides[tileDimension] == 0;
}

static SmallVector<int64_t> getBackendTileShape(ViewMetadata view,
                                                ArrayRef<int64_t> tileShape) {
  SmallVector<int64_t> shape(tileShape);
  for (size_t dimension = 0; dimension < shape.size(); ++dimension) {
    if (isBroadcastTileDimension(view, dimension)) {
      shape[dimension] = 1;
    }
  }
  return shape;
}

struct BackendPartition {
  Value value;
  SmallVector<int64_t> tileShape;
};

static BackendPartition
createBackendPartition(ConversionPatternRewriter &rewriter, Location loc,
                       ViewMetadata sourceView, Value backendView,
                       ArrayRef<int64_t> sourceTileShape,
                       TilePaddingAttr padding) {
  SmallVector<int64_t> backendTileShape =
      getBackendTileShape(sourceView, sourceTileShape);
  SmallVector<int32_t> partitionTileShape;
  SmallVector<int32_t> dimMap;
  partitionTileShape.reserve(backendTileShape.size());
  dimMap.reserve(backendTileShape.size());
  for (auto [dimension, extent] : llvm::enumerate(backendTileShape)) {
    partitionTileShape.push_back(extent);
    dimMap.push_back(dimension);
  }
  auto partitionType = cuda_tile::PartitionViewType::get(
      rewriter.getContext(),
      DenseI32ArrayAttr::get(rewriter.getContext(), partitionTileShape),
      cast<cuda_tile::TensorViewType>(backendView.getType()), dimMap,
      convertPaddingValue(rewriter.getContext(), padding));
  Value partition = cuda_tile::MakePartitionViewOp::create(
      rewriter, loc, partitionType, backendView);
  return {partition, std::move(backendTileShape)};
}

/// Materialize static coordinates in the backend's scalar index type.
static SmallVector<Value> convertIndices(ConversionPatternRewriter &rewriter,
                                         Location loc,
                                         ArrayRef<int64_t> staticIndices,
                                         ValueRange dynamicIndices) {
  SmallVector<Value> indices;
  auto dynamic = dynamicIndices.begin();
  for (int64_t index : staticIndices) {
    if (ShapedType::isDynamic(index)) {
      indices.push_back(*dynamic++);
    } else {
      Value coordinate = createConstant(
          rewriter, loc,
          cuda_tile::TileType::get(ArrayRef<int64_t>{}, rewriter.getI32Type()),
          index);
      indices.push_back(coordinate);
    }
  }
  return indices;
}

static Value zeroBroadcastCoordinates(ConversionPatternRewriter &rewriter,
                                      Location loc, ViewMetadata view,
                                      MutableArrayRef<Value> coordinates) {
  Value zero;
  for (size_t dimension = 0; dimension < coordinates.size(); ++dimension) {
    if (!isBroadcastTileDimension(view, dimension)) {
      continue;
    }
    if (!zero) {
      zero = createConstant(
          rewriter, loc,
          cuda_tile::TileType::get(
              ArrayRef<int64_t>{},
              IntegerType::get(rewriter.getContext(), /*width=*/32)),
          int64_t{0});
    }
    coordinates[dimension] = zero;
  }
  return zero;
}

/// Type conversion for the staged ABI. A memref becomes a pointer followed by
/// dynamic sizes and explicit dynamic strides. The uniform ABI includes every
/// size and stride, including values that are statically known in the IR.
class OutlinedKernelTypeConverter : public TypeConverter {
public:
  OutlinedKernelTypeConverter(MLIRContext *context, bool uniformSignature)
      : uniformSignature(uniformSignature) {
    addConversion([](cuda_tile::TileType type) -> Type { return type; });
    addConversion([](cuda_tile::TensorViewType type) -> Type { return type; });
    addConversion(
        [](cuda_tile::PartitionViewType type) -> Type { return type; });

    addConversion(
        [context](IndexType) -> Type { return scalarIndex(context); });
    addConversion([](FloatType type) -> Type {
      return cuda_tile::TileType::get(ArrayRef<int64_t>{}, type);
    });
    addConversion([context](IntegerType type) -> Type {
      return cuda_tile::TileType::get(ArrayRef<int64_t>{},
                                      convertElementType(context, type));
    });
    addConversion([context](RankedTensorType type) -> Type {
      return cuda_tile::TileType::get(
          type.getShape(), convertElementType(context, type.getElementType()));
    });
    addConversion([context](ptr::PtrType) -> Type {
      Type bytePointer =
          cuda_tile::PointerType::get(IntegerType::get(context, /*width=*/8));
      return cuda_tile::TileType::get(ArrayRef<int64_t>{}, bytePointer);
    });
    addConversion(
        [this, context](MemRefType type, SmallVectorImpl<Type> &results)
            -> std::optional<LogicalResult> {
          SmallVector<int64_t> strides;
          int64_t offset;
          if (failed(type.getStridesAndOffset(strides, offset))) {
            return failure();
          }
          Type elementType = convertElementType(context, type.getElementType());
          Type pointerType = cuda_tile::PointerType::get(elementType);
          results.push_back(
              cuda_tile::TileType::get(ArrayRef<int64_t>{}, pointerType));
          Type indexType = scalarIndex(context);
          for (int64_t size : type.getShape()) {
            if (this->uniformSignature || ShapedType::isDynamic(size)) {
              results.push_back(indexType);
            }
          }
          if (this->uniformSignature || !type.getLayout().isIdentity()) {
            for (int64_t stride : strides) {
              if (this->uniformSignature || ShapedType::isDynamic(stride)) {
                results.push_back(indexType);
              }
            }
          }
          return success();
        });
  }

  bool usesUniformSignature() const { return uniformSignature; }

  static cuda_tile::TileType scalarIndex(MLIRContext *context) {
    return cuda_tile::TileType::get(ArrayRef<int64_t>{},
                                    IntegerType::get(context, 32));
  }

private:
  bool uniformSignature;
};

static FailureOr<Value> only(ValueRange range) {
  if (range.size() != 1) {
    return failure();
  }
  return range.front();
}

class KernelFuncConversion : public OpConversionPattern<func::FuncOp> {
public:
  KernelFuncConversion(const TypeConverter &converter, MLIRContext *context,
                       cuda_tile::OptimizationHintsAttr optimizationHints)
      : OpConversionPattern(converter, context),
        optimizationHints(optimizationHints) {}

  LogicalResult
  matchAndRewrite(func::FuncOp func, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!func->hasAttr(gpu::GPUDialect::getKernelFuncAttrName())) {
      return rewriter.notifyMatchFailure(func, "expected gpu.kernel marker");
    }
    if (!func.getResultTypes().empty()) {
      return func.emitError("outlined TensorIR kernels must return void");
    }

    TypeConverter::SignatureConversion signature(func.getNumArguments());
    if (failed(typeConverter->convertSignatureArgs(
            func.getFunctionType().getInputs(), signature))) {
      return func.emitError("failed to convert outlined kernel signature");
    }
    auto functionType = FunctionType::get(rewriter.getContext(),
                                          signature.getConvertedTypes(), {});
    auto entry = cuda_tile::EntryOp::create(
        rewriter, func.getLoc(), func.getName(), functionType,
        /*arg_attrs=*/{}, /*res_attrs=*/{}, optimizationHints);
    rewriter.inlineRegionBefore(func.getBody(), entry.getBody(),
                                entry.getBody().end());
    rewriter.applySignatureConversion(&entry.getBody().front(), signature,
                                      typeConverter);
    rewriter.eraseOp(func);
    return success();
  }

private:
  cuda_tile::OptimizationHintsAttr optimizationHints;
};

class ReturnConversion : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cuda_tile::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

class UnrealizedCastConversion
    : public OpConversionPattern<UnrealizedConversionCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange inputs = adaptor.getInputs();
    if (op.getNumResults() != 1 || inputs.size() != 1) {
      return op.emitError("staged signedness bridge must be a one-to-one cast");
    }
    Type resultType = typeConverter->convertType(op.getResult(0).getType());
    if (!resultType || resultType != inputs.front().getType()) {
      return op.emitError(
          "staged signedness bridge must erase during conversion");
    }
    rewriter.replaceOp(op, inputs.front());
    return success();
  }
};

static Value applyOffsetAndAlignment(ConversionPatternRewriter &rewriter,
                                     Location loc, Value pointer, Value offset,
                                     int64_t staticOffset, int64_t alignment,
                                     Type elementType) {
  if (!offset && staticOffset != 0) {
    offset = createConstant(
        rewriter, loc,
        OutlinedKernelTypeConverter::scalarIndex(rewriter.getContext()),
        staticOffset);
  }
  if (offset) {
    pointer = cuda_tile::OffsetOp::create(rewriter, loc, pointer, offset);
  }

  unsigned elementBytes = std::max(1u, elementType.getIntOrFloatBitWidth() / 8);
  if (alignment > elementBytes) {
    auto predicate = cuda_tile::DivByAttr::get(rewriter.getContext(), alignment,
                                               /*every=*/std::nullopt,
                                               /*along=*/std::nullopt);
    pointer = cuda_tile::AssumeOp::create(rewriter, loc, pointer, predicate);
  }
  return pointer;
}

static FailureOr<Value>
createBackendTensorView(ConversionPatternRewriter &rewriter, Location loc,
                        ViewMetadata view, Value base, ValueRange dynamicShapes,
                        ValueRange dynamicStrides, ValueRange dynamicOffset,
                        Type sourceElementType) {
  Type elementType =
      convertElementType(rewriter.getContext(), sourceElementType);
  auto pointerType = cuda_tile::TileType::get(
      ArrayRef<int64_t>{}, cuda_tile::PointerType::get(elementType));
  Value pointer = base;
  if (pointer.getType() != pointerType) {
    // Bufferization's ptr.to_ptr result is lowered through an opaque i8
    // pointer. Recover the already typed source when this access immediately
    // casts it back to the memref element type.
    auto cast = pointer.getDefiningOp<cuda_tile::PtrToPtrOp>();
    if (cast && cast.getSource().getType() == pointerType) {
      pointer = cast.getSource();
    } else {
      pointer =
          cuda_tile::PtrToPtrOp::create(rewriter, loc, pointerType, pointer);
    }
  }

  FailureOr<Value> convertedOffset = only(dynamicOffset);
  if (failed(convertedOffset) && !dynamicOffset.empty()) {
    return failure();
  }
  Value offset = dynamicOffset.empty() ? Value{} : *convertedOffset;
  pointer = applyOffsetAndAlignment(rewriter, loc, pointer, offset, view.offset,
                                    view.alignment, elementType);

  BackendTensorViewLayout layout = getBackendLayout(view);
  auto resultType = cuda_tile::TensorViewType::get(
      rewriter.getContext(), elementType, layout.shape, layout.strides);
  SmallVector<Value> backendDynamicShapes;
  backendDynamicShapes.reserve(dynamicShapes.size());
  size_t dynamicShapeIndex = 0;
  for (auto [dimension, extent] : llvm::enumerate(view.shape)) {
    if (!ShapedType::isDynamic(extent)) {
      continue;
    }
    if (dynamicShapeIndex >= dynamicShapes.size()) {
      return failure();
    }
    Value dynamicShape = dynamicShapes[dynamicShapeIndex++];
    if (view.strides[dimension] != 0) {
      backendDynamicShapes.push_back(dynamicShape);
    }
  }
  if (dynamicShapeIndex != dynamicShapes.size()) {
    return failure();
  }
  return cuda_tile::MakeTensorViewOp::create(rewriter, loc, resultType, pointer,
                                             backendDynamicShapes,
                                             dynamicStrides)
      .getResult();
}

class ToPtrConversion : public OpConversionPattern<ptr::ToPtrOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ptr::ToPtrOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<ptr::GenericSpaceAttr>(op.getType().getMemorySpace())) {
      return op.emitError(
          "staged CudaTile lowering requires generic pointer space");
    }
    ValueRange source = adaptor.getPtr();
    if (source.empty()) {
      return rewriter.notifyMatchFailure(op,
                                         "missing converted source pointer");
    }
    auto resultType =
        typeConverter->convertType<cuda_tile::TileType>(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "pointer type is not convertible");
    }
    Value result = source.front();
    if (result.getType() != resultType) {
      result = cuda_tile::PtrToPtrOp::create(rewriter, op.getLoc(), resultType,
                                             result);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class LoadConversion : public OpConversionPattern<LoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileType =
        typeConverter->convertType<cuda_tile::TileType>(op.getType());
    if (!tileType) {
      return failure();
    }
    ViewMetadata view = getViewMetadata(op);
    FailureOr<Value> backendView = createBackendTensorView(
        rewriter, op.getLoc(), view, adaptor.getBase(), adaptor.getSizes(),
        adaptor.getStrides(), adaptor.getOffsets(),
        op.getType().getElementType());
    if (failed(backendView)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to form backend tensor view");
    }
    auto sourceTileType = cast<RankedTensorType>(op.getType());
    BackendPartition partition =
        createBackendPartition(rewriter, op.getLoc(), view, *backendView,
                               sourceTileType.getShape(), op.getPaddingAttr());
    SmallVector<Value> indices = convertIndices(
        rewriter, op.getLoc(), op.getStaticIndices(), adaptor.getIndices());
    zeroBroadcastCoordinates(rewriter, op.getLoc(), view, indices);
    auto loadType = cuda_tile::TileType::get(partition.tileShape,
                                             tileType.getElementType());
    auto load = cuda_tile::LoadViewTkoOp::create(
        rewriter, op.getLoc(), loadType,
        cuda_tile::TokenType::get(rewriter.getContext()),
        cuda_tile::MemoryOrderingSemantics::WEAK, /*memory_scope=*/nullptr,
        partition.value, indices, /*token=*/nullptr,
        /*optimization_hints=*/nullptr);
    Value result = load.getResult(0);
    if (loadType != tileType) {
      result = cuda_tile::BroadcastOp::create(rewriter, op.getLoc(), tileType,
                                              result);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class StoreConversion : public OpConversionPattern<StoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value tile = adaptor.getTile();
    ViewMetadata view = getViewMetadata(op);
    FailureOr<Value> backendView = createBackendTensorView(
        rewriter, op.getLoc(), view, adaptor.getBase(), adaptor.getSizes(),
        adaptor.getStrides(), adaptor.getOffsets(),
        op.getTile().getType().getElementType());
    if (failed(backendView)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to form backend tensor view");
    }
    auto sourceTileType = cast<RankedTensorType>(op.getTile().getType());
    BackendPartition partition = createBackendPartition(
        rewriter, op.getLoc(), view, *backendView, sourceTileType.getShape(),
        /*padding=*/TilePaddingAttr{});
    SmallVector<Value> indices = convertIndices(
        rewriter, op.getLoc(), op.getStaticIndices(), adaptor.getIndices());
    Value zero = zeroBroadcastCoordinates(rewriter, op.getLoc(), view, indices);
    auto tileType = cast<cuda_tile::TileType>(tile.getType());
    auto storeType = cuda_tile::TileType::get(partition.tileShape,
                                              tileType.getElementType());
    if (storeType != tileType) {
      SmallVector<Value> extractIndices(indices.size(), zero);
      tile = cuda_tile::ExtractOp::create(rewriter, op.getLoc(), storeType,
                                          tile, extractIndices);
    }
    cuda_tile::StoreViewTkoOp::create(
        rewriter, op.getLoc(), cuda_tile::TokenType::get(rewriter.getContext()),
        cuda_tile::MemoryOrderingSemantics::WEAK, /*memory_scope=*/nullptr,
        tile, partition.value, indices, /*token=*/nullptr,
        /*optimization_hints=*/nullptr);
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename GpuOp, typename CudaOp>
class LaunchCoordinateConversion : public OpConversionPattern<GpuOp> {
public:
  using OpConversionPattern<GpuOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GpuOp op, typename OpConversionPattern<GpuOp>::OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getDimension() != gpu::Dimension::x) {
      return op.emitError("staged TensorIR kernels require the x grid axis");
    }
    auto coordinates = CudaOp::create(rewriter, op.getLoc());
    rewriter.replaceOp(op, coordinates.getResult(0));
    return success();
  }
};

using BlockIdConversion =
    LaunchCoordinateConversion<gpu::BlockIdOp, cuda_tile::GetTileBlockIdOp>;
using GridDimConversion =
    LaunchCoordinateConversion<gpu::GridDimOp, cuda_tile::GetNumTileBlocksOp>;

static SmallVector<ValueRange> packResults(Operation *op,
                                           Operation *replacement,
                                           const TypeConverter &converter) {
  SmallVector<ValueRange> packed;
  unsigned offset = 0;
  for (Type type : op->getResultTypes()) {
    SmallVector<Type> converted;
    if (failed(converter.convertType(type, converted))) {
      return {};
    }
    packed.push_back(replacement->getResults().slice(offset, converted.size()));
    offset += converted.size();
  }
  return packed;
}

class ForConversion : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> lower = only(adaptor.getLowerBound());
    FailureOr<Value> upper = only(adaptor.getUpperBound());
    FailureOr<Value> step = only(adaptor.getStep());
    if (failed(lower) || failed(upper) || failed(step) ||
        failed(rewriter.convertRegionTypes(&op.getRegion(), *typeConverter))) {
      return failure();
    }
    SmallVector<Value> init = flatten(adaptor.getInitArgs());
    if (llvm::any_of(init, [](Value value) {
          return isa<cuda_tile::TileView>(value.getType());
        })) {
      return op.emitError("CUDA Tile loops cannot carry view values");
    }
    auto loop = cuda_tile::ForOp::create(rewriter, op.getLoc(), *lower, *upper,
                                         *step, init);
    rewriter.eraseBlock(loop.getBody());
    rewriter.inlineRegionBefore(op.getRegion(), loop.getRegion(),
                                loop.getRegion().end());
    rewriter.replaceOpWithMultiple(
        op, packResults(op, loop.getOperation(), *typeConverter));
    return success();
  }
};

class IfConversion : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), resultTypes))) {
      return failure();
    }
    auto ifOp = cuda_tile::IfOp::create(rewriter, op.getLoc(), resultTypes,
                                        adaptor.getCondition());
    rewriter.inlineRegionBefore(op.getThenRegion(), ifOp.getThenRegion(),
                                ifOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), ifOp.getElseRegion(),
                                ifOp.getElseRegion().end());
    rewriter.replaceOpWithMultiple(
        op, packResults(op, ifOp.getOperation(), *typeConverter));
    return success();
  }
};

class YieldConversion : public OpConversionPattern<scf::YieldOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> values = flatten(adaptor.getOperands());
    Operation *parent = op->getParentOp();
    if (isa<scf::ForOp, cuda_tile::ForOp>(parent)) {
      rewriter.replaceOpWithNewOp<cuda_tile::ContinueOp>(op, values);
    } else if (isa<scf::IfOp, cuda_tile::IfOp>(parent)) {
      rewriter.replaceOpWithNewOp<cuda_tile::YieldOp>(op, values);
    } else {
      return op.emitError("unsupported SCF yield parent in staged kernel");
    }
    return success();
  }
};

class ArithConstantConversion : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        typeConverter->convertType<cuda_tile::TileType>(op.getType());
    if (!resultType) {
      return failure();
    }
    if (auto integer = dyn_cast<IntegerAttr>(op.getValue())) {
      rewriter.replaceOp(op, createConstant(rewriter, op.getLoc(), resultType,
                                            integer.getInt()));
      return success();
    }
    if (auto floating = dyn_cast<FloatAttr>(op.getValue())) {
      rewriter.replaceOp(op, createConstant(rewriter, op.getLoc(), resultType,
                                            floating.getValueAsDouble()));
      return success();
    }
    if (auto dense = dyn_cast<DenseElementsAttr>(op.getValue());
        dense && dense.isSplat()) {
      auto value = cast<DenseTypedElementsAttr>(dense.resizeSplat(resultType));
      rewriter.replaceOpWithNewOp<cuda_tile::ConstantOp>(op, resultType, value);
      return success();
    }
    return op.emitError("unsupported staged-kernel constant");
  }
};

class IndexCastConversion : public OpConversionPattern<arith::IndexCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getIn();
    auto resultType =
        typeConverter->convertType<cuda_tile::TileType>(op.getType());
    if (!resultType) {
      return failure();
    }
    unsigned inputWidth =
        cast<ShapedType>(input.getType()).getElementTypeBitWidth();
    unsigned resultWidth = resultType.getElementTypeBitWidth();
    if (inputWidth == resultWidth) {
      rewriter.replaceOp(op, input);
    } else if (inputWidth < resultWidth) {
      rewriter.replaceOpWithNewOp<cuda_tile::ExtIOp>(
          op, resultType, input, cuda_tile::Signedness::Unsigned);
    } else {
      rewriter.replaceOpWithNewOp<cuda_tile::TruncIOp>(op, resultType, input);
    }
    return success();
  }
};

class SIToFPConversion : public OpConversionPattern<arith::SIToFPOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SIToFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getIn();
    auto resultType =
        typeConverter->convertType<cuda_tile::TileType>(op.getType());
    if (!resultType) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<cuda_tile::IToFOp>(
        op, resultType, input, cuda_tile::Signedness::Signed,
        cuda_tile::RoundingMode::NEAREST_EVEN);
    return success();
  }
};

class UIToFPConversion : public OpConversionPattern<arith::UIToFPOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::UIToFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getIn();
    auto resultType =
        typeConverter->convertType<cuda_tile::TileType>(op.getType());
    if (!resultType) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<cuda_tile::IToFOp>(
        op, resultType, input, cuda_tile::Signedness::Unsigned,
        cuda_tile::RoundingMode::NEAREST_EVEN);
    return success();
  }
};

/// Select a dynamic dimension from the expanded memref ABI.
class DimConversion : public OpConversionPattern<memref::DimOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DimOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<int64_t> dimension = op.getConstantIndex();
    if (!dimension) {
      return op.emitError("staged kernels require a constant memref.dim index");
    }
    auto memrefType = cast<MemRefType>(op.getSource().getType());
    int64_t size = memrefType.getDimSize(*dimension);
    if (!ShapedType::isDynamic(size)) {
      rewriter.replaceOp(
          op, createConstant(rewriter, op.getLoc(),
                             OutlinedKernelTypeConverter::scalarIndex(
                                 rewriter.getContext()),
                             size));
      return success();
    }

    ValueRange descriptor = adaptor.getSource();
    const auto &converter =
        *static_cast<const OutlinedKernelTypeConverter *>(typeConverter);
    unsigned field = 1;
    for (int64_t dim = 0; dim < *dimension; ++dim) {
      if (converter.usesUniformSignature() || memrefType.isDynamicDim(dim)) {
        ++field;
      }
    }
    if (field >= descriptor.size()) {
      return op.emitError("missing dynamic size in converted memref ABI");
    }
    rewriter.replaceOp(op, descriptor[field]);
    return success();
  }
};

static FailureOr<Value>
getDescriptorSize(ValueRange descriptor, MemRefType type, unsigned dimension,
                  const OutlinedKernelTypeConverter &converter) {
  if (!type.isDynamicDim(dimension)) {
    return failure();
  }
  unsigned field = 1;
  for (unsigned dim = 0; dim < dimension; ++dim) {
    if (converter.usesUniformSignature() || type.isDynamicDim(dim)) {
      ++field;
    }
  }
  if (field >= descriptor.size()) {
    return failure();
  }
  return descriptor[field];
}

static FailureOr<Value>
getDescriptorStride(ValueRange descriptor, MemRefType type,
                    ArrayRef<int64_t> strides, unsigned dimension,
                    const OutlinedKernelTypeConverter &converter) {
  if (!ShapedType::isDynamic(strides[dimension])) {
    return failure();
  }
  unsigned field = 1;
  field += converter.usesUniformSignature()
               ? type.getRank()
               : llvm::count_if(type.getShape(), ShapedType::isDynamic);
  for (unsigned dim = 0; dim < dimension; ++dim) {
    if (converter.usesUniformSignature() ||
        ShapedType::isDynamic(strides[dim])) {
      ++field;
    }
  }
  if (field >= descriptor.size()) {
    return failure();
  }
  return descriptor[field];
}

/// Reconstruct strided metadata from the flattened kernel ABI. The pointer ABI
/// argument already denotes the logical memref origin, so the extracted offset
/// is zero; dynamic sizes and strides select their corresponding ABI fields.
class ExtractStridedMetadataConversion
    : public OpConversionPattern<memref::ExtractStridedMetadataOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ExtractStridedMetadataOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange descriptor = adaptor.getSource();
    if (descriptor.empty()) {
      return failure();
    }
    auto sourceType = cast<MemRefType>(op.getSource().getType());
    SmallVector<int64_t> strides;
    int64_t sourceOffset;
    if (failed(sourceType.getStridesAndOffset(strides, sourceOffset))) {
      return failure();
    }
    const auto &converter =
        *static_cast<const OutlinedKernelTypeConverter *>(typeConverter);
    auto constant = [&](int64_t value) {
      return createConstant(
          rewriter, op.getLoc(),
          OutlinedKernelTypeConverter::scalarIndex(rewriter.getContext()),
          value);
    };

    SmallVector<Value> replacements;
    replacements.reserve(2 + 2 * sourceType.getRank());
    replacements.push_back(descriptor.front());
    replacements.push_back(constant(0));
    SmallVector<Value> sizes;
    sizes.reserve(sourceType.getRank());
    for (auto [dimension, size] : llvm::enumerate(sourceType.getShape())) {
      if (!ShapedType::isDynamic(size)) {
        sizes.push_back(constant(size));
        continue;
      }
      FailureOr<Value> dynamic =
          getDescriptorSize(descriptor, sourceType, dimension, converter);
      if (failed(dynamic)) {
        return op.emitError("missing dynamic size in converted memref ABI");
      }
      sizes.push_back(*dynamic);
    }
    llvm::append_range(replacements, sizes);
    if (sourceType.getLayout().isIdentity() &&
        !converter.usesUniformSignature()) {
      SmallVector<Value> identityStrides(sourceType.getRank());
      Value running = constant(1);
      for (size_t position = sourceType.getRank(); position > 0; --position) {
        size_t dimension = position - 1;
        identityStrides[dimension] = running;
        running = cuda_tile::MulIOp::create(rewriter, op.getLoc(), running,
                                            sizes[dimension]);
      }
      llvm::append_range(replacements, identityStrides);
      rewriter.replaceOp(op, replacements);
      return success();
    }
    for (auto [dimension, stride] : llvm::enumerate(strides)) {
      if (!ShapedType::isDynamic(stride)) {
        replacements.push_back(constant(stride));
        continue;
      }
      FailureOr<Value> dynamic = getDescriptorStride(
          descriptor, sourceType, strides, dimension, converter);
      if (failed(dynamic)) {
        return op.emitError("missing dynamic stride in converted memref ABI");
      }
      replacements.push_back(*dynamic);
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

/// Consume only dispatch wrappers whose staging launch resolves to one of the
/// outlined kernels collected for this conversion.
static void eraseDispatchWrappers(ModuleOp module,
                                  ArrayRef<func::FuncOp> kernels) {
  llvm::SmallDenseSet<Operation *> kernelOps;
  for (func::FuncOp kernel : kernels) {
    kernelOps.insert(kernel);
  }

  SymbolTableCollection symbolTables;
  SmallVector<func::FuncOp> wrappers;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (func->hasAttr(gpu::GPUDialect::getKernelFuncAttrName()) ||
        !func.getName().contains("_dispatch")) {
      continue;
    }
    bool launchesKernel = false;
    func.walk([&](gpu::LaunchFuncOp launch) {
      Operation *callee =
          symbolTables.lookupNearestSymbolFrom(launch, launch.getKernelAttr());
      launchesKernel |= kernelOps.contains(callee);
    });
    if (launchesKernel) {
      wrappers.push_back(func);
    }
  }
  for (func::FuncOp wrapper : wrappers) {
    wrapper.erase();
  }
}

static LogicalResult verifyOutlinedKernel(func::FuncOp kernel) {
  if (kernel.isExternal() || !kernel.getBody().hasOneBlock()) {
    return kernel.emitError(
        "outlined TensorIR kernels must define a single-block body");
  }
  for (Type type : kernel.getFunctionType().getInputs()) {
    if (isa<IndexType, IntegerType, FloatType>(type)) {
      continue;
    }
    auto memrefType = dyn_cast<MemRefType>(type);
    SmallVector<int64_t> strides;
    int64_t offset;
    if (!memrefType ||
        failed(memrefType.getStridesAndOffset(strides, offset))) {
      return kernel.emitError()
             << "outlined TensorIR kernel arguments must be scalars or "
                "strided memrefs, but got "
             << type;
    }
  }
  return success();
}

} // namespace

LogicalResult convertOutlinedKernels(
    ModuleOp module, cuda_tile::OptimizationHintsAttr optimizationHints,
    bool uniformSignature, bool enableExperimentalCudaTileOps) {
  SmallVector<func::FuncOp> kernels;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (func->hasAttr(gpu::GPUDialect::getKernelFuncAttrName())) {
      return func.emitError(
          "outlined TensorIR kernels must be nested in a gpu.module");
    }
  }

  SmallVector<gpu::GPUModuleOp> gpuModules;
  llvm::SmallDenseSet<StringRef> kernelNames;
  for (gpu::GPUModuleOp gpuModule : module.getOps<gpu::GPUModuleOp>()) {
    gpuModules.push_back(gpuModule);
    bool foundKernel = false;
    for (Operation &op : gpuModule.getBody()->without_terminator()) {
      auto kernel = dyn_cast<func::FuncOp>(op);
      if (!kernel ||
          !kernel->hasAttr(gpu::GPUDialect::getKernelFuncAttrName())) {
        return op.emitError(
            "outlined TensorIR gpu.module may contain only gpu.kernel "
            "func.func entry points");
      }
      foundKernel = true;
      if (!kernelNames.insert(kernel.getName()).second) {
        return kernel.emitError()
               << "duplicate outlined kernel name @" << kernel.getName();
      }
      kernels.push_back(kernel);
    }
    if (!foundKernel) {
      return gpuModule.emitError(
          "outlined TensorIR gpu.module must contain a gpu.kernel function");
    }
  }
  for (func::FuncOp kernel : kernels) {
    if (failed(verifyOutlinedKernel(kernel))) {
      return failure();
    }
  }
  eraseDispatchWrappers(module, kernels);

  MLIRContext *context = module.getContext();
  OutlinedKernelTypeConverter typeConverter(context, uniformSignature);
  ConversionTarget target(*context);
  target.addLegalDialect<cuda_tile::CudaTileDialect>();
  target.addIllegalDialect<
      TensorIRDialect, arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
      memref::MemRefDialect, ptr::PtrDialect, scf::SCFDialect>();
  target.addIllegalOp<UnrealizedConversionCastOp>();

  for (func::FuncOp kernel : kernels) {
    RewritePatternSet patterns(context);
    patterns.add<KernelFuncConversion>(typeConverter, context,
                                       optimizationHints);
    patterns
        .add<ReturnConversion, UnrealizedCastConversion, ToPtrConversion,
             LoadConversion, StoreConversion, BlockIdConversion,
             GridDimConversion, ForConversion, IfConversion, YieldConversion,
             ArithConstantConversion, IndexCastConversion, SIToFPConversion,
             UIToFPConversion, DimConversion, ExtractStridedMetadataConversion>(
            typeConverter, context);
    registerOutlinedComputePatterns(patterns, typeConverter,
                                    enableExperimentalCudaTileOps);
    if (failed(applyFullConversion(kernel, target, std::move(patterns)))) {
      return failure();
    }
  }
  for (gpu::GPUModuleOp gpuModule : gpuModules) {
    for (cuda_tile::EntryOp entry :
         llvm::make_early_inc_range(gpuModule.getOps<cuda_tile::EntryOp>())) {
      entry->moveBefore(gpuModule);
    }
    gpuModule.erase();
  }
  SmallVector<cuda_tile::PtrToPtrOp> deadPointerCasts;
  module.walk([&](cuda_tile::PtrToPtrOp cast) {
    if (cast->use_empty()) {
      deadPointerCasts.push_back(cast);
    }
  });
  for (cuda_tile::PtrToPtrOp cast : deadPointerCasts) {
    cast.erase();
  }
  WalkResult remnants = module.walk([&](Operation *op) {
    StringRef dialect = op->getName().getDialectNamespace();
    if (dialect != TensorIRDialect::getDialectNamespace() &&
        dialect != func::FuncDialect::getDialectNamespace() &&
        dialect != gpu::GPUDialect::getDialectNamespace() &&
        dialect != ptr::PtrDialect::getDialectNamespace() &&
        dialect != scf::SCFDialect::getDialectNamespace() &&
        !isa<UnrealizedConversionCastOp>(op)) {
      return WalkResult::advance();
    }
    op->emitError(
        "outlined TensorIR conversion left an illegal operation behind");
    return WalkResult::interrupt();
  });
  if (remnants.wasInterrupted()) {
    return failure();
  }
  module->removeAttr(gpu::GPUDialect::getContainerModuleAttrName());
  return success();
}

} // namespace mlir::nv_tensor_ir::tensor_to_cuda_tile
