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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

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

struct ViewMetadata {
  ArrayRef<int64_t> shape;
  SmallVector<int64_t> strides;
  int64_t offset;
  int64_t alignment;
};

template <typename OpTy>
static ViewMetadata getViewMetadata(OpTy op) {
  auto type = cast<MemRefType>(op.getView().getType());
  auto [strides, offset] = type.getStridesAndOffset();
  return {type.getShape(), std::move(strides), offset,
          op.getAlignmentAttr().getInt()};
}

static cuda_tile::MakePartitionViewOp
createBackendPartition(ConversionPatternRewriter &rewriter, Location loc,
                       Value backendView, ArrayRef<int64_t> sourceTileShape,
                       TilePaddingAttr padding) {
  SmallVector<int32_t> partitionTileShape;
  SmallVector<int32_t> dimMap;
  partitionTileShape.reserve(sourceTileShape.size());
  dimMap.reserve(sourceTileShape.size());
  for (auto [dimension, extent] : llvm::enumerate(sourceTileShape)) {
    partitionTileShape.push_back(extent);
    dimMap.push_back(dimension);
  }
  auto partitionType = cuda_tile::PartitionViewType::get(
      rewriter.getContext(),
      DenseI32ArrayAttr::get(rewriter.getContext(), partitionTileShape),
      cast<cuda_tile::TensorViewType>(backendView.getType()), dimMap,
      convertPaddingValue(rewriter.getContext(), padding));
  return cuda_tile::MakePartitionViewOp::create(rewriter, loc, partitionType,
                                                backendView);
}

/// Materialize static coordinates in the backend's scalar index type.
static SmallVector<Value> convertIndices(ConversionPatternRewriter &rewriter,
                                         Location loc,
                                         ArrayRef<int64_t> staticIndices,
                                         ArrayRef<ValueRange> dynamicIndices) {
  SmallVector<Value> indices;
  auto dynamic = dynamicIndices.begin();
  for (int64_t index : staticIndices) {
    if (ShapedType::isDynamic(index)) {
      assert(dynamic->size() == 1 && "index must convert to one value");
      indices.push_back((dynamic++)->front());
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

/// Kernel argument memrefs have statically zero offsets and expand to a pointer,
/// dynamic sizes, and explicit dynamic strides. Internal views additionally
/// carry a dynamic offset, when present, immediately after the pointer. Static
/// fields come from the memref type. Identity-layout strides are derived from
/// sizes. Reinterpret casts replace the offset relative to the aligned pointer,
/// leaving that pointer unchanged.
class OutlinedKernelTypeConverter : public TypeConverter {
public:
  explicit OutlinedKernelTypeConverter(MLIRContext *context) {
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
    addConversion([context](MemRefType type, SmallVectorImpl<Type> &results)
                      -> std::optional<LogicalResult> {
      if (type.getMemorySpace()) {
        return failure();
      }
      SmallVector<int64_t> strides;
      int64_t offset;
      if (failed(type.getStridesAndOffset(strides, offset))) {
        return failure();
      }
      Type elementType = convertElementType(context, type.getElementType());
      Type pointerType = cuda_tile::PointerType::get(elementType);
      results.push_back(
          cuda_tile::TileType::get(ArrayRef<int64_t>{}, pointerType));
      unsigned numDynamicFields =
          ShapedType::isDynamic(offset) + type.getNumDynamicDims();
      if (!type.getLayout().isIdentity()) {
        numDynamicFields += llvm::count_if(strides, ShapedType::isDynamic);
      }
      results.append(numDynamicFields, scalarIndex(context));
      return success();
    });
  }

  static cuda_tile::TileType scalarIndex(MLIRContext *context) {
    return cuda_tile::TileType::get(ArrayRef<int64_t>{},
                                    IntegerType::get(context, 32));
  }
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
      return rewriter.notifyMatchFailure(func, "not an entrypoint");
    }
    if (!func.getResultTypes().empty()) {
      return rewriter.notifyMatchFailure(func, "entrypoint must return void");
    }

    TypeConverter::SignatureConversion signature(func.getNumArguments());
    if (failed(typeConverter->convertSignatureArgs(
            func.getFunctionType().getInputs(), signature))) {
      return rewriter.notifyMatchFailure(
          func, "failed to convert outlined kernel signature");
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
    if (auto ordinalAttr = func->getAttrOfType<IntegerAttr>(
            TensorIRDialect::getKernelOrdinalAttrName())) {
      entry->setAttr(TensorIRDialect::getKernelOrdinalAttrName(), ordinalAttr);
    }
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
  if (staticOffset != 0) {
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

/// Decode the compact memref descriptor. Static fields come from the type;
/// identity-layout strides are derived from sizes rather than ABI arguments.
struct ConvertedMemRef {
  Value base;
  Value offset;
  SmallVector<Value> sizes;
  SmallVector<Value> strides;
};

static FailureOr<ConvertedMemRef>
decodeMemRef(ConversionPatternRewriter &rewriter, Location loc, MemRefType type,
             ValueRange descriptor) {
  if (descriptor.empty()) {
    return failure();
  }
  auto [strides, offset] = type.getStridesAndOffset();
  auto constant = [&](int64_t value) {
    return createConstant(
        rewriter, loc,
        OutlinedKernelTypeConverter::scalarIndex(rewriter.getContext()), value);
  };
  unsigned field = 1;
  auto read = [&](int64_t value) -> Value {
    if (!ShapedType::isDynamic(value)) {
      return constant(value);
    }
    return field < descriptor.size() ? descriptor[field++] : Value{};
  };
  ConvertedMemRef result;
  result.base = descriptor.front();
  result.offset = read(offset);
  if (!result.offset) {
    return failure();
  }
  for (int64_t size : type.getShape()) {
    Value value = read(size);
    if (!value) {
      return failure();
    }
    result.sizes.push_back(value);
  }
  result.strides.resize(type.getRank());
  if (type.getLayout().isIdentity()) {
    Value running = constant(1);
    for (int64_t dim = type.getRank() - 1; dim >= 0; --dim) {
      result.strides[dim] = running;
      if (dim != 0) {
        running = cuda_tile::MulIOp::create(rewriter, loc, running,
                                            result.sizes[dim]);
      }
    }
  } else {
    for (unsigned dim = 0; dim < strides.size(); ++dim) {
      result.strides[dim] = read(strides[dim]);
      if (!result.strides[dim]) {
        return failure();
      }
    }
  }
  if (field != descriptor.size()) {
    return failure();
  }
  return result;
}

static FailureOr<Value>
createBackendMemRefView(ConversionPatternRewriter &rewriter, Location loc,
                        MemRefType type, ValueRange descriptor,
                        const ViewMetadata &view) {
  auto decoded = decodeMemRef(rewriter, loc, type, descriptor);
  if (failed(decoded)) {
    return failure();
  }
  Type elementType =
      convertElementType(rewriter.getContext(), type.getElementType());
  auto pointerType = cuda_tile::TileType::get(
      ArrayRef<int64_t>{}, cuda_tile::PointerType::get(elementType));
  if (decoded->base.getType() != pointerType) {
    return failure();
  }
  Value pointer =
      applyOffsetAndAlignment(rewriter, loc, decoded->base, decoded->offset,
                              view.offset, view.alignment, elementType);
  SmallVector<int64_t> shape(view.shape), strides(view.strides);
  SmallVector<Value> dynamicSizes, dynamicStrides;
  for (auto [dim, stride] : llvm::enumerate(strides)) {
    if (ShapedType::isDynamic(shape[dim])) {
      dynamicSizes.push_back(decoded->sizes[dim]);
    }
    if (ShapedType::isDynamic(stride)) {
      dynamicStrides.push_back(decoded->strides[dim]);
    }
  }
  auto resultType = cuda_tile::TensorViewType::get(rewriter.getContext(),
                                                   elementType, shape, strides);
  return cuda_tile::MakeTensorViewOp::create(rewriter, loc, resultType, pointer,
                                             dynamicSizes, dynamicStrides)
      .getResult();
}

template <typename OpTy>
class TiledMemoryConversion : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op,
                  typename OpConversionPattern<OpTy>::OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileType =
        this->getTypeConverter()->template convertType<cuda_tile::TileType>(
            op.getTile().getType());
    if (!tileType) {
      return failure();
    }
    ViewMetadata view = getViewMetadata(op);
    FailureOr<Value> backendView = createBackendMemRefView(
        rewriter, op.getLoc(), cast<MemRefType>(op.getView().getType()),
        adaptor.getView(), view);
    if (failed(backendView)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to form backend tensor view");
    }
    TilePaddingAttr padding;
    if constexpr (std::is_same_v<OpTy, LoadOp>) {
      padding = op.getPaddingAttr();
    }
    auto partition = createBackendPartition(rewriter, op.getLoc(), *backendView,
                                            tileType.getShape(), padding);
    SmallVector<Value> indices = convertIndices(
        rewriter, op.getLoc(), op.getStaticIndices(), adaptor.getIndices());
    auto hints = createLoadStoreOptimizationHints(
        rewriter.getContext(), op.getAllowTma(), /*latency=*/-1);
    if constexpr (std::is_same_v<OpTy, LoadOp>) {
      auto load = cuda_tile::LoadViewTkoOp::create(
          rewriter, op.getLoc(), tileType,
          cuda_tile::TokenType::get(rewriter.getContext()),
          cuda_tile::MemoryOrderingSemantics::WEAK, /*memory_scope=*/nullptr,
          partition, indices, /*token=*/nullptr, hints);
      rewriter.replaceOp(op, load.getResult(0));
    } else {
      Value tile = adaptor.getTile().front();
      cuda_tile::StoreViewTkoOp::create(
          rewriter, op.getLoc(),
          cuda_tile::TokenType::get(rewriter.getContext()),
          cuda_tile::MemoryOrderingSemantics::WEAK, /*memory_scope=*/nullptr,
          tile, partition, indices, /*token=*/nullptr, hints);
      rewriter.eraseOp(op);
    }
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
      Value result = createConstant(
          rewriter, op.getLoc(),
          OutlinedKernelTypeConverter::scalarIndex(rewriter.getContext()),
          size);
      rewriter.replaceOp(op, result);
      return success();
    }

    ValueRange descriptor = adaptor.getSource();
    auto [strides, offset] = memrefType.getStridesAndOffset();
    unsigned field = 1 + ShapedType::isDynamic(offset) +
                     memrefType.getDynamicDimIndex(*dimension);
    if (field >= descriptor.size()) {
      return op.emitError("missing dynamic size in converted memref ABI");
    }
    rewriter.replaceOp(op, descriptor[field]);
    return success();
  }
};

class ExtractStridedMetadataConversion
    : public OpConversionPattern<memref::ExtractStridedMetadataOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::ExtractStridedMetadataOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto decoded = decodeMemRef(rewriter, op.getLoc(),
                                cast<MemRefType>(op.getSource().getType()),
                                adaptor.getSource());
    if (failed(decoded)) {
      return failure();
    }
    SmallVector<Value> values{decoded->base, decoded->offset};
    llvm::append_range(values, decoded->sizes);
    llvm::append_range(values, decoded->strides);
    rewriter.replaceOp(op, values);
    return success();
  }
};

/// Reinterpret casts replace descriptor metadata without moving the base.
class ReinterpretCastConversion
    : public OpConversionPattern<memref::ReinterpretCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = op.getType();
    auto [strides, offset] = type.getStridesAndOffset();
    SmallVector<Value> descriptor{adaptor.getSource().front()};
    auto appendDynamic = [&](ArrayRef<OpFoldResult> mixed,
                             ArrayRef<ValueRange> converted,
                             ArrayRef<int64_t> resultStaticFields) {
      unsigned operandIndex = 0;
      for (auto [value, field] : llvm::zip_equal(mixed, resultStaticFields)) {
        Value lowered;
        // Consume each SSA operand even when the result type makes its field
        // static and therefore omits it from the converted descriptor.
        if (isa<Value>(value)) {
          lowered = converted[operandIndex++].front();
        }
        if (!ShapedType::isDynamic(field)) {
          continue;
        }
        if (!lowered) {
          lowered = createConstant(
              rewriter, op.getLoc(),
              OutlinedKernelTypeConverter::scalarIndex(rewriter.getContext()),
              *getConstantIntValue(value));
        }
        descriptor.push_back(lowered);
      }
    };
    appendDynamic(op.getMixedOffsets(), adaptor.getOffsets(), {offset});
    appendDynamic(op.getMixedSizes(), adaptor.getSizes(), type.getShape());
    if (!type.getLayout().isIdentity()) {
      appendDynamic(op.getMixedStrides(), adaptor.getStrides(), strides);
    }
    rewriter.replaceOpWithMultiple(op, {descriptor});
    return success();
  }
};

/// Convert gpu::GPUModuleOp to cuda_tile::ModuleOp
class ModuleConversion : public OpConversionPattern<gpu::GPUModuleOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gpu::GPUModuleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    std::string cudaTileModuleName =
        parentModule.getName().has_value()
            ? "cuda_tile_" + parentModule.getName()->str()
            : "cuda_tile_module";

    auto moduleOp = cuda_tile::ModuleOp::create(rewriter, parentModule.getLoc(),
                                                cudaTileModuleName);
    Block &body = moduleOp.getBody().front();
    rewriter.inlineBlockBefore(op.getBody(), &body, body.end());
    rewriter.replaceOp(op, moduleOp);
    return success();
  }
};
} // namespace

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
    if (offset != 0) {
      return kernel.emitError()
             << "outlined TensorIR kernel memref arguments must have a "
                "statically zero offset, but got "
             << type;
    }
  }
  return success();
}

LogicalResult
convertOutlinedKernels(ModuleOp module,
                       cuda_tile::OptimizationHintsAttr optimizationHints,
                       bool enableExperimentalCudaTileOps,
                       cuda_tile::BytecodeVersion targetBytecodeVersion) {
  llvm::SmallDenseSet<StringRef> kernelNames;
  for (gpu::GPUModuleOp gpuModule : module.getOps<gpu::GPUModuleOp>()) {
    for (Operation &op : gpuModule.getBody()->without_terminator()) {
      auto kernel = dyn_cast<func::FuncOp>(op);
      if (!kernel ||
          !kernel->hasAttr(gpu::GPUDialect::getKernelFuncAttrName())) {
        return op.emitError(
            "outlined TensorIR gpu.module may contain only gpu.kernel "
            "func.func entry points");
      }
      if (failed(verifyOutlinedKernel(kernel))) {
        return failure();
      }
    }
  }

  MLIRContext *context = module.getContext();
  OutlinedKernelTypeConverter typeConverter(context);
  ConversionTarget target(*context);
  target.addLegalDialect<cuda_tile::CudaTileDialect, LLVM::LLVMDialect>();
  target.addIllegalDialect<TensorIRDialect, arith::ArithDialect,
                           func::FuncDialect, gpu::GPUDialect,
                           memref::MemRefDialect, scf::SCFDialect>();
  target.addIllegalOp<UnrealizedConversionCastOp>();
  target.addLegalOp<mlir::ModuleOp>();
  target.addDynamicallyLegalOp<func::FuncOp>(
      [&](func::FuncOp op) { return op->getParentOp() == module; });
  target.markOpRecursivelyLegal<func::FuncOp>();
  target.addLegalOp<LLVM::LLVMFuncOp>();
  target.markOpRecursivelyLegal<LLVM::LLVMFuncOp>();

  RewritePatternSet patterns(context);
  patterns.add<KernelFuncConversion>(typeConverter, context, optimizationHints);
  patterns.add<ReturnConversion, UnrealizedCastConversion,
               TiledMemoryConversion<LoadOp>, TiledMemoryConversion<StoreOp>,
               BlockIdConversion, GridDimConversion, ForConversion,
               IfConversion, YieldConversion, ArithConstantConversion,
               IndexCastConversion, SIToFPConversion, UIToFPConversion,
               DimConversion, ExtractStridedMetadataConversion,
               ReinterpretCastConversion, ModuleConversion>(typeConverter,
                                                            context);

  registerOutlinedComputePatterns(patterns, typeConverter,
                                  enableExperimentalCudaTileOps,
                                  targetBytecodeVersion);
  if (failed(applyFullConversion(module, target, std::move(patterns)))) {
    return failure();
  }

  return success();
}

} // namespace mlir::nv_tensor_ir::tensor_to_cuda_tile
