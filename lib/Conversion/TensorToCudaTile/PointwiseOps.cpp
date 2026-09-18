// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// \file PointwiseOps.cpp
/// Shared pointwise TensorIR-to-CUDA-Tile lowering.
///
/// Affine-map lowering adapts operands through `ConversionState`; outlined
/// lowering receives explicit tiles from the conversion adaptor. Both paths
/// delegate operation emission to `lowerPointwiseOp`.

#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTileInternal.h"
#include "tensor_ir/Utils/Utils.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

namespace mlir::nv_tensor_ir {
namespace tensor_to_cuda_tile {

//===----------------------------------------------------------------------===//
// Constants and splat
//===----------------------------------------------------------------------===//

/// Converts `ConstantOp` to `cuda_tile::ConstantOp`.
class ConstantOpConversion : public ConversionPattern<ConstantOp> {
public:
  using ConversionPattern<ConstantOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    // Pre-rewrite legality: reject tensor-like literals that are not splats
    // (the "unsupported constant value" case below). The scalar/splat literal
    // conversions further down depend on the converted tile type and tile
    // shape, so they stay inline in this pattern.
    MLIR_RETURN_IF_ERROR(validateConstantOpLowerable(op));

    // Handle scalar constants (e.g. "constant 0.0 : f32").
    if (!isa<TensorType>(op.getType())) {
      auto tileType =
          cast<ShapedType>(getTypeConverter()->convertType(op.getType()));
      auto literal = dyn_cast_or_null<DenseTypedElementsAttr>(
          fromScalarLiteral(tileType, op.getValue()));
      if (!literal) {
        return op.emitError("expected scalar literal");
      }
      rewriter.replaceOpWithNewOp<cuda_tile::ConstantOp>(op, tileType, literal);
      return success();
    }

    // Handle splat constants (e.g. "constant dense<0.0> : tensor<16xf32>").
    // Use iteration space shape for the resulting tile type.
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
        denseAttr && denseAttr.isSplat()) {
      auto tileType =
          cast<ShapedType>(getTypeConverter()->convertType(op.getType()))
              .clone(state.getTileShape());
      auto literal = dyn_cast_or_null<DenseTypedElementsAttr>(
          fromSplatLiteral(tileType, denseAttr));
      if (!literal) {
        return op.emitError("expected splat literal");
      }
      rewriter.replaceOpWithNewOp<cuda_tile::ConstantOp>(op, tileType, literal);
      return success();
    }

    // Tensor-like constants are not supported.
    return op.emitError("unsupported constant value");
  }

private:
  /// Create constant literal from scalar attribute.
  DenseElementsAttr fromScalarLiteral(ShapedType type, Attribute value) const {
    if (auto floatAttr = dyn_cast<FloatAttr>(value)) {
      return DenseElementsAttr::get(type, floatAttr.getValue());
    }
    if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
      return DenseElementsAttr::get(type, intAttr.getValue());
    }
    return nullptr;
  }

  /// Create constant literal from dense attribute.
  DenseElementsAttr fromSplatLiteral(ShapedType type,
                                     DenseElementsAttr value) const {
    Type elementType = type.getElementType();
    if (elementType.isFloat()) {
      return DenseElementsAttr::get(type, value.getSplatValue<APFloat>());
    }
    if (elementType.isInteger()) {
      return DenseElementsAttr::get(type, value.getSplatValue<APInt>());
    }
    return nullptr;
  }
};

/// Converts `SplatOp` to `cuda_tile::ConstantOp`.
class SplatOpConversion : public ConversionPattern<SplatOp> {
public:
  using ConversionPattern<SplatOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    // Pre-rewrite legality: the splat input must be scalar.
    MLIR_RETURN_IF_ERROR(validatePointwiseOpLowerable(op));

    // The converted input is the tile form of the (scalar) splat input.
    auto inputType = cast<ShapedType>(adaptor.getInput().getType());

    // Use iteration space shape for the resulting tile type.
    assert(isa<cuda_tile::TileType>(inputType) && "expected tile type");
    ShapedType tileType = inputType.clone(state.getTileShape());

    // Handle splat of a constant.
    auto constantOp = dyn_cast_if_present<cuda_tile::ConstantOp>(
        adaptor.getInput().getDefiningOp());
    if (constantOp) {
      auto literal = cast<DenseTypedElementsAttr>(
          constantOp.getValue().resizeSplat(tileType));
      rewriter.replaceOpWithNewOp<cuda_tile::ConstantOp>(op, tileType, literal);

      // Remove the scalar constant if it is now trivially dead.
      if (mlir::isOpTriviallyDead(constantOp)) {
        rewriter.eraseOp(constantOp);
      }
      return success();
    }

    // Handle splat of an input (reshape + broadcast).
    SmallVector<int64_t> unitShape(tileType.getShape().size(), 1);
    ShapedType unitType = inputType.clone(unitShape);

    Value reshaped = cuda_tile::ReshapeOp::create(rewriter, op.getLoc(),
                                                  unitType, adaptor.getInput());
    rewriter.replaceOpWithNewOp<cuda_tile::BroadcastOp>(op, tileType, reshaped);
    return success();
  }
};
static bool isLowPrecisionFloat(Type type) {
  return type.isF16() || type.isBF16();
}

/// Build exp lowering matching XLA's CUDA policy:
/// - f32/f64 use the full libdevice-backed exp path.
/// - f16/bf16 are computed with f32 approximate exp and rounded back.
Value buildExp(OpBuilder &rewriter, Location loc, Value tile,
               Type resultElementType) {
  auto tileType = cast<ShapedType>(tile.getType());
  if (isLowPrecisionFloat(resultElementType)) {
    auto f32TileType =
        tileType.clone(tileType.getShape(), rewriter.getF32Type());
    Value f32Tile =
        cuda_tile::FToFOp::create(rewriter, loc, f32TileType, tile,
                                  cuda_tile::RoundingMode::NEAREST_EVEN);
    Value exp = cuda_tile::ExpOp::create(rewriter, loc, f32Tile,
                                         cuda_tile::RoundingMode::APPROX);
    return cuda_tile::FToFOp::create(rewriter, loc, tileType, exp,
                                     cuda_tile::RoundingMode::NEAREST_EVEN);
  }

  return cuda_tile::ExpOp::create(rewriter, loc, tile);
}

//===----------------------------------------------------------------------===//
// Pointwise operations
//===----------------------------------------------------------------------===//

/// Adapts affine-map operands to the shared pointwise emitter.
class PointwiseOpConversion
    : public OpInterfaceConversionPattern<PointwiseOpInterface> {
public:
  PointwiseOpConversion(ConversionState &state,
                        const TypeConverter &typeConverter, MLIRContext *ctx,
                        bool enableExperimentalCudaTileOps)
      : OpInterfaceConversionPattern(typeConverter, ctx), state(state),
        enableExperimentalCudaTileOps(enableExperimentalCudaTileOps) {}

  LogicalResult
  matchAndRewrite(PointwiseOpInterface pointwise,
                  ArrayRef<ValueRange> convertedOperands,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *op = pointwise.getOperation();
    if (isa<SplatOp>(op)) {
      return rewriter.notifyMatchFailure(
          op, "handled by the dedicated splat conversion");
    }
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    MLIR_RETURN_IF_ERROR(validatePointwiseOpLowerable(op));

    if (convertedOperands.size() != op->getNumOperands()) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }
    SmallVector<Value> operands;
    operands.reserve(convertedOperands.size());
    for (auto [source, converted] :
         llvm::zip_equal(op->getOperands(), convertedOperands)) {
      if (converted.size() != 1) {
        return rewriter.notifyMatchFailure(
            op, "expected one converted value per operand");
      }
      Value value = converted.front();
      operands.push_back(
          isa<TensorType>(source.getType()) ? state.getTile(value) : value);
    }

    auto convertedType = cast<cuda_tile::TileType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    auto resultType = cuda_tile::TileType::get(state.getTileShape(),
                                               convertedType.getElementType());
    return lowerPointwiseOp(pointwise, operands, resultType, rewriter,
                            enableExperimentalCudaTileOps);
  }

private:
  ConversionState &state;
  bool enableExperimentalCudaTileOps;
};

/// Sigmoid computation builder (common).
Value buildSigmoid(OpBuilder &rewriter, Value tile) {
  ShapedType tileType = cast<ShapedType>(tile.getType());
  Value oneTile = createConstant(rewriter, tile.getLoc(), tileType, 1.0);

  // Compute exp(-x)
  Value negX = cuda_tile::NegFOp::create(rewriter, tile.getLoc(), tile);
  Value expNegX =
      buildExp(rewriter, tile.getLoc(), negX, tileType.getElementType());

  // Compute 1 / (1 + exp(-x))
  auto roundingMode = cuda_tile::RoundingMode::NEAREST_EVEN;
  Value onePlusExp = cuda_tile::AddFOp::create(rewriter, tile.getLoc(), oneTile,
                                               expNegX, roundingMode);
  return cuda_tile::DivFOp::create(rewriter, tile.getLoc(), oneTile, onePlusExp,
                                   roundingMode);
}
//===----------------------------------------------------------------------===//
// Erf and GELU operations
//===----------------------------------------------------------------------===//

// Abramowitz and Stegun 7.1.26 coefficients for erf(x), maximum absolute
// error approximately 1.5e-7 for real x.
constexpr double kErfApproxP = 0.3275911;
constexpr double kErfApproxA1 = 0.254829592;
constexpr double kErfApproxA2 = -0.284496736;
constexpr double kErfApproxA3 = 1.421413741;
constexpr double kErfApproxA4 = -1.453152027;
constexpr double kErfApproxA5 = 1.061405429;

/// Approximate erf(x) with Abramowitz-Stegun 7.1.26 using CUDA Tile pointwise
/// ops:
///   erf(x) ~= sign(x) * (1 - poly(t) * exp(-abs(x)^2)),
/// where t = 1 / (1 + p * abs(x)).
Value buildErfApprox(OpBuilder &rewriter, Location loc, Value tile) {
  auto tileType = cast<ShapedType>(tile.getType());
  auto roundingMode = cuda_tile::RoundingMode::NEAREST_EVEN;

  Value zero = createConstant(rewriter, loc, tileType, 0.0);
  Value one = createConstant(rewriter, loc, tileType, 1.0);
  Value p = createConstant(rewriter, loc, tileType, kErfApproxP);
  Value a1 = createConstant(rewriter, loc, tileType, kErfApproxA1);
  Value a2 = createConstant(rewriter, loc, tileType, kErfApproxA2);
  Value a3 = createConstant(rewriter, loc, tileType, kErfApproxA3);
  Value a4 = createConstant(rewriter, loc, tileType, kErfApproxA4);
  Value a5 = createConstant(rewriter, loc, tileType, kErfApproxA5);

  Value isNonNegative = cuda_tile::CmpFOp::create(
      rewriter, loc, cuda_tile::ComparisonPredicate::GREATER_THAN_OR_EQUAL,
      cuda_tile::ComparisonOrdering::ORDERED, tile, zero);
  Value negTile = cuda_tile::NegFOp::create(rewriter, loc, tile);
  Value absX =
      cuda_tile::SelectOp::create(rewriter, loc, isNonNegative, tile, negTile);

  Value pTimesX =
      cuda_tile::MulFOp::create(rewriter, loc, p, absX, roundingMode);
  Value denominator =
      cuda_tile::AddFOp::create(rewriter, loc, one, pTimesX, roundingMode);
  Value t =
      cuda_tile::DivFOp::create(rewriter, loc, one, denominator, roundingMode);

  Value poly = a5;
  Value polyTimesT =
      cuda_tile::MulFOp::create(rewriter, loc, poly, t, roundingMode);
  poly = cuda_tile::AddFOp::create(rewriter, loc, polyTimesT, a4, roundingMode);
  polyTimesT = cuda_tile::MulFOp::create(rewriter, loc, poly, t, roundingMode);
  poly = cuda_tile::AddFOp::create(rewriter, loc, polyTimesT, a3, roundingMode);
  polyTimesT = cuda_tile::MulFOp::create(rewriter, loc, poly, t, roundingMode);
  poly = cuda_tile::AddFOp::create(rewriter, loc, polyTimesT, a2, roundingMode);
  polyTimesT = cuda_tile::MulFOp::create(rewriter, loc, poly, t, roundingMode);
  poly = cuda_tile::AddFOp::create(rewriter, loc, polyTimesT, a1, roundingMode);
  polyTimesT = cuda_tile::MulFOp::create(rewriter, loc, poly, t, roundingMode);

  Value xSquared =
      cuda_tile::MulFOp::create(rewriter, loc, absX, absX, roundingMode);
  Value negXSquared = cuda_tile::NegFOp::create(rewriter, loc, xSquared);
  Value expTerm =
      buildExp(rewriter, loc, negXSquared, tileType.getElementType());
  Value scaledExp = cuda_tile::MulFOp::create(rewriter, loc, polyTimesT,
                                              expTerm, roundingMode);
  Value positiveErf =
      cuda_tile::SubFOp::create(rewriter, loc, one, scaledExp, roundingMode);
  Value negativeErf = cuda_tile::NegFOp::create(rewriter, loc, positiveErf);
  return cuda_tile::SelectOp::create(rewriter, loc, isNonNegative, positiveErf,
                                     negativeErf);
}
namespace {

/// Whether a `ConvertOp` describes a type conversion the lowering can emit.
/// Mirrors the dispatch in `ConvertOpConversion::matchAndRewrite`: the only
/// unsupported combination is a conversion to/from a non-float, non-integer
/// element type.
bool isConvertSupported(ConvertOp op) {
  Type inputType = op.getInput().getType().getElementType();
  Type outputType = op.getType().getElementType();
  if (outputType.isInteger(1)) {
    // Conversion to boolean is supported from i1, float, or integer inputs.
    return inputType.isInteger(1) || inputType.isFloat() ||
           inputType.isInteger();
  }
  // All other conversions require both sides to be float or integer.
  return (inputType.isFloat() || inputType.isInteger()) &&
         (outputType.isFloat() || outputType.isInteger());
}

} // namespace

/// Shared per-op legality check for pointwise operations. This is the single
/// source of truth for the element-type guards that the pointwise conversion
/// patterns enforce in both the affine-map and outlined-kernel paths. Ops
/// without an element-type restriction trivially succeed.
LogicalResult validatePointwiseOpLowerable(Operation *op) {
  // Common predicate: result element type is float or integer.
  auto resultIsFloatOrInt = [](Operation *op) {
    auto elementType =
        cast<ShapedType>(op->getResult(0).getType()).getElementType();
    return elementType.isFloat() || elementType.isInteger();
  };

  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<AbsOp, AddOp, DivOp, ModOp, RemOp, MulOp, SubOp, MinOp, MaxOp,
            AddSquareOp>([&](auto typedOp) -> LogicalResult {
        if (!resultIsFloatOrInt(typedOp)) {
          return typedOp.emitError("unsupported element type");
        }
        return success();
      })
      .Case<LogicalNotOp>([&](LogicalNotOp typedOp) -> LogicalResult {
        if (!typedOp.getType().getElementType().isInteger(1)) {
          return typedOp.emitError("unsupported element type");
        }
        return success();
      })
      .Case<PowOp>([&](PowOp typedOp) -> LogicalResult {
        if (!typedOp.getType().getElementType().isFloat()) {
          return typedOp.emitError("unsupported element type");
        }
        return success();
      })
      .Case<CmpOp>([&](CmpOp typedOp) -> LogicalResult {
        Type elementType = typedOp.getLhs().getType().getElementType();
        if (!elementType.isFloat() && !elementType.isInteger()) {
          return typedOp.emitError("unsupported element type");
        }
        return success();
      })
      .Case<ConvertOp>([&](ConvertOp typedOp) -> LogicalResult {
        if (!isConvertSupported(typedOp)) {
          return typedOp.emitError("unsupported type conversion");
        }
        return success();
      })
      .Case<ErfOp, GeluFwdOp>([&](auto typedOp) -> LogicalResult {
        Type elementType = typedOp.getType().getElementType();
        if (!elementType.isF32() && !elementType.isF64()) {
          return typedOp.emitError("unsupported element type");
        }
        return success();
      })
      .Case<Atan2Op>([&](Atan2Op typedOp) -> LogicalResult {
        Type lhsElemType = typedOp.getLhs().getType().getElementType();
        Type rhsElemType = typedOp.getRhs().getType().getElementType();
        if (!lhsElemType.isFloat() || !rhsElemType.isFloat()) {
          return typedOp.emitError(
              "atan2 requires floating-point element types");
        }
        return success();
      })
      .Case<SplatOp>([&](SplatOp typedOp) -> LogicalResult {
        // The splat input must be a scalar (non-shaped) value. This mirrors the
        // 0-D-tile check the pattern performs on the converted operand: a valid
        // TensorIR splat carries a scalar input, which the type converter maps
        // to a 0-D tile.
        if (isa<ShapedType>(typedOp.getInput().getType())) {
          return typedOp.emitError("expected scalar input");
        }
        return success();
      })
      .Default([](Operation *) { return success(); });
}

/// Shared pre-rewrite legality check for `ConstantOp`. Single source of truth
/// for the "unsupported constant value" guard: a tensor-typed constant must
/// carry a splat literal. The scalar / splat literal *conversions* the pattern
/// performs depend on the converted tile type and iteration-space tile shape,
/// so they remain inline in `ConstantOpConversion::matchAndRewrite`.
LogicalResult validateConstantOpLowerable(ConstantOp op) {
  // Scalar constants are handled by the pattern's scalar literal path; their
  // legality depends on the converted tile type, so defer to the pattern.
  if (!isa<TensorType>(op.getType())) {
    return success();
  }
  // Tensor-typed constants must be splats; any other dense literal is
  // unsupported.
  auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
  if (!denseAttr || !denseAttr.isSplat()) {
    return op.emitError("unsupported constant value");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerPointwisePatterns(RewritePatternSet &patterns,
                               ConversionState &state,
                               const TypeConverter &typeConverter,
                               bool enableExperimentalCudaTileOps) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<ConstantOpConversion, SplatOpConversion>(state, typeConverter,
                                                        ctx);
  patterns.add<PointwiseOpConversion>(state, typeConverter, ctx,
                                      enableExperimentalCudaTileOps);
}

} // namespace tensor_to_cuda_tile
} // namespace mlir::nv_tensor_ir
