// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// \file PointwiseOps.cpp
/// Conversion patterns for pointwise TensorIR operations.
///
/// This module implements conversion patterns that lower pointwise operations
/// from the TensorIR dialect (inherited from `TensorIR_PointWiseOpBase`) to
/// their CudaTile counterparts. The patterns include:
/// - Splat operation and equivalent constants with dense literal value
/// - Unary operations (abs, exp, log, sin, cos, ceil, floor, sqrt, etc.)
/// - Binary operations (add, sub, mul, div, mod, min, max, pow, etc.)
/// - Comparison and binary select operations
/// - Type conversions (float-to-int, int-to-float, etc.)
/// - Activation functions (forward only at the moment)
///
/// Each pattern is inherited from `ConversionPattern` base class which
/// maintains the conversion state that allows converting the tensor shape to
/// the tile shape (depends on the iteration space of the op being converted).
///
/// The `ConversionState` interface provides the following methods:
/// - `update` initializes the conversion state for the current operation, must
///    be called in every `matchAndRewrite` implementation (may fail);
/// - `getTile` retrieves the operand's tile value (`cuda_tile::TileType`);
/// - `getTileShape` returns the tile shape for the current iteration space.
///
/// The patterns are registered via `registerPointwisePatterns`.

#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"
#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTileInternal.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Utils/Utils.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/TypeSwitch.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include <cmath>

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

//===----------------------------------------------------------------------===//
// Unary pointwise operations
//===----------------------------------------------------------------------===//

/// Conversion pattern template for unary pointwise Tensor IR ops to their
/// CudaTile counterparts. `TensorOpTy` is the tensor-dialect op (one tensor
/// operand); `TileOpTy` is the tile op with the same elementwise semantics.
template <typename TensorOpTy, typename TileOpTy>
class UnaryPointwiseOpConversion : public ConversionPattern<TensorOpTy> {
public:
  using Base = ConversionPattern<TensorOpTy>;
  using Base::ConversionPattern;

  LogicalResult
  matchAndRewrite(TensorOpTy op, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(this->state.update(rewriter, op));
    Value tile = this->state.getTile(adaptor.getInput());

    rewriter.replaceOpWithNewOp<TileOpTy>(op, tile);
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

/// Converts `ExpOp` using XLA-compatible precision policy for f16/bf16.
class ExpOpConversion : public ConversionPattern<ExpOp> {
public:
  using ConversionPattern<ExpOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(ExpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    Value tile = state.getTile(adaptor.getInput());
    Type elementType = op.getType().getElementType();

    rewriter.replaceOp(op, buildExp(rewriter, op.getLoc(), tile, elementType));
    return success();
  }
};

using CeilOpConversion = UnaryPointwiseOpConversion<CeilOp, cuda_tile::CeilOp>;
using CosOpConversion = UnaryPointwiseOpConversion<CosOp, cuda_tile::CosOp>;
using FloorOpConversion =
    UnaryPointwiseOpConversion<FloorOp, cuda_tile::FloorOp>;
using LogOpConversion = UnaryPointwiseOpConversion<LogOp, cuda_tile::LogOp>;
using NegOpConversion = UnaryPointwiseOpConversion<NegOp, cuda_tile::NegFOp>;
using RsqrtOpConversion =
    UnaryPointwiseOpConversion<RsqrtOp, cuda_tile::RsqrtOp>;
using SinOpConversion = UnaryPointwiseOpConversion<SinOp, cuda_tile::SinOp>;
using TanOpConversion = UnaryPointwiseOpConversion<TanOp, cuda_tile::TanOp>;
using TanhFwdOpConversion =
    UnaryPointwiseOpConversion<TanhFwdOp, cuda_tile::TanHOp>;

/// Converts `SqrtOp` to `cuda_tile::SqrtOp` with explicit `RoundingMode`.
class SqrtOpConversion : public ConversionPattern<SqrtOp> {
public:
  using ConversionPattern<SqrtOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(SqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    Value tile = state.getTile(adaptor.getInput());

    rewriter.replaceOpWithNewOp<cuda_tile::SqrtOp>(
        op, tile, cuda_tile::RoundingMode::NEAREST_EVEN);
    return success();
  }
};

/// Converts `ReciprocalOp` to `cuda_tile::DivFOp`.
class ReciprocalOpConversion : public ConversionPattern<ReciprocalOp> {
public:
  using ConversionPattern<ReciprocalOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(ReciprocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    Value tile = state.getTile(adaptor.getInput());

    Value one = createConstant(rewriter, op.getLoc(),
                               cast<ShapedType>(tile.getType()), 1.0);
    rewriter.replaceOpWithNewOp<cuda_tile::DivFOp>(
        op, one, tile, cuda_tile::RoundingMode::NEAREST_EVEN);
    return success();
  }
};

/// Converts `AbsOp` to `cuda_tile::AbsFOp` or `cuda_tile::AbsIOp`.
class AbsOpConversion : public ConversionPattern<AbsOp> {
public:
  using ConversionPattern<AbsOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(AbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    MLIR_RETURN_IF_ERROR(validatePointwiseOpLowerable(op));
    Value tile = state.getTile(adaptor.getInput());

    Type elementType = op.getType().getElementType();
    if (elementType.isFloat()) {
      rewriter.replaceOpWithNewOp<cuda_tile::AbsFOp>(op, tile);
    } else if (elementType.isSignedInteger()) {
      rewriter.replaceOpWithNewOp<cuda_tile::AbsIOp>(op, tile);
    } else {
      // Unsigned integer absolute value is the identity.
      rewriter.replaceOp(op, tile);
    }
    return success();
  }
};

/// Converts `LogicalNotOp` to `cuda_tile::XOrIOp`.
class LogicalNotOpConversion : public ConversionPattern<LogicalNotOp> {
public:
  using ConversionPattern<LogicalNotOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(LogicalNotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    MLIR_RETURN_IF_ERROR(validatePointwiseOpLowerable(op));
    Value tile = state.getTile(adaptor.getInput());

    Value one = createConstant(rewriter, op.getLoc(),
                               cast<ShapedType>(tile.getType()), int64_t{1});
    rewriter.replaceOpWithNewOp<cuda_tile::XOrIOp>(op, tile, one);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Binary pointwise operations
//===----------------------------------------------------------------------===//

/// Converts `AddOp` to `cuda_tile::AddFOp` or `cuda_tile::AddIOp`.
class AddOpConversion : public ConversionPattern<AddOp> {
public:
  using ConversionPattern<AddOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    MLIR_RETURN_IF_ERROR(validatePointwiseOpLowerable(op));
    Value lhs = state.getTile(adaptor.getLhs());
    Value rhs = state.getTile(adaptor.getRhs());

    Type elementType = op.getType().getElementType();
    if (elementType.isFloat()) {
      rewriter.replaceOpWithNewOp<cuda_tile::AddFOp>(
          op, lhs, rhs, cuda_tile::RoundingMode::NEAREST_EVEN);
    } else {
      rewriter.replaceOpWithNewOp<cuda_tile::AddIOp>(op, lhs, rhs);
    }
    return success();
  }
};

/// Converts `DivOp` to `cuda_tile::DivFOp` or `cuda_tile::DivIOp`.
class DivOpConversion : public ConversionPattern<DivOp> {
public:
  using ConversionPattern<DivOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    MLIR_RETURN_IF_ERROR(validatePointwiseOpLowerable(op));
    Value lhs = state.getTile(adaptor.getLhs());
    Value rhs = state.getTile(adaptor.getRhs());

    Type elementType = op.getType().getElementType();
    if (elementType.isFloat()) {
      if (isLowPrecisionFloat(elementType)) {
        // Match XLA's effective CUDA lowering for low-precision division:
        // compute the f32 divide at full precision, then round back. XLA
        // builds with -nvptx-prec-divf32=1, so it never emits div.approx.f32;
        // a splat-1.0 numerator is folded to rcp.approx.f32 during backend
        // lowering, which is bit-exact with div.full.f32.
        auto resultTileType = cast<ShapedType>(lhs.getType());
        auto f32TileType = resultTileType.clone(resultTileType.getShape(),
                                                rewriter.getF32Type());
        Value f32Lhs =
            cuda_tile::FToFOp::create(rewriter, op.getLoc(), f32TileType, lhs,
                                      cuda_tile::RoundingMode::NEAREST_EVEN);
        Value f32Rhs =
            cuda_tile::FToFOp::create(rewriter, op.getLoc(), f32TileType, rhs,
                                      cuda_tile::RoundingMode::NEAREST_EVEN);
        Value f32Div =
            cuda_tile::DivFOp::create(rewriter, op.getLoc(), f32Lhs, f32Rhs,
                                      cuda_tile::RoundingMode::FULL);
        rewriter.replaceOpWithNewOp<cuda_tile::FToFOp>(
            op, resultTileType, f32Div, cuda_tile::RoundingMode::NEAREST_EVEN);
        return success();
      }

      auto roundingMode = elementType.isF32()
                              ? cuda_tile::RoundingMode::FULL
                              : cuda_tile::RoundingMode::NEAREST_EVEN;
      rewriter.replaceOpWithNewOp<cuda_tile::DivFOp>(op, lhs, rhs,
                                                     roundingMode);
    } else {
      rewriter.replaceOpWithNewOp<cuda_tile::DivIOp>(
          op, lhs, rhs, getSignedness(elementType));
    }
    return success();
  }
};

/// Converts `ModOp` to a floored remainder `a - floor(a / b) * b`, built
/// from `cuda_tile::DivF/IOp` (with negative-infinity rounding for signed
/// integers), `cuda_tile::FloorOp`, `cuda_tile::MulF/IOp`, and
/// `cuda_tile::SubF/IOp`. The result has the sign of the divisor.
/// For the truncated counterpart see `RemOpConversion` below.
class ModOpConversion : public ConversionPattern<ModOp> {
public:
  using ConversionPattern<ModOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(ModOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    MLIR_RETURN_IF_ERROR(validatePointwiseOpLowerable(op));
    Value lhs = state.getTile(adaptor.getLhs());
    Value rhs = state.getTile(adaptor.getRhs());

    // `ModOp` semantics follow Python/PyTorch/JAX: a % b = a - floor(a / b) * b
    // This is different from C-style remainder (which `cuda_tile::RemFOp`
    // and `cuda_tile::RemIOp` implement).

    Type elementType = op.getType().getElementType();
    if (elementType.isFloat()) {
      // For floats: mod(a, b) = a - floor(a / b) * b
      auto roundingMode = cuda_tile::RoundingMode::NEAREST_EVEN;
      Value div = cuda_tile::DivFOp::create(rewriter, op.getLoc(), lhs, rhs,
                                            roundingMode);
      Value floor_div = cuda_tile::FloorOp::create(rewriter, op.getLoc(), div);
      Value mul = cuda_tile::MulFOp::create(rewriter, op.getLoc(), floor_div,
                                            rhs, roundingMode);
      rewriter.replaceOpWithNewOp<cuda_tile::SubFOp>(op, lhs, mul,
                                                     roundingMode);
    } else if (elementType.isInteger()) {
      // For integers: mod(a, b) = a - (a / b) * b
      // For signed integers: round towards negative infinity
      // For unsigned integers: round towards zero
      auto roundingMode = elementType.isSignedInteger()
                              ? cuda_tile::RoundingMode::NEGATIVE_INF
                              : cuda_tile::RoundingMode::ZERO;
      Value floor_div =
          cuda_tile::DivIOp::create(rewriter, op.getLoc(), lhs, rhs,
                                    getSignedness(elementType), roundingMode);
      Value mul =
          cuda_tile::MulIOp::create(rewriter, op.getLoc(), floor_div, rhs);
      rewriter.replaceOpWithNewOp<cuda_tile::SubIOp>(op, lhs, mul);
    }
    return success();
  }
};

/// Converts `RemOp` to `cuda_tile::RemFOp` or `cuda_tile::RemIOp`.
///
/// `RemOp` is the truncated remainder: the result has the sign of the
/// dividend. This is the direct counterpart of `cuda_tile::RemFOp` and
/// `cuda_tile::RemIOp`, and is distinct from `ModOp`, whose lowering
/// computes a floored remainder (result has the sign of the divisor).
class RemOpConversion : public ConversionPattern<RemOp> {
public:
  using ConversionPattern<RemOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(RemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    MLIR_RETURN_IF_ERROR(validatePointwiseOpLowerable(op));
    Value lhs = state.getTile(adaptor.getLhs());
    Value rhs = state.getTile(adaptor.getRhs());

    Type elementType = op.getType().getElementType();
    if (elementType.isFloat()) {
      rewriter.replaceOpWithNewOp<cuda_tile::RemFOp>(op, lhs, rhs);
    } else {
      rewriter.replaceOpWithNewOp<cuda_tile::RemIOp>(
          op, lhs, rhs, getSignedness(elementType));
    }
    return success();
  }
};

/// Converts `MulOp` to `cuda_tile::MulFOp` or `cuda_tile::MulIOp`.
class MulOpConversion : public ConversionPattern<MulOp> {
public:
  using ConversionPattern<MulOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    MLIR_RETURN_IF_ERROR(validatePointwiseOpLowerable(op));
    Value lhs = state.getTile(adaptor.getLhs());
    Value rhs = state.getTile(adaptor.getRhs());

    Type elementType = op.getType().getElementType();
    if (elementType.isFloat()) {
      rewriter.replaceOpWithNewOp<cuda_tile::MulFOp>(
          op, lhs, rhs, cuda_tile::RoundingMode::NEAREST_EVEN);
    } else {
      rewriter.replaceOpWithNewOp<cuda_tile::MulIOp>(op, lhs, rhs);
    }
    return success();
  }
};

/// Converts `SubOp` to `cuda_tile::SubFOp` or `cuda_tile::SubIOp`.
class SubOpConversion : public ConversionPattern<SubOp> {
public:
  using ConversionPattern<SubOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    MLIR_RETURN_IF_ERROR(validatePointwiseOpLowerable(op));
    Value lhs = state.getTile(adaptor.getLhs());
    Value rhs = state.getTile(adaptor.getRhs());

    Type elementType = op.getType().getElementType();
    if (elementType.isFloat()) {
      rewriter.replaceOpWithNewOp<cuda_tile::SubFOp>(
          op, lhs, rhs, cuda_tile::RoundingMode::NEAREST_EVEN);
    } else {
      rewriter.replaceOpWithNewOp<cuda_tile::SubIOp>(op, lhs, rhs);
    }
    return success();
  }
};

/// Converts `MinOp` to `cuda_tile::MinFOp` or `cuda_tile::MinIOp`.
class MinOpConversion : public ConversionPattern<MinOp> {
public:
  using ConversionPattern<MinOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(MinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    MLIR_RETURN_IF_ERROR(validatePointwiseOpLowerable(op));
    Value lhs = state.getTile(adaptor.getLhs());
    Value rhs = state.getTile(adaptor.getRhs());

    Type elementType = op.getType().getElementType();
    if (elementType.isFloat()) {
      rewriter.replaceOpWithNewOp<cuda_tile::MinFOp>(op, lhs, rhs,
                                                     /*propagate_nan=*/true);
    } else {
      rewriter.replaceOpWithNewOp<cuda_tile::MinIOp>(
          op, lhs, rhs, getSignedness(elementType));
    }
    return success();
  }
};

/// Converts `MaxOp` to `cuda_tile::MaxFOp` or `cuda_tile::MaxIOp`.
class MaxOpConversion : public ConversionPattern<MaxOp> {
public:
  using ConversionPattern<MaxOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(MaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    MLIR_RETURN_IF_ERROR(validatePointwiseOpLowerable(op));
    Value lhs = state.getTile(adaptor.getLhs());
    Value rhs = state.getTile(adaptor.getRhs());

    Type elementType = op.getType().getElementType();
    if (elementType.isFloat()) {
      rewriter.replaceOpWithNewOp<cuda_tile::MaxFOp>(op, lhs, rhs,
                                                     /*propagate_nan=*/true);
    } else {
      rewriter.replaceOpWithNewOp<cuda_tile::MaxIOp>(
          op, lhs, rhs, getSignedness(elementType));
    }
    return success();
  }
};

/// Converts `PowOp` to `cuda_tile::PowOp` (floating-point).
/// Integer inputs are not supported currently.
class PowOpConversion : public ConversionPattern<PowOp> {
public:
  using ConversionPattern<PowOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(PowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    MLIR_RETURN_IF_ERROR(validatePointwiseOpLowerable(op));
    Value lhs = state.getTile(adaptor.getLhs());
    Value rhs = state.getTile(adaptor.getRhs());

    Type exponentType = op.getRhs().getType().getElementType();
    if (exponentType.isInteger()) {
      // Convert integer exponent to floating-point.
      rhs = cuda_tile::IToFOp::create(rewriter, op.getLoc(), lhs.getType(), rhs,
                                      getSignedness(exponentType),
                                      cuda_tile::RoundingMode::NEAREST_EVEN);
    }
    rewriter.replaceOpWithNewOp<cuda_tile::PowOp>(op, lhs, rhs);
    return success();
  }
};

class Atan2OpConversion : public ConversionPattern<Atan2Op> {
public:
  using ConversionPattern<Atan2Op>::ConversionPattern;

  LogicalResult
  matchAndRewrite(Atan2Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    MLIR_RETURN_IF_ERROR(validatePointwiseOpLowerable(op));

    Value lhs = state.getTile(adaptor.getLhs());
    Value rhs = state.getTile(adaptor.getRhs());
    rewriter.replaceOpWithNewOp<cuda_tile::Atan2Op>(op, lhs, rhs);
    return success();
  }
};

/// Converts `AddSquareOp` (lhs + rhs²).
class AddSquareOpConversion : public ConversionPattern<AddSquareOp> {
public:
  using ConversionPattern<AddSquareOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(AddSquareOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    MLIR_RETURN_IF_ERROR(validatePointwiseOpLowerable(op));
    Value lhs = state.getTile(adaptor.getLhs());
    Value rhs = state.getTile(adaptor.getRhs());

    Type elementType = op.getType().getElementType();
    if (elementType.isFloat()) {
      rhs = cuda_tile::MulFOp::create(rewriter, op.getLoc(), rhs, rhs,
                                      cuda_tile::RoundingMode::NEAREST_EVEN);
      rewriter.replaceOpWithNewOp<cuda_tile::AddFOp>(
          op, lhs, rhs, cuda_tile::RoundingMode::NEAREST_EVEN);
    } else {
      rhs = cuda_tile::MulIOp::create(rewriter, op.getLoc(), rhs, rhs);
      rewriter.replaceOpWithNewOp<cuda_tile::AddIOp>(op, lhs, rhs);
    }
    return success();
  }
};

/// Converts `LogicalAndOp` to `cuda_tile::AndIOp`.
class LogicalAndOpConversion : public ConversionPattern<LogicalAndOp> {
public:
  using ConversionPattern<LogicalAndOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(LogicalAndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    Value lhs = state.getTile(adaptor.getLhs());
    Value rhs = state.getTile(adaptor.getRhs());

    rewriter.replaceOpWithNewOp<cuda_tile::AndIOp>(op, lhs, rhs);
    return success();
  }
};

/// Converts `LogicalOrOp` to `cuda_tile::OrIOp`.
class LogicalOrOpConversion : public ConversionPattern<LogicalOrOp> {
public:
  using ConversionPattern<LogicalOrOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(LogicalOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    Value lhs = state.getTile(adaptor.getLhs());
    Value rhs = state.getTile(adaptor.getRhs());

    rewriter.replaceOpWithNewOp<cuda_tile::OrIOp>(op, lhs, rhs);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Miscellaneous operations
//===----------------------------------------------------------------------===//

/// Converts `BinarySelectOp` to `cuda_tile::SelectOp`.
class BinarySelectOpConversion : public ConversionPattern<BinarySelectOp> {
public:
  using ConversionPattern<BinarySelectOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(BinarySelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    Value selector = state.getTile(adaptor.getSelector());
    Value lhs = state.getTile(adaptor.getLhs());
    Value rhs = state.getTile(adaptor.getRhs());

    rewriter.replaceOpWithNewOp<cuda_tile::SelectOp>(op, selector, lhs, rhs);
    return success();
  }
};

/// Converts `CmpOp` to `cuda_tile::CmpFOp` or `cuda_tile::CmpIOp`.
class CmpOpConversion : public ConversionPattern<CmpOp> {
public:
  using ConversionPattern<CmpOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(CmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    MLIR_RETURN_IF_ERROR(validatePointwiseOpLowerable(op));
    Value lhs = state.getTile(adaptor.getLhs());
    Value rhs = state.getTile(adaptor.getRhs());
    Comparator comparator = op.getComparator();

    Type elementType = op.getLhs().getType().getElementType();
    if (elementType.isFloat()) {
      rewriter.replaceOpWithNewOp<cuda_tile::CmpFOp>(
          op, getPredicate(comparator), getOrdering(comparator), lhs, rhs);
    } else {
      rewriter.replaceOpWithNewOp<cuda_tile::CmpIOp>(
          op, getPredicate(comparator), lhs, rhs, getSignedness(elementType));
    }
    return success();
  }

private:
  cuda_tile::ComparisonPredicate getPredicate(Comparator comparator) const {
    switch (comparator) {
    case Comparator::eq:
    case Comparator::oeq:
    case Comparator::ueq:
      return cuda_tile::ComparisonPredicate::EQUAL;
    case Comparator::neq:
    case Comparator::one:
    case Comparator::une:
      return cuda_tile::ComparisonPredicate::NOT_EQUAL;
    case Comparator::lt:
    case Comparator::olt:
    case Comparator::ult:
      return cuda_tile::ComparisonPredicate::LESS_THAN;
    case Comparator::le:
    case Comparator::ole:
    case Comparator::ule:
      return cuda_tile::ComparisonPredicate::LESS_THAN_OR_EQUAL;
    case Comparator::gt:
    case Comparator::ogt:
    case Comparator::ugt:
      return cuda_tile::ComparisonPredicate::GREATER_THAN;
    case Comparator::ge:
    case Comparator::oge:
    case Comparator::uge:
      return cuda_tile::ComparisonPredicate::GREATER_THAN_OR_EQUAL;
    default:
      llvm_unreachable("unsupported comparator");
    }
  }

  cuda_tile::ComparisonOrdering getOrdering(Comparator comparator) const {
    if (comparator == Comparator::ueq || comparator == Comparator::une ||
        comparator == Comparator::ult || comparator == Comparator::ule ||
        comparator == Comparator::ugt || comparator == Comparator::uge) {
      return cuda_tile::ComparisonOrdering::UNORDERED;
    }
    return cuda_tile::ComparisonOrdering::ORDERED;
  }
};

/// Implements type conversion (F2F, F2I, I2F, I2I).
/// Conversion to boolean type (i1) is implemented as a comparison against zero:
/// non-zero values become `1` and zero values become `0`.
class ConvertOpConversion : public ConversionPattern<ConvertOp> {
public:
  using ConversionPattern<ConvertOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(ConvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    MLIR_RETURN_IF_ERROR(validatePointwiseOpLowerable(op));
    Value tile = state.getTile(adaptor.getInput());

    Type inputType = op.getInput().getType().getElementType();
    Type outputType = op.getType().getElementType();
    auto tileShapedType = cast<ShapedType>(tile.getType());
    Type targetType =
        cast<ShapedType>(getTypeConverter()->convertType(outputType))
            .clone(tileShapedType.getShape());

    if (outputType.isInteger(1)) {
      // Conversion to boolean: emit a not-equal-to-zero comparison.
      if (inputType.isInteger(1)) {
        // i1-to-i1 is a no-op.
        rewriter.replaceOp(op, tile);
      } else if (inputType.isFloat()) {
        Value zero = createConstant(rewriter, op.getLoc(), tileShapedType, 0.0);
        rewriter.replaceOpWithNewOp<cuda_tile::CmpFOp>(
            op, cuda_tile::ComparisonPredicate::NOT_EQUAL,
            cuda_tile::ComparisonOrdering::ORDERED, tile, zero);
      } else if (inputType.isInteger()) {
        Value zero = createConstant(rewriter, op.getLoc(), tileShapedType,
                                    static_cast<int64_t>(0));
        rewriter.replaceOpWithNewOp<cuda_tile::CmpIOp>(
            op, cuda_tile::ComparisonPredicate::NOT_EQUAL, tile, zero,
            getSignedness(inputType));
      } else {
        return op.emitError("unsupported type conversion");
      }
    } else if (inputType.isFloat() && outputType.isFloat()) {
      // Float-to-float conversion.
      rewriter.replaceOpWithNewOp<cuda_tile::FToFOp>(
          op, targetType, tile, cuda_tile::RoundingMode::NEAREST_EVEN);
    } else if (inputType.isFloat() && outputType.isInteger()) {
      // Float-to-integer conversion.
      rewriter.replaceOpWithNewOp<cuda_tile::FToIOp>(
          op, targetType, tile, getSignedness(outputType),
          cuda_tile::RoundingMode::NEAREST_INT_TO_ZERO);
    } else if (inputType.isInteger() && outputType.isFloat()) {
      // Integer-to-float conversion.
      rewriter.replaceOpWithNewOp<cuda_tile::IToFOp>(
          op, targetType, tile, getSignedness(inputType),
          cuda_tile::RoundingMode::NEAREST_EVEN);
    } else if (inputType.isInteger() && outputType.isInteger()) {
      // Integer-to-integer conversion.
      int inputWidth = inputType.getIntOrFloatBitWidth();
      int outputWidth = outputType.getIntOrFloatBitWidth();
      if (inputWidth < outputWidth) {
        // Extend integer width.
        rewriter.replaceOpWithNewOp<cuda_tile::ExtIOp>(
            op, targetType, tile, getSignedness(inputType));
      } else if (inputWidth > outputWidth) {
        // Truncate integer width.
        rewriter.replaceOpWithNewOp<cuda_tile::TruncIOp>(op, targetType, tile);
      } else {
        // Same bit width, no-op.
        rewriter.replaceOp(op, tile);
      }
    } else {
      return op.emitError("unsupported type conversion");
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Activation functions (forward)
//===----------------------------------------------------------------------===//

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

/// Computes RELU(x) = max(0, x)
class ReluFwdOpConversion : public ConversionPattern<ReluFwdOp> {
public:
  using ConversionPattern<ReluFwdOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(ReluFwdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    Value tile = state.getTile(adaptor.getInput());

    Value zero = createConstant(rewriter, op.getLoc(),
                                cast<ShapedType>(tile.getType()), 0.0);
    rewriter.replaceOpWithNewOp<cuda_tile::MaxFOp>(op, zero, tile,
                                                   /*propagate_nan=*/true);
    return success();
  }
};

/// Computes SIGMOID(x) = 1 / (1 + exp(-x))
class SigmoidFwdOpConversion : public ConversionPattern<SigmoidFwdOp> {
public:
  using ConversionPattern<SigmoidFwdOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(SigmoidFwdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    Value tile = state.getTile(adaptor.getInput());

    rewriter.replaceOp(op, buildSigmoid(rewriter, tile));
    return success();
  }
};

/// Computes GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
class GeluApproxTanhFwdOpConversion
    : public ConversionPattern<GeluApproxTanhFwdOp> {
public:
  using ConversionPattern<GeluApproxTanhFwdOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(GeluApproxTanhFwdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    Value tile = state.getTile(adaptor.getInput());

    // Create constants.
    ShapedType tileType = cast<ShapedType>(tile.getType());
    Value sqrt2OverPiTile = createConstant(rewriter, op.getLoc(), tileType,
                                           std::sqrt(2.0 / llvm::numbers::pi));
    Value coeffTile = createConstant(rewriter, op.getLoc(), tileType, 0.044715);
    Value oneTile = createConstant(rewriter, op.getLoc(), tileType, 1.0);
    Value halfTile = createConstant(rewriter, op.getLoc(), tileType, 0.5);

    // Compute x + 0.044715 * x^3
    auto roundingMode = cuda_tile::RoundingMode::NEAREST_EVEN;
    Value xSquare = cuda_tile::MulFOp::create(rewriter, op.getLoc(), tile, tile,
                                              roundingMode);
    Value xCube = cuda_tile::MulFOp::create(rewriter, op.getLoc(), xSquare,
                                            tile, roundingMode);
    Value coeffMulXCube = cuda_tile::MulFOp::create(
        rewriter, op.getLoc(), coeffTile, xCube, roundingMode);
    Value tanhInputPart = cuda_tile::AddFOp::create(
        rewriter, op.getLoc(), tile, coeffMulXCube, roundingMode);

    // Compute 1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))
    Value tanhInput = cuda_tile::MulFOp::create(
        rewriter, op.getLoc(), sqrt2OverPiTile, tanhInputPart, roundingMode);
    Value tanhValue =
        cuda_tile::TanHOp::create(rewriter, op.getLoc(), tanhInput);
    Value onePlusTanh = cuda_tile::AddFOp::create(
        rewriter, op.getLoc(), oneTile, tanhValue, roundingMode);

    // Compute 0.5 * x * (1 + tanh(...))
    Value xHalved = cuda_tile::MulFOp::create(rewriter, op.getLoc(), tile,
                                              halfTile, roundingMode);
    rewriter.replaceOpWithNewOp<cuda_tile::MulFOp>(op, xHalved, onePlusTanh,
                                                   roundingMode);
    return success();
  }
};

/// Computes SOFTPLUS(x) = 1/β * log(1 + exp(β * x))
class SoftplusFwdOpConversion : public ConversionPattern<SoftplusFwdOp> {
public:
  using ConversionPattern<SoftplusFwdOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(SoftplusFwdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    Value tile = state.getTile(adaptor.getX());
    double beta = adaptor.getBeta().convertToDouble();

    // Create constants.
    ShapedType tileType = cast<ShapedType>(tile.getType());
    Value oneTile = createConstant(rewriter, op.getLoc(), tileType, 1.0);
    Value betaTile = createConstant(rewriter, op.getLoc(), tileType, beta);
    Value oneOverBetaTile =
        createConstant(rewriter, op.getLoc(), tileType, 1.0 / beta);

    // Compute 1 + exp(β * x)
    auto roundingMode = cuda_tile::RoundingMode::NEAREST_EVEN;
    Value xScaled = cuda_tile::MulFOp::create(rewriter, op.getLoc(), tile,
                                              betaTile, roundingMode);
    Value expValue =
        buildExp(rewriter, op.getLoc(), xScaled, tileType.getElementType());
    Value onePlusExp = cuda_tile::AddFOp::create(rewriter, op.getLoc(), oneTile,
                                                 expValue, roundingMode);

    // Compute 1/β * log(...)
    Value logValue =
        cuda_tile::LogOp::create(rewriter, op.getLoc(), onePlusExp);
    rewriter.replaceOpWithNewOp<cuda_tile::MulFOp>(op, oneOverBetaTile,
                                                   logValue, roundingMode);
    return success();
  }
};

/// Computes SWISH(x) = x / (1 + exp(-β * x))
class SwishFwdOpConversion : public ConversionPattern<SwishFwdOp> {
public:
  using ConversionPattern<SwishFwdOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(SwishFwdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    Value tile = state.getTile(adaptor.getX());
    double beta = adaptor.getBeta().convertToDouble();

    // Compute sigmoid(β * x)
    auto roundingMode = cuda_tile::RoundingMode::NEAREST_EVEN;
    Value betaTile = createConstant(rewriter, op.getLoc(),
                                    cast<ShapedType>(tile.getType()), beta);
    Value xScaled = cuda_tile::MulFOp::create(rewriter, op.getLoc(), tile,
                                              betaTile, roundingMode);
    Value sigmoid = buildSigmoid(rewriter, xScaled);

    // Compute x * sigmoid(...)
    rewriter.replaceOpWithNewOp<cuda_tile::MulFOp>(op, tile, sigmoid,
                                                   roundingMode);
    return success();
  }
};

/// Computes ELU(x) = {x if x > 0; β * (exp(x) - 1) if x ≤ 0}
class EluFwdOpConversion : public ConversionPattern<EluFwdOp> {
public:
  using ConversionPattern<EluFwdOp>::ConversionPattern;

  LogicalResult
  matchAndRewrite(EluFwdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    Value tile = state.getTile(adaptor.getX());
    double beta = adaptor.getBeta().convertToDouble();

    // Create constants.
    ShapedType tileType = cast<ShapedType>(tile.getType());
    Value zeroTile = createConstant(rewriter, op.getLoc(), tileType, 0.0);
    Value oneTile = createConstant(rewriter, op.getLoc(), tileType, 1.0);
    Value betaTile = createConstant(rewriter, op.getLoc(), tileType, beta);

    // Compute β * (exp(x) - 1)
    auto roundingMode = cuda_tile::RoundingMode::NEAREST_EVEN;
    Value expValue =
        buildExp(rewriter, op.getLoc(), tile, tileType.getElementType());
    Value expMinusOne = cuda_tile::SubFOp::create(
        rewriter, op.getLoc(), expValue, oneTile, roundingMode);
    Value expMinusOneScaled = cuda_tile::MulFOp::create(
        rewriter, op.getLoc(), expMinusOne, betaTile, roundingMode);

    // Select value based on the sign.
    Value pred = cuda_tile::CmpFOp::create(
        rewriter, op.getLoc(), cuda_tile::ComparisonPredicate::GREATER_THAN,
        cuda_tile::ComparisonOrdering::ORDERED, tile, zeroTile);
    rewriter.replaceOpWithNewOp<cuda_tile::SelectOp>(op, pred, tile,
                                                     expMinusOneScaled);
    return success();
  }
};

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

/// Converts `ErfOp`.
class ErfOpConversion : public ConversionPattern<ErfOp> {
public:
  ErfOpConversion(ConversionState &state, const TypeConverter &typeConverter,
                  MLIRContext *ctx, bool enableExperimentalCudaTileOps)
      : ConversionPattern<ErfOp>(state, typeConverter, ctx),
        enableExperimentalCudaTileOps(enableExperimentalCudaTileOps) {}

  LogicalResult
  matchAndRewrite(ErfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    MLIR_RETURN_IF_ERROR(validatePointwiseOpLowerable(op));
    Value tile = state.getTile(adaptor.getInput());

    if (enableExperimentalCudaTileOps) {
    }

    rewriter.replaceOp(op, buildErfApprox(rewriter, op.getLoc(), tile));
    return success();
  }

private:
  bool enableExperimentalCudaTileOps;
};

/// Computes GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))).
class GeluFwdOpConversion : public ConversionPattern<GeluFwdOp> {
public:
  GeluFwdOpConversion(ConversionState &state,
                      const TypeConverter &typeConverter, MLIRContext *ctx,
                      bool enableExperimentalCudaTileOps)
      : ConversionPattern<GeluFwdOp>(state, typeConverter, ctx),
        enableExperimentalCudaTileOps(enableExperimentalCudaTileOps) {}

  LogicalResult
  matchAndRewrite(GeluFwdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIR_RETURN_IF_ERROR(state.update(rewriter, op));
    MLIR_RETURN_IF_ERROR(validatePointwiseOpLowerable(op));
    Value tile = state.getTile(adaptor.getInput());

    Location loc = op.getLoc();
    ShapedType tileType = cast<ShapedType>(tile.getType());
    Value sqrt2Tile = createConstant(rewriter, loc, tileType, std::sqrt(2.0));
    Value oneTile = createConstant(rewriter, loc, tileType, 1.0);
    Value halfTile = createConstant(rewriter, loc, tileType, 0.5);

    auto roundingMode = cuda_tile::RoundingMode::NEAREST_EVEN;
    Value xOverSqrt2 =
        cuda_tile::DivFOp::create(rewriter, loc, tile, sqrt2Tile, roundingMode);
    Value erfValue;
    if (enableExperimentalCudaTileOps) {
    }
    if (!erfValue) {
      erfValue = buildErfApprox(rewriter, loc, xOverSqrt2);
    }
    Value onePlusErf = cuda_tile::AddFOp::create(rewriter, loc, oneTile,
                                                 erfValue, roundingMode);

    Value xHalved =
        cuda_tile::MulFOp::create(rewriter, loc, tile, halfTile, roundingMode);
    Value gelu = cuda_tile::MulFOp::create(rewriter, loc, xHalved, onePlusErf,
                                           roundingMode);
    rewriter.replaceOp(op, gelu);
    return success();
  }

private:
  bool enableExperimentalCudaTileOps;
};

//===----------------------------------------------------------------------===//
// Per-op legality
//===----------------------------------------------------------------------===//

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
/// patterns enforce: it is invoked both by those patterns (right after the
/// per-op state update) and by the pre-flight feasibility check, so the two
/// cannot diverge. Ops without an element-type restriction trivially succeed.
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
  patterns.add<
      // Operations with non-tensor inputs
      ConstantOpConversion, SplatOpConversion,
      // Unary pointwise operations
      CeilOpConversion, CosOpConversion, ExpOpConversion, FloorOpConversion,
      LogOpConversion, NegOpConversion, RsqrtOpConversion, SinOpConversion,
      TanOpConversion, TanhFwdOpConversion, SqrtOpConversion,
      ReciprocalOpConversion, AbsOpConversion, LogicalNotOpConversion,
      Atan2OpConversion,
      // Binary pointwise operations
      AddOpConversion, DivOpConversion, ModOpConversion, RemOpConversion,
      MulOpConversion, SubOpConversion, MinOpConversion, MaxOpConversion,
      PowOpConversion, AddSquareOpConversion, LogicalAndOpConversion,
      LogicalOrOpConversion,
      // Miscellaneous operations
      BinarySelectOpConversion, CmpOpConversion, ConvertOpConversion,
      // Activation functions
      ReluFwdOpConversion, SigmoidFwdOpConversion,
      GeluApproxTanhFwdOpConversion, SoftplusFwdOpConversion,
      SwishFwdOpConversion, EluFwdOpConversion>(state, typeConverter, ctx);
  patterns.add<ErfOpConversion, GeluFwdOpConversion>(
      state, typeConverter, ctx, enableExperimentalCudaTileOps);
}

} // namespace tensor_to_cuda_tile
} // namespace mlir::nv_tensor_ir
