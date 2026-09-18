// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- OutlinedComputeOps.cpp - Lower explicit TensorIR tiles ------------===//
//
// Stateless conversions for compute in an outlined tiled program. The tile
// shape is part of each ranked-tensor result type and every input tile arrives
// through the conversion adaptor. No layout-propagation attribute or mutable
// conversion state is consulted here.
//
//===----------------------------------------------------------------------===//

#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTileInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include <cmath>
#include <type_traits>

namespace mlir::nv_tensor_ir::tensor_to_cuda_tile {
namespace {

static FailureOr<cuda_tile::TileType>
getResultType(Operation *op, const TypeConverter &typeConverter) {
  if (op->getNumResults() != 1) {
    return failure();
  }
  auto type = typeConverter.convertType<cuda_tile::TileType>(
      op->getResult(0).getType());
  if (!type || !type.hasStaticShape()) {
    return failure();
  }
  return type;
}

static bool isLowPrecisionFloat(Type type) {
  return type.isF16() || type.isBF16();
}

static LogicalResult lowerFloatingDiv(Operation *op, Value lhs, Value rhs,
                                      Type elementType,
                                      ConversionPatternRewriter &rewriter) {
  if (isLowPrecisionFloat(elementType)) {
    auto sourceType = cast<cuda_tile::TileType>(lhs.getType());
    auto f32Type = sourceType.clone(rewriter.getF32Type());
    Value convertedLhs =
        cuda_tile::FToFOp::create(rewriter, op->getLoc(), f32Type, lhs,
                                  cuda_tile::RoundingMode::NEAREST_EVEN);
    Value convertedRhs =
        cuda_tile::FToFOp::create(rewriter, op->getLoc(), f32Type, rhs,
                                  cuda_tile::RoundingMode::NEAREST_EVEN);
    Value quotient =
        cuda_tile::DivFOp::create(rewriter, op->getLoc(), convertedLhs,
                                  convertedRhs, cuda_tile::RoundingMode::FULL);
    rewriter.replaceOpWithNewOp<cuda_tile::FToFOp>(
        op, sourceType, quotient, cuda_tile::RoundingMode::NEAREST_EVEN);
    return success();
  }
  auto rounding = elementType.isF32() ? cuda_tile::RoundingMode::FULL
                                      : cuda_tile::RoundingMode::NEAREST_EVEN;
  rewriter.replaceOpWithNewOp<cuda_tile::DivFOp>(op, lhs, rhs, rounding);
  return success();
}

template <typename CudaOp>
static LogicalResult replaceUnary(Operation *op, Value input,
                                  ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<CudaOp>(op, input);
  return success();
}

static cuda_tile::ComparisonPredicate getPredicate(Comparator comparator) {
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
  }
  llvm_unreachable("unsupported comparator");
}

static cuda_tile::ComparisonOrdering getOrdering(Comparator comparator) {
  switch (comparator) {
  case Comparator::ueq:
  case Comparator::une:
  case Comparator::ult:
  case Comparator::ule:
  case Comparator::ugt:
  case Comparator::uge:
    return cuda_tile::ComparisonOrdering::UNORDERED;
  default:
    return cuda_tile::ComparisonOrdering::ORDERED;
  }
}

class TensorIRConstantConversion : public OpConversionPattern<ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(validateConstantOpLowerable(op))) {
      return failure();
    }
    FailureOr<cuda_tile::TileType> resultType =
        getResultType(op, *typeConverter);
    if (failed(resultType)) {
      return op.emitError("staged constants require a static tile result");
    }

    Attribute value = op.getValue();
    DenseTypedElementsAttr literal;
    if (auto dense = dyn_cast<DenseElementsAttr>(value)) {
      if (!dense.isSplat()) {
        return op.emitError("unsupported constant value");
      }
      if ((*resultType).getElementType().isFloat()) {
        literal = cast<DenseTypedElementsAttr>(DenseElementsAttr::get(
            *resultType, dense.getSplatValue<APFloat>()));
      } else if ((*resultType).getElementType().isInteger()) {
        literal = cast<DenseTypedElementsAttr>(
            DenseElementsAttr::get(*resultType, dense.getSplatValue<APInt>()));
      }
    } else if (auto floatValue = dyn_cast<FloatAttr>(value)) {
      literal = cast<DenseTypedElementsAttr>(
          DenseElementsAttr::get(*resultType, floatValue.getValue()));
    } else if (auto integerValue = dyn_cast<IntegerAttr>(value)) {
      literal = cast<DenseTypedElementsAttr>(
          DenseElementsAttr::get(*resultType, integerValue.getValue()));
    }
    if (!literal) {
      return op.emitError("expected integer or floating-point literal");
    }
    rewriter.replaceOpWithNewOp<cuda_tile::ConstantOp>(op, *resultType,
                                                       literal);
    return success();
  }
};

class PointwiseEmitter {
public:
  static LogicalResult lower(PointwiseOpInterface pointwise,
                             ArrayRef<Value> operands,
                             cuda_tile::TileType resultType,
                             ConversionPatternRewriter &rewriter,
                             bool enableExperimentalCudaTileOps) {
    Operation *op = pointwise.getOperation();
    if (operands.empty()) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }

    Type resultElementType =
        cast<ShapedType>(op->getResult(0).getType()).getElementType();
    Location loc = op->getLoc();
    Value input = operands[0];

    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<CeilOp>([&](auto) {
          return replaceUnary<cuda_tile::CeilOp>(op, input, rewriter);
        })
        .Case<CosOp>([&](auto) {
          return replaceUnary<cuda_tile::CosOp>(op, input, rewriter);
        })
        .Case<FloorOp>([&](auto) {
          return replaceUnary<cuda_tile::FloorOp>(op, input, rewriter);
        })
        .Case<LogOp>([&](auto) {
          return replaceUnary<cuda_tile::LogOp>(op, input, rewriter);
        })
        .Case<NegOp>([&](auto) {
          return replaceUnary<cuda_tile::NegFOp>(op, input, rewriter);
        })
        .Case<RsqrtOp>([&](auto) {
          return replaceUnary<cuda_tile::RsqrtOp>(op, input, rewriter);
        })
        .Case<SinOp>([&](auto) {
          return replaceUnary<cuda_tile::SinOp>(op, input, rewriter);
        })
        .Case<TanOp>([&](auto) {
          return replaceUnary<cuda_tile::TanOp>(op, input, rewriter);
        })
        .Case<TanhFwdOp>([&](auto) {
          return replaceUnary<cuda_tile::TanHOp>(op, input, rewriter);
        })
        .Case<ExpOp>([&](auto) {
          rewriter.replaceOp(op,
                             buildExp(rewriter, loc, input, resultElementType));
          return success();
        })
        .Case<SqrtOp>([&](auto) {
          rewriter.replaceOpWithNewOp<cuda_tile::SqrtOp>(
              op, input, cuda_tile::RoundingMode::NEAREST_EVEN);
          return success();
        })
        .Case<ReciprocalOp>([&](auto) {
          Value one = createConstant(rewriter, loc, resultType, 1.0);
          rewriter.replaceOpWithNewOp<cuda_tile::DivFOp>(
              op, one, input, cuda_tile::RoundingMode::NEAREST_EVEN);
          return success();
        })
        .Case<AbsOp>([&](auto) {
          if (resultElementType.isFloat()) {
            rewriter.replaceOpWithNewOp<cuda_tile::AbsFOp>(op, input);
          } else if (resultElementType.isSignedInteger()) {
            rewriter.replaceOpWithNewOp<cuda_tile::AbsIOp>(op, input);
          } else {
            rewriter.replaceOp(op, input);
          }
          return success();
        })
        .Case<LogicalNotOp>([&](auto) {
          Value one = createConstant(rewriter, loc, resultType, int64_t{1});
          rewriter.replaceOpWithNewOp<cuda_tile::XOrIOp>(op, input, one);
          return success();
        })
        .Case<AddOp>([&](auto) {
          return lowerAdd(op, operands, resultElementType, rewriter);
        })
        .Case<SubOp>([&](auto) {
          return lowerSub(op, operands, resultElementType, rewriter);
        })
        .Case<MulOp>([&](auto) {
          return lowerMul(op, operands, resultElementType, rewriter);
        })
        .Case<DivOp>([&](auto) {
          return lowerDiv(op, operands, resultElementType, rewriter);
        })
        .Case<ModOp>([&](auto) {
          return lowerMod(op, operands, resultElementType, rewriter);
        })
        .Case<RemOp>([&](auto) {
          if (resultElementType.isFloat()) {
            rewriter.replaceOpWithNewOp<cuda_tile::RemFOp>(op, operands[0],
                                                           operands[1]);
          } else {
            rewriter.replaceOpWithNewOp<cuda_tile::RemIOp>(
                op, operands[0], operands[1], getSignedness(resultElementType));
          }
          return success();
        })
        .Case<MinOp>([&](auto) {
          if (resultElementType.isFloat()) {
            rewriter.replaceOpWithNewOp<cuda_tile::MinFOp>(
                op, operands[0], operands[1], /*propagate_nan=*/true);
          } else {
            rewriter.replaceOpWithNewOp<cuda_tile::MinIOp>(
                op, operands[0], operands[1], getSignedness(resultElementType));
          }
          return success();
        })
        .Case<MaxOp>([&](auto) {
          if (resultElementType.isFloat()) {
            rewriter.replaceOpWithNewOp<cuda_tile::MaxFOp>(
                op, operands[0], operands[1], /*propagate_nan=*/true);
          } else {
            rewriter.replaceOpWithNewOp<cuda_tile::MaxIOp>(
                op, operands[0], operands[1], getSignedness(resultElementType));
          }
          return success();
        })
        .Case<PowOp>([&](PowOp pow) {
          Value rhs = operands[1];
          Type exponentType = pow.getRhs().getType().getElementType();
          if (exponentType.isInteger()) {
            rhs = cuda_tile::IToFOp::create(
                rewriter, loc, operands[0].getType(), rhs,
                getSignedness(exponentType),
                cuda_tile::RoundingMode::NEAREST_EVEN);
          }
          rewriter.replaceOpWithNewOp<cuda_tile::PowOp>(op, operands[0], rhs);
          return success();
        })
        .Case<Atan2Op>([&](auto) {
          rewriter.replaceOpWithNewOp<cuda_tile::Atan2Op>(op, operands[0],
                                                          operands[1]);
          return success();
        })
        .Case<AddSquareOp>([&](auto) {
          Value rhs;
          if (resultElementType.isFloat()) {
            rhs = cuda_tile::MulFOp::create(
                rewriter, loc, operands[1], operands[1],
                cuda_tile::RoundingMode::NEAREST_EVEN);
            rewriter.replaceOpWithNewOp<cuda_tile::AddFOp>(
                op, operands[0], rhs, cuda_tile::RoundingMode::NEAREST_EVEN);
          } else {
            rhs = cuda_tile::MulIOp::create(rewriter, loc, operands[1],
                                            operands[1]);
            rewriter.replaceOpWithNewOp<cuda_tile::AddIOp>(op, operands[0],
                                                           rhs);
          }
          return success();
        })
        .Case<LogicalAndOp>([&](auto) {
          rewriter.replaceOpWithNewOp<cuda_tile::AndIOp>(op, operands[0],
                                                         operands[1]);
          return success();
        })
        .Case<LogicalOrOp>([&](auto) {
          rewriter.replaceOpWithNewOp<cuda_tile::OrIOp>(op, operands[0],
                                                        operands[1]);
          return success();
        })
        .Case<BinarySelectOp>([&](auto) {
          rewriter.replaceOpWithNewOp<cuda_tile::SelectOp>(
              op, operands[0], operands[1], operands[2]);
          return success();
        })
        .Case<CmpOp>([&](CmpOp cmp) {
          Type elementType = cmp.getLhs().getType().getElementType();
          if (elementType.isFloat()) {
            rewriter.replaceOpWithNewOp<cuda_tile::CmpFOp>(
                op, getPredicate(cmp.getComparator()),
                getOrdering(cmp.getComparator()), operands[0], operands[1]);
          } else {
            rewriter.replaceOpWithNewOp<cuda_tile::CmpIOp>(
                op, getPredicate(cmp.getComparator()), operands[0], operands[1],
                getSignedness(elementType));
          }
          return success();
        })
        .Case<ConvertOp>([&](ConvertOp convert) {
          return lowerConvert(convert, input, resultType, rewriter);
        })
        .Case<SplatOp>([&](SplatOp) {
          if (auto constant = input.getDefiningOp<cuda_tile::ConstantOp>()) {
            auto literal = cast<DenseTypedElementsAttr>(
                constant.getValue().resizeSplat(resultType));
            rewriter.replaceOpWithNewOp<cuda_tile::ConstantOp>(op, resultType,
                                                               literal);
            if (mlir::isOpTriviallyDead(constant)) {
              rewriter.eraseOp(constant);
            }
            return success();
          }

          SmallVector<int64_t> unitShape(resultType.getRank(), 1);
          auto unitType =
              cuda_tile::TileType::get(unitShape, resultType.getElementType());
          Value reshaped =
              cuda_tile::ReshapeOp::create(rewriter, loc, unitType, input);
          rewriter.replaceOpWithNewOp<cuda_tile::BroadcastOp>(op, resultType,
                                                              reshaped);
          return success();
        })
        .Case<ReluFwdOp>([&](auto) {
          Value zero = createConstant(rewriter, loc, resultType, 0.0);
          rewriter.replaceOpWithNewOp<cuda_tile::MaxFOp>(
              op, zero, input, /*propagate_nan=*/true);
          return success();
        })
        .Case<SigmoidFwdOp>([&](auto) {
          rewriter.replaceOp(op, buildSigmoid(rewriter, input));
          return success();
        })
        .Case<GeluApproxTanhFwdOp>([&](auto) {
          rewriter.replaceOp(op, buildGeluApproxTanh(rewriter, loc, input));
          return success();
        })
        .Case<SoftplusFwdOp>([&](SoftplusFwdOp activation) {
          rewriter.replaceOp(
              op, buildSoftplus(rewriter, loc, input,
                                activation.getBeta().convertToDouble()));
          return success();
        })
        .Case<SwishFwdOp>([&](SwishFwdOp activation) {
          rewriter.replaceOp(
              op, buildSwish(rewriter, loc, input,
                             activation.getBeta().convertToDouble()));
          return success();
        })
        .Case<EluFwdOp>([&](EluFwdOp activation) {
          rewriter.replaceOp(op,
                             buildElu(rewriter, loc, input,
                                      activation.getBeta().convertToDouble()));
          return success();
        })
        .Case<ErfOp>([&](auto) {
          Value result;
          if (enableExperimentalCudaTileOps) {
          }
          if (!result) {
            result = buildErfApprox(rewriter, loc, input);
          }
          rewriter.replaceOp(op, result);
          return success();
        })
        .Case<GeluFwdOp>([&](auto) {
          rewriter.replaceOp(op, buildGelu(rewriter, loc, input,
                                           enableExperimentalCudaTileOps));
          return success();
        })
        .Default([&](Operation *) {
          return rewriter.notifyMatchFailure(
              op, "pointwise operation is not supported in staged kernels");
        });
  }

private:
  static LogicalResult lowerAdd(Operation *op, ArrayRef<Value> operands,
                                Type elementType,
                                ConversionPatternRewriter &rewriter) {
    if (elementType.isFloat()) {
      rewriter.replaceOpWithNewOp<cuda_tile::AddFOp>(
          op, operands[0], operands[1], cuda_tile::RoundingMode::NEAREST_EVEN);
    } else {
      rewriter.replaceOpWithNewOp<cuda_tile::AddIOp>(op, operands[0],
                                                     operands[1]);
    }
    return success();
  }

  static LogicalResult lowerSub(Operation *op, ArrayRef<Value> operands,
                                Type elementType,
                                ConversionPatternRewriter &rewriter) {
    if (elementType.isFloat()) {
      rewriter.replaceOpWithNewOp<cuda_tile::SubFOp>(
          op, operands[0], operands[1], cuda_tile::RoundingMode::NEAREST_EVEN);
    } else {
      rewriter.replaceOpWithNewOp<cuda_tile::SubIOp>(op, operands[0],
                                                     operands[1]);
    }
    return success();
  }

  static LogicalResult lowerMul(Operation *op, ArrayRef<Value> operands,
                                Type elementType,
                                ConversionPatternRewriter &rewriter) {
    if (elementType.isFloat()) {
      rewriter.replaceOpWithNewOp<cuda_tile::MulFOp>(
          op, operands[0], operands[1], cuda_tile::RoundingMode::NEAREST_EVEN);
    } else {
      rewriter.replaceOpWithNewOp<cuda_tile::MulIOp>(op, operands[0],
                                                     operands[1]);
    }
    return success();
  }

  static LogicalResult lowerDiv(Operation *op, ArrayRef<Value> operands,
                                Type elementType,
                                ConversionPatternRewriter &rewriter) {
    if (!elementType.isFloat()) {
      rewriter.replaceOpWithNewOp<cuda_tile::DivIOp>(
          op, operands[0], operands[1], getSignedness(elementType));
      return success();
    }
    return lowerFloatingDiv(op, operands[0], operands[1], elementType,
                            rewriter);
  }

  static LogicalResult lowerMod(Operation *op, ArrayRef<Value> operands,
                                Type elementType,
                                ConversionPatternRewriter &rewriter) {
    Value quotient;
    Value product;
    if (elementType.isFloat()) {
      quotient = cuda_tile::DivFOp::create(
          rewriter, op->getLoc(), operands[0], operands[1],
          cuda_tile::RoundingMode::NEAREST_EVEN);
      quotient = cuda_tile::FloorOp::create(rewriter, op->getLoc(), quotient);
      product = cuda_tile::MulFOp::create(
          rewriter, op->getLoc(), quotient, operands[1],
          cuda_tile::RoundingMode::NEAREST_EVEN);
      rewriter.replaceOpWithNewOp<cuda_tile::SubFOp>(
          op, operands[0], product, cuda_tile::RoundingMode::NEAREST_EVEN);
      return success();
    }
    auto rounding = elementType.isSignedInteger()
                        ? cuda_tile::RoundingMode::NEGATIVE_INF
                        : cuda_tile::RoundingMode::ZERO;
    quotient = cuda_tile::DivIOp::create(rewriter, op->getLoc(), operands[0],
                                         operands[1],
                                         getSignedness(elementType), rounding);
    product = cuda_tile::MulIOp::create(rewriter, op->getLoc(), quotient,
                                        operands[1]);
    rewriter.replaceOpWithNewOp<cuda_tile::SubIOp>(op, operands[0], product);
    return success();
  }

  static LogicalResult lowerConvert(ConvertOp op, Value input,
                                    cuda_tile::TileType resultType,
                                    ConversionPatternRewriter &rewriter) {
    Type inputType = op.getInput().getType().getElementType();
    Type outputType = op.getType().getElementType();
    if (outputType.isInteger(1)) {
      if (inputType.isInteger(1)) {
        rewriter.replaceOp(op, input);
      } else if (inputType.isFloat()) {
        Value zero = createConstant(rewriter, op.getLoc(),
                                    cast<ShapedType>(input.getType()), 0.0);
        rewriter.replaceOpWithNewOp<cuda_tile::CmpFOp>(
            op, cuda_tile::ComparisonPredicate::NOT_EQUAL,
            cuda_tile::ComparisonOrdering::UNORDERED, input, zero);
      } else {
        Value zero =
            createConstant(rewriter, op.getLoc(),
                           cast<ShapedType>(input.getType()), int64_t{0});
        rewriter.replaceOpWithNewOp<cuda_tile::CmpIOp>(
            op, cuda_tile::ComparisonPredicate::NOT_EQUAL, input, zero,
            getSignedness(inputType));
      }
    } else if (inputType.isFloat() && outputType.isFloat()) {
      rewriter.replaceOpWithNewOp<cuda_tile::FToFOp>(
          op, resultType, input, cuda_tile::RoundingMode::NEAREST_EVEN);
    } else if (inputType.isFloat() && outputType.isInteger()) {
      rewriter.replaceOpWithNewOp<cuda_tile::FToIOp>(
          op, resultType, input, getSignedness(outputType),
          cuda_tile::RoundingMode::NEAREST_INT_TO_ZERO);
    } else if (inputType.isInteger() && outputType.isFloat()) {
      rewriter.replaceOpWithNewOp<cuda_tile::IToFOp>(
          op, resultType, input, getSignedness(inputType),
          cuda_tile::RoundingMode::NEAREST_EVEN);
    } else if (inputType.isInteger() && outputType.isInteger()) {
      unsigned inputWidth = inputType.getIntOrFloatBitWidth();
      unsigned outputWidth = outputType.getIntOrFloatBitWidth();
      if (inputWidth < outputWidth) {
        rewriter.replaceOpWithNewOp<cuda_tile::ExtIOp>(
            op, resultType, input, getSignedness(inputType));
      } else if (inputWidth > outputWidth) {
        rewriter.replaceOpWithNewOp<cuda_tile::TruncIOp>(op, resultType, input);
      } else {
        rewriter.replaceOp(op, input);
      }
    } else {
      return op.emitError("unsupported type conversion");
    }
    return success();
  }

  static Value buildGeluApproxTanh(OpBuilder &builder, Location loc,
                                   Value input) {
    auto type = cast<ShapedType>(input.getType());
    auto rounding = cuda_tile::RoundingMode::NEAREST_EVEN;
    Value scale =
        createConstant(builder, loc, type, std::sqrt(2.0 / llvm::numbers::pi));
    Value coefficient = createConstant(builder, loc, type, 0.044715);
    Value one = createConstant(builder, loc, type, 1.0);
    Value half = createConstant(builder, loc, type, 0.5);
    Value square =
        cuda_tile::MulFOp::create(builder, loc, input, input, rounding);
    Value cube =
        cuda_tile::MulFOp::create(builder, loc, square, input, rounding);
    Value polynomial =
        cuda_tile::MulFOp::create(builder, loc, coefficient, cube, rounding);
    polynomial =
        cuda_tile::AddFOp::create(builder, loc, input, polynomial, rounding);
    polynomial =
        cuda_tile::MulFOp::create(builder, loc, scale, polynomial, rounding);
    polynomial = cuda_tile::TanHOp::create(builder, loc, polynomial);
    polynomial =
        cuda_tile::AddFOp::create(builder, loc, one, polynomial, rounding);
    Value scaledInput =
        cuda_tile::MulFOp::create(builder, loc, half, input, rounding);
    return cuda_tile::MulFOp::create(builder, loc, scaledInput, polynomial,
                                     rounding);
  }

  static Value buildSoftplus(OpBuilder &builder, Location loc, Value input,
                             double beta) {
    auto type = cast<ShapedType>(input.getType());
    auto rounding = cuda_tile::RoundingMode::NEAREST_EVEN;
    Value betaTile = createConstant(builder, loc, type, beta);
    Value one = createConstant(builder, loc, type, 1.0);
    Value inverseBeta = createConstant(builder, loc, type, 1.0 / beta);
    Value scaled =
        cuda_tile::MulFOp::create(builder, loc, input, betaTile, rounding);
    Value exponent = buildExp(builder, loc, scaled, type.getElementType());
    Value sum =
        cuda_tile::AddFOp::create(builder, loc, one, exponent, rounding);
    Value logarithm = cuda_tile::LogOp::create(builder, loc, sum);
    return cuda_tile::MulFOp::create(builder, loc, inverseBeta, logarithm,
                                     rounding);
  }

  static Value buildSwish(OpBuilder &builder, Location loc, Value input,
                          double beta) {
    auto type = cast<ShapedType>(input.getType());
    auto rounding = cuda_tile::RoundingMode::NEAREST_EVEN;
    Value betaTile = createConstant(builder, loc, type, beta);
    Value scaled =
        cuda_tile::MulFOp::create(builder, loc, input, betaTile, rounding);
    Value sigmoid = buildSigmoid(builder, scaled);
    return cuda_tile::MulFOp::create(builder, loc, input, sigmoid, rounding);
  }

  static Value buildElu(OpBuilder &builder, Location loc, Value input,
                        double beta) {
    auto type = cast<ShapedType>(input.getType());
    auto rounding = cuda_tile::RoundingMode::NEAREST_EVEN;
    Value zero = createConstant(builder, loc, type, 0.0);
    Value one = createConstant(builder, loc, type, 1.0);
    Value betaTile = createConstant(builder, loc, type, beta);
    Value exponent = buildExp(builder, loc, input, type.getElementType());
    Value shifted =
        cuda_tile::SubFOp::create(builder, loc, exponent, one, rounding);
    Value negative =
        cuda_tile::MulFOp::create(builder, loc, shifted, betaTile, rounding);
    Value positive = cuda_tile::CmpFOp::create(
        builder, loc, cuda_tile::ComparisonPredicate::GREATER_THAN,
        cuda_tile::ComparisonOrdering::ORDERED, input, zero);
    return cuda_tile::SelectOp::create(builder, loc, positive, input, negative);
  }

  static Value buildGelu(OpBuilder &builder, Location loc, Value input,
                         bool enableExperimentalCudaTileOps) {
    auto type = cast<ShapedType>(input.getType());
    auto rounding = cuda_tile::RoundingMode::NEAREST_EVEN;
    Value sqrtTwo = createConstant(builder, loc, type, std::sqrt(2.0));
    Value one = createConstant(builder, loc, type, 1.0);
    Value half = createConstant(builder, loc, type, 0.5);
    Value scaled =
        cuda_tile::DivFOp::create(builder, loc, input, sqrtTwo, rounding);
    Value erf;
    if (enableExperimentalCudaTileOps) {
    }
    if (!erf) {
      erf = buildErfApprox(builder, loc, scaled);
    }
    Value factor = cuda_tile::AddFOp::create(builder, loc, one, erf, rounding);
    Value scaledInput =
        cuda_tile::MulFOp::create(builder, loc, input, half, rounding);
    return cuda_tile::MulFOp::create(builder, loc, scaledInput, factor,
                                     rounding);
  }
};

class PointwiseConversion
    : public OpInterfaceConversionPattern<PointwiseOpInterface> {
public:
  PointwiseConversion(const TypeConverter &typeConverter, MLIRContext *context,
                      bool enableExperimentalCudaTileOps)
      : OpInterfaceConversionPattern(typeConverter, context),
        enableExperimentalCudaTileOps(enableExperimentalCudaTileOps) {}

  LogicalResult
  matchAndRewrite(PointwiseOpInterface pointwise, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *op = pointwise.getOperation();
    if (failed(validatePointwiseOpLowerable(op))) {
      return failure();
    }
    FailureOr<cuda_tile::TileType> resultType =
        getResultType(op, *typeConverter);
    if (failed(resultType)) {
      return op->emitError(
          "staged pointwise operations require a static tile result");
    }
    return lowerPointwiseOp(pointwise, operands, *resultType, rewriter,
                            enableExperimentalCudaTileOps);
  }

private:
  bool enableExperimentalCudaTileOps;
};

class IotaConversion : public OpConversionPattern<IotaOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IotaOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<cuda_tile::TileType> resultType =
        getResultType(op, *typeConverter);
    if (failed(resultType)) {
      return op.emitError("staged iota requires a static tile result");
    }
    unsigned dimension = op.getDimension();
    if (dimension >= (*resultType).getRank()) {
      return op.emitError("iota dimension is out of range");
    }
    Type sourceElementType = op.getType().getElementType();
    Type iotaElementType = sourceElementType.isFloat()
                               ? rewriter.getI32Type()
                               : (*resultType).getElementType();
    int64_t extent = (*resultType).getDimSize(dimension);
    auto oneDimensionalType =
        cuda_tile::TileType::get({extent}, iotaElementType);
    Value result =
        cuda_tile::IotaOp::create(rewriter, op.getLoc(), oneDimensionalType);
    SmallVector<int64_t> unitShape((*resultType).getRank(), 1);
    unitShape[dimension] = extent;
    auto rankedType = cuda_tile::TileType::get(unitShape, iotaElementType);
    if (rankedType != oneDimensionalType) {
      result = cuda_tile::ReshapeOp::create(rewriter, op.getLoc(), rankedType,
                                            result);
    }
    auto integerResultType =
        cuda_tile::TileType::get((*resultType).getShape(), iotaElementType);
    if (rankedType != integerResultType) {
      result = cuda_tile::BroadcastOp::create(rewriter, op.getLoc(),
                                              integerResultType, result);
    }
    if (sourceElementType.isFloat()) {
      result =
          cuda_tile::IToFOp::create(rewriter, op.getLoc(), *resultType, result,
                                    cuda_tile::Signedness::Unsigned,
                                    cuda_tile::RoundingMode::NEAREST_EVEN);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename TensorOp, typename CudaOp>
class UnaryShapeConversion : public OpConversionPattern<TensorOp> {
public:
  using Base = OpConversionPattern<TensorOp>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(TensorOp op, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    FailureOr<cuda_tile::TileType> resultType =
        getResultType(op, *this->typeConverter);
    if (failed(resultType)) {
      return op.emitError("staged shape operation requires static tile types");
    }
    if constexpr (std::is_same_v<TensorOp, ReshapeOp>) {
      if (input.getType() == *resultType) {
        rewriter.replaceOp(op, input);
        return success();
      }
      if (auto previous = input.template getDefiningOp<cuda_tile::ReshapeOp>();
          previous && previous.getSource().getType() == *resultType) {
        rewriter.replaceOp(op, previous.getSource());
        return success();
      }
    }
    rewriter.replaceOpWithNewOp<CudaOp>(op, *resultType, input);
    return success();
  }
};

using BroadcastConversion =
    UnaryShapeConversion<BroadcastOp, cuda_tile::BroadcastOp>;
using ReshapeConversion = UnaryShapeConversion<ReshapeOp, cuda_tile::ReshapeOp>;

class TransposeConversion : public OpConversionPattern<TransposeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    FailureOr<cuda_tile::TileType> resultType =
        getResultType(op, *typeConverter);
    if (failed(resultType)) {
      return op.emitError("staged transpose requires static tile types");
    }
    SmallVector<int32_t> permutation;
    for (int64_t dimension : op.getPermutation()) {
      permutation.push_back(static_cast<int32_t>(dimension));
    }
    rewriter.replaceOpWithNewOp<cuda_tile::PermuteOp>(op, *resultType, input,
                                                      permutation);
    return success();
  }
};

class SliceConversion : public OpConversionPattern<SliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    FailureOr<cuda_tile::TileType> resultType =
        getResultType(op, *typeConverter);
    if (failed(resultType)) {
      return op.emitError("staged slice requires static tile types");
    }
    if (llvm::any_of(op.getStrides(),
                     [](int64_t stride) { return stride != 1; })) {
      return op.emitError("staged tile slice requires unit strides");
    }
    auto indexType =
        cuda_tile::TileType::get(ArrayRef<int64_t>{}, rewriter.getI32Type());
    SmallVector<Value> indices;
    for (int64_t start : op.getStarts()) {
      indices.push_back(
          createConstant(rewriter, op.getLoc(), indexType, start));
    }
    rewriter.replaceOpWithNewOp<cuda_tile::ExtractOp>(op, *resultType, input,
                                                      indices);
    return success();
  }
};

class ConcatenateConversion : public OpConversionPattern<ConcatenateOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConcatenateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange inputs = adaptor.getInputs();
    FailureOr<cuda_tile::TileType> resultType =
        getResultType(op, *typeConverter);
    if (inputs.empty() || failed(resultType)) {
      return op.emitError("staged concatenate requires static tile types");
    }
    Value result = inputs.front();
    unsigned dimension = op.getDimension();
    for (Value input : llvm::drop_begin(inputs)) {
      SmallVector<int64_t> shape(cast<ShapedType>(result.getType()).getShape());
      shape[dimension] +=
          cast<ShapedType>(input.getType()).getDimSize(dimension);
      auto intermediateType =
          cuda_tile::TileType::get(shape, (*resultType).getElementType());
      result = cuda_tile::CatOp::create(rewriter, op.getLoc(), intermediateType,
                                        result, input, dimension);
    }
    if (result.getType() != *resultType) {
      return op.emitError("concatenated tile shape does not match result type");
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ReductionConversion : public OpConversionPattern<ReduceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    FailureOr<cuda_tile::TileType> resultType =
        getResultType(op, *typeConverter);
    if (failed(resultType)) {
      return op.emitError("staged reduction requires static tile types");
    }
    Type elementType = op.getType().getElementType();
    if ((!elementType.isFloat() && !elementType.isInteger()) ||
        op.getReductionMode() == ReductionMode::customize) {
      return op.emitError("Unsupported reduction type");
    }
    if (op.getReductionMode() == ReductionMode::norm2 &&
        elementType.isInteger()) {
      return op.emitError("Unsupported reduction mode");
    }

    auto inputType = cast<cuda_tile::TileType>(input.getType());
    SmallVector<int32_t> dimensions(op.getDimensions().begin(),
                                    op.getDimensions().end());
    if (dimensions.empty()) {
      return op.emitError("staged reduction has no dimensions");
    }
    int64_t firstDimension = dimensions.front();
    for (auto [index, dimension] : llvm::enumerate(dimensions)) {
      if (dimension != firstDimension + static_cast<int64_t>(index) ||
          dimension >= inputType.getRank()) {
        return op.emitError(
            "staged reduction dimensions must be trailing and contiguous");
      }
    }
    if (firstDimension + static_cast<int64_t>(dimensions.size()) !=
        inputType.getRank()) {
      return op.emitError("staged reduction dimensions must be trailing");
    }

    ReductionEmissionHelper emission{op.getReductionMode(), elementType};
    Value reductionInput = input;
    if (!op->hasAttr("nv_tensor_ir.reduction_prologue_applied")) {
      reductionInput = emission.buildPrologue(rewriter, reductionInput);
    }
    int64_t localReductionSize = 1;
    for (int32_t dimension : dimensions) {
      localReductionSize *= inputType.getDimSize(dimension);
    }
    if (dimensions.size() > 1) {
      SmallVector<int64_t> mergedShape(
          inputType.getShape().take_front(firstDimension));
      mergedShape.push_back(localReductionSize);
      auto mergedType =
          cuda_tile::TileType::get(mergedShape, inputType.getElementType());
      reductionInput = cuda_tile::ReshapeOp::create(rewriter, op.getLoc(),
                                                    mergedType, reductionInput);
    }

    auto identities = rewriter.getArrayAttr({emission.getIdentity(rewriter)});
    auto tileReduce = cuda_tile::ReduceOp::create(
        rewriter, op.getLoc(), reductionInput, firstDimension, identities);
    Block &body = tileReduce.getBodyRegion().emplaceBlock();
    auto scalarType = cuda_tile::TileType::get(ArrayRef<int64_t>{},
                                               inputType.getElementType());
    BlockArgument value = body.addArgument(scalarType, op.getLoc());
    BlockArgument accumulator = body.addArgument(scalarType, op.getLoc());
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(&body);
      Value combined = emission.buildReduction(rewriter, accumulator, value);
      cuda_tile::YieldOp::create(rewriter, op.getLoc(), combined);
    }

    rewriter.setInsertionPointAfter(tileReduce);
    int64_t fullReductionSize = localReductionSize;
    if (auto extent =
            op->getAttrOfType<IntegerAttr>("nv_tensor_ir.reduction_extent")) {
      fullReductionSize = extent.getInt();
    }
    Value result = emission.buildEpilogue(rewriter, tileReduce.getResult(0),
                                          fullReductionSize);
    if (result.getType() != *resultType) {
      result = cuda_tile::ReshapeOp::create(rewriter, op.getLoc(), *resultType,
                                            result);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

static bool isSupportedStagedMatmulTypes(Type lhsType, Type rhsType,
                                         Type resultType) {
  if (lhsType.isF16() && rhsType == lhsType) {
    return resultType.isF16() || resultType.isF32();
  }
  if ((lhsType.isBF16() || lhsType.isF32()) && rhsType == lhsType) {
    return resultType.isF32();
  }
  if (lhsType.isF64() && rhsType == lhsType) {
    return resultType.isF64();
  }
  return lhsType.isInteger(8) && rhsType.isInteger(8) &&
         resultType.isSignedInteger(32);
}

class MatmulConversion : public OpConversionPattern<MatmulOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(validateOutlinedMatmulLowerable(op))) {
      return failure();
    }
    SmallVector<Value> operands(adaptor.getOperands());
    FailureOr<cuda_tile::TileType> resultType =
        getResultType(op, *typeConverter);
    if (failed(resultType)) {
      return op.emitError("staged matmul requires a static tile result");
    }

    auto lhsType = dyn_cast<cuda_tile::TileType>(operands[0].getType());
    auto rhsType = dyn_cast<cuda_tile::TileType>(operands[1].getType());
    if (!lhsType || !rhsType || !lhsType.hasStaticShape() ||
        !rhsType.hasStaticShape()) {
      return op.emitError("staged matmul requires static tile operands");
    }
    if (lhsType.getRank() != rhsType.getRank() ||
        lhsType.getRank() != resultType->getRank() ||
        (lhsType.getRank() != 2 && lhsType.getRank() != 3)) {
      return op.emitError("staged matmul requires consistently ranked 2D or "
                          "3D tiles");
    }
    ArrayRef<int64_t> lhsShape = lhsType.getShape();
    ArrayRef<int64_t> rhsShape = rhsType.getShape();
    ArrayRef<int64_t> outputShape = resultType->getShape();
    if (lhsShape.take_front(lhsShape.size() - 2) !=
            rhsShape.take_front(rhsShape.size() - 2) ||
        lhsShape.take_front(lhsShape.size() - 2) !=
            outputShape.take_front(outputShape.size() - 2) ||
        lhsShape[lhsShape.size() - 2] != outputShape[outputShape.size() - 2] ||
        lhsShape.back() != rhsShape[rhsShape.size() - 2] ||
        rhsShape.back() != outputShape.back()) {
      return op.emitError("staged matmul tile shapes are incompatible");
    }

    Type lhsElementType = op.getA().getType().getElementType();
    Type rhsElementType = op.getB().getType().getElementType();
    Type resultElementType = op.getType().getElementType();
    if (!isSupportedStagedMatmulTypes(lhsElementType, rhsElementType,
                                      resultElementType)) {
      return op.emitError("unsupported input/output type combination");
    }
    if (operands.size() == 3 && operands[2].getType() != *resultType) {
      return op.emitError("staged matmul accumulator must match the result "
                          "tile type");
    }
    if (operands.size() == 2) {
      operands.push_back(
          resultElementType.isFloat()
              ? createConstant(rewriter, op.getLoc(), *resultType, 0.0)
              : createConstant(rewriter, op.getLoc(), *resultType, int64_t{0}));
    }

    Value result;
    if (resultElementType.isFloat()) {
      result = cuda_tile::MmaFOp::create(rewriter, op.getLoc(),
                                         TypeRange{*resultType}, operands);
    } else {
      result = cuda_tile::MmaIOp::create(rewriter, op.getLoc(), *resultType,
                                         operands[0], operands[1], operands[2],
                                         getSignedness(lhsElementType),
                                         getSignedness(rhsElementType));
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class UserDefinedReductionConversion : public OpConversionPattern<ReduceUDOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReduceUDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange inputs = adaptor.getInputs();
    if (inputs.empty() || inputs.size() != op.getNumResults()) {
      return op.emitError(
          "staged user-defined reduction requires one result per input");
    }
    auto inputType = dyn_cast<cuda_tile::TileType>(inputs.front().getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return op.emitError(
          "staged user-defined reduction requires static tile inputs");
    }

    SmallVector<int32_t> dimensions(op.getDimensions().begin(),
                                    op.getDimensions().end());
    if (dimensions.empty()) {
      return op.emitError("staged reduction has no dimensions");
    }
    int64_t firstDimension = dimensions.front();
    for (auto [index, dimension] : llvm::enumerate(dimensions)) {
      if (dimension != firstDimension + static_cast<int64_t>(index) ||
          dimension >= inputType.getRank()) {
        return op.emitError(
            "staged reduction dimensions must be trailing and contiguous");
      }
    }
    if (firstDimension + static_cast<int64_t>(dimensions.size()) !=
        inputType.getRank()) {
      return op.emitError("staged reduction dimensions must be trailing");
    }

    SmallVector<Value> reductionInputs(inputs);
    if (dimensions.size() > 1) {
      SmallVector<int64_t> mergedShape(
          inputType.getShape().take_front(firstDimension));
      int64_t mergedExtent = 1;
      for (int32_t dimension : dimensions) {
        mergedExtent *= inputType.getDimSize(dimension);
      }
      mergedShape.push_back(mergedExtent);
      for (Value &input : reductionInputs) {
        auto mergedType = cuda_tile::TileType::get(
            mergedShape,
            cast<cuda_tile::TileType>(input.getType()).getElementType());
        input = cuda_tile::ReshapeOp::create(rewriter, op.getLoc(), mergedType,
                                             input);
      }
    }

    SmallVector<Attribute> identities;
    identities.reserve(op.getIdentity().size());
    for (Attribute identity : op.getIdentity()) {
      if (auto integer = dyn_cast<IntegerAttr>(identity)) {
        auto signless =
            rewriter.getIntegerType(integer.getType().getIntOrFloatBitWidth());
        identity = rewriter.getIntegerAttr(signless, integer.getValue());
      }
      identities.push_back(identity);
    }
    auto tileReduce = cuda_tile::ReduceOp::create(
        rewriter, op.getLoc(), reductionInputs, firstDimension,
        rewriter.getArrayAttr(identities));

    size_t numInputs = inputs.size();
    TypeConverter::SignatureConversion signature(numInputs * 2);
    Block &sourceBlock = op.getBodyRegion().front();
    for (size_t index = 0; index < numInputs; ++index) {
      Type input = typeConverter->convertType(
          sourceBlock.getArgument(numInputs + index).getType());
      if (!input) {
        return op.emitError(
            "failed to convert user-defined reduction body argument");
      }
      // TensorIR orders all accumulators before all values. CUDA Tile orders
      // each value immediately before its accumulator.
      signature.addInputs(numInputs + index, input);
      signature.addInputs(index, input);
    }
    Region &targetRegion = tileReduce.getBodyRegion();
    rewriter.inlineRegionBefore(op.getBodyRegion(), targetRegion,
                                targetRegion.end());
    rewriter.applySignatureConversion(&targetRegion.front(), signature,
                                      typeConverter);

    SmallVector<Value> replacements;
    replacements.reserve(tileReduce.getNumResults());
    for (auto [original, reduced] :
         llvm::zip_equal(op.getResults(), tileReduce.getResults())) {
      auto resultType =
          typeConverter->convertType<cuda_tile::TileType>(original.getType());
      if (!resultType || !resultType.hasStaticShape()) {
        return op.emitError(
            "staged user-defined reduction requires static result tiles");
      }
      Value result = reduced;
      if (result.getType() != resultType) {
        result = cuda_tile::ReshapeOp::create(rewriter, op.getLoc(), resultType,
                                              result);
      }
      replacements.push_back(result);
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

class ReductionYieldConversion : public OpConversionPattern<YieldOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cuda_tile::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};

static bool isStagedReductionArithmetic(Operation *op) {
  return op->hasAttr("nv_tensor_ir.strip_mined_reduction_body") ||
         op->getParentOfType<ReduceUDOp>() ||
         op->getParentOfType<cuda_tile::ReduceOp>();
}

template <typename SourceOp, typename TargetOp, bool ReductionOnly = true>
class SimpleArithmeticConversion : public OpConversionPattern<SourceOp> {
public:
  using Base = OpConversionPattern<SourceOp>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(SourceOp op, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (ReductionOnly && !isStagedReductionArithmetic(op)) {
      return rewriter.notifyMatchFailure(op, "expected reduction arithmetic");
    }
    rewriter.replaceOpWithNewOp<TargetOp>(op, adaptor.getOperands());
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
class RoundedFloatConversion : public OpConversionPattern<SourceOp> {
public:
  using Base = OpConversionPattern<SourceOp>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(SourceOp op, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isStagedReductionArithmetic(op)) {
      return rewriter.notifyMatchFailure(op, "expected reduction arithmetic");
    }
    rewriter.replaceOpWithNewOp<TargetOp>(
        op, adaptor.getLhs(), adaptor.getRhs(),
        cuda_tile::RoundingMode::NEAREST_EVEN);
    return success();
  }
};

class ReductionBodyDivFConversion : public OpConversionPattern<arith::DivFOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::DivFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isStagedReductionArithmetic(op)) {
      return rewriter.notifyMatchFailure(op, "expected reduction arithmetic");
    }
    Type resultType = op.getType();
    Type elementType = isa<ShapedType>(resultType)
                           ? cast<ShapedType>(resultType).getElementType()
                           : resultType;
    return lowerFloatingDiv(op, adaptor.getLhs(), adaptor.getRhs(), elementType,
                            rewriter);
  }
};

template <typename SourceOp, typename TargetOp>
class NaNPropagatingFloatConversion : public OpConversionPattern<SourceOp> {
public:
  using Base = OpConversionPattern<SourceOp>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(SourceOp op, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isStagedReductionArithmetic(op)) {
      return rewriter.notifyMatchFailure(op, "expected reduction arithmetic");
    }
    rewriter.replaceOpWithNewOp<TargetOp>(op, adaptor.getLhs(),
                                          adaptor.getRhs(),
                                          /*propagateNan=*/true);
    return success();
  }
};

template <typename SourceOp, typename TargetOp,
          cuda_tile::Signedness Signedness, bool ReductionOnly = true>
class SignedArithmeticConversion : public OpConversionPattern<SourceOp> {
public:
  using Base = OpConversionPattern<SourceOp>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(SourceOp op, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (ReductionOnly && !isStagedReductionArithmetic(op)) {
      return rewriter.notifyMatchFailure(op, "expected reduction arithmetic");
    }
    rewriter.replaceOpWithNewOp<TargetOp>(op, adaptor.getLhs(),
                                          adaptor.getRhs(), Signedness);
    return success();
  }
};

static std::optional<Comparator>
convertComparator(arith::CmpFPredicate predicate) {
  switch (predicate) {
  case arith::CmpFPredicate::UEQ:
    return Comparator::ueq;
  case arith::CmpFPredicate::UNE:
    return Comparator::une;
  case arith::CmpFPredicate::ULT:
    return Comparator::ult;
  case arith::CmpFPredicate::ULE:
    return Comparator::ule;
  case arith::CmpFPredicate::UGT:
    return Comparator::ugt;
  case arith::CmpFPredicate::UGE:
    return Comparator::uge;
  case arith::CmpFPredicate::OEQ:
    return Comparator::oeq;
  case arith::CmpFPredicate::ONE:
    return Comparator::one;
  case arith::CmpFPredicate::OLT:
    return Comparator::olt;
  case arith::CmpFPredicate::OLE:
    return Comparator::ole;
  case arith::CmpFPredicate::OGT:
    return Comparator::ogt;
  case arith::CmpFPredicate::OGE:
    return Comparator::oge;
  default:
    return std::nullopt;
  }
}

class ReductionBodyCmpFConversion : public OpConversionPattern<arith::CmpFOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isStagedReductionArithmetic(op)) {
      return failure();
    }
    std::optional<Comparator> comparator = convertComparator(op.getPredicate());
    if (!comparator) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<cuda_tile::CmpFOp>(
        op, getPredicate(*comparator), getOrdering(*comparator),
        adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

class IntegerComparisonConversion : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    cuda_tile::ComparisonPredicate predicate;
    switch (op.getPredicate()) {
    case arith::CmpIPredicate::eq:
      predicate = cuda_tile::ComparisonPredicate::EQUAL;
      break;
    case arith::CmpIPredicate::ne:
      predicate = cuda_tile::ComparisonPredicate::NOT_EQUAL;
      break;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      predicate = cuda_tile::ComparisonPredicate::LESS_THAN;
      break;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      predicate = cuda_tile::ComparisonPredicate::LESS_THAN_OR_EQUAL;
      break;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      predicate = cuda_tile::ComparisonPredicate::GREATER_THAN;
      break;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      predicate = cuda_tile::ComparisonPredicate::GREATER_THAN_OR_EQUAL;
      break;
    }
    bool isUnsigned = llvm::is_contained(
        {arith::CmpIPredicate::ult, arith::CmpIPredicate::ule,
         arith::CmpIPredicate::ugt, arith::CmpIPredicate::uge},
        op.getPredicate());
    rewriter.replaceOpWithNewOp<cuda_tile::CmpIOp>(
        op, predicate, adaptor.getLhs(), adaptor.getRhs(),
        isUnsigned ? cuda_tile::Signedness::Unsigned
                   : cuda_tile::Signedness::Signed);
    return success();
  }
};

using ReductionBodyAddF =
    RoundedFloatConversion<arith::AddFOp, cuda_tile::AddFOp>;
using ReductionBodyDivF = ReductionBodyDivFConversion;
using ReductionBodyMulF =
    RoundedFloatConversion<arith::MulFOp, cuda_tile::MulFOp>;
using ReductionBodySubF =
    RoundedFloatConversion<arith::SubFOp, cuda_tile::SubFOp>;
using ReductionBodyAndI =
    SimpleArithmeticConversion<arith::AndIOp, cuda_tile::AndIOp>;
using ReductionBodyOrI =
    SimpleArithmeticConversion<arith::OrIOp, cuda_tile::OrIOp>;
using ReductionBodyXOrI =
    SimpleArithmeticConversion<arith::XOrIOp, cuda_tile::XOrIOp>;
using ReductionBodyMaximumF =
    NaNPropagatingFloatConversion<arith::MaximumFOp, cuda_tile::MaxFOp>;
using ReductionBodyMinimumF =
    NaNPropagatingFloatConversion<arith::MinimumFOp, cuda_tile::MinFOp>;
using ReductionBodyNegF =
    SimpleArithmeticConversion<arith::NegFOp, cuda_tile::NegFOp>;
using ReductionBodyRemF =
    SimpleArithmeticConversion<arith::RemFOp, cuda_tile::RemFOp>;
using ReductionBodyMaxSI =
    SignedArithmeticConversion<arith::MaxSIOp, cuda_tile::MaxIOp,
                               cuda_tile::Signedness::Signed>;
using ReductionBodyMaxUI =
    SignedArithmeticConversion<arith::MaxUIOp, cuda_tile::MaxIOp,
                               cuda_tile::Signedness::Unsigned>;
using ReductionBodyMinSI =
    SignedArithmeticConversion<arith::MinSIOp, cuda_tile::MinIOp,
                               cuda_tile::Signedness::Signed>;
using ReductionBodyMinUI =
    SignedArithmeticConversion<arith::MinUIOp, cuda_tile::MinIOp,
                               cuda_tile::Signedness::Unsigned>;

using AddIConversion =
    SimpleArithmeticConversion<arith::AddIOp, cuda_tile::AddIOp, false>;
using SubIConversion =
    SimpleArithmeticConversion<arith::SubIOp, cuda_tile::SubIOp, false>;
using MulIConversion =
    SimpleArithmeticConversion<arith::MulIOp, cuda_tile::MulIOp, false>;
using DivUIConversion =
    SignedArithmeticConversion<arith::DivUIOp, cuda_tile::DivIOp,
                               cuda_tile::Signedness::Unsigned, false>;
using RemUIConversion =
    SignedArithmeticConversion<arith::RemUIOp, cuda_tile::RemIOp,
                               cuda_tile::Signedness::Unsigned, false>;
using SelectConversion =
    SimpleArithmeticConversion<arith::SelectOp, cuda_tile::SelectOp, false>;

} // namespace

LogicalResult lowerPointwiseOp(PointwiseOpInterface op,
                               ArrayRef<Value> operands,
                               cuda_tile::TileType resultType,
                               ConversionPatternRewriter &rewriter,
                               bool enableExperimentalCudaTileOps) {
  return PointwiseEmitter::lower(op, operands, resultType, rewriter,
                                 enableExperimentalCudaTileOps);
}

LogicalResult validateOutlinedMatmulLowerable(MatmulOp op) {
  Type lhsElementType = op.getA().getType().getElementType();
  Type rhsElementType = op.getB().getType().getElementType();
  Type resultElementType = op.getType().getElementType();
  if (!isSupportedStagedMatmulTypes(lhsElementType, rhsElementType,
                                    resultElementType)) {
    return op.emitError("unsupported input/output type combination");
  }

  auto lhsType = dyn_cast<RankedTensorType>(op.getA().getType());
  auto rhsType = dyn_cast<RankedTensorType>(op.getB().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getType());
  if (!lhsType || !rhsType || !resultType || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape() || !resultType.hasStaticShape()) {
    return op.emitError("staged matmul requires static ranked tensor tiles");
  }
  if (lhsType.getRank() != rhsType.getRank() ||
      lhsType.getRank() != resultType.getRank() ||
      (lhsType.getRank() != 2 && lhsType.getRank() != 3)) {
    return op.emitError(
        "staged matmul requires consistently ranked 2D or 3D tiles");
  }

  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ArrayRef<int64_t> rhsShape = rhsType.getShape();
  ArrayRef<int64_t> outputShape = resultType.getShape();
  if (lhsShape.take_front(lhsShape.size() - 2) !=
          rhsShape.take_front(rhsShape.size() - 2) ||
      lhsShape.take_front(lhsShape.size() - 2) !=
          outputShape.take_front(outputShape.size() - 2) ||
      lhsShape[lhsShape.size() - 2] != outputShape[outputShape.size() - 2] ||
      lhsShape.back() != rhsShape[rhsShape.size() - 2] ||
      rhsShape.back() != outputShape.back()) {
    return op.emitError("staged matmul tile shapes are incompatible");
  }
  if (Value accumulator = op.getAcc();
      accumulator && accumulator.getType() != op.getType()) {
    return op.emitError("staged matmul accumulator must match the result tile "
                        "type");
  }
  return success();
}

void registerOutlinedComputePatterns(RewritePatternSet &patterns,
                                     const TypeConverter &typeConverter,
                                     bool enableExperimentalCudaTileOps) {
  MLIRContext *context = patterns.getContext();
  patterns.add<TensorIRConstantConversion, IotaConversion, BroadcastConversion,
               ReshapeConversion, TransposeConversion, SliceConversion,
               ConcatenateConversion, ReductionConversion, MatmulConversion,
               UserDefinedReductionConversion, ReductionYieldConversion>(
      typeConverter, context);
  patterns.add<PointwiseConversion>(typeConverter, context,
                                    enableExperimentalCudaTileOps);
  patterns.add<AddIConversion, SubIConversion, MulIConversion, DivUIConversion,
               RemUIConversion, SelectConversion, IntegerComparisonConversion,
               ReductionBodyCmpFConversion, ReductionBodyAddF,
               ReductionBodyDivF, ReductionBodyMulF, ReductionBodySubF,
               ReductionBodyAndI, ReductionBodyOrI, ReductionBodyXOrI,
               ReductionBodyMaximumF, ReductionBodyMinimumF, ReductionBodyNegF,
               ReductionBodyRemF, ReductionBodyMaxSI, ReductionBodyMaxUI,
               ReductionBodyMinSI, ReductionBodyMinUI>(typeConverter, context);
}

} // namespace mlir::nv_tensor_ir::tensor_to_cuda_tile
