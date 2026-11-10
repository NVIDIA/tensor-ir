// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// Lower the public, portable TensorIR operation subset to tensor-semantics
/// Linalg and Tensor operations.

#include "tensor_ir/Conversion/TensorToLinalg/TensorToLinalg.h"

#include "tensor_ir/Dialect/TensorIR.h" // IWYU pragma: keep

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ScalarBuilder.h"

namespace mlir::nv_tensor_ir {

#define GEN_PASS_DEF_TENSORTOLINALGCONVERSIONPASS
#include "tensor_ir/Conversion/TensorToLinalg/Passes.h.inc"
} // namespace mlir::nv_tensor_ir

using namespace mlir;
using namespace mlir::nv_tensor_ir;

static Value getDim(OpBuilder &builder, Location loc, Value tensor,
                    int64_t dim) {
  return tensor::DimOp::create(builder, loc, tensor, dim);
}

static FailureOr<Value> createEmptyTensor(OpBuilder &builder, Location loc,
                                          RankedTensorType resultType,
                                          Value shapeSource = {}) {
  SmallVector<Value> dynamicSizes;
  for (auto [dim, size] : llvm::enumerate(resultType.getShape())) {
    if (!ShapedType::isDynamic(size)) {
      continue;
    }
    auto sourceType = dyn_cast_if_present<RankedTensorType>(
        shapeSource ? shapeSource.getType() : Type());
    if (!sourceType || sourceType.getRank() != resultType.getRank()) {
      return failure();
    }
    Value dynamicSize = getDim(builder, loc, shapeSource, dim);
    dynamicSizes.push_back(dynamicSize);
  }
  return tensor::EmptyOp::create(builder, loc, resultType, dynamicSizes)
      .getResult();
}

static FailureOr<Value> createEmptyTensor(OpBuilder &builder, Location loc,
                                          RankedTensorType resultType,
                                          ValueRange dynamicSizes) {
  if (dynamicSizes.size() != resultType.getNumDynamicDims()) {
    return failure();
  }
  return tensor::EmptyOp::create(builder, loc, resultType, dynamicSizes)
      .getResult();
}

static TypedAttr getSignlessConstant(TypedAttr attr) {
  if (auto integer = dyn_cast<IntegerAttr>(attr)) {
    return IntegerAttr::get(getSignlessType(integer.getType()),
                            integer.getValue());
  }
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    Type elementType = getSignlessType(dense.getElementType());
    return dense.mapValues(elementType,
                           [](const APInt &value) { return value; });
  }
  return attr;
}

static Value castFrom(ScalarBuilder &scalar, Value input, Type outputType,
                      Type sourceType) {
  return isUnsignedInteger(sourceType) ? scalar.ucast(input, outputType)
                                       : scalar.scast(input, outputType);
}

/// Model MMA with widened products. Integer accumulation uses MmaIOp's default
/// signed saturation; floating-point arithmetic is a portable reference, not a
/// bit-exact reproduction of hardware MMA's intermediate precision or ordering.
static Value buildMmaScalar(OpBuilder &builder, Location loc, Value lhs,
                            Value rhs, Value accumulator, Type lhsType,
                            Type rhsType) {
  ScalarBuilder scalar(builder, loc);
  Type resultType = accumulator.getType();
  if (isa<IntegerType>(lhsType) && isa<IntegerType>(rhsType) &&
      isa<IntegerType>(resultType)) {
    unsigned resultWidth = resultType.getIntOrFloatBitWidth();
    unsigned productWidth =
        lhsType.getIntOrFloatBitWidth() + rhsType.getIntOrFloatBitWidth();
    // Represent an unsigned product as signed, then reserve a carry bit for
    // adding the signed accumulator without overflow before clamping.
    auto wideType =
        builder.getIntegerType(std::max(productWidth + 1, resultWidth) + 1);
    Value wideLhs = castFrom(scalar, lhs, wideType, lhsType);
    Value wideRhs = castFrom(scalar, rhs, wideType, rhsType);
    Value product = scalar.mul(wideLhs, wideRhs);
    Value wideAccumulator = scalar.scast(accumulator, wideType);
    Value sum = scalar.add(wideAccumulator, product);
    auto minAttr = builder.getIntegerAttr(
        wideType,
        APInt::getSignedMinValue(resultWidth).sext(wideType.getWidth()));
    auto maxAttr = builder.getIntegerAttr(
        wideType,
        APInt::getSignedMaxValue(resultWidth).sext(wideType.getWidth()));
    Value minimum = arith::ConstantOp::create(builder, loc, minAttr);
    Value maximum = arith::ConstantOp::create(builder, loc, maxAttr);
    Value lowerClamped = scalar.smax(sum, minimum);
    Value clamped = scalar.smin(lowerClamped, maximum);
    return scalar.scast(clamped, resultType);
  }
  Value convertedLhs = castFrom(scalar, lhs, resultType, lhsType);
  Value convertedRhs = castFrom(scalar, rhs, resultType, rhsType);
  Value product = scalar.mul(convertedLhs, convertedRhs);
  return scalar.add(accumulator, product);
}

static Value buildSigmoid(ScalarBuilder &scalar, Value input) {
  Value one = scalar.constant(input.getType(), 1);
  Value neg = scalar.neg(input);
  Value exp = scalar.exp(neg);
  Value denominator = scalar.add(one, exp);
  return scalar.sdiv(one, denominator);
}

static Value buildGeluApproxTanh(ScalarBuilder &scalar, Value input) {
  Type type = input.getType();
  Value half = scalar.constant(type, 0.5);
  Value one = scalar.constant(type, 1.0);
  Value coefficient = scalar.constant(type, 0.044715);
  constexpr double kPi = 3.14159265358979323846;
  Value scale = scalar.constant(type, std::sqrt(2.0 / kPi));
  Value squared = scalar.mul(input, input);
  Value cubed = scalar.mul(squared, input);
  Value scaledCube = scalar.mul(coefficient, cubed);
  Value polynomial = scalar.add(input, scaledCube);
  Value scaledPolynomial = scalar.mul(scale, polynomial);
  Value tanh = scalar.tanh(scaledPolynomial);
  Value factor = scalar.add(one, tanh);
  Value halfInput = scalar.mul(half, input);
  return scalar.mul(halfInput, factor);
}

static Value buildSelectPositive(ScalarBuilder &scalar, Value input,
                                 Value positive, Value nonPositive) {
  Value zero = scalar.constant(input.getType(), 0);
  Value predicate = scalar.sgt(input, zero);
  return scalar.select(predicate, positive, nonPositive);
}

/// Scalar callback factories defer IR emission until their TypeSwitch case matches.
template <typename Builder>
static auto unaryScalar(ScalarBuilder &scalar, ValueRange values,
                        Value (Builder::*method)(Value)) {
  return [&scalar, values, method](auto) -> FailureOr<Value> {
    return (scalar.*method)(values[0]);
  };
}

template <typename Builder>
static auto binaryScalar(ScalarBuilder &scalar, ValueRange values,
                         Value (Builder::*method)(Value, Value)) {
  return [&scalar, values, method](auto) -> FailureOr<Value> {
    return (scalar.*method)(values[0], values[1]);
  };
}

static FailureOr<Value> buildPointwiseScalar(Operation *operation,
                                             ValueRange values,
                                             OpBuilder &builder, Location loc,
                                             Type resultType) {
  Type inputType = cast<RankedTensorType>(operation->getOperand(0).getType())
                       .getElementType();
  bool isUnsigned = isUnsignedInteger(inputType);
  ScalarBuilder scalar(builder, loc);
  auto div = isUnsigned ? &ScalarBuilder::udiv : &ScalarBuilder::sdiv;
  auto rem = isUnsigned ? &ScalarBuilder::urem : &ScalarBuilder::srem;
  auto min = isUnsigned ? &ScalarBuilder::umin : &ScalarBuilder::smin;
  auto max = isUnsigned ? &ScalarBuilder::umax : &ScalarBuilder::smax;
  auto floorDiv =
      isUnsigned ? &ScalarBuilder::ufloorDiv : &ScalarBuilder::sfloorDiv;
  auto pow = isUnsigned ? &ScalarBuilder::upow : &ScalarBuilder::spow;
  auto cmp = isUnsigned ? &ScalarBuilder::ucmp : &ScalarBuilder::scmp;
  return llvm::TypeSwitch<Operation *, FailureOr<Value>>(operation)
      .Case<AddOp>(binaryScalar(scalar, values, &ScalarBuilder::add))
      .Case<SubOp>(binaryScalar(scalar, values, &ScalarBuilder::sub))
      .Case<MulOp>(binaryScalar(scalar, values, &ScalarBuilder::mul))
      .Case<MulHighOp>(binaryScalar(scalar, values, &ScalarBuilder::mulHigh))
      .Case<FmaOp>([&](FmaOp op) {
        return FailureOr<Value>(
            scalar.fma(values[0], values[1], values[2], op.getFtz()));
      })
      .Case<DivOp>(binaryScalar(scalar, values, div))
      .Case<RemOp>(binaryScalar(scalar, values, rem))
      .Case<MinOp>(binaryScalar(scalar, values, min))
      .Case<MaxOp>(binaryScalar(scalar, values, max))
      .Case<AddSquareOp>([&](auto) {
        Value square = scalar.mul(values[1], values[1]);
        return FailureOr<Value>(scalar.add(values[0], square));
      })
      .Case<ModOp>([&](auto) {
        Value quotient = (scalar.*floorDiv)(values[0], values[1]);
        Value product = scalar.mul(quotient, values[1]);
        return FailureOr<Value>(scalar.sub(values[0], product));
      })
      .Case<AbsOp>([&](auto) {
        return FailureOr<Value>(isUnsigned ? values[0] : scalar.abs(values[0]));
      })
      .Case<CeilOp>(unaryScalar(scalar, values, &ScalarBuilder::ceil))
      .Case<CosOp>(unaryScalar(scalar, values, &ScalarBuilder::cos))
      .Case<CoshOp>(unaryScalar(scalar, values, &ScalarBuilder::cosh))
      .Case<Exp2Op>([&](Exp2Op op) {
        return FailureOr<Value>(scalar.exp2(values[0], op.getFtz()));
      })
      .Case<ExpOp>(unaryScalar(scalar, values, &ScalarBuilder::exp))
      .Case<FloorOp>(unaryScalar(scalar, values, &ScalarBuilder::floor))
      .Case<LogOp>(unaryScalar(scalar, values, &ScalarBuilder::log))
      .Case<Log2Op>(unaryScalar(scalar, values, &ScalarBuilder::log2))
      .Case<NegIOp>([&](NegIOp op) {
        Value zero = scalar.constant(values[0].getType(), 0);
        auto overflow = op.getOverflow() == IntegerOverflow::NONE
                            ? arith::IntegerOverflowFlags::none
                            : arith::IntegerOverflowFlags::nsw;
        Value result =
            arith::SubIOp::create(builder, loc, zero, values[0], overflow);
        return FailureOr<Value>(result);
      })
      .Case<NegOp>(unaryScalar(scalar, values, &ScalarBuilder::neg))
      .Case<RsqrtOp>(unaryScalar(scalar, values, &ScalarBuilder::rsqrt))
      .Case<SinOp>(unaryScalar(scalar, values, &ScalarBuilder::sin))
      .Case<SinhOp>(unaryScalar(scalar, values, &ScalarBuilder::sinh))
      .Case<SqrtOp>(unaryScalar(scalar, values, &ScalarBuilder::sqrt))
      .Case<TanOp>(unaryScalar(scalar, values, &ScalarBuilder::tan))
      .Case<ErfOp>(unaryScalar(scalar, values, &ScalarBuilder::erf))
      .Case<TanhFwdOp>(unaryScalar(scalar, values, &ScalarBuilder::tanh))
      .Case<Atan2Op>(binaryScalar(scalar, values, &ScalarBuilder::atan2))
      .Case<ReciprocalOp>([&](auto) {
        Value one = scalar.constant(resultType, 1);
        return FailureOr<Value>((scalar.*div)(one, values[0]));
      })
      .Case<PowOp>([&](auto) {
        Type exponentType =
            cast<RankedTensorType>(operation->getOperand(1).getType())
                .getElementType();
        Value exponent = castFrom(scalar, values[1], resultType, exponentType);
        Value result = (scalar.*pow)(values[0], exponent);
        return FailureOr<Value>(result);
      })
      .Case<AndOp>(binaryScalar(scalar, values, &ScalarBuilder::_and))
      .Case<OrOp>(binaryScalar(scalar, values, &ScalarBuilder::_or))
      .Case<XorOp>(binaryScalar(scalar, values, &ScalarBuilder::_xor))
      .Case<ShiftLeftOp>([&](ShiftLeftOp op) {
        return FailureOr<Value>(
            scalar.shl(values[0], values[1], op.getOverflow()));
      })
      .Case<ShiftRightOp>(binaryScalar(scalar, values,
                                       inputType.isSignedInteger()
                                           ? &ScalarBuilder::sshr
                                           : &ScalarBuilder::ushr))
      .Case<LogicalNotOp>([&](auto) {
        Value one = scalar.constant(resultType, 1);
        Value result = scalar._xor(values[0], one);
        return FailureOr<Value>(result);
      })
      .Case<BinarySelectOp>([&](auto) {
        Value result = scalar.select(values[0], values[1], values[2]);
        return FailureOr<Value>(result);
      })
      .Case<CmpOp>([&](CmpOp op) {
        return FailureOr<Value>(
            (scalar.*cmp)(op.getComparator(), values[0], values[1]));
      })
      .Case<ConvertOp>([&](auto) {
        return FailureOr<Value>(
            castFrom(scalar, values[0], resultType, inputType));
      })
      .Case<ReluFwdOp>([&](auto) {
        Value zero = scalar.constant(resultType, 0);
        return FailureOr<Value>((scalar.*max)(zero, values[0]));
      })
      .Case<SigmoidFwdOp>([&](auto) {
        return FailureOr<Value>(buildSigmoid(scalar, values[0]));
      })
      .Case<GeluApproxTanhFwdOp>([&](auto) {
        return FailureOr<Value>(buildGeluApproxTanh(scalar, values[0]));
      })
      .Case<GeluFwdOp>([&](auto) {
        Value half = scalar.constant(resultType, 0.5);
        Value one = scalar.constant(resultType, 1);
        constexpr double kInverseSqrtTwo = 0.70710678118654752440;
        Value invSqrtTwo = scalar.constant(resultType, kInverseSqrtTwo);
        Value scaled = scalar.mul(values[0], invSqrtTwo);
        Value erf = scalar.erf(scaled);
        Value factor = scalar.add(one, erf);
        Value halfInput = scalar.mul(half, values[0]);
        return FailureOr<Value>(scalar.mul(halfInput, factor));
      })
      .Case<ReluBwdOp>([&](auto) {
        Value zero = scalar.constant(resultType, 0);
        return FailureOr<Value>(
            buildSelectPositive(scalar, values[0], values[1], zero));
      })
      .Case<SigmoidBwdOp>([&](auto) {
        Value one = scalar.constant(resultType, 1);
        Value sigmoid = buildSigmoid(scalar, values[0]);
        Value oneMinusSigmoid = scalar.sub(one, sigmoid);
        Value derivative = scalar.mul(sigmoid, oneMinusSigmoid);
        return FailureOr<Value>(scalar.mul(values[1], derivative));
      })
      .Case<TanhBwdOp>([&](auto) {
        Value one = scalar.constant(resultType, 1);
        Value tanh = scalar.tanh(values[0]);
        Value tanhSquared = scalar.mul(tanh, tanh);
        Value derivative = scalar.sub(one, tanhSquared);
        return FailureOr<Value>(scalar.mul(values[1], derivative));
      })
      .Case<GeluBwdOp>([&](auto) {
        constexpr double kInverseSqrtTwo = 0.70710678118654752440;
        constexpr double kInverseSqrtTwoPi = 0.39894228040143267794;
        Value half = scalar.constant(resultType, 0.5);
        Value one = scalar.constant(resultType, 1);
        Value invSqrtTwo = scalar.constant(resultType, kInverseSqrtTwo);
        Value invSqrtTwoPi = scalar.constant(resultType, kInverseSqrtTwoPi);
        Value scaled = scalar.mul(values[0], invSqrtTwo);
        Value erf = scalar.erf(scaled);
        Value onePlusErf = scalar.add(one, erf);
        Value cdf = scalar.mul(half, onePlusErf);
        Value squared = scalar.mul(values[0], values[0]);
        Value exponent = scalar.mul(half, squared);
        exponent = scalar.neg(exponent);
        Value density = scalar.exp(exponent);
        density = scalar.mul(values[0], density);
        density = scalar.mul(density, invSqrtTwoPi);
        Value derivative = scalar.add(cdf, density);
        return FailureOr<Value>(scalar.mul(values[1], derivative));
      })
      .Case<GeluApproxTanhBwdOp>([&](auto) {
        Value half = scalar.constant(resultType, 0.5);
        Value one = scalar.constant(resultType, 1);
        Value scale = scalar.constant(resultType, 0.7978845608028654);
        Value coefficient = scalar.constant(resultType, 0.044715);
        Value threeCoefficient = scalar.constant(resultType, 0.134145);
        Value squared = scalar.mul(values[0], values[0]);
        Value cubed = scalar.mul(squared, values[0]);
        Value scaledCube = scalar.mul(coefficient, cubed);
        Value polynomial = scalar.add(values[0], scaledCube);
        Value tanhInput = scalar.mul(scale, polynomial);
        Value tanh = scalar.tanh(tanhInput);
        Value tanhSquared = scalar.mul(tanh, tanh);
        Value tanhDerivative = scalar.sub(one, tanhSquared);
        Value scaledSquared = scalar.mul(threeCoefficient, squared);
        Value polynomialDerivative = scalar.add(one, scaledSquared);
        Value tanhInputDerivative = scalar.mul(scale, polynomialDerivative);
        Value product = scalar.mul(tanhDerivative, tanhInputDerivative);
        product = scalar.mul(values[0], product);
        Value onePlusTanh = scalar.add(one, tanh);
        Value derivativeSum = scalar.add(onePlusTanh, product);
        Value derivative = scalar.mul(half, derivativeSum);
        return FailureOr<Value>(scalar.mul(values[1], derivative));
      })
      .Case<SoftplusFwdOp>([&](SoftplusFwdOp op) {
        double betaValue = op.getBeta().convertToDouble();
        Value one = scalar.constant(resultType, 1);
        Value beta = scalar.constant(resultType, betaValue);
        Value scaled = scalar.mul(beta, values[0]);
        Value exp = scalar.exp(scaled);
        Value onePlusExp = scalar.add(one, exp);
        Value log = scalar.log(onePlusExp);
        Value inverseBeta = scalar.constant(resultType, 1.0 / betaValue);
        return FailureOr<Value>(scalar.mul(inverseBeta, log));
      })
      .Case<SwishFwdOp>([&](SwishFwdOp op) {
        Value beta =
            scalar.constant(resultType, op.getBeta().convertToDouble());
        Value betaInput = scalar.mul(beta, values[0]);
        Value sigmoid = buildSigmoid(scalar, betaInput);
        return FailureOr<Value>(scalar.mul(values[0], sigmoid));
      })
      .Case<EluFwdOp>([&](EluFwdOp op) {
        Value one = scalar.constant(resultType, 1);
        Value beta =
            scalar.constant(resultType, op.getBeta().convertToDouble());
        Value exp = scalar.exp(values[0]);
        Value expMinusOne = scalar.sub(exp, one);
        Value negative = scalar.mul(beta, expMinusOne);
        return FailureOr<Value>(
            buildSelectPositive(scalar, values[0], values[0], negative));
      })
      .Case<SoftplusBwdOp>([&](SoftplusBwdOp op) {
        Value beta =
            scalar.constant(resultType, op.getBeta().convertToDouble());
        Value betaInput = scalar.mul(beta, values[0]);
        Value sigmoid = buildSigmoid(scalar, betaInput);
        return FailureOr<Value>(scalar.mul(values[1], sigmoid));
      })
      .Case<SwishBwdOp>([&](SwishBwdOp op) {
        Value one = scalar.constant(resultType, 1);
        Value beta =
            scalar.constant(resultType, op.getBeta().convertToDouble());
        Value betaInput = scalar.mul(beta, values[0]);
        Value sigmoid = buildSigmoid(scalar, betaInput);
        Value oneMinusSigmoid = scalar.sub(one, sigmoid);
        Value sigmoidDerivative = scalar.mul(sigmoid, oneMinusSigmoid);
        Value derivative = scalar.mul(betaInput, sigmoidDerivative);
        derivative = scalar.add(sigmoid, derivative);
        return FailureOr<Value>(scalar.mul(values[1], derivative));
      })
      .Case<EluBwdOp>([&](EluBwdOp op) {
        Value one = scalar.constant(resultType, 1);
        Value beta =
            scalar.constant(resultType, op.getBeta().convertToDouble());
        Value exp = scalar.exp(values[0]);
        Value negativeDerivative = scalar.mul(beta, exp);
        Value derivative =
            buildSelectPositive(scalar, values[0], one, negativeDerivative);
        return FailureOr<Value>(scalar.mul(values[1], derivative));
      })
      .Default([&](Operation *) -> FailureOr<Value> { return failure(); });
}

static Value createParallelGeneric(
    OpBuilder &builder, Location loc, ValueRange inputs, Value destination,
    function_ref<Value(OpBuilder &, Location, ValueRange)> scalarBuilder) {
  auto resultType = cast<RankedTensorType>(destination.getType());
  SmallVector<AffineMap> maps(
      inputs.size() + 1, builder.getMultiDimIdentityMap(resultType.getRank()));
  SmallVector<utils::IteratorType> iterators(resultType.getRank(),
                                             utils::IteratorType::parallel);
  auto generic = linalg::GenericOp::create(
      builder, loc, TypeRange{resultType}, inputs, ValueRange{destination},
      maps, iterators,
      [&](OpBuilder &bodyBuilder, Location bodyLoc, ValueRange args) {
        Value result = scalarBuilder(bodyBuilder, bodyLoc, args.drop_back());
        linalg::YieldOp::create(bodyBuilder, bodyLoc, result);
      });
  return generic.getResult(0);
}

namespace {
template <typename OpTy>
class PointwiseOpConversion : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op->getResult(0).getType()));
    if constexpr (std::is_same_v<OpTy, PowOp>) {
      Type elementType = op.getType().getElementType();
      if (isUnsignedInteger(elementType) &&
          elementType.getIntOrFloatBitWidth() > 32) {
        return rewriter.notifyMatchFailure(
            op, "unsigned integer pow supports input widths up to 32 bits");
      }
    }
    SmallVector<Value> inputs(adaptor.getOperands());
    Value shapeSource = inputs.empty() ? Value() : inputs.front();
    FailureOr<Value> empty =
        createEmptyTensor(rewriter, op.getLoc(), resultType, shapeSource);
    if (failed(empty)) {
      return rewriter.notifyMatchFailure(
          op, "cannot derive dynamic result dimensions from the first input");
    }

    Value result = createParallelGeneric(
        rewriter, op.getLoc(), inputs, *empty,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          FailureOr<Value> scalar = buildPointwiseScalar(
              op.getOperation(), args, builder, loc,
              cast<RankedTensorType>(op->getResult(0).getType())
                  .getElementType());
          assert(succeeded(scalar) && "registered pointwise op must lower");
          return *scalar;
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

class GraphOpConversion : public OpConversionPattern<GraphOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GraphOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!sortTopologically(op.getBody())) {
      return op.emitError("cannot convert a graph with cyclic dependencies");
    }

    auto function =
        func::FuncOp::create(rewriter, op.getLoc(), op.getSymName(),
                             cast<FunctionType>(getTypeConverter()->convertType(
                                 op.getFunctionType())));
    if (ArrayAttr argAttrs = op.getArgAttrsAttr()) {
      function.setArgAttrsAttr(argAttrs);
    }
    if (ArrayAttr resultAttrs = op.getResAttrsAttr()) {
      function.setResAttrsAttr(resultAttrs);
    }
    function.setSymVisibilityAttr(op.getSymVisibilityAttr());
    for (NamedAttribute attribute : op->getAttrs()) {
      StringRef name = attribute.getName().strref();
      if (name == op.getSymNameAttrName() || name == "function_type" ||
          name == "arg_attrs" || name == "res_attrs" ||
          name == op.getSymVisibilityAttrName()) {
        continue;
      }
      function->setAttr(attribute.getName(), attribute.getValue());
    }

    rewriter.inlineRegionBefore(op.getGraphBody(), function.getBody(),
                                function.getBody().end());
    if (failed(rewriter.convertRegionTypes(&function.getBody(),
                                           *getTypeConverter()))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class ResultsOpConversion : public OpConversionPattern<ResultsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ResultsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getResults());
    return success();
  }
};

class ConstantOpConversion : public OpConversionPattern<ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, getSignlessConstant(op.getValue()));
    return success();
  }
};

class DimOpConversion : public OpConversionPattern<DimOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tensor::DimOp>(
        op, adaptor.getInput(), static_cast<int64_t>(op.getDimension()));
    return success();
  }
};

class SplatOpConversion : public OpConversionPattern<SplatOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
    FailureOr<Value> empty = createEmptyTensor(
        rewriter, op.getLoc(), resultType, adaptor.getDynamicSizes());
    if (failed(empty)) {
      return rewriter.notifyMatchFailure(
          op, "dynamic result dimensions require dynamic_dims operands");
    }
    auto fill = linalg::FillOp::create(rewriter, op.getLoc(),
                                       adaptor.getInput(), *empty);
    rewriter.replaceOp(op, fill.getResults());
    return success();
  }
};

class ConcatenateOpConversion : public OpConversionPattern<ConcatenateOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConcatenateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
    ValueRange inputs = adaptor.getInputs();
    if (inputs.size() == 1 && inputs.front().getType() == resultType) {
      rewriter.replaceOp(op, inputs.front());
      return success();
    }
    Location loc = op.getLoc();
    int64_t dimension = op.getDimension();
    // This MLIR version bufferizes static empty concat operands into invalid
    // boundary subviews. Empty results need only their runtime shape.
    if (llvm::is_contained(resultType.getShape(), 0)) {
      SmallVector<Value> dynamicSizes;
      for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
        if (!resultType.isDynamicDim(dim)) {
          continue;
        }
        Value size = getDim(rewriter, loc, inputs.front(), dim);
        if (dim == dimension) {
          for (Value input : inputs.drop_front()) {
            Value extent = getDim(rewriter, loc, input, dim);
            size = rewriter.createOrFold<arith::AddIOp>(loc, size, extent);
          }
        }
        dynamicSizes.push_back(size);
      }
      FailureOr<Value> empty =
          createEmptyTensor(rewriter, loc, resultType, dynamicSizes);
      assert(succeeded(empty) && "constructed all empty concat dimensions");
      rewriter.replaceOp(op, *empty);
      return success();
    }
    SmallVector<Value> nonemptyInputs;
    for (Value input : inputs) {
      auto inputType = cast<RankedTensorType>(input.getType());
      if (inputType.getDimSize(dimension) != 0) {
        nonemptyInputs.push_back(input);
      }
    }
    rewriter.replaceOpWithNewOp<tensor::ConcatOp>(op, resultType, dimension,
                                                  nonemptyInputs);
    return success();
  }
};

class BroadcastOpConversion : public OpConversionPattern<BroadcastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
    auto resultType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
    SmallVector<int64_t> broadcastDimensions;
    SmallVector<ReassociationIndices> reassociation(1);
    for (int64_t index = 0; index < inputType.getRank(); ++index) {
      int64_t inputDim = inputType.getDimSize(index);
      int64_t resultDim = resultType.getDimSize(index);
      reassociation.back().push_back(index);
      if (inputDim != resultDim) {
        broadcastDimensions.push_back(index);
        continue;
      }
      reassociation.emplace_back();
    }
    if (reassociation.size() <= 1) {
      reassociation.clear();
    } else {
      llvm::append_range(reassociation[reassociation.size() - 2],
                         reassociation.back());
      reassociation.pop_back();
    }

    Value collapsed = tensor::CollapseShapeOp::create(
        rewriter, op.getLoc(), adaptor.getInput(), reassociation);
    FailureOr<Value> empty = createEmptyTensor(
        rewriter, op.getLoc(), resultType, adaptor.getDynamicSizes());
    if (failed(empty)) {
      return rewriter.notifyMatchFailure(
          op, "dynamic result dimensions require dynamic_dims operands");
    }
    auto broadcast = linalg::BroadcastOp::create(
        rewriter, op.getLoc(), collapsed, *empty, broadcastDimensions);
    rewriter.replaceOp(op, broadcast.getResults());
    return success();
  }
};

class ReshapeOpConversion : public OpConversionPattern<ReshapeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
    auto resultType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
    if (inputType == resultType && inputType.hasStaticShape()) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    if (auto reassociation =
            getReassociationIndicesForReshape(inputType, resultType)) {
      if (inputType.getRank() > resultType.getRank()) {
        rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
            op, resultType, adaptor.getInput(), *reassociation);
        return success();
      }
      if (adaptor.getDynamicSizes().size() == resultType.getNumDynamicDims()) {
        SmallVector<OpFoldResult> outputShape = getMixedValues(
            resultType.getShape(), adaptor.getDynamicSizes(), rewriter);
        rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
            op, resultType, adaptor.getInput(), *reassociation, outputShape);
        return success();
      }
    }

    if (adaptor.getDynamicSizes().size() != resultType.getNumDynamicDims()) {
      return rewriter.notifyMatchFailure(
          op, "non-reassociative dynamic reshape requires dynamic_dims");
    }
    SmallVector<OpFoldResult> mixedShape = getMixedValues(
        resultType.getShape(), adaptor.getDynamicSizes(), rewriter);
    SmallVector<Value> shapeElements =
        getValueOrCreateConstantIndexOp(rewriter, op.getLoc(), mixedShape);
    Value shape =
        tensor::FromElementsOp::create(rewriter, op.getLoc(), shapeElements);
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, resultType,
                                                   adaptor.getInput(), shape);
    return success();
  }
};

class TransposeOpConversion : public OpConversionPattern<TransposeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
    SmallVector<Value> dynamicSizes;
    for (auto [resultDim, size] : llvm::enumerate(resultType.getShape())) {
      if (!ShapedType::isDynamic(size)) {
        continue;
      }
      Value dynamicSize = getDim(rewriter, op.getLoc(), adaptor.getInput(),
                                 op.getPermutation()[resultDim]);
      dynamicSizes.push_back(dynamicSize);
    }
    FailureOr<Value> empty =
        createEmptyTensor(rewriter, op.getLoc(), resultType, dynamicSizes);
    assert(succeeded(empty) && "constructed all dynamic dimensions");
    auto transpose = linalg::TransposeOp::create(
        rewriter, op.getLoc(), adaptor.getInput(), *empty,
        llvm::to_vector(op.getPermutation()));
    rewriter.replaceOp(op, transpose.getResults());
    return success();
  }
};

class SliceOpConversion : public OpConversionPattern<SliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
    SmallVector<OpFoldResult> offsets =
        llvm::map_to_vector(op.getStarts(), [&](int64_t value) -> OpFoldResult {
          return rewriter.getIndexAttr(value);
        });
    SmallVector<OpFoldResult> sizes = llvm::map_to_vector(
        resultType.getShape(), [&](int64_t value) -> OpFoldResult {
          return rewriter.getIndexAttr(value);
        });
    SmallVector<OpFoldResult> strides = llvm::map_to_vector(
        op.getStrides(), [&](int64_t value) -> OpFoldResult {
          return rewriter.getIndexAttr(value);
        });
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, resultType, adaptor.getInput(), offsets, sizes, strides);
    return success();
  }
};

class IotaOpConversion : public OpConversionPattern<IotaOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IotaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
    FailureOr<Value> empty = createEmptyTensor(
        rewriter, op.getLoc(), resultType, adaptor.getDynamicSizes());
    if (failed(empty)) {
      return rewriter.notifyMatchFailure(
          op, "dynamic result dimensions require dynamic_dims operands");
    }
    Value result = createParallelGeneric(
        rewriter, op.getLoc(), ValueRange(), *empty,
        [&](OpBuilder &builder, Location loc, ValueRange) {
          ScalarBuilder scalar(builder, loc);
          Value index = linalg::IndexOp::create(
              builder, loc, static_cast<int64_t>(op.getDimension()));
          return scalar.scast(index, resultType.getElementType());
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

static Value createReductionIdentity(OpBuilder &builder, Location loc,
                                     Type type, ReductionMode mode) {
  ScalarBuilder scalar(builder, loc);
  if (mode == ReductionMode::add || mode == ReductionMode::avg ||
      mode == ReductionMode::norm1 || mode == ReductionMode::norm2) {
    return scalar.constant(type, 0);
  }
  if (mode == ReductionMode::mul || mode == ReductionMode::mul_no_zeros) {
    return scalar.constant(type, 1);
  }
  if (auto floatType = dyn_cast<FloatType>(type)) {
    const llvm::fltSemantics &semantics = floatType.getFloatSemantics();
    bool negative = mode != ReductionMode::min;
    APFloat value = APFloat::semanticsHasInf(semantics)
                        ? APFloat::getInf(semantics, negative)
                        : APFloat::getLargest(semantics, negative);
    return arith::ConstantOp::create(builder, loc,
                                     builder.getFloatAttr(floatType, value));
  }
  auto integerType = cast<IntegerType>(type);
  unsigned width = integerType.getWidth();
  APInt value =
      mode == ReductionMode::min
          ? (integerType.isUnsigned() ? APInt::getMaxValue(width)
                                      : APInt::getSignedMaxValue(width))
          : (integerType.isUnsigned() ? APInt::getMinValue(width)
                                      : APInt::getSignedMinValue(width));
  return arith::ConstantOp::create(
      builder, loc,
      builder.getIntegerAttr(getSignlessType(integerType), value));
}

static Value buildReductionScalar(OpBuilder &builder, Location loc,
                                  ReductionMode mode, Value input,
                                  Value accumulator, Type sourceType) {
  bool isUnsigned = isUnsignedInteger(sourceType);
  ScalarBuilder scalar(builder, loc);
  if (!isUnsigned &&
      (mode == ReductionMode::amax || mode == ReductionMode::norm1)) {
    input = scalar.abs(input);
  }
  if (mode == ReductionMode::norm2) {
    input = scalar.mul(input, input);
  }
  if (mode == ReductionMode::mul_no_zeros) {
    Value zero = scalar.constant(input.getType(), 0);
    Comparator comparison =
        isa<FloatType>(input.getType()) ? Comparator::oeq : Comparator::eq;
    Value isZero = scalar.scmp(comparison, input, zero);
    Value product = scalar.mul(accumulator, input);
    return scalar.select(isZero, accumulator, product);
  }
  if (mode == ReductionMode::mul) {
    return scalar.mul(accumulator, input);
  }
  if (mode == ReductionMode::min) {
    return isUnsigned ? scalar.umin(accumulator, input)
                      : scalar.smin(accumulator, input);
  }
  if (mode == ReductionMode::max || mode == ReductionMode::amax) {
    return isUnsigned ? scalar.umax(accumulator, input)
                      : scalar.smax(accumulator, input);
  }
  return scalar.add(accumulator, input);
}

static SmallVector<Value>
getDynamicResultSizes(OpBuilder &builder, IterationDomainOpInterface domain,
                      RankedTensorType resultType, ValueRange inputs,
                      unsigned resultNumber = 0) {
  auto inputMaps = domain.getIndexingMapsArray();
  AffineMap resultMap = domain.getResultIndexingMaps()[resultNumber];
  Location loc = domain.getLoc();
  SmallVector<Value> dynamicSizes;
  for (auto [dimension, size] : llvm::enumerate(resultType.getShape())) {
    if (!ShapedType::isDynamic(size)) {
      continue;
    }
    Value extent;
    for (auto [input, map] : llvm::zip(inputs, inputMaps)) {
      auto inputDimension =
          map.getResultPosition(resultMap.getResult(dimension));
      if (!inputDimension) {
        continue;
      }
      extent = getDim(builder, loc, input, *inputDimension);
      break;
    }
    assert(extent && "dynamic result dimension must be supplied by an operand");
    dynamicSizes.push_back(extent);
  }
  return dynamicSizes;
}

namespace {
class ReductionOpConversion
    : public OpInterfaceConversionPattern<IterationDomainOpInterface> {
public:
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

  LogicalResult
  matchAndRewrite(IterationDomainOpInterface op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto builtin = dyn_cast<ReduceOp>(op.getOperation());
    auto custom = dyn_cast<ReduceUDOp>(op.getOperation());
    if (!builtin && !custom) {
      return failure();
    }
    ReductionMode mode =
        builtin ? builtin.getReductionMode() : ReductionMode::customize;
    if (builtin && mode == ReductionMode::customize) {
      return rewriter.notifyMatchFailure(
          op, "custom reductions use nv_tensor_ir.reduce_ud");
    }
    Type sourceType =
        cast<RankedTensorType>(op->getResult(0).getType()).getElementType();
    if (mode == ReductionMode::norm2 &&
        !isa<FloatType>(cast<RankedTensorType>(operands.front().getType())
                            .getElementType())) {
      return rewriter.notifyMatchFailure(op,
                                         "norm2 requires floating-point input");
    }

    // Refine compatible input shapes to the same static dimensions for Linalg.
    SmallVector<Value> inputs(operands);
    SmallVector<int64_t> shape(
        cast<RankedTensorType>(inputs.front().getType()).getShape());
    for (Value input : llvm::drop_begin(inputs)) {
      for (auto [dim, size] : llvm::enumerate(
               cast<RankedTensorType>(input.getType()).getShape())) {
        if (!ShapedType::isDynamic(size)) {
          shape[dim] = size;
        }
      }
    }
    for (Value &input : inputs) {
      auto type = cast<RankedTensorType>(input.getType());
      auto mergedType = type.clone(shape);
      if (type != mergedType) {
        input =
            tensor::CastOp::create(rewriter, op.getLoc(), mergedType, input);
      }
    }
    SmallVector<Value> inits;
    SmallVector<Type> resultTypes;
    for (OpResult result : op->getResults()) {
      auto resultType = cast<RankedTensorType>(
          getTypeConverter()->convertType(result.getType()));
      SmallVector<Value> dynamicSizes = getDynamicResultSizes(
          rewriter, op, resultType, inputs, result.getResultNumber());
      Value empty = tensor::EmptyOp::create(rewriter, op.getLoc(), resultType,
                                            dynamicSizes);
      Value identity;
      if (builtin) {
        identity =
            createReductionIdentity(rewriter, op.getLoc(), sourceType, mode);
      } else {
        identity = arith::ConstantOp::create(
            rewriter, op.getLoc(),
            getSignlessConstant(cast<TypedAttr>(
                custom.getIdentity()[result.getResultNumber()])));
      }
      Value init =
          linalg::FillOp::create(rewriter, op.getLoc(), identity, empty)
              .getResult(0);
      inits.push_back(init);
      resultTypes.push_back(resultType);
    }
    auto reduce = linalg::GenericOp::create(
        rewriter, op.getLoc(), resultTypes, inputs, inits,
        op.getIterationDomainMaps(), op.getLoopIteratorTypes(),
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          SmallVector<Value> yielded;
          if (builtin) {
            Value result = buildReductionScalar(
                builder, loc, mode, args[0], args[1],
                builtin.getInput().getType().getElementType());
            yielded.push_back(result);
          } else {
            Block &source = custom.getBody().front();
            unsigned count = custom.getNumResults();
            IRMapping mapping;
            mapping.map(source.getArguments().take_front(count),
                        args.drop_front(count));
            mapping.map(source.getArguments().drop_front(count),
                        args.take_front(count));
            for (Operation &nested : source.without_terminator()) {
              builder.clone(nested, mapping);
            }
            for (Value value : source.getTerminator()->getOperands()) {
              yielded.push_back(mapping.lookup(value));
            }
          }
          linalg::YieldOp::create(builder, loc, yielded);
        });
    SmallVector<Value> results(reduce.getResults());
    if (mode == ReductionMode::norm2 || mode == ReductionMode::avg) {
      Value divisor;
      if (mode == ReductionMode::avg) {
        Value count = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 1);
        for (int64_t dimension : op.getReductionDims()) {
          Value extent =
              getDim(rewriter, op.getLoc(), inputs.front(), dimension);
          count = arith::MulIOp::create(rewriter, op.getLoc(), count, extent);
        }
        ScalarBuilder scalar(rewriter, op.getLoc());
        divisor = scalar.scast(count, sourceType);
      }
      auto resultType = cast<RankedTensorType>(results.front().getType());
      FailureOr<Value> empty =
          createEmptyTensor(rewriter, op.getLoc(), resultType, results.front());
      assert(succeeded(empty) && "result supplies its own dynamic dimensions");
      results.front() = createParallelGeneric(
          rewriter, op.getLoc(), ValueRange{results.front()}, *empty,
          [&](OpBuilder &builder, Location loc, ValueRange args) {
            ScalarBuilder scalar(builder, loc);
            if (mode == ReductionMode::norm2) {
              return scalar.sqrt(args.front());
            }
            return isUnsignedInteger(sourceType)
                       ? scalar.udiv(args.front(), divisor)
                       : scalar.sdiv(args.front(), divisor);
          });
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

template <typename OpTy>
class ContractionOpConversion : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using typename OpConversionPattern<OpTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    Value inputs[] = {adaptor.getA(), adaptor.getB()};
    RankedTensorType inputTypes[] = {op.getA().getType(), op.getB().getType()};
    if (std::is_same_v<OpTy, ContractionOp> &&
        inputTypes[0].getElementType() != inputTypes[1].getElementType()) {
      return rewriter.notifyMatchFailure(
          op, "contraction inputs must have the same element type");
    }

    SmallVector<AffineMap> maps = op.getIterationDomainMaps();
    if (op.getAcc()) {
      maps.erase(maps.begin() + 2);
    }
    SmallVector<Value> dynamicSizes =
        getDynamicResultSizes(rewriter, op, resultType, inputs);

    Value init = adaptor.getAcc();
    if (!init) {
      FailureOr<Value> empty =
          createEmptyTensor(rewriter, op.getLoc(), resultType, dynamicSizes);
      assert(succeeded(empty) && "constructed all dynamic dimensions");
      ScalarBuilder scalar(rewriter, op.getLoc());
      Value zero = scalar.constant(resultType.getElementType(), 0);
      init = linalg::FillOp::create(rewriter, op.getLoc(), zero, *empty)
                 .getResult(0);
    }
    if constexpr (std::is_same_v<OpTy, MatmulOp>) {
      // Named matmuls support at most one batch dimension and floating-point
      // accumulation. Integer MMA requires the scalar saturation semantics.
      if (!isa<IntegerType>(inputTypes[0].getElementType()) &&
          !isa<IntegerType>(resultType.getElementType()) &&
          resultType.getRank() <= 3) {
        if (resultType.getRank() == 2) {
          rewriter.replaceOpWithNewOp<linalg::MatmulOp>(op, ValueRange(inputs),
                                                        ValueRange{init});
        } else {
          rewriter.replaceOpWithNewOp<linalg::BatchMatmulOp>(
              op, ValueRange(inputs), ValueRange{init});
        }
        return success();
      }
    }
    auto generic = linalg::GenericOp::create(
        rewriter, op.getLoc(), TypeRange{resultType}, ValueRange(inputs),
        ValueRange{init}, maps, op.getLoopIteratorTypes(),
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value sum = buildMmaScalar(builder, loc, args[0], args[1], args[2],
                                     inputTypes[0].getElementType(),
                                     inputTypes[1].getElementType());
          linalg::YieldOp::create(builder, loc, sum);
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

using SupportedPointwiseOps =
    std::tuple<AbsOp, AddOp, AddSquareOp, Atan2Op, BinarySelectOp, CeilOp,
               CmpOp, CosOp, CoshOp, DivOp, EluBwdOp, EluFwdOp, ErfOp, ExpOp,
               Exp2Op, FloorOp, FmaOp, GeluApproxTanhBwdOp, GeluApproxTanhFwdOp,
               GeluBwdOp, GeluFwdOp, LogOp, Log2Op, AndOp, LogicalNotOp, OrOp,
               MaxOp, MinOp, ModOp, MulOp, MulHighOp, NegOp, NegIOp, PowOp,
               ReciprocalOp, ReluBwdOp, ReluFwdOp, RemOp, RsqrtOp, SigmoidBwdOp,
               SigmoidFwdOp, SinOp, SinhOp, SoftplusBwdOp, SoftplusFwdOp,
               SqrtOp, SubOp, SwishBwdOp, SwishFwdOp, TanOp, TanhBwdOp,
               TanhFwdOp, ConvertOp, XorOp, ShiftLeftOp, ShiftRightOp>;

template <typename... OpTys>
void addPointwisePatterns(RewritePatternSet &patterns,
                          const TypeConverter &typeConverter,
                          std::tuple<OpTys...> *) {
  patterns.add<PointwiseOpConversion<OpTys>...>(typeConverter,
                                                patterns.getContext());
}

template <typename... OpTys>
void markPointwiseOpsIllegal(ConversionTarget &target, std::tuple<OpTys...> *) {
  target.addIllegalOp<OpTys...>();
}

class TensorToLinalgConversionPass
    : public nv_tensor_ir::impl::TensorToLinalgConversionPassBase<
          TensorToLinalgConversionPass> {
public:
  using TensorToLinalgConversionPassBase::TensorToLinalgConversionPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    TypeConverter typeConverter;
    typeConverter.addConversion(getSignlessType);
    // Preserve the boundary to operations outside this pass's conversion
    // contract, whose TensorIR integer types must remain unchanged.
    auto materialize = [](OpBuilder &builder, Type type, ValueRange inputs,
                          Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
          .getResult(0);
    };
    typeConverter.addSourceMaterialization(materialize);
    typeConverter.addTargetMaterialization(materialize);

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           linalg::LinalgDialect, math::MathDialect,
                           tensor::TensorDialect>();
    target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(
        [&](Operation *op) { return typeConverter.isLegal(op); });
    target.addIllegalOp<GraphOp, ResultsOp, ConstantOp, DimOp, SplatOp,
                        BroadcastOp, ReshapeOp, TransposeOp, SliceOp, IotaOp,
                        MatmulOp, ContractionOp, ConcatenateOp, ReduceOp,
                        ReduceUDOp>();
    markPointwiseOpsIllegal(target,
                            static_cast<SupportedPointwiseOps *>(nullptr));

    RewritePatternSet patterns(context);
    patterns.add<GraphOpConversion, ResultsOpConversion, ConstantOpConversion,
                 DimOpConversion, SplatOpConversion, BroadcastOpConversion,
                 ReshapeOpConversion, TransposeOpConversion, SliceOpConversion,
                 IotaOpConversion, ContractionOpConversion<MatmulOp>,
                 ContractionOpConversion<ContractionOp>,
                 ConcatenateOpConversion, ReductionOpConversion>(typeConverter,
                                                                 context);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    addPointwisePatterns(patterns, typeConverter,
                         static_cast<SupportedPointwiseOps *>(nullptr));

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
