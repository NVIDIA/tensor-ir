// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// \file
/// Lower the public, portable TensorIR operation subset to tensor-semantics
/// Linalg and Tensor operations.

#include "tensor_ir/Conversion/TensorToLinalg/TensorToLinalg.h"

#include "tensor_ir/Dialect/TensorIR.h" // IWYU pragma: keep

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
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

// Model MMA with widened products. Integer accumulation uses MmaIOp's default
// signed saturation; floating-point arithmetic is a portable reference, not a
// bit-exact reproduction of hardware MMA's intermediate precision or ordering.
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

// Scalar callback factories defer IR emission until their TypeSwitch case matches.
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
      .Case<ExpOp>(unaryScalar(scalar, values, &ScalarBuilder::exp))
      .Case<FloorOp>(unaryScalar(scalar, values, &ScalarBuilder::floor))
      .Case<LogOp>(unaryScalar(scalar, values, &ScalarBuilder::log))
      .Case<NegOp>(unaryScalar(scalar, values, &ScalarBuilder::neg))
      .Case<RsqrtOp>(unaryScalar(scalar, values, &ScalarBuilder::rsqrt))
      .Case<SinOp>(unaryScalar(scalar, values, &ScalarBuilder::sin))
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
      .Case<LogicalAndOp>(binaryScalar(scalar, values, &ScalarBuilder::_and))
      .Case<LogicalOrOp>(binaryScalar(scalar, values, &ScalarBuilder::_or))
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

SmallVector<OpFoldResult> getMixedResultShape(OpBuilder &builder, Location loc,
                                              RankedTensorType resultType,
                                              ValueRange dynamicSizes) {
  SmallVector<OpFoldResult> shape;
  auto dynamicSize = dynamicSizes.begin();
  for (int64_t size : resultType.getShape()) {
    if (ShapedType::isDynamic(size)) {
      assert(dynamicSize != dynamicSizes.end());
      shape.push_back(*dynamicSize++);
    } else {
      shape.push_back(builder.getIndexAttr(size));
    }
  }
  return shape;
}

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
        SmallVector<OpFoldResult> outputShape = getMixedResultShape(
            rewriter, op.getLoc(), resultType, adaptor.getDynamicSizes());
        rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
            op, resultType, adaptor.getInput(), *reassociation, outputShape);
        return success();
      }
      if (resultType.hasStaticShape()) {
        rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
            op, resultType, adaptor.getInput(), *reassociation);
        return success();
      }
    }

    if (adaptor.getDynamicSizes().size() != resultType.getNumDynamicDims()) {
      return rewriter.notifyMatchFailure(
          op, "non-reassociative dynamic reshape requires dynamic_dims");
    }
    SmallVector<Value> shapeElements;
    auto dynamicSize = adaptor.getDynamicSizes().begin();
    for (int64_t size : resultType.getShape()) {
      if (ShapedType::isDynamic(size)) {
        shapeElements.push_back(*dynamicSize++);
      } else {
        Value shapeElement =
            arith::ConstantIndexOp::create(rewriter, op.getLoc(), size);
        shapeElements.push_back(shapeElement);
      }
    }
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

static FailureOr<Value> mapReducedTensor(
    ConversionPatternRewriter &rewriter, Location loc, Value input,
    function_ref<Value(OpBuilder &, Location, Value)> scalarBuilder) {
  auto resultType = cast<RankedTensorType>(input.getType());
  FailureOr<Value> empty = createEmptyTensor(rewriter, loc, resultType, input);
  if (failed(empty)) {
    return failure();
  }
  return createParallelGeneric(
      rewriter, loc, ValueRange{input}, *empty,
      [&](OpBuilder &builder, Location bodyLoc, ValueRange args) {
        return scalarBuilder(builder, bodyLoc, args.front());
      });
}

struct ReductionShape {
  SmallVector<int64_t> shape;
  SmallVector<Value> dynamicSizes;
};

static ReductionShape getReductionShape(OpBuilder &builder, Location loc,
                                        RankedTensorType inputType, Value input,
                                        ArrayRef<int64_t> reductionDims) {
  ReductionShape reduced;
  for (auto [dim, size] : llvm::enumerate(inputType.getShape())) {
    if (llvm::is_contained(reductionDims, dim)) {
      continue;
    }
    reduced.shape.push_back(size);
    if (ShapedType::isDynamic(size)) {
      Value dynamicSize = getDim(builder, loc, input, dim);
      reduced.dynamicSizes.push_back(dynamicSize);
    }
  }
  return reduced;
}

static Value createReductionInit(OpBuilder &builder, Location loc,
                                 RankedTensorType resultType,
                                 const ReductionShape &reduced,
                                 function_ref<Value()> buildIdentity) {
  auto reducedType = RankedTensorType::get(
      reduced.shape, resultType.getElementType(), resultType.getEncoding());
  FailureOr<Value> empty =
      createEmptyTensor(builder, loc, reducedType, reduced.dynamicSizes);
  assert(succeeded(empty) && "constructed all reduced dynamic dimensions");
  Value identity = buildIdentity();
  return linalg::FillOp::create(builder, loc, identity, *empty).getResult(0);
}

static Value broadcastReductionResult(OpBuilder &builder, Location loc,
                                      RankedTensorType resultType, Value input,
                                      Value reduced,
                                      ArrayRef<int64_t> reductionDims) {
  if (reduced.getType() == resultType) {
    return reduced;
  }
  SmallVector<Value> dynamicSizes;
  for (auto [dim, size] : llvm::enumerate(resultType.getShape())) {
    if (!ShapedType::isDynamic(size)) {
      continue;
    }
    if (llvm::is_contained(reductionDims, dim)) {
      Value dynamicSize = arith::ConstantIndexOp::create(builder, loc, 1);
      dynamicSizes.push_back(dynamicSize);
    } else {
      Value dynamicSize = getDim(builder, loc, input, dim);
      dynamicSizes.push_back(dynamicSize);
    }
  }
  FailureOr<Value> empty =
      createEmptyTensor(builder, loc, resultType, dynamicSizes);
  assert(succeeded(empty) && "constructed all result dynamic dimensions");
  auto broadcast =
      linalg::BroadcastOp::create(builder, loc, reduced, *empty, reductionDims);
  return broadcast.getResult().front();
}

namespace {
class ReduceOpConversion : public OpConversionPattern<ReduceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ReductionMode mode = op.getReductionMode();
    if (mode == ReductionMode::customize) {
      return rewriter.notifyMatchFailure(
          op, "custom reductions use nv_tensor_ir.reduce_ud");
    }

    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
    auto resultType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
    if (mode == ReductionMode::norm2 &&
        !isa<FloatType>(inputType.getElementType())) {
      return rewriter.notifyMatchFailure(op,
                                         "norm2 requires floating-point input");
    }
    SmallVector<int64_t> dimensions =
        llvm::to_vector_of<int64_t>(op.getDimensions());
    llvm::sort(dimensions);
    ReductionShape reduced = getReductionShape(rewriter, op.getLoc(), inputType,
                                               adaptor.getInput(), dimensions);
    Value init =
        createReductionInit(rewriter, op.getLoc(), resultType, reduced, [&]() {
          return createReductionIdentity(rewriter, op.getLoc(),
                                         op.getType().getElementType(), mode);
        });
    auto reduce = linalg::ReduceOp::create(
        rewriter, op.getLoc(), ValueRange{adaptor.getInput()}, ValueRange{init},
        dimensions, [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value result =
              buildReductionScalar(builder, loc, mode, args[0], args[1],
                                   op.getInput().getType().getElementType());
          linalg::YieldOp::create(builder, loc, result);
        });

    Value result = broadcastReductionResult(rewriter, op.getLoc(), resultType,
                                            adaptor.getInput(),
                                            reduce.getResult(0), dimensions);

    if (mode == ReductionMode::norm2) {
      FailureOr<Value> mapped = mapReducedTensor(
          rewriter, op.getLoc(), result,
          [](OpBuilder &builder, Location loc, Value value) -> Value {
            ScalarBuilder scalar(builder, loc);
            return scalar.sqrt(value);
          });
      assert(succeeded(mapped) && "result shape is available");
      result = *mapped;
    } else if (mode == ReductionMode::avg) {
      Value count = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 1);
      for (int64_t dim : dimensions) {
        Value extent = getDim(rewriter, op.getLoc(), adaptor.getInput(), dim);
        count = arith::MulIOp::create(rewriter, op.getLoc(), count, extent);
      }
      Type sourceElementType = op.getType().getElementType();
      ScalarBuilder scalar(rewriter, op.getLoc());
      Value typedCount = scalar.scast(count, sourceElementType);
      FailureOr<Value> mapped =
          mapReducedTensor(rewriter, op.getLoc(), result,
                           [&](OpBuilder &builder, Location loc, Value value) {
                             ScalarBuilder scalar(builder, loc);
                             return isUnsignedInteger(sourceElementType)
                                        ? scalar.udiv(value, typedCount)
                                        : scalar.sdiv(value, typedCount);
                           });
      assert(succeeded(mapped) && "result shape is available");
      result = *mapped;
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

class ReduceUDOpConversion : public OpConversionPattern<ReduceUDOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReduceUDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<int64_t> dimensions =
        llvm::to_vector_of<int64_t>(op.getDimensions());
    llvm::sort(dimensions);
    auto inputType =
        cast<RankedTensorType>(adaptor.getInputs().front().getType());
    SmallVector<int64_t> shape(inputType.getShape());
    for (Value input : adaptor.getInputs()) {
      auto type = cast<RankedTensorType>(input.getType());
      for (auto [dim, size] : llvm::enumerate(type.getShape())) {
        if (!ShapedType::isDynamic(size)) {
          shape[dim] = size;
        }
      }
    }

    // The verifier merges compatible input shapes to determine the result
    // shape. Linalg requires identical input shapes, including dynamic dims.
    SmallVector<Value> inputs;
    for (Value input : adaptor.getInputs()) {
      auto type = cast<RankedTensorType>(input.getType());
      auto mergedType = type.clone(shape);
      if (type != mergedType) {
        input =
            tensor::CastOp::create(rewriter, op.getLoc(), mergedType, input);
      }
      inputs.push_back(input);
    }
    ReductionShape reduced = getReductionShape(
        rewriter, op.getLoc(), cast<RankedTensorType>(inputs.front().getType()),
        inputs.front(), dimensions);

    SmallVector<Value> inits;
    for (auto [result, identityAttr] :
         llvm::zip_equal(op.getResults(), op.getIdentity())) {
      auto resultType = cast<RankedTensorType>(
          getTypeConverter()->convertType(result.getType()));
      Value init = createReductionInit(
          rewriter, op.getLoc(), resultType, reduced, [&]() -> Value {
            return arith::ConstantOp::create(
                rewriter, op.getLoc(),
                getSignlessConstant(cast<TypedAttr>(identityAttr)));
          });
      inits.push_back(init);
    }

    auto reduce = linalg::ReduceOp::create(
        rewriter, op.getLoc(), inputs, inits, dimensions,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Block &source = op.getBody().front();
          unsigned resultCount = op.getNumResults();
          IRMapping mapping;
          for (unsigned i = 0; i < resultCount; ++i) {
            mapping.map(source.getArgument(i), args[resultCount + i]);
            mapping.map(source.getArgument(resultCount + i), args[i]);
          }
          for (Operation &nested : source.without_terminator()) {
            builder.clone(nested, mapping);
          }
          auto yield = cast<YieldOp>(source.getTerminator());
          SmallVector<Value> yielded =
              llvm::map_to_vector(yield.getOperands(), [&](Value value) {
                return mapping.lookup(value);
              });
          linalg::YieldOp::create(builder, loc, yielded);
        });

    SmallVector<Value> replacements;
    for (auto [index, result] : llvm::enumerate(op.getResults())) {
      auto resultType = cast<RankedTensorType>(
          getTypeConverter()->convertType(result.getType()));
      Value resultValue = broadcastReductionResult(
          rewriter, op.getLoc(), resultType, inputs.front(),
          reduce.getResult(index), dimensions);
      replacements.push_back(resultValue);
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

// Dynamic batch dimensions can be either unit or full-sized at runtime. Expand
// them before the contraction so its affine maps need no runtime conditionals.
static Value broadcastContractionInput(OpBuilder &builder, Location loc,
                                       Value input, AffineMap inputMap,
                                       RankedTensorType resultType,
                                       ArrayRef<int64_t> resultPositions,
                                       unsigned numBatchDims,
                                       ArrayRef<Value> resultExtents) {
  auto inputType = cast<RankedTensorType>(input.getType());
  SmallVector<int64_t> shape(inputType.getShape());
  SmallVector<int64_t> batchPositions(inputType.getRank(), -1);
  for (auto [dim, expr] : llvm::enumerate(inputMap.getResults())) {
    int64_t resultDim =
        resultPositions[cast<AffineDimExpr>(expr).getPosition()];
    if (resultDim >= 0 && resultDim < numBatchDims &&
        inputType.isDynamicDim(dim)) {
      batchPositions[dim] = resultDim;
      shape[dim] = resultType.getDimSize(resultDim);
    }
  }
  if (llvm::all_of(batchPositions, [](int64_t dim) { return dim < 0; })) {
    return input;
  }

  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  SmallVector<Value> dynamicSizes;
  SmallVector<Value> isUnit(inputType.getRank());
  for (auto [dim, size] : llvm::enumerate(shape)) {
    if (batchPositions[dim] >= 0) {
      Value extent = getDim(builder, loc, input, dim);
      isUnit[dim] = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::eq, extent, one);
      if (ShapedType::isDynamic(size)) {
        dynamicSizes.push_back(resultExtents[batchPositions[dim]]);
      }
    } else if (ShapedType::isDynamic(size)) {
      Value extent = getDim(builder, loc, input, dim);
      dynamicSizes.push_back(extent);
    }
  }
  auto broadcastType = RankedTensorType::get(shape, inputType.getElementType(),
                                             inputType.getEncoding());
  return tensor::GenerateOp::create(
      builder, loc, broadcastType, dynamicSizes,
      [&](OpBuilder &bodyBuilder, Location bodyLoc, ValueRange indices) {
        Value zero = arith::ConstantIndexOp::create(bodyBuilder, bodyLoc, 0);
        SmallVector<Value> sourceIndices(indices);
        for (auto [dim, predicate] : llvm::enumerate(isUnit)) {
          if (predicate) {
            sourceIndices[dim] = arith::SelectOp::create(
                bodyBuilder, bodyLoc, predicate, zero, indices[dim]);
          }
        }
        Value element = tensor::ExtractOp::create(bodyBuilder, bodyLoc, input,
                                                  sourceIndices);
        tensor::YieldOp::create(bodyBuilder, bodyLoc, element);
      });
}

class ContractionOpConversion : public OpConversionPattern<ContractionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ContractionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
    Value inputs[] = {adaptor.getA(), adaptor.getB()};
    RankedTensorType inputTypes[] = {op.getA().getType(), op.getB().getType()};
    if (inputTypes[0].getElementType() != inputTypes[1].getElementType()) {
      return rewriter.notifyMatchFailure(
          op, "contraction inputs must have the same element type");
    }

    SmallVector<AffineMap> maps{op.getLhsIndexingMap(), op.getRhsIndexingMap(),
                                op.getResultIndexingMap()};
    unsigned numDims = maps[2].getNumDims();
    unsigned numBatchDims = op.getNumBatchDims();
    SmallVector<int64_t> inputPositions[2] = {
        SmallVector<int64_t>(numDims, -1), SmallVector<int64_t>(numDims, -1)};
    for (unsigned input = 0; input < 2; ++input) {
      for (auto [dim, expr] : llvm::enumerate(maps[input].getResults())) {
        inputPositions[input][cast<AffineDimExpr>(expr).getPosition()] = dim;
      }
    }
    SmallVector<int64_t> resultPositions(numDims, -1);
    SmallVector<utils::IteratorType> iterators(numDims,
                                               utils::IteratorType::reduction);
    SmallVector<Value> resultExtents(resultType.getRank());
    SmallVector<Value> dynamicSizes;
    for (auto [dim, expr] : llvm::enumerate(maps[2].getResults())) {
      unsigned domainDim = cast<AffineDimExpr>(expr).getPosition();
      resultPositions[domainDim] = dim;
      iterators[domainDim] = utils::IteratorType::parallel;
      if (!resultType.isDynamicDim(dim)) {
        continue;
      }
      int64_t aDim = inputPositions[0][domainDim];
      int64_t bDim = inputPositions[1][domainDim];
      Value extent;
      if (aDim < 0 || (bDim >= 0 && inputTypes[0].getDimSize(aDim) == 1)) {
        extent = getDim(rewriter, op.getLoc(), inputs[1], bDim);
      } else if (bDim < 0 || inputTypes[1].getDimSize(bDim) == 1) {
        extent = getDim(rewriter, op.getLoc(), inputs[0], aDim);
      } else {
        Value aExtent = getDim(rewriter, op.getLoc(), inputs[0], aDim);
        Value bExtent = getDim(rewriter, op.getLoc(), inputs[1], bDim);
        Value one = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 1);
        Value isUnit = arith::CmpIOp::create(
            rewriter, op.getLoc(), arith::CmpIPredicate::eq, aExtent, one);
        // Select rather than max: broadcasting a unit extent to zero is empty.
        extent = arith::SelectOp::create(rewriter, op.getLoc(), isUnit, bExtent,
                                         aExtent);
      }
      resultExtents[dim] = extent;
      dynamicSizes.push_back(extent);
    }

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
    for (unsigned input = 0; input < 2; ++input) {
      inputs[input] = broadcastContractionInput(
          rewriter, op.getLoc(), inputs[input], maps[input], resultType,
          resultPositions, numBatchDims, resultExtents);
      SmallVector<AffineExpr> expressions(maps[input].getResults());
      for (auto [dim, expr] : llvm::enumerate(expressions)) {
        int64_t resultDim =
            resultPositions[cast<AffineDimExpr>(expr).getPosition()];
        if (resultDim >= 0 && resultDim < numBatchDims &&
            inputTypes[input].getDimSize(dim) == 1) {
          expressions[dim] = rewriter.getAffineConstantExpr(0);
        }
      }
      maps[input] =
          AffineMap::get(numDims, 0, expressions, rewriter.getContext());
    }
    auto generic = linalg::GenericOp::create(
        rewriter, op.getLoc(), TypeRange{resultType}, ValueRange(inputs),
        ValueRange{init}, maps, iterators,
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

class MatmulOpConversion : public OpConversionPattern<MatmulOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
    Value init = adaptor.getAcc();
    if (!init) {
      SmallVector<Value> dynamicSizes;
      auto aType = cast<RankedTensorType>(adaptor.getA().getType());
      auto bType = cast<RankedTensorType>(adaptor.getB().getType());
      for (auto [unsignedDim, size] : llvm::enumerate(resultType.getShape())) {
        if (!ShapedType::isDynamic(size)) {
          continue;
        }
        int64_t dim = static_cast<int64_t>(unsignedDim);
        Value source;
        int64_t sourceDim;
        if (dim == resultType.getRank() - 2) {
          source = adaptor.getA();
          sourceDim = aType.getRank() - 2;
        } else if (dim == resultType.getRank() - 1) {
          source = adaptor.getB();
          sourceDim = bType.getRank() - 1;
        } else {
          source = adaptor.getA();
          sourceDim = dim;
        }
        dynamicSizes.push_back(
            getDim(rewriter, op.getLoc(), source, sourceDim));
      }
      FailureOr<Value> empty =
          createEmptyTensor(rewriter, op.getLoc(), resultType, dynamicSizes);
      assert(succeeded(empty) && "constructed all dynamic dimensions");
      ScalarBuilder scalar(rewriter, op.getLoc());
      Value zero = scalar.constant(resultType.getElementType(), 0);
      init = linalg::FillOp::create(rewriter, op.getLoc(), zero, *empty)
                 .getResult(0);
    }

    Type inputElementType = op.getA().getType().getElementType();
    if (isa<IntegerType>(inputElementType) ||
        isa<IntegerType>(resultType.getElementType()) ||
        resultType.getRank() > 3) {
      // Named matmul supports at most one batch dimension and uses wrapping
      // integer addition. Use a scalar region for higher ranks and MMA's
      // signed saturating accumulator.
      unsigned rank = resultType.getRank();
      SmallVector<AffineExpr> dims;
      for (unsigned i = 0; i <= rank; ++i) {
        dims.push_back(rewriter.getAffineDimExpr(i));
      }
      SmallVector<AffineExpr> aDims(dims.begin(), dims.begin() + rank - 1);
      aDims.push_back(dims[rank]);
      SmallVector<AffineExpr> bDims(dims.begin(), dims.begin() + rank - 2);
      bDims.append({dims[rank], dims[rank - 1]});
      SmallVector<AffineMap> maps{
          AffineMap::get(rank + 1, 0, aDims, rewriter.getContext()),
          AffineMap::get(rank + 1, 0, bDims, rewriter.getContext()),
          AffineMap::get(rank + 1, 0, ArrayRef(dims).take_front(rank),
                         rewriter.getContext())};
      SmallVector<utils::IteratorType> iterators(rank,
                                                 utils::IteratorType::parallel);
      iterators.push_back(utils::IteratorType::reduction);
      auto matmul = linalg::GenericOp::create(
          rewriter, op.getLoc(), TypeRange{resultType},
          ValueRange{adaptor.getA(), adaptor.getB()}, ValueRange{init}, maps,
          iterators, [&](OpBuilder &builder, Location loc, ValueRange args) {
            Value sum = buildMmaScalar(builder, loc, args[0], args[1], args[2],
                                       inputElementType,
                                       op.getB().getType().getElementType());
            linalg::YieldOp::create(builder, loc, sum);
          });
      rewriter.replaceOp(op, matmul.getResults());
      return success();
    }

    if (resultType.getRank() == 2) {
      auto matmul = linalg::MatmulOp::create(
          rewriter, op.getLoc(), ValueRange{adaptor.getA(), adaptor.getB()},
          ValueRange{init});
      rewriter.replaceOp(op, matmul.getResults());
      return success();
    }
    auto matmul = linalg::BatchMatmulOp::create(
        rewriter, op.getLoc(), ValueRange{adaptor.getA(), adaptor.getB()},
        ValueRange{init});
    rewriter.replaceOp(op, matmul.getResults());
    return success();
  }
};

using SupportedPointwiseOps =
    std::tuple<AbsOp, AddOp, AddSquareOp, Atan2Op, BinarySelectOp, CeilOp,
               CmpOp, CosOp, DivOp, EluBwdOp, EluFwdOp, ErfOp, ExpOp, FloorOp,
               GeluApproxTanhBwdOp, GeluApproxTanhFwdOp, GeluBwdOp, GeluFwdOp,
               LogOp, LogicalAndOp, LogicalNotOp, LogicalOrOp, MaxOp, MinOp,
               ModOp, MulOp, NegOp, PowOp, ReciprocalOp, ReluBwdOp, ReluFwdOp,
               RemOp, RsqrtOp, SigmoidBwdOp, SigmoidFwdOp, SinOp, SoftplusBwdOp,
               SoftplusFwdOp, SqrtOp, SubOp, SwishBwdOp, SwishFwdOp, TanOp,
               TanhBwdOp, TanhFwdOp, ConvertOp>;

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
    patterns
        .add<GraphOpConversion, ResultsOpConversion, ConstantOpConversion,
             DimOpConversion, SplatOpConversion, BroadcastOpConversion,
             ReshapeOpConversion, TransposeOpConversion, SliceOpConversion,
             IotaOpConversion, MatmulOpConversion, ContractionOpConversion,
             ConcatenateOpConversion, ReduceOpConversion, ReduceUDOpConversion>(
            typeConverter, context);
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
