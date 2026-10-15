// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// \file
/// Scalar Arith and Math builders for TensorIR element operations.

#include "ScalarBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::nv_tensor_ir;

bool mlir::nv_tensor_ir::isUnsignedInteger(Type type) {
  auto integerType = dyn_cast<IntegerType>(type);
  return integerType && integerType.isUnsigned();
}

Type mlir::nv_tensor_ir::getSignlessType(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type)) {
    return IntegerType::get(type.getContext(), integer.getWidth());
  }
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    return tensor.clone(getSignlessType(tensor.getElementType()));
  }
  if (auto function = dyn_cast<FunctionType>(type)) {
    auto inputs = llvm::map_to_vector(function.getInputs(), getSignlessType);
    auto results = llvm::map_to_vector(function.getResults(), getSignlessType);
    return FunctionType::get(type.getContext(), inputs, results);
  }
  return type;
}

template <typename ScalarOp>
Value ScalarBuilder::unary(Value input) {
  return ScalarOp::create(builder, loc, input);
}

template <typename ScalarOp>
Value ScalarBuilder::binary(Value lhs, Value rhs) {
  return ScalarOp::create(builder, loc, lhs, rhs);
}

Value ScalarBuilder::constant(Type type, double value) {
  if (auto floatType = dyn_cast<FloatType>(type)) {
    auto attr = builder.getFloatAttr(floatType, value);
    return arith::ConstantOp::create(builder, loc, attr);
  }
  auto integerType = mlir::cast<IntegerType>(getSignlessType(type));
  auto attr = builder.getIntegerAttr(integerType, static_cast<int64_t>(value));
  return arith::ConstantOp::create(builder, loc, attr);
}

template <typename FloatOp, typename SignedOp, typename UnsignedOp>
Value ScalarBuilder::integerOrFloat(Value lhs, Value rhs, bool isUnsigned) {
  if (isa<FloatType>(lhs.getType())) {
    return binary<FloatOp>(lhs, rhs);
  }
  if (isUnsigned) {
    return binary<UnsignedOp>(lhs, rhs);
  }
  return binary<SignedOp>(lhs, rhs);
}

Value ScalarBuilder::sdiv(Value lhs, Value rhs) {
  return integerOrFloat<arith::DivFOp, arith::DivSIOp, arith::DivUIOp>(lhs, rhs,
                                                                       false);
}

Value ScalarBuilder::udiv(Value lhs, Value rhs) {
  return integerOrFloat<arith::DivFOp, arith::DivSIOp, arith::DivUIOp>(lhs, rhs,
                                                                       true);
}

Value ScalarBuilder::srem(Value lhs, Value rhs) {
  return integerOrFloat<arith::RemFOp, arith::RemSIOp, arith::RemUIOp>(lhs, rhs,
                                                                       false);
}

Value ScalarBuilder::urem(Value lhs, Value rhs) {
  return integerOrFloat<arith::RemFOp, arith::RemSIOp, arith::RemUIOp>(lhs, rhs,
                                                                       true);
}

Value ScalarBuilder::smin(Value lhs, Value rhs) {
  return integerOrFloat<arith::MinimumFOp, arith::MinSIOp, arith::MinUIOp>(
      lhs, rhs, false);
}

Value ScalarBuilder::umin(Value lhs, Value rhs) {
  return integerOrFloat<arith::MinimumFOp, arith::MinSIOp, arith::MinUIOp>(
      lhs, rhs, true);
}

Value ScalarBuilder::smax(Value lhs, Value rhs) {
  return integerOrFloat<arith::MaximumFOp, arith::MaxSIOp, arith::MaxUIOp>(
      lhs, rhs, false);
}

Value ScalarBuilder::umax(Value lhs, Value rhs) {
  return integerOrFloat<arith::MaximumFOp, arith::MaxSIOp, arith::MaxUIOp>(
      lhs, rhs, true);
}

Value ScalarBuilder::scmp(Comparator comparison, Value lhs, Value rhs) {
  return cmp(comparison, lhs, rhs, false);
}

Value ScalarBuilder::ucmp(Comparator comparison, Value lhs, Value rhs) {
  return cmp(comparison, lhs, rhs, true);
}

Value ScalarBuilder::cmp(Comparator comparison, Value lhs, Value rhs,
                         bool isUnsigned) {
  if (isa<FloatType>(lhs.getType())) {
    arith::CmpFPredicate predicate =
        llvm::StringSwitch<arith::CmpFPredicate>(
            stringifyComparator(comparison))
            .Case("ueq", arith::CmpFPredicate::UEQ)
            .Case("une", arith::CmpFPredicate::UNE)
            .Case("ugt", arith::CmpFPredicate::UGT)
            .Case("uge", arith::CmpFPredicate::UGE)
            .Case("ult", arith::CmpFPredicate::ULT)
            .Case("ule", arith::CmpFPredicate::ULE)
            .Case("oeq", arith::CmpFPredicate::OEQ)
            .Case("one", arith::CmpFPredicate::ONE)
            .Case("ogt", arith::CmpFPredicate::OGT)
            .Case("oge", arith::CmpFPredicate::OGE)
            .Case("olt", arith::CmpFPredicate::OLT)
            .Case("ole", arith::CmpFPredicate::OLE)
            .Default(arith::CmpFPredicate::AlwaysFalse);
    return arith::CmpFOp::create(builder, loc, predicate, lhs, rhs);
  }

  StringRef comparator = stringifyComparator(comparison);
  arith::CmpIPredicate predicate;
  if (comparator == "eq") {
    predicate = arith::CmpIPredicate::eq;
  } else if (comparator == "neq") {
    predicate = arith::CmpIPredicate::ne;
  } else if (comparator == "gt") {
    predicate =
        isUnsigned ? arith::CmpIPredicate::ugt : arith::CmpIPredicate::sgt;
  } else if (comparator == "ge") {
    predicate =
        isUnsigned ? arith::CmpIPredicate::uge : arith::CmpIPredicate::sge;
  } else if (comparator == "lt") {
    predicate =
        isUnsigned ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt;
  } else {
    predicate =
        isUnsigned ? arith::CmpIPredicate::ule : arith::CmpIPredicate::sle;
  }
  return arith::CmpIOp::create(builder, loc, predicate, lhs, rhs);
}

Value ScalarBuilder::scast(Value input, Type outputType) {
  return cast(input, outputType, false);
}

Value ScalarBuilder::ucast(Value input, Type outputType) {
  return cast(input, outputType, true);
}

Value ScalarBuilder::cast(Value input, Type outputType, bool inputIsUnsigned) {
  bool outputIsUnsigned = isUnsignedInteger(outputType);
  outputType = getSignlessType(outputType);
  Type inputType = input.getType();
  // TensorIR treats i1 as boolean: true widens to one, never minus one.
  inputIsUnsigned |= inputType.isInteger(1);
  if (inputType == outputType) {
    return input;
  }
  if (isa<IndexType>(inputType)) {
    if (isa<IntegerType>(outputType)) {
      return arith::IndexCastOp::create(builder, loc, outputType, input);
    }
    Value integer =
        arith::IndexCastOp::create(builder, loc, builder.getI64Type(), input);
    return cast(integer, outputType, inputIsUnsigned);
  }
  if (outputType.isInteger(1)) {
    Value zero = constant(inputType, 0);
    if (isa<FloatType>(inputType)) {
      return arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::UNE,
                                   input, zero);
    }
    return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne, input,
                                 zero);
  }
  if (isa<FloatType>(inputType) && isa<FloatType>(outputType)) {
    unsigned inputWidth = inputType.getIntOrFloatBitWidth();
    unsigned outputWidth = outputType.getIntOrFloatBitWidth();
    if (inputWidth == outputWidth) {
      return arith::ConvertFOp::create(builder, loc, outputType, input);
    }
    if (inputWidth < outputWidth) {
      return arith::ExtFOp::create(builder, loc, outputType, input);
    }
    return arith::TruncFOp::create(builder, loc, outputType, input);
  }
  if (isa<FloatType>(inputType) && isa<IntegerType>(outputType)) {
    if (outputIsUnsigned) {
      return arith::FPToUIOp::create(builder, loc, outputType, input);
    }
    return arith::FPToSIOp::create(builder, loc, outputType, input);
  }
  if (isa<IntegerType>(inputType) && isa<FloatType>(outputType)) {
    if (inputIsUnsigned) {
      return arith::UIToFPOp::create(builder, loc, outputType, input);
    }
    return arith::SIToFPOp::create(builder, loc, outputType, input);
  }

  unsigned inputWidth = inputType.getIntOrFloatBitWidth();
  unsigned outputWidth = outputType.getIntOrFloatBitWidth();
  if (inputWidth < outputWidth) {
    if (inputIsUnsigned) {
      return arith::ExtUIOp::create(builder, loc, outputType, input);
    }
    return arith::ExtSIOp::create(builder, loc, outputType, input);
  }
  if (inputWidth > outputWidth) {
    return arith::TruncIOp::create(builder, loc, outputType, input);
  }
  return input;
}

Value ScalarBuilder::abs(Value input) {
  if (isa<FloatType>(input.getType())) {
    return unary<math::AbsFOp>(input);
  }
  return unary<math::AbsIOp>(input);
}

Value ScalarBuilder::ceil(Value input) { return unary<math::CeilOp>(input); }
Value ScalarBuilder::cos(Value input) { return unary<math::CosOp>(input); }
Value ScalarBuilder::exp(Value input) { return unary<math::ExpOp>(input); }
Value ScalarBuilder::floor(Value input) { return unary<math::FloorOp>(input); }
Value ScalarBuilder::log(Value input) { return unary<math::LogOp>(input); }
Value ScalarBuilder::neg(Value input) { return unary<arith::NegFOp>(input); }
Value ScalarBuilder::rsqrt(Value input) { return unary<math::RsqrtOp>(input); }
Value ScalarBuilder::sin(Value input) { return unary<math::SinOp>(input); }
Value ScalarBuilder::sqrt(Value input) { return unary<math::SqrtOp>(input); }
Value ScalarBuilder::tan(Value input) { return unary<math::TanOp>(input); }
Value ScalarBuilder::erf(Value input) { return unary<math::ErfOp>(input); }
Value ScalarBuilder::tanh(Value input) { return unary<math::TanhOp>(input); }

Value ScalarBuilder::atan2(Value lhs, Value rhs) {
  return binary<math::Atan2Op>(lhs, rhs);
}

Value ScalarBuilder::spow(Value lhs, Value rhs) {
  if (isa<FloatType>(lhs.getType())) {
    return binary<math::PowFOp>(lhs, rhs);
  }
  return binary<math::IPowIOp>(lhs, rhs);
}

Value ScalarBuilder::upow(Value lhs, Value rhs) {
  if (isa<FloatType>(lhs.getType())) {
    return binary<math::PowFOp>(lhs, rhs);
  }
  // IPowI interprets the exponent as signed. Reserve a sign bit so the
  // entire unsigned exponent range remains nonnegative. Use a power-of-two
  // width so the widened arithmetic fits a standard integer type.
  Type resultType = lhs.getType();
  auto wideType = builder.getIntegerType(
      llvm::NextPowerOf2(resultType.getIntOrFloatBitWidth()));
  Value wideLhs = ucast(lhs, wideType);
  Value wideRhs = ucast(rhs, wideType);
  Value result = binary<math::IPowIOp>(wideLhs, wideRhs);
  return ucast(result, resultType);
}

Value ScalarBuilder::_or(Value lhs, Value rhs) {
  return binary<arith::OrIOp>(lhs, rhs);
}

Value ScalarBuilder::_xor(Value lhs, Value rhs) {
  return binary<arith::XOrIOp>(lhs, rhs);
}

Value ScalarBuilder::sfloorDiv(Value lhs, Value rhs) {
  if (isa<FloatType>(lhs.getType())) {
    Value quotient = sdiv(lhs, rhs);
    return floor(quotient);
  }
  return binary<arith::FloorDivSIOp>(lhs, rhs);
}

Value ScalarBuilder::ufloorDiv(Value lhs, Value rhs) {
  if (isa<FloatType>(lhs.getType())) {
    Value quotient = udiv(lhs, rhs);
    return floor(quotient);
  }
  return udiv(lhs, rhs);
}
