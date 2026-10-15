// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// \file
/// Scalar Arith and Math builders for TensorIR element operations.

#ifndef TENSOR_IR_LIB_CONVERSION_TENSORTOLINALG_SCALARBUILDER_H
#define TENSOR_IR_LIB_CONVERSION_TENSORTOLINALG_SCALARBUILDER_H

#include "tensor_ir/Dialect/TensorIR.h" // IWYU pragma: keep

#include "mlir/Dialect/Arith/Utils/Utils.h"

namespace mlir::nv_tensor_ir {

bool isUnsignedInteger(Type type);

/// Arith represents signedness in its operations, not in integer types. Keep
/// source TensorIR types for operation selection and convert emitted SSA types.
Type getSignlessType(Type type);

/// Builds scalar Arith and Math operations on signless SSA values.
///
/// The operands have already been converted by getSignlessType, so integer
/// signedness cannot be recovered from their types. Like ArithBuilder::sgt and
/// ArithBuilder::slt, methods prefixed with `s` or `u` select the signed or
/// unsigned integer operation and emit the floating-point operation for
/// floating-point operands. Callers choose the prefix from the source TensorIR
/// element type.
class ScalarBuilder : public ArithBuilder {
public:
  ScalarBuilder(OpBuilder &builder, Location loc)
      : ArithBuilder(builder, loc), builder(builder), loc(loc) {}

  Value constant(Type type, double value);
  Value sdiv(Value lhs, Value rhs);
  Value udiv(Value lhs, Value rhs);
  Value srem(Value lhs, Value rhs);
  Value urem(Value lhs, Value rhs);
  Value smin(Value lhs, Value rhs);
  Value umin(Value lhs, Value rhs);
  Value smax(Value lhs, Value rhs);
  Value umax(Value lhs, Value rhs);
  Value scmp(Comparator comparison, Value lhs, Value rhs);
  Value ucmp(Comparator comparison, Value lhs, Value rhs);
  Value sfloorDiv(Value lhs, Value rhs);
  Value ufloorDiv(Value lhs, Value rhs);
  Value spow(Value lhs, Value rhs);
  /// Supports operand widths up to 32 bits; the exponent is widened to keep
  /// it nonnegative under IPowI's signed interpretation.
  Value upow(Value lhs, Value rhs);

  /// Converts `input` to `outputType`, reading `input` as signed (`scast`) or
  /// unsigned (`ucast`). An i1 input is a boolean and always widens to one.
  /// The output signedness comes from `outputType`, which may be a TensorIR
  /// unsigned integer type.
  Value scast(Value input, Type outputType);
  Value ucast(Value input, Type outputType);

  /// Absolute value of a floating-point or signed integer operand.
  Value abs(Value input);
  Value ceil(Value input);
  Value cos(Value input);
  Value exp(Value input);
  Value floor(Value input);
  Value log(Value input);
  Value neg(Value input);
  Value rsqrt(Value input);
  Value sin(Value input);
  Value sqrt(Value input);
  Value tan(Value input);
  Value erf(Value input);
  Value tanh(Value input);
  Value atan2(Value lhs, Value rhs);
  Value _or(Value lhs, Value rhs);
  Value _xor(Value lhs, Value rhs);

private:
  template <typename ScalarOp>
  Value unary(Value input);
  template <typename ScalarOp>
  Value binary(Value lhs, Value rhs);
  template <typename FloatOp, typename SignedOp, typename UnsignedOp>
  Value integerOrFloat(Value lhs, Value rhs, bool isUnsigned);
  Value cmp(Comparator comparison, Value lhs, Value rhs, bool isUnsigned);
  Value cast(Value input, Type outputType, bool inputIsUnsigned);

  OpBuilder &builder;
  Location loc;
};

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_LIB_CONVERSION_TENSORTOLINALG_SCALARBUILDER_H
