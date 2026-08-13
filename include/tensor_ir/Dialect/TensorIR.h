// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_DIALECT_TENSOR_IR_H_
#define TENSOR_IR_DIALECT_TENSOR_IR_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace nv_tensor_ir {} // namespace nv_tensor_ir
} // namespace mlir

#include "tensor_ir/Dialect/TensorDialect.h.inc"
#include "tensor_ir/Dialect/TensorIRAttrs.h"

namespace mlir::nv_tensor_ir {
/// Compatibility alias for TensorIR's builtin ranked tensor representation.
using TensorType = ::mlir::RankedTensorType;

/// Returns whether `type` is a ranked tensor supported by TensorIR.
bool isTensorType(Type type);

/// Returns true if `value` is a block argument of a TensorIR graph.
bool isGraphInput(Value value);

/// Returns true if `value` is consumed by a TensorIR graph results operation.
bool isGraphOutput(Value value);

/// Returns the extent produced by broadcasting two dimensions, or failure when
/// two known, non-unit dimensions are incompatible. Zero is a known extent:
/// it broadcasts with one to zero, but does not broadcast with other static
/// extents.
FailureOr<int64_t> mergeBroadcastDimensions(int64_t lhs, int64_t rhs);

/// Returns the common extent of two dimensions, treating a dynamic extent as
/// an unknown value that may be refined by a static extent.
FailureOr<int64_t> mergeEqualDimensions(int64_t lhs, int64_t rhs);

/// Verifies the shared A (..., M, K), B (..., K, N), C (..., M, N) shape
/// contract used by matrix multiplication operations.
LogicalResult verifyMatmulShapes(Operation *op, TensorType aType,
                                 TensorType bType, TensorType cType);

namespace detail {
/// Verify the operand/result contract shared by DynamicDimsOpInterface ops.
LogicalResult verifyDynamicDimsOpInterface(Operation *op);

/// Converts a flat tcutegen shape to MLIR ranked tensor dimensions.
SmallVector<int64_t> getFlatTensorShape(const tcutegen::Shape &shape);
} // namespace detail

/// Converts the dimensions of `tensorType` to a flat tcutegen shape.
tcutegen::Shape getShapeRef(TensorType tensorType);
} // namespace mlir::nv_tensor_ir

// Disable clang-format to prevent the includes to be rearranged
// and break dependencies between them.
// clang-format off
#include "tensor_ir/Dialect/TensorOpInterfaces.h.inc"
// clang-format on

#define GET_OP_CLASSES
#include "tensor_ir/Dialect/TensorOps.h.inc"

#endif // TENSOR_IR_DIALECT_TENSOR_IR_H_
