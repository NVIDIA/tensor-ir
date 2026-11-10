// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_LIB_ANALYSIS_SHAPERELATIONMODELS_H
#define TENSOR_IR_LIB_ANALYSIS_SHAPERELATIONMODELS_H

#include "tensor_ir/Analysis/PolyExprs.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Support/LLVM.h"

#include <optional>
#include <utility>

namespace mlir::nv_tensor_ir::detail {

/// Receives the relations that operation models derive.
class ShapeRelationBuilder {
public:
  virtual ~ShapeRelationBuilder();

  virtual PolyExprContext &getContext() = 0;

  /// Expression of an index value or of dimension `dim` of a ranked shaped
  /// value.
  virtual PolyExpr getExpr(Value value,
                           std::optional<int64_t> dim = std::nullopt) = 0;

  /// Records `lhs == rhs`, reporting contradictions on `op`. A contradicted
  /// assumption also reports dimension `dim`.
  virtual void equate(PolyExpr lhs, PolyExpr rhs, Operation *op,
                      std::optional<int64_t> dim = std::nullopt,
                      bool assumption = false) = 0;

  /// Records that dimension `dim` of `result` (or `result` itself) equals
  /// `expr(operands)`. Floor, ceil, and mod wait until their divisions are
  /// exact.
  virtual void equateAffine(Operation *op, Value result,
                            std::optional<int64_t> dim, AffineExpr expr,
                            ArrayRef<PolyExpr> operands, unsigned numDims) = 0;

  /// Dynamic argument dimensions and index arguments.
  virtual const ValueDimList &getArgumentDependencies() const = 0;

  /// Whether `value` depends on a constant dimension index outside its
  /// source's rank, which value-bounds models assert against.
  virtual bool hasOutOfRangeDim(Value value) = 0;

  PolyExpr getExpr(OpFoldResult value);
  PolyExpr getConstant(int64_t value);
  SmallVector<PolyExpr> getShape(Value value);
};

/// Records the shape relations of `op`.
void populateShapeRelations(Operation *op, ShapeRelationBuilder &builder);

/// Source and index of a `DimOp`, `tensor.dim`, or `memref.dim` with a
/// constant index, without range checks.
std::optional<std::pair<Value, int64_t>> matchConstantDim(Operation *op);

} // namespace mlir::nv_tensor_ir::detail

#endif // TENSOR_IR_LIB_ANALYSIS_SHAPERELATIONMODELS_H
