// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_ANALYSIS_SHAPEORACLE_H
#define TENSOR_IR_ANALYSIS_SHAPEORACLE_H

#include "tensor_ir/Analysis/PolyExprs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"

#include <memory>
#include <optional>
#include <utility>

namespace mlir::nv_tensor_ir {

/// Values that materialization reuses across calls. Cached values must
/// dominate later insertion points.
struct MaterializationCache {
  /// The value materialized for each symbol.
  llvm::DenseMap<VariableId, Value> symbols;
};

/// Polynomial facts about the index values and shaped-value dimensions of a
/// function. Construction models every operation, solves the resulting
/// equalities, adds the residuals it cannot solve to the reduction basis of
/// its context, and freezes the context. Returned expressions use entry-block
/// arguments as their basis: index arguments and the extents of shaped
/// arguments. Values the models cannot relate to arguments keep their own
/// symbols. Symbols range over
/// non-negative integers.
///
/// Queries have no side effects. A query that touches a symbol related to a
/// contradiction answers that it does not know; only verify() reports
/// contradictions. IR mutation invalidates the oracle.
class ShapeOracle {
public:
  /// What a symbol stands for: an index value, or the extent of dimension
  /// `dim` of a ranked shaped value.
  struct Symbol {
    Value value;
    std::optional<int64_t> dim;
  };

  /// Builds the oracle of `scope`, a FunctionOpInterface operation.
  explicit ShapeOracle(Operation *scope);
  ~ShapeOracle();

  /// Context of every returned expression. Its definitions and reduction
  /// basis are frozen; clients may build new expressions in it.
  PolyExprContext &getContext() const;

  /// What each symbol stands for, indexed by VariableId. Symbols below
  /// getNumArgumentSymbols() stand for entry-block arguments.
  ArrayRef<Symbol> getSymbols() const;
  unsigned getNumArgumentSymbols() const;

  /// Normalized extent of dimension `dim` of a ranked shaped value defined in
  /// the function.
  FailureOr<PolyExpr> getExtent(Value shaped, int64_t dim) const;
  /// Normalized extents of a ranked shaped value defined in the function.
  FailureOr<SmallVector<PolyExpr>> getShape(Value shaped) const;
  /// Normalized expression of an index value defined in the function.
  FailureOr<PolyExpr> getIndex(Value index) const;

  /// Whether `lhs == rhs`: true or false when proven, std::nullopt otherwise.
  std::optional<bool> areEqual(PolyExpr lhs, PolyExpr rhs) const;
  /// Whether `extent == 1`, as areEqual.
  std::optional<bool> isOne(PolyExpr extent) const;
  /// A positive integer that divides `extent`: the gcd of its coefficients,
  /// or 1 when unknown. Zero for the zero extent.
  int64_t getDivisor(PolyExpr extent) const;
  /// An upper bound of `expression` over non-negative symbols: its constant
  /// term when no other term has a positive coefficient.
  std::optional<int64_t> getConstantUpperBound(PolyExpr expression) const;

  /// Computes `extent` from entry-block arguments through dimension queries.
  /// Returns an index attribute for a constant extent. Fails if `extent` depends on other symbols or has a
  /// coefficient outside int64_t.
  FailureOr<OpFoldResult> materialize(OpBuilder &builder, Location loc,
                                      PolyExpr extent,
                                      MaterializationCache &cache) const;

  /// Reports every contradiction among the constraints, in IR order, and
  /// fails if there is any.
  LogicalResult verify() const;

  bool isInvalidated(const AnalysisManager::PreservedAnalyses &preserved) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_ANALYSIS_SHAPEORACLE_H
