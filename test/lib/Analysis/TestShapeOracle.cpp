// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Analysis/ShapeOracle.h"
#include "tensor_ir/Dialect/TensorIR.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::nv_tensor_ir;

/// Prints the value or dimension that each symbol of `expression` stands for.
static void printSymbols(const ShapeOracle &oracle, PolyExpr expression,
                         llvm::raw_ostream &os) {
  os << ", symbols = {";
  llvm::interleaveComma(expression.getVariables(), os, [&](VariableId id) {
    auto [value, dim] = oracle.getSymbols()[id];
    os << "s" << id << " = ";
    if (id < oracle.getNumArgumentSymbols()) {
      os << "arg" << cast<BlockArgument>(value).getArgNumber();
    } else {
      os << "internal";
    }
    if (dim) {
      os << ".dim" << *dim;
    }
  });
  os << "}";
}

static void printExpression(FailureOr<PolyExpr> expression,
                            llvm::raw_ostream &os) {
  if (succeeded(expression)) {
    os << *expression;
  } else {
    os << "unknown";
  }
}

/// Prints the shape of a shaped value or the expression of an index value.
static void printValue(const ShapeOracle &oracle, Value value,
                       llvm::raw_ostream &os) {
  if (value.getType().isIndex()) {
    printExpression(oracle.getIndex(value), os);
    return;
  }
  FailureOr<SmallVector<PolyExpr>> shape = oracle.getShape(value);
  if (failed(shape)) {
    os << "unknown";
    return;
  }
  os << '[';
  llvm::interleaveComma(*shape, os);
  os << ']';
}

/// Remarks on `graph` with the shape or expression of every argument and
/// result.
static void printSignature(const ShapeOracle &oracle,
                           FunctionOpInterface graph) {
  auto print = [&](StringRef kind, unsigned index, Value value) {
    if (!value.getType().isIndex() && !isa<ShapedType>(value.getType())) {
      return;
    }
    std::string text;
    llvm::raw_string_ostream os(text);
    os << kind << ' ' << index << ": ";
    printValue(oracle, value, os);
    graph->emitRemark(text);
  };
  for (auto [index, argument] : llvm::enumerate(graph.getArguments())) {
    print("argument", index, argument);
  }
  for (auto [index, result] : llvm::enumerate(
           graph.getFunctionBody().front().getTerminator()->getOperands())) {
    print("result", index, result);
  }
}

/// Remarks on `op` with the answer to the query it is tagged with: equality
/// of the operands of an `arith.cmpi eq`, or the expression of an index
/// result, optionally with its divisor and upper bound.
static LogicalResult printQuery(Operation *op, const ShapeOracle &oracle) {
  auto name = op->getAttrOfType<StringAttr>("test.shape_query");
  std::string text;
  llvm::raw_string_ostream os(text);
  os << name.getValue() << ": ";
  if (auto compare = dyn_cast<arith::CmpIOp>(op)) {
    if (compare.getPredicate() != arith::CmpIPredicate::eq) {
      return op->emitError("shape equality queries require eq");
    }
    FailureOr<PolyExpr> lhs = oracle.getIndex(compare.getLhs());
    FailureOr<PolyExpr> rhs = oracle.getIndex(compare.getRhs());
    std::optional<bool> equal;
    if (succeeded(lhs) && succeeded(rhs)) {
      equal = oracle.areEqual(*lhs, *rhs);
    }
    os << (!equal ? "unknown" : *equal ? "true" : "false");
    op->emitRemark(text);
    return success();
  }
  if (op->getNumResults() != 1 || !op->getResult(0).getType().isIndex()) {
    return op->emitError("shape query requires index values");
  }
  FailureOr<PolyExpr> expression = oracle.getIndex(op->getResult(0));
  printExpression(expression, os);
  if (succeeded(expression) && op->hasAttr("test.symbols")) {
    printSymbols(oracle, *expression, os);
  }
  if (succeeded(expression) && op->hasAttr("test.bounds")) {
    os << ", divisor = " << oracle.getDivisor(*expression)
       << ", upper_bound = ";
    if (std::optional<int64_t> bound =
            oracle.getConstantUpperBound(*expression)) {
      os << *bound;
    } else {
      os << "none";
    }
  }
  op->emitRemark(text);
  return success();
}

static LogicalResult printQueries(FunctionOpInterface graph,
                                  const ShapeOracle &oracle) {
  SmallVector<Operation *> queries;
  graph.walk([&](Operation *op) {
    if (op->hasAttr("test.shape_query")) {
      queries.push_back(op);
    }
  });
  for (Operation *op : queries) {
    if (failed(printQuery(op, oracle))) {
      return failure();
    }
  }
  return success();
}

/// Replaces every index result tagged `test.materialize` with its extent
/// materialized from the function arguments.
static LogicalResult materializeTestExpressions(RewriterBase &rewriter,
                                                FunctionOpInterface graph,
                                                const ShapeOracle &oracle) {
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<std::pair<Operation *, PolyExpr>> expressions;
  WalkResult walk = graph.walk([&](Operation *op) {
    if (!op->hasAttr("test.materialize")) {
      return WalkResult::advance();
    }
    FailureOr<PolyExpr> expression = op->getNumResults() == 1
                                         ? oracle.getIndex(op->getResult(0))
                                         : FailureOr<PolyExpr>(failure());
    if (failed(expression)) {
      op->emitError("failed to get expression");
      return WalkResult::interrupt();
    }
    expressions.emplace_back(op, *expression);
    return WalkResult::advance();
  });
  if (walk.wasInterrupted()) {
    return failure();
  }
  MaterializationCache cache;
  for (auto [op, expression] : expressions) {
    rewriter.setInsertionPoint(op);
    FailureOr<OpFoldResult> materialized =
        oracle.materialize(rewriter, op->getLoc(), expression, cache);
    if (failed(materialized)) {
      std::string text;
      llvm::raw_string_ostream os(text);
      os << expression;
      return op->emitError() << "cannot materialize extent " << text;
    }
    Value value = dyn_cast<Value>(*materialized);
    if (!value) {
      value = arith::ConstantIndexOp::create(
          rewriter, op->getLoc(),
          cast<IntegerAttr>(cast<Attribute>(*materialized)).getInt());
    }
    rewriter.replaceOp(op, value);
  }
  return success();
}

namespace {
struct TestShapeOraclePass
    : public PassWrapper<TestShapeOraclePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestShapeOraclePass)

  TestShapeOraclePass() = default;
  TestShapeOraclePass(const TestShapeOraclePass &other) : PassWrapper(other) {}

  Option<bool> materialize{
      *this, "materialize",
      llvm::cl::desc("Replace index results tagged test.materialize with "
                     "extents materialized from function arguments"),
      llvm::cl::init(false)};

  StringRef getArgument() const override { return "test-shape-relations"; }
  StringRef getDescription() const override {
    return "Verify the shape oracle of every function and remark on the "
           "answers to tagged queries";
  }

  void runOnOperation() override;
};
} // namespace

void TestShapeOraclePass::runOnOperation() {
  IRRewriter rewriter(getOperation());
  for (auto graph : getOperation().getOps<FunctionOpInterface>()) {
    if (graph.isExternal()) {
      continue;
    }
    const ShapeOracle &oracle =
        getChildAnalysis<ShapeOracle>(graph.getOperation());
    (void)oracle.verify();
    if (graph->hasAttr("test.print_signature")) {
      printSignature(oracle, graph);
    }
    if (failed(printQueries(graph, oracle))) {
      return signalPassFailure();
    }
    if (materialize) {
      if (failed(materializeTestExpressions(rewriter, graph, oracle))) {
        return signalPassFailure();
      }
      mlir::eliminateTriviallyDeadOps(rewriter, graph.getFunctionBody());
    }
  }
}

namespace mlir::test {
void registerTestShapeOraclePass() { PassRegistration<TestShapeOraclePass>(); }
} // namespace mlir::test
