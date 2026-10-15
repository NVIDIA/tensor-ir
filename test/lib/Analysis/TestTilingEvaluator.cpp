// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Analysis/Tiling/Evaluate.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Utils/ComputeCapability.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::nv_tensor_ir;
using namespace mlir::nv_tensor_ir::tiling_analysis;

namespace {

// Test pass that drives TilingEvaluator so its behavior can be exercised with
// FileCheck instead of C++ unit tests. Runs TilingEvaluator::create() plus
// evaluate() with the tile-shape/reduction-tile-size pass options and stashes
// the string form of the result (or the evaluation failure) as an
// "tiling-evaluation" attribute on the graph.
struct TestTilingEvaluatorPass
    : public PassWrapper<TestTilingEvaluatorPass, OperationPass<GraphOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTilingEvaluatorPass);

  using Base = PassWrapper<TestTilingEvaluatorPass, OperationPass<GraphOp>>;

  TestTilingEvaluatorPass() : Base() {}
  TestTilingEvaluatorPass(const TestTilingEvaluatorPass &other) : Base(other) {
    tileShape = other.tileShape;
    reductionTileSize = other.reductionTileSize;
    computeCapability = other.computeCapability;
  }

  StringRef getArgument() const final { return "test-tiling-evaluator"; }
  StringRef getDescription() const final {
    return "Run TilingEvaluator and attach an \"tiling-evaluation\" string "
           "attribute with the result, for FileCheck-based testing";
  }

  void runOnOperation() override {
    GraphOp graphOp = getOperation();

    auto cc = symbolizeComputeCapability(computeCapability);
    if (!cc) {
      graphOp.emitError("invalid compute capability: ")
          << computeCapability.getValue();
      return signalPassFailure();
    }

    FailureOr<TilingEvaluator> evaluator =
        TilingEvaluator::create(graphOp, getArchInfo(*cc, smCount));
    if (failed(evaluator)) {
      // TilingEvaluator::create() already emits a diagnostic on failure.
      return signalPassFailure();
    }

    FailureOr<TilingConfig> config = TilingConfig::get(
        SmallVector<int32_t>(tileShape.begin(), tileShape.end()),
        reductionTileSize);
    if (failed(config)) {
      graphOp.emitError("invalid tile shape");
      return signalPassFailure();
    }

    FailureOr<TilingEvaluation> eval = evaluator->evaluate(*config);
    if (failed(eval)) {
      graphOp.emitError("tiling evaluation failed");
      return signalPassFailure();
    }

    std::string evalStr;
    llvm::raw_string_ostream(evalStr) << *eval;
    graphOp->setAttr("tiling-evaluation",
                     StringAttr::get(graphOp.getContext(), evalStr));
  }

private:
  Pass::ListOption<int32_t> tileShape{*this, "tile-shape",
                                      llvm::cl::desc("Tile shape to evaluate")};
  Pass::Option<int32_t> reductionTileSize{
      *this, "reduction-tile-size",
      llvm::cl::desc("Reduction tile size to evaluate"), llvm::cl::init(1)};
  Pass::Option<int32_t> computeCapability{
      *this, "compute-capability",
      llvm::cl::desc(
          "Target GPU compute capability as an integer (e.g., 90/100/120)"),
      llvm::cl::init(100)};
  Pass::Option<int32_t> smCount{
      *this, "sm-count",
      llvm::cl::desc("Number of Streaming Multiprocessors (SMs)"),
      llvm::cl::init(148)};
};

} // namespace

namespace mlir::test {
void registerTestTilingEvaluatorPass() {
  PassRegistration<TestTilingEvaluatorPass>();
}
} // namespace mlir::test
