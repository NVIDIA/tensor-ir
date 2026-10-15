// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Analysis/Tiling/Enumerate.h"
#include "tensor_ir/Analysis/Tiling/Evaluate.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Utils/ComputeCapability.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::nv_tensor_ir;
using namespace mlir::nv_tensor_ir::tiling_analysis;

namespace {

// Test pass that drives enumerateTilings() so its behavior can be exercised
// with FileCheck instead of C++ unit tests. Runs TilingEvaluator::create()
// plus enumerateTilings() with the max-results pass option and stashes the
// ordered list of candidate configs as a "tiling-candidates" array-of-strings
// attribute on the graph, one string per config.
struct TestTilingEnumeratorPass
    : public PassWrapper<TestTilingEnumeratorPass, OperationPass<GraphOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTilingEnumeratorPass);

  using Base = PassWrapper<TestTilingEnumeratorPass, OperationPass<GraphOp>>;

  TestTilingEnumeratorPass() : Base() {}
  TestTilingEnumeratorPass(const TestTilingEnumeratorPass &other)
      : Base(other) {
    maxResults = other.maxResults;
    computeCapability = other.computeCapability;
  }

  StringRef getArgument() const final { return "test-tiling-enumerator"; }
  StringRef getDescription() const final {
    return "Run enumerateTilings() and attach a \"tiling-candidates\" "
           "array-of-strings attribute with the result, for FileCheck-based "
           "testing";
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

    SmallVector<TilingConfig> candidates =
        enumerateTilings(*evaluator, maxResults);

    MLIRContext *ctx = graphOp.getContext();
    SmallVector<Attribute> candidateAttrs;
    candidateAttrs.reserve(candidates.size());
    for (const TilingConfig &config : candidates) {
      std::string str;
      llvm::raw_string_ostream(str) << config;
      candidateAttrs.push_back(StringAttr::get(ctx, str));
    }
    graphOp->setAttr("tiling-candidates", ArrayAttr::get(ctx, candidateAttrs));
  }

private:
  Pass::Option<size_t> maxResults{
      *this, "max-results",
      llvm::cl::desc("Maximum number of candidates to generate"),
      llvm::cl::init(8)};
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
void registerTestTilingEnumeratorPass() {
  PassRegistration<TestTilingEnumeratorPass>();
}
} // namespace mlir::test
