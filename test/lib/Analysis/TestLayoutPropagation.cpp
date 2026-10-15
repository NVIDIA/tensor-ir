// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Analysis/LayoutPropagation.h"
#include "tensor_ir/Dialect/TensorIR.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::nv_tensor_ir;

namespace {

/// Print the forward layout of every analyzed operation of each graph, in the
/// order the analysis visits them.
struct TestLayoutPropagationPass
    : public PassWrapper<TestLayoutPropagationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLayoutPropagationPass);

  StringRef getArgument() const final { return "test-layout-propagation"; }
  StringRef getDescription() const final {
    return "Print the forward layout-propagation analysis of TensorIR graphs";
  }

  void runOnOperation() override {
    llvm::raw_ostream &os = llvm::outs();
    getOperation().walk([&](GraphOp graph) {
      FailureOr<LayoutBoundary> boundary = LayoutBoundary::get(graph);
      if (failed(boundary)) {
        return signalPassFailure();
      }
      FailureOr<LayoutPropagationAnalysis> propagation =
          LayoutPropagationAnalysis::compute(*boundary);
      if (failed(propagation)) {
        return signalPassFailure();
      }
      os << "layout propagation for @" << graph.getSymName() << "\n";
      graph.getBody()->walk([&](Operation *op) {
        if (op->getNumResults() == 0) {
          return;
        }
        if (LayoutSourceAttrInterface layout =
                propagation->getLayout(op->getResult(0))) {
          os << "  " << op->getName() << ": " << layout << "\n";
        }
      });
    });
  }
};

/// Print the normalized iteration space and per-result metadata of each graph.
struct TestLayoutNormalizationPass
    : public PassWrapper<TestLayoutNormalizationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLayoutNormalizationPass);

  StringRef getArgument() const final { return "test-layout-normalization"; }
  StringRef getDescription() const final {
    return "Print the normalized layout-propagation iteration space of "
           "TensorIR graphs";
  }

  void runOnOperation() override {
    llvm::raw_ostream &os = llvm::outs();
    getOperation().walk([&](GraphOp graph) {
      FailureOr<LayoutNormalization> normalization =
          LayoutNormalization::compute(graph);
      if (failed(normalization)) {
        return signalPassFailure();
      }
      os << "layout normalization for @" << graph.getSymName() << "\n";
      os << "  iteration_space: " << normalization->iterationSpace << "\n";
      if (normalization->resultLayouts.empty()) {
        return;
      }
      os << "  result_layouts: [";
      llvm::interleaveComma(normalization->resultLayouts, os);
      os << "]\n  result_views: [";
      llvm::interleaveComma(normalization->resultViews, os);
      os << "]\n";
    });
  }
};

} // namespace

namespace mlir::test {
void registerTestLayoutPropagationPasses() {
  PassRegistration<TestLayoutPropagationPass>();
  PassRegistration<TestLayoutNormalizationPass>();
}
} // namespace mlir::test
