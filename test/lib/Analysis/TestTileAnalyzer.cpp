// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Analysis/TileAnalyzer.h"
#include "tensor_ir/Dialect/TensorIR.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::nv_tensor_ir;

namespace {

// Test pass that exercises TileAnalyzer::getTileExplanation
struct TestTileAnalyzerPass
    : public PassWrapper<TestTileAnalyzerPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTileAnalyzerPass);

  StringRef getArgument() const final { return "test-tile-analyzer"; }
  StringRef getDescription() const final {
    return "Test TileAnalyzer::getTileExplanation coverage";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    moduleOp.walk([&](GraphOp graphOp) {
      // Create TileAnalyzer instance with default GPU architecture
      // SM100: warpSize=32, l1CacheLineSize=128
      GpuArchitecture gpu(32, 128);
      TileAnalyzer analyzer(gpu);

      // Find a reduction operation to analyze
      graphOp.walk([&](ReduceOp reduceOp) {
        auto inputType = llvm::dyn_cast<nv_tensor_ir::TensorType>(
            reduceOp.getInput().getType());
        if (!inputType || !inputType.hasStaticShape()) {
          return;
        }

        auto shape = inputType.getShape();
        llvm::SmallVector<int64_t> shapeVec(shape.begin(), shape.end());
        llvm::SmallVector<int64_t> strideVec(shape.size());

        // Compute row-major strides
        int64_t stride = 1;
        for (int64_t i = shape.size() - 1; i >= 0; --i) {
          strideVec[i] = stride;
          stride *= shape[i];
        }

        // Get reduction dimensions (convert to size_t as required)
        auto reductionDims = reduceOp.getDimensions();
        llvm::SmallVector<size_t> reductionDimsVec;
        for (auto dim : reductionDims) {
          reductionDimsVec.push_back(static_cast<size_t>(dim));
        }

        // Perform tile analysis
        int64_t bytesPerElement = 4; // Assume f32
        if (failed(analyzer.analyzeTileSize(
                shapeVec, strideVec, bytesPerElement, reductionDimsVec))) {
          reduceOp.emitError("failed to analyze tile size");
          signalPassFailure();
          return;
        }

        // Call getTileExplanation to trigger coverage
        std::optional<std::string> explanation = analyzer.getTileExplanation();
        if (!explanation) {
          reduceOp.emitError("tile size explanation was not recorded");
          signalPassFailure();
          return;
        }

        // Output explanation for FileCheck to verify
        llvm::outs() << "Tile explanation:\n" << *explanation;
      });
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestTileAnalyzerPass() {
  PassRegistration<TestTileAnalyzerPass>();
}
} // namespace test
} // namespace mlir
