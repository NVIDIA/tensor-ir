// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Compiler/CudaTile/Pipelines.h"
#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Registration/Registration.h"
#include "tensor_ir/Transform/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/ErrorHandling.h"

#ifdef TENSOR_IR_INCLUDE_TESTS
#include "../test/lib/Registration/RegisterPasses.h"
#endif

using namespace llvm;
using namespace mlir;

namespace {
struct TensorToCudaTilePipelineCLOptions
    : public PassPipelineOptions<TensorToCudaTilePipelineCLOptions> {
  PassOptions::ListOption<int32_t> tileSize{
      *this, "tile-size", llvm::cl::desc("A list of integers for tile size"),
      llvm::cl::list_init<int32_t>({})};
  PassOptions::Option<int64_t> reductionTileSize{
      *this, "reduction-tile-size",
      llvm::cl::desc("Tile size for contracting dimensions"),
      llvm::cl::init(nv_tensor_ir::kDefaultReductionTileSize)};
};
} // namespace

static void buildTensorToCudaTileConversionPassPipeline(
    OpPassManager &pm, const TensorToCudaTilePipelineCLOptions &opts) {
  nv_tensor_ir::TensorToCudaTilePipelineOptions options;
  options.tileSize.assign(opts.tileSize.begin(), opts.tileSize.end());
  options.reductionTileSize = opts.reductionTileSize;
  options.codegenStrategy =
      nv_tensor_ir::CudaTileCodegenStrategy::LayoutPropagation;
  nv_tensor_ir::buildTensorToCudaTileConversionPipeline(pm, options);
}

namespace {

/// Runs the layout-propagation lowerability check on each
/// graph without materializing any IR. Exercises the per-op validators (which
/// the conversion driver would otherwise swallow as a generic "failed to
/// legalize" diagnostic) so they can be covered by -verify-diagnostics tests.
struct TestVerifyLayoutPropLowerablePass
    : public PassWrapper<TestVerifyLayoutPropLowerablePass,
                         OperationPass<nv_tensor_ir::GraphOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestVerifyLayoutPropLowerablePass)

  void runOnOperation() override {
    nv_tensor_ir::GraphOp graphOp = getOperation();
    if (failed(nv_tensor_ir::tensor_to_cuda_tile::verifyLayoutPropLowerable(
            graphOp))) {
      signalPassFailure();
    }
  }
};
} // namespace

/// Pipeline that runs the layout-propagation analysis (annotation,
/// normalization, tile selection, graph splitting) and then the lowerability
/// check, but does not run the IR-materializing conversion.
static void buildVerifyLayoutPropLowerablePassPipeline(
    OpPassManager &pm, const TensorToCudaTilePipelineCLOptions &opts) {
  nv_tensor_ir::TensorToCudaTilePipelineOptions options;
  options.tileSize.assign(opts.tileSize.begin(), opts.tileSize.end());
  options.codegenStrategy =
      nv_tensor_ir::CudaTileCodegenStrategy::LayoutPropagation;
  nv_tensor_ir::buildGraphAnalysisPipeline(pm, options);
  nv_tensor_ir::buildTileSelectionPipeline(pm, options);
  pm.addNestedPass<nv_tensor_ir::GraphOp>(
      std::make_unique<TestVerifyLayoutPropLowerablePass>());
}

static void registerToolPipelines() {

  // Layout propagation pipeline
  // Pass ordering is critical and must be preserved:
  // 1. LayoutPropagationAnnotation: Annotates ops with layout information
  // 2. LayoutPropagationNormalization: Computes the main iteration space
  // 3. TileAnalyzer: Analyzes the iteration spaces and adds tile candidates
  // 4. TileSelection: Selects tile configuration based on a heuristic
  // 5. GraphSplitting: Back-propagates iteration_space layouts and enforces
  //    that non-unary elementwise ops have composite layouts with separated
  //    operand layouts (required for TensorToCudaTile codegen)
  // 6. TensorToCudaTileConversion: Converts the graph to CudaTile dialect
  PassPipelineRegistration<TensorToCudaTilePipelineCLOptions>(
      "layout-propagation-pipeline",
      "Pipeline to convert TensorIR to CudaTile using layout propagation",
      buildTensorToCudaTileConversionPassPipeline);

  PassPipelineRegistration<TensorToCudaTilePipelineCLOptions>(
      "verify-layout-prop-lowerable",
      "Pipeline that runs the layout-propagation analysis and then checks "
      "lowerability, without materializing the conversion.",
      buildVerifyLayoutPropLowerablePassPipeline);

  mlir::nv_tensor_ir::registerTensorToCudaTileConversionPasses();
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  mlir::nv_tensor_ir::registerDialects(registry);

  registerToolPipelines();

#ifdef TENSOR_IR_INCLUDE_TESTS
  mlir::nv_tensor_ir::test::registerAllPasses();
#endif

  mlir::nv_tensor_ir::registerNVTensorIRTransformPasses();
  mlir::registerTransformsPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Tensor IR optimizer test driver\n", registry));
}
