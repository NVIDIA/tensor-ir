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
  PassOptions::Option<nv_tensor_ir::PersistenceMode> persistence{
      *this, "persistence", llvm::cl::desc("Kernel persistence mode."),
      llvm::cl::init(nv_tensor_ir::PersistenceMode::None),
      llvm::cl::values(clEnumValN(nv_tensor_ir::PersistenceMode::None, "none",
                                  "Normal kernel launch."),
                       clEnumValN(nv_tensor_ir::PersistenceMode::Static,
                                  "static", "Static persistent kernel."))};
  PassOptions::Option<int32_t> smCount{
      *this, "sm-count",
      llvm::cl::desc("Runtime SM count for persistent kernels."),
      llvm::cl::init(0)};
  PassOptions::Option<int32_t> occupancy{
      *this, "occupancy",
      llvm::cl::desc("Target occupancy for persistent kernels."),
      llvm::cl::init(1)};
};
} // namespace

static void buildTensorToCudaTileConversionPassPipeline(
    OpPassManager &pm, const TensorToCudaTilePipelineCLOptions &opts) {
  nv_tensor_ir::TensorToCudaTilePipelineOptions options;
  options.tileSize.assign(opts.tileSize.begin(), opts.tileSize.end());
  options.reductionTileSize = opts.reductionTileSize;
  options.persistence = opts.persistence;
  options.smCount = opts.smCount;
  options.occupancy = opts.occupancy;
  options.codegenStrategy =
      nv_tensor_ir::CudaTileCodegenStrategy::LayoutPropagation;
  nv_tensor_ir::buildTensorToCudaTileConversionPipeline(pm, options);
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
  //    operand layouts
  // 6. FormGridPass: Forms the flattened destination-passing tiled grid
  // 7. TileReductionsPass: Materializes contraction/persistence loops
  // 8. OutlineKernelPass: Outlines the mapped forall body
  // 9. TensorToCudaTileConversion: Converts the explicit program to CudaTile
  PassPipelineRegistration<TensorToCudaTilePipelineCLOptions>(
      "layout-propagation-pipeline",
      "Pipeline to convert TensorIR to CudaTile using layout propagation",
      buildTensorToCudaTileConversionPassPipeline);

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
