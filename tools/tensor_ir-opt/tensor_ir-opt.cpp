// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Compiler/CudaTile/Pipelines.h"
#include "tensor_ir/Options/Options.h"
#include "tensor_ir/Registration/Registration.h"
#include "tensor_ir/Support/TIROverrideOptions.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#ifdef TENSOR_IR_INCLUDE_TESTS
#include "../test/lib/Registration/RegisterPasses.h"
#endif
using namespace llvm;
using namespace mlir;

namespace {

struct TestTensorToCudaTileOptions
    : public PassPipelineOptions<TestTensorToCudaTileOptions> {
  PassOptions::Option<std::string> bytecodeVersion{
      *this, "bytecode-version", llvm::cl::desc("CUDA Tile bytecode target."),
      llvm::cl::init("default")};
  PassOptions::ListOption<int32_t> tileSize{
      *this, "tile-size", llvm::cl::desc("A list of integers for tile size"),
      llvm::cl::list_init<int32_t>({})};
  PassOptions::Option<int64_t> reductionTileSize{
      *this, "reduction-tile-size",
      llvm::cl::desc(
          "Tile size for contracting dimensions in layout propagation"),
      llvm::cl::init(nv_tensor_ir::CudaTileOptions().reductionTileSize)};
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

nv_tensor_ir::CompilerInvocation
getCudaTileConversionOptions(const TestTensorToCudaTileOptions &opts) {
  nv_tensor_ir::CompilerInvocation options;
  options.codegen.tileSize.assign(opts.tileSize.begin(), opts.tileSize.end());
  options.cudaTile.bytecodeVersion = opts.bytecodeVersion;
  options.cudaTile.reductionTileSize = opts.reductionTileSize;
  options.cudaTile.persistenceMode = opts.persistence;
  options.cudaTile.smCount = opts.smCount;
  options.cudaTile.occupancy = opts.occupancy;
  options.cudaTile.codegenStrategy =
      nv_tensor_ir::CudaTileCodegenStrategy::LayoutPropagation;
  return options;
}

nv_tensor_ir::Status buildTensorToCudaTileConversionPassPipeline(
    OpPassManager &pm, const TestTensorToCudaTileOptions &opts) {
  nv_tensor_ir::CompilerInvocation options = getCudaTileConversionOptions(opts);
  return nv_tensor_ir::buildTensorToCudaTileConversionPipeline(pm, options);
}

template <typename Options>
void registerCheckedPipeline(StringRef name, StringRef description,
                             nv_tensor_ir::Status (*builder)(OpPassManager &,
                                                             const Options &)) {
  registerPassPipeline(
      name, description,
      [builder](OpPassManager &pm, StringRef text,
                llvm::function_ref<LogicalResult(const Twine &)> errorHandler) {
        Options options;
        if (failed(options.parseFromString(text))) {
          return failure();
        }
        auto status = builder(pm, options);
        if (!status.ok()) {
          return errorHandler(status.message());
        }
        return success();
      },
      [](llvm::function_ref<void(const mlir::detail::PassOptions &)> handler) {
        handler(Options());
      });
}

} // namespace

namespace mlir::nv_tensor_ir::test {
void registerTestLoweringPipelines() {

  // Layout propagation pipeline
  // Pass ordering is critical and must be preserved. Passes compute the
  // layout-propagation analyses on demand:
  // 1. TileAnalyzer: Analyzes the iteration spaces and adds tile candidates
  // 2. TileSelection: Selects tile configuration based on a heuristic
  // 3. FormGridPass: Back-propagates iteration-space layouts and forms the
  //    flattened destination-passing tiled grid
  // 4. TileReductionsPass: Materializes contraction/persistence loops
  // 5. OutlineKernelPass: Outlines the mapped forall body
  // 6. TensorToCudaTileConversion: Converts the explicit program to CudaTile
  registerCheckedPipeline<TestTensorToCudaTileOptions>(
      "layout-propagation-pipeline",
      "Pipeline to convert TensorIR to CudaTile using layout propagation",
      buildTensorToCudaTileConversionPassPipeline);
}
} // namespace mlir::nv_tensor_ir::test

int main(int argc, char **argv) {
  DialectRegistry registry;
  mlir::nv_tensor_ir::registerDialects(registry);
  mlir::nv_tensor_ir::registerPasses();
  nv_tensor_ir::test::registerTestLoweringPipelines();
#ifdef TENSOR_IR_INCLUDE_TESTS
  mlir::nv_tensor_ir::test::registerAllPasses();
#endif

  const std::string helpHeader =
      mlir::registerCLIOptions("Tensor IR optimizer test driver\n", registry);
  mlir::nv_tensor_ir::TIRCommandLineOverride overriddenArgv(argc, argv);
  auto [inputFilename, outputFilename] = mlir::parseCLIOptions(
      overriddenArgv.argc(), overriddenArgv.argv(), helpHeader);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(overriddenArgv.argc(), overriddenArgv.argv(),
                        inputFilename, outputFilename, registry));
}
