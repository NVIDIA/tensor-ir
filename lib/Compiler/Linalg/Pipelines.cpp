// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Compiler/Linalg/Pipelines.h"

#include "tensor_ir/Conversion/TensorToLinalg/TensorToLinalg.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/MathToFuncs/MathToFuncs.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::nv_tensor_ir;

namespace {
struct LinalgPipelineCLIOptions
    : public PassPipelineOptions<LinalgPipelineCLIOptions> {
  Option<LinalgPipelinePhase> stopAfter{
      *this, "stop-after", llvm::cl::desc("Last pipeline phase to run"),
      llvm::cl::init(LinalgPipelinePhase::LLVM),
      llvm::cl::values(clEnumValN(LinalgPipelinePhase::Linalg, "linalg",
                                  "Stop after conversion to Linalg."),
                       clEnumValN(LinalgPipelinePhase::LinalgOpt, "linalg-opt",
                                  "Stop after Linalg optimization."),
                       clEnumValN(LinalgPipelinePhase::Bufferize, "bufferize",
                                  "Stop after bufferization."),
                       clEnumValN(LinalgPipelinePhase::LLVM, "llvm",
                                  "Stop after lowering to LLVM."))};
  Option<bool> useLibm{*this, "use-libm",
                       llvm::cl::desc("Lower math operations to libm calls"),
                       llvm::cl::init(false)};
};
} // namespace

void mlir::nv_tensor_ir::buildLinalgPipeline(
    OpPassManager &pm, const LinalgPipelineOptions &options) {
  pm.addPass(createTensorToLinalgConversionPass());
  if (options.stopAfter == LinalgPipelinePhase::Linalg) {
    return;
  }

  pm.addNestedPass<func::FuncOp>(createLinalgFoldIntoElementwisePass());
  pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (options.stopAfter == LinalgPipelinePhase::LinalgOpt) {
    return;
  }

  bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizationOptions));
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  if (options.stopAfter == LinalgPipelinePhase::Bufferize) {
    return;
  }

  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<func::FuncOp>(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertMathToFuncs());
  pm.addPass(createSCFToControlFlowPass());
  if (options.useLibm) {
    pm.addPass(createConvertMathToLibmPass());
  }
  pm.addPass(createConvertToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

void mlir::nv_tensor_ir::registerLinalgPipeline() {
  static PassPipelineRegistration<LinalgPipelineCLIOptions> registration(
      "tir-linalg-pipeline", "Lower TensorIR through Linalg to LLVM",
      [](OpPassManager &pm, const LinalgPipelineCLIOptions &options) {
        buildLinalgPipeline(pm, {options.stopAfter, options.useLibm});
      });
}
