// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Compiler/CudaTile/Pipelines.h"

#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Transform/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::nv_tensor_ir {

namespace {

static StringRef getPersistenceName(PersistenceMode persistence) {
  switch (persistence) {
  case PersistenceMode::None:
    return "none";
  case PersistenceMode::Static:
    return "static";
  }
  llvm_unreachable("unknown TensorIR persistence mode");
}

} // namespace

void buildGraphAnalysisPipeline(mlir::OpPassManager &pm,
                                const TensorToCudaTilePipelineOptions &opts) {
  pm.addNestedPass<nv_tensor_ir::GraphOp>(
      nv_tensor_ir::createMaterializeDefaultStridesPass());
  if (opts.codegenStrategy == CudaTileCodegenStrategy::LayoutPropagation) {
    pm.addNestedPass<nv_tensor_ir::GraphOp>(
        nv_tensor_ir::createLayoutPropagationAnnotationPass());
    pm.addNestedPass<nv_tensor_ir::GraphOp>(
        nv_tensor_ir::createLayoutPropagationNormalizationPass());

    if (opts.maxCandidates > 0) {
      TileAnalyzerPassOptions tileOpts;
      tileOpts.tile_size.assign(opts.tileSize.begin(), opts.tileSize.end());
      tileOpts.max_candidates = opts.maxCandidates;
      tileOpts.computeCapability = opts.computeCapability;
      tileOpts.sm_count = opts.smCount;
      pm.addNestedPass<nv_tensor_ir::GraphOp>(
          nv_tensor_ir::createTileAnalyzerPass(tileOpts));
    }

    pm.addNestedPass<nv_tensor_ir::GraphOp>(
        nv_tensor_ir::createGraphSplittingPass());
  } else {
    pm.addNestedPass<nv_tensor_ir::GraphOp>(
        nv_tensor_ir::createDiscoverIterationSpaceInfoPass());
  }
}

void buildTileSelectionPipeline(mlir::OpPassManager &pm,
                                const TensorToCudaTilePipelineOptions &opts) {
  if (opts.codegenStrategy != CudaTileCodegenStrategy::LayoutPropagation) {
    return;
  }
  TileSelectionPassOptions selOpts;
  selOpts.tile_size.assign(opts.tileSize.begin(), opts.tileSize.end());
  pm.addNestedPass<nv_tensor_ir::GraphOp>(
      nv_tensor_ir::createTileSelectionPass(selOpts));
}

void buildTensorIRTiledProgramFormationPipeline(
    mlir::OpPassManager &pm, const TensorToCudaTilePipelineOptions &opts) {
  if (opts.codegenStrategy != CudaTileCodegenStrategy::LayoutPropagation) {
    return;
  }

  pm.addPass(nv_tensor_ir::createBufferizePass());
  pm.addNestedPass<func::FuncOp>(nv_tensor_ir::createFormGridPass());

  TileReductionsPassOptions stripMiningOptions;
  stripMiningOptions.reduction_tile_size = opts.reductionTileSize;
  stripMiningOptions.persistence = getPersistenceName(opts.persistence).str();
  stripMiningOptions.sm_count = opts.smCount;
  stripMiningOptions.occupancy = opts.occupancy;
  pm.addNestedPass<func::FuncOp>(
      nv_tensor_ir::createTileReductionsPass(stripMiningOptions));
}

void buildTensorIRKernelOutliningPipeline(mlir::OpPassManager &pm) {
  pm.addPass(nv_tensor_ir::createOutlineKernelPass());
}

void buildTensorIRTiledProgramPipeline(
    mlir::OpPassManager &pm, const TensorToCudaTilePipelineOptions &opts) {
  buildTensorIRTiledProgramFormationPipeline(pm, opts);
  if (opts.codegenStrategy == CudaTileCodegenStrategy::LayoutPropagation) {
    buildTensorIRKernelOutliningPipeline(pm);
  }
}

void buildTensorToCudaTileConversionOnlyPipeline(
    mlir::OpPassManager &pm, const TensorToCudaTilePipelineOptions &opts) {
  TensorToCudaTileConversionPassOptions options;
  // AffineMap reads tile_size directly from the pass option. The staged
  // layout-propagation path consumes its selected tile during formation.
  options.num_ctas = opts.numCTAs;
  options.occupancy = opts.occupancy;
  options.num_warps = opts.numWarps;
  options.codegen_strategy = opts.codegenStrategy;
  options.uniform_signature = opts.uniformSignature;
  if (opts.codegenStrategy == CudaTileCodegenStrategy::AffineMap) {
    options.tile_size.assign(opts.tileSize.begin(), opts.tileSize.end());
    options.sm_count = opts.smCount;
    options.persistence = opts.persistence;
  }

  pm.addPass(nv_tensor_ir::createTensorToCudaTileConversionPass(options));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void buildTensorToCudaTileConversionPipeline(
    mlir::OpPassManager &pm, const TensorToCudaTilePipelineOptions &opts) {
  buildGraphAnalysisPipeline(pm, opts);
  buildTileSelectionPipeline(pm, opts);
  buildTensorIRTiledProgramPipeline(pm, opts);
  buildTensorToCudaTileConversionOnlyPipeline(pm, opts);
}

} // namespace mlir::nv_tensor_ir
