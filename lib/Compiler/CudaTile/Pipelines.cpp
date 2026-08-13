// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Compiler/CudaTile/Pipelines.h"

#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Transform/Passes.h"

#include "mlir/Transforms/Passes.h"

namespace mlir::nv_tensor_ir {

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

void buildTensorToCudaTileConversionOnlyPipeline(
    mlir::OpPassManager &pm, const TensorToCudaTilePipelineOptions &opts) {
  TensorToCudaTileConversionPassOptions options;
  // AffineMap reads tile_size directly from the pass option. In
  // LayoutPropagation, TileSelectionPass sets tile_size on GraphOp and the
  // conversion reads it from there.
  options.tile_size.assign(opts.tileSize.begin(), opts.tileSize.end());
  options.num_ctas = opts.numCTAs;
  options.occupancy = opts.occupancy;
  options.num_warps = opts.numWarps;
  options.sm_count = opts.smCount;
  options.reduction_tile_size = opts.reductionTileSize;
  options.persistence = opts.persistence;
  options.codegen_strategy = opts.codegenStrategy;
  options.uniform_signature = opts.uniformSignature;

  pm.addPass(nv_tensor_ir::createTensorToCudaTileConversionPass(options));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void buildTensorToCudaTileConversionPipeline(
    mlir::OpPassManager &pm, const TensorToCudaTilePipelineOptions &opts) {
  buildGraphAnalysisPipeline(pm, opts);
  buildTileSelectionPipeline(pm, opts);
  buildTensorToCudaTileConversionOnlyPipeline(pm, opts);
}

} // namespace mlir::nv_tensor_ir
