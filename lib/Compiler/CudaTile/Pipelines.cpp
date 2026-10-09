// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Compiler/CudaTile/Pipelines.h"

#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Options/Options.h"
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

Status buildGraphAnalysisPipeline(mlir::OpPassManager &pm,
                                  const CompilerInvocation &opts) {
  auto target = opts.getKernelSmTarget();
  if (failed(target)) {
    return Status::InvalidArgument("Invalid target SM '" +
                                   opts.deviceTarget.kernelTarget + "'");
  }
  pm.addNestedPass<nv_tensor_ir::GraphOp>(
      nv_tensor_ir::createMaterializeDefaultStridesPass());
  if (opts.cudaTile.codegenStrategy ==
      CudaTileCodegenStrategy::LayoutPropagation) {
    pm.addNestedPass<nv_tensor_ir::GraphOp>(
        nv_tensor_ir::createLayoutPropagationAnnotationPass());
    pm.addNestedPass<nv_tensor_ir::GraphOp>(
        nv_tensor_ir::createLayoutPropagationNormalizationPass());

    if (opts.codegen.tileSize.empty() && opts.cudaTile.maxTileCandidates > 0) {
      TileAnalyzerPassOptions tileOpts;
      tileOpts.max_candidates = opts.cudaTile.maxTileCandidates;
      tileOpts.computeCapability = target->getComputeCapabilityVersion();
      tileOpts.sm_count = opts.cudaTile.smCount;
      pm.addNestedPass<nv_tensor_ir::GraphOp>(
          nv_tensor_ir::createTileAnalyzerPass(tileOpts));
    }

    pm.addNestedPass<nv_tensor_ir::GraphOp>(
        nv_tensor_ir::createGraphSplittingPass());
  } else {
    pm.addNestedPass<nv_tensor_ir::GraphOp>(
        nv_tensor_ir::createDiscoverIterationSpaceInfoPass());
  }
  return Status::Ok();
}

void buildTileSelectionPipeline(mlir::OpPassManager &pm,
                                const CompilerInvocation &opts) {
  if (opts.cudaTile.codegenStrategy !=
      CudaTileCodegenStrategy::LayoutPropagation) {
    return;
  }
  TileSelectionPassOptions selOpts;
  selOpts.tile_size.assign(opts.codegen.tileSize.begin(),
                           opts.codegen.tileSize.end());
  pm.addNestedPass<nv_tensor_ir::GraphOp>(
      nv_tensor_ir::createTileSelectionPass(selOpts));
}

void buildTensorIRTiledProgramFormationPipeline(
    mlir::OpPassManager &pm, const CompilerInvocation &opts) {
  if (opts.cudaTile.codegenStrategy !=
      CudaTileCodegenStrategy::LayoutPropagation) {
    return;
  }

  pm.addPass(nv_tensor_ir::createBufferizePass());
  pm.addNestedPass<func::FuncOp>(nv_tensor_ir::createFormGridPass());

  TileReductionsPassOptions stripMiningOptions;
  stripMiningOptions.reduction_tile_size = opts.cudaTile.reductionTileSize;
  stripMiningOptions.persistence =
      getPersistenceName(opts.cudaTile.persistenceMode).str();
  stripMiningOptions.sm_count = opts.cudaTile.smCount;
  stripMiningOptions.occupancy = opts.cudaTile.occupancy;
  pm.addNestedPass<func::FuncOp>(
      nv_tensor_ir::createTileReductionsPass(stripMiningOptions));
}

void buildTensorIRKernelOutliningPipeline(mlir::OpPassManager &pm) {
  pm.addPass(nv_tensor_ir::createOutlineKernelPass());
}

void buildTensorIRTiledProgramPipeline(mlir::OpPassManager &pm,
                                       const CompilerInvocation &opts) {
  buildTensorIRTiledProgramFormationPipeline(pm, opts);
  if (opts.cudaTile.codegenStrategy ==
      CudaTileCodegenStrategy::LayoutPropagation) {
    buildTensorIRKernelOutliningPipeline(pm);
  }
}

void buildTensorToCudaTileConversionOnlyPipeline(
    mlir::OpPassManager &pm, const CompilerInvocation &opts) {
  TensorToCudaTileConversionPassOptions options;
  // AffineMap reads tile_size directly from the pass option. The staged
  // layout-propagation path consumes its selected tile during formation.
  options.num_ctas = opts.codegen.ctaCount;
  options.occupancy = opts.cudaTile.occupancy;
  options.num_warps = opts.cudaTile.warpCount;
  options.codegen_strategy = opts.cudaTile.codegenStrategy;
  if (opts.cudaTile.codegenStrategy == CudaTileCodegenStrategy::AffineMap) {
    options.tile_size.assign(opts.codegen.tileSize.begin(),
                             opts.codegen.tileSize.end());
    options.sm_count = opts.cudaTile.smCount;
    options.persistence = opts.cudaTile.persistenceMode;
  }

  pm.addPass(nv_tensor_ir::createTensorToCudaTileConversionPass(options));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

Status buildTensorToCudaTileConversionPipeline(mlir::OpPassManager &pm,
                                               const CompilerInvocation &opts) {
  TIR_RETURN_IF_ERROR(buildGraphAnalysisPipeline(pm, opts));
  buildTileSelectionPipeline(pm, opts);
  buildTensorIRTiledProgramPipeline(pm, opts);
  buildTensorToCudaTileConversionOnlyPipeline(pm, opts);
  return Status::Ok();
}

} // namespace mlir::nv_tensor_ir
