// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Compiler/CudaTile/Pipelines.h"

#include "tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Options/Options.h"
#include "tensor_ir/Transform/Passes.h"

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h"

namespace mlir::nv_tensor_ir {

Status buildGraphAnalysisPipeline(mlir::OpPassManager &pm,
                                  const CompilerInvocation &opts) {
  auto target = opts.getKernelSmTarget();
  if (failed(target)) {
    return Status::InvalidArgument("Invalid target SM '" +
                                   opts.deviceTarget.kernelTarget + "'");
  }

  pm.addNestedPass<nv_tensor_ir::GraphOp>(
      nv_tensor_ir::createMaterializeDefaultStridesPass());

  pm.addPass(nv_tensor_ir::createRewriteUnsupportedOpsPass());
  if (opts.cudaTile.codegenStrategy ==
      CudaTileCodegenStrategy::LayoutPropagation) {
    if (opts.codegen.tileSize.empty() && opts.cudaTile.maxTileCandidates > 0) {
      TileAnalyzerPassOptions tileOpts;
      tileOpts.max_candidates = opts.cudaTile.maxTileCandidates;
      tileOpts.computeCapability = target->getComputeCapabilityVersion();
      tileOpts.sm_count = opts.cudaTile.smCount;
      pm.addNestedPass<nv_tensor_ir::GraphOp>(
          nv_tensor_ir::createTileAnalyzerPass(tileOpts));
    }
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
  pm.addNestedPass<func::FuncOp>(
      nv_tensor_ir::createTileReductionsPass(stripMiningOptions));
  if (opts.cudaTile.persistenceMode == PersistenceMode::Static) {
    StaticPersistencePassOptions persistenceOptions;
    persistenceOptions.sm_count = opts.cudaTile.smCount;
    persistenceOptions.occupancy = opts.cudaTile.occupancy;
    pm.addNestedPass<func::FuncOp>(
        nv_tensor_ir::createStaticPersistencePass(persistenceOptions));
  }
  pm.addPass(nv_tensor_ir::createOutlineKernelPass());

  buildHostToLLVMPipeline(pm, opts);
}

void buildHostToLLVMPipeline(OpPassManager &pm,
                             const CompilerInvocation &opts) {
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createSCFToControlFlowPass());
  pm.addPass(createHostAbiBridgePass());
  pm.addPass(createHostToLLVMPass());
  pm.addNestedPass<LLVM::LLVMFuncOp>(createCanonicalizerPass());
  pm.addNestedPass<LLVM::LLVMFuncOp>(createCSEPass());

  if (opts.getEffectiveHostTarget() == HostTarget::Native) {
    HostToObjectPassOptions options{};
    options.hostArch = opts.codegen.hostArch;
    options.hostOS = opts.codegen.hostOS;
    pm.addPass(createHostToObjectPass(options));
  }
}

namespace {
struct HostToLLVMPassOptions
    : public PassPipelineOptions<HostToLLVMPassOptions> {
  Option<HostArch> hostArch{
      *this, "host-arch", llvm::cl::init(HostArch::Native),
      llvm::cl::values(
          clEnumValN(HostArch::Native, "native", "Host architecture"),
          clEnumValN(HostArch::X86_64, "x86_64", "X86_64"),
          clEnumValN(HostArch::AArch64, "aarch64", "AArch64"))};
  Option<HostOS> hostOS{
      *this, "host-os", llvm::cl::init(HostOS::Linux),
      llvm::cl::values(clEnumValN(HostOS::Linux, "linux", "Linux"),
                       clEnumValN(HostOS::Windows, "windows", "Windows"))};
};
} // namespace

void registerHostToLLVMPipeline() {
  PassPipelineRegistration<HostToLLVMPassOptions> registration(
      "tir-host-to-llvm-pipeline",
      "Lower outlined host dispatch to LLVM, preserving GPU kernel bodies",
      [](OpPassManager &pm, const HostToLLVMPassOptions &options) {
        CompilerInvocation inv;
        inv.codegen.hostTarget = HostTarget::Native;
        inv.codegen.hostArch = options.hostArch;
        inv.codegen.hostOS = options.hostOS;
        buildHostToLLVMPipeline(pm, inv);
      });
}

Status
buildTensorToCudaTileConversionOnlyPipeline(mlir::OpPassManager &pm,
                                            const CompilerInvocation &opts) {
  AutoGdcInsertion autoGdc = opts.cudaTile.autoGdcInsertion;
  if (autoGdc != AutoGdcInsertion::None &&
      autoGdc != AutoGdcInsertion::Conservative) {
    return Status::InvalidArgument(
        "Only conservative automatic GDC insertion is supported");
  }
  auto target = opts.getKernelSmTarget();
  if (failed(target)) {
    return Status::InvalidArgument("Invalid target SM '" +
                                   opts.deviceTarget.kernelTarget + "'");
  }

  TensorToCudaTileConversionPassOptions options;
  // AffineMap reads tile_size directly from the pass option. The staged
  // layout-propagation path consumes its selected tile during formation.
  options.targetBytecodeVersion = opts.cudaTile.bytecodeVersion;
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
  pm.addNestedPass<mlir::cuda_tile::ModuleOp>(
      mlir::cuda_tile::createSynthesizeDebugInfoScopesPass());
  // GDC (PDL) instructions require Hopper or newer.
  if (autoGdc == AutoGdcInsertion::Conservative &&
      target->getComputeCapabilityVersion() >= 90) {
    mlir::OpPassManager &entryPm =
        pm.nest<mlir::cuda_tile::ModuleOp>().nest<mlir::cuda_tile::EntryOp>();
    entryPm.addPass(nv_tensor_ir::createCudaTileInsertGdcWaitPass());
  }
  return Status::Ok();
}

Status buildTensorToCudaTileConversionPipeline(mlir::OpPassManager &pm,
                                               const CompilerInvocation &opts) {
  TIR_RETURN_IF_ERROR(buildGraphAnalysisPipeline(pm, opts));
  buildTileSelectionPipeline(pm, opts);
  buildTensorIRTiledProgramFormationPipeline(pm, opts);

  return buildTensorToCudaTileConversionOnlyPipeline(pm, opts);
}

} // namespace mlir::nv_tensor_ir
