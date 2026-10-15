// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_COMPILER_CUDATILE_PIPELINES_H_
#define TENSOR_IR_COMPILER_CUDATILE_PIPELINES_H_

#include "tensor_ir/Support/Status.h"

#include "mlir/Pass/PassManager.h"

namespace mlir::nv_tensor_ir {

struct CompilerInvocation;

/// Add layout and tile analysis passes for the selected CUDA Tile codegen
/// strategy. Invalid kernel targets return an error without adding passes.
Status buildGraphAnalysisPipeline(mlir::OpPassManager &pm,
                                  const CompilerInvocation &opts);

/// Resolve a tile from the analyzed candidates or explicit conversion option.
void buildTileSelectionPipeline(mlir::OpPassManager &pm,
                                const CompilerInvocation &opts);

/// Form the flattened grid and materialize contraction and persistence loops
/// for layout-propagation lowering.
void buildTensorIRTiledProgramFormationPipeline(mlir::OpPassManager &pm,
                                                const CompilerInvocation &opts);

/// Lower top-level outlined dispatch completely to LLVM dialect. Device kernel
/// bodies remain untouched in GPU modules for subsequent device conversion.
void buildHostToLLVMPipeline(OpPassManager &pm, const CompilerInvocation &opts);

/// Register the same host pipeline for textual IR tests and reproducer replay.
void registerHostToLLVMPipeline();

/// Add TensorIR-to-CudaTile conversion and cleanup passes without analysis.
/// With `auto-gdc-insertion=conservative` on SM90+, also add the PDL wait
/// insertion pass. Unsupported insertion modes and invalid kernel targets
/// return an error without adding passes.
Status
buildTensorToCudaTileConversionOnlyPipeline(mlir::OpPassManager &pm,
                                            const CompilerInvocation &opts);

/// Add the complete TensorIR-to-CudaTile analysis and conversion pipeline.
/// Invalid kernel targets return an error without adding passes.
Status buildTensorToCudaTileConversionPipeline(mlir::OpPassManager &pm,
                                               const CompilerInvocation &opts);

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_COMPILER_CUDATILE_PIPELINES_H_
