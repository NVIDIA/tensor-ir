// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_COMPILER_CUDATILE_PIPELINES_H_
#define TENSOR_IR_COMPILER_CUDATILE_PIPELINES_H_

#include "tensor_ir/Conversion/TensorToCudaTile/Options.h"

#include "mlir/Pass/PassManager.h"

namespace mlir::nv_tensor_ir {

/// Add layout and tile analysis passes for the selected CUDA Tile codegen
/// strategy.
void buildGraphAnalysisPipeline(mlir::OpPassManager &pm,
                                const TensorToCudaTilePipelineOptions &opts);

/// Resolve a tile from the analyzed candidates or explicit conversion option.
void buildTileSelectionPipeline(mlir::OpPassManager &pm,
                                const TensorToCudaTilePipelineOptions &opts);

/// Add TensorIR-to-CudaTile conversion and cleanup passes without analysis.
void buildTensorToCudaTileConversionOnlyPipeline(
    mlir::OpPassManager &pm, const TensorToCudaTilePipelineOptions &opts);

/// Add the complete TensorIR-to-CudaTile analysis and conversion pipeline.
void buildTensorToCudaTileConversionPipeline(
    mlir::OpPassManager &pm, const TensorToCudaTilePipelineOptions &opts);

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_COMPILER_CUDATILE_PIPELINES_H_
