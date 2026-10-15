// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_COMPILER_LINALG_PIPELINES_H_
#define TENSOR_IR_COMPILER_LINALG_PIPELINES_H_

namespace mlir {
class OpPassManager;

namespace nv_tensor_ir {

enum class LinalgPipelinePhase { Linalg, LinalgOpt, Bufferize, LLVM };

struct LinalgPipelineOptions {
  LinalgPipelinePhase stopAfter = LinalgPipelinePhase::LLVM;
  bool useLibm = false;
};

/// Populate a module pipeline through the requested phase. The context must
/// have TensorIR's registerDialects interfaces, including bufferization and
/// ConvertToLLVM models, registered before running the pipeline.
void buildLinalgPipeline(OpPassManager &pm,
                         const LinalgPipelineOptions &options);

/// Register tir-linalg-pipeline for command-line use and textual pipelines.
void registerLinalgPipeline();

} // namespace nv_tensor_ir
} // namespace mlir

#endif // TENSOR_IR_COMPILER_LINALG_PIPELINES_H_
