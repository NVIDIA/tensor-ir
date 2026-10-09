// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_COMPILER_CUDATILE_KERNEL_METADATA_EXTRACTION_H_
#define TENSOR_IR_COMPILER_CUDATILE_KERNEL_METADATA_EXTRACTION_H_

#include "tensor_ir/Artifact/KernelMetadata.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>

namespace mlir::nv_tensor_ir {
struct CodegenOptions;
struct CudaTileOptions;
class GraphOp;
} // namespace mlir::nv_tensor_ir

namespace mlir::nv_tensor_ir::compiler::cuda_tile {

/// Build compiled kernel launch metadata by inspecting a TensorIR graph's
/// types, stride attributes, and generated compiler options. Fails (with an
/// MLIR diagnostic on graphOp) if an argument has an unsupported element type.
::mlir::FailureOr<::tensor_ir::rt::KernelMetadata>
extractKernelMetadata(GraphOp graphOp, const CodegenOptions &codegen,
                      const CudaTileOptions &cudaTile);

} // namespace mlir::nv_tensor_ir::compiler::cuda_tile

#endif // TENSOR_IR_COMPILER_CUDATILE_KERNEL_METADATA_EXTRACTION_H_
