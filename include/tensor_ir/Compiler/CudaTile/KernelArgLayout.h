// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_COMPILER_CUDATILE_KERNEL_ARG_LAYOUT_H_
#define TENSOR_IR_COMPILER_CUDATILE_KERNEL_ARG_LAYOUT_H_

#include "tensor_ir/Runtime/CudaTile/KernelArgLayout.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>

namespace mlir::nv_tensor_ir {
class GraphOp;
struct TensorToCudaTilePipelineOptions;
} // namespace mlir::nv_tensor_ir

namespace tensor_ir {

/// Build runtime kernel argument ABI metadata by inspecting a TensorIR graph's
/// types, stride attributes, and conversion options. Fails (with an MLIR
/// diagnostic on graphOp) if an argument's element type is not one of the
/// recognized runtime dtypes.
::mlir::FailureOr<rt::KernelArgLayout> extractKernelArgLayout(
    ::mlir::nv_tensor_ir::GraphOp graphOp,
    const ::mlir::nv_tensor_ir::TensorToCudaTilePipelineOptions &options);

} // namespace tensor_ir

#endif // TENSOR_IR_COMPILER_CUDATILE_KERNEL_ARG_LAYOUT_H_
