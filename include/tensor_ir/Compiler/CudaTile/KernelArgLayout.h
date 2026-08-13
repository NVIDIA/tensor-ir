// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_COMPILER_CUDATILE_KERNEL_ARG_LAYOUT_H_
#define TENSOR_IR_COMPILER_CUDATILE_KERNEL_ARG_LAYOUT_H_

#include "tensor_ir/Runtime/CudaTile/KernelArgLayout.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>

namespace mlir::nv_tensor_ir {
class GraphOp;
} // namespace mlir::nv_tensor_ir

namespace tensor_ir {

/// Build runtime kernel argument ABI metadata by inspecting a TensorIR graph's
/// types and stride attributes with the selected tile and signature policy.
rt::KernelArgLayout
extractKernelArgLayout(::mlir::nv_tensor_ir::GraphOp graphOp,
                       llvm::ArrayRef<int32_t> tileSizes,
                       bool uniformSignature);

} // namespace tensor_ir

#endif // TENSOR_IR_COMPILER_CUDATILE_KERNEL_ARG_LAYOUT_H_
