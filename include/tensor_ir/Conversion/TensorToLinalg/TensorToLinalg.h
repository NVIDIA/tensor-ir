// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_CONVERSION_TENSORTOLINALG_TENSORTOLINALG_H
#define TENSOR_IR_CONVERSION_TENSORTOLINALG_TENSORTOLINALG_H

#include "tensor_ir/Dialect/TensorIR.h"

#include "mlir/Pass/Pass.h" // IWYU pragma: keep

namespace mlir::nv_tensor_ir {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "tensor_ir/Conversion/TensorToLinalg/Passes.h.inc"

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_CONVERSION_TENSORTOLINALG_TENSORTOLINALG_H
