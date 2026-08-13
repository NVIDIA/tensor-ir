// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Passes.h - TensorIR transformation passes ---------------*- C++ -*-===//

#ifndef TENSOR_IR_TRANSFORM_PASSES_H
#define TENSOR_IR_TRANSFORM_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace nv_tensor_ir {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "tensor_ir/Transform/Passes.h.inc"

} // namespace nv_tensor_ir
} // namespace mlir

#endif // TENSOR_IR_TRANSFORM_PASSES_H
