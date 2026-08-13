// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_REGISTRATION_REGISTRATION_H
#define TENSOR_IR_REGISTRATION_REGISTRATION_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace mlir::nv_tensor_ir {

/// Registers the dialects and extensions used by the TensorIR compiler and
/// Python package.
void registerDialects(DialectRegistry &registry);

} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_REGISTRATION_REGISTRATION_H
