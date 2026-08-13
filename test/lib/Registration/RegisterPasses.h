// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_TEST_LIB_REGISTRATION_REGISTER_PASSES_H
#define TENSOR_IR_TEST_LIB_REGISTRATION_REGISTER_PASSES_H

namespace mlir::nv_tensor_ir::test {

/// Registers all Tensor IR test passes so they can be consumed by
/// tensor_ir-opt and other testing utilities.
void registerAllPasses();

} // namespace mlir::nv_tensor_ir::test

#endif // TENSOR_IR_TEST_LIB_REGISTRATION_REGISTER_PASSES_H
