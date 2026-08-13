// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Registration/Registration.h"

#include "tensor_ir/Dialect/TensorIR.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;

void nv_tensor_ir::registerDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<
    arith::ArithDialect,
    func::FuncDialect,
    nv_tensor_ir::TensorIRDialect
  >();
  // clang-format on

  func::registerInlinerExtension(registry);
}
