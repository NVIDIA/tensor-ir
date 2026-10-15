// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_CONVERSION_TENSORTOLINALG_PASSES
#define TENSOR_IR_CONVERSION_TENSORTOLINALG_PASSES

include "mlir/Pass/PassBase.td"

def TensorToLinalgConversionPass
    : Pass<"convert-tensor-to-linalg", "mlir::ModuleOp"> {
  let summary = "Convert public TensorIR operations to Linalg and Tensor";
  let description = [{
    Converts the supported public TensorIR operation set to tensor-semantics
    Linalg, Tensor, Arith, Math, and Func operations. TensorIR operations that
    are not in the OSS release are deliberately outside this
    pass's conversion contract and remain legal.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::math::MathDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif // TENSOR_IR_CONVERSION_TENSORTOLINALG_PASSES
