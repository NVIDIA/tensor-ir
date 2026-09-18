// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tensor_ir/Dialect/TensorIR.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::nv_tensor_ir;

#define GET_TYPEDEF_CLASSES
#include "tensor_ir/Dialect/TensorTypes.cpp.inc"

void TensorIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "tensor_ir/Dialect/TensorTypes.cpp.inc"
      >();
}
