// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_DIALECT_TENSORIRATTRS_H_
#define TENSOR_IR_DIALECT_TENSORIRATTRS_H_

#include "tensor_ir/Support/TCutegen.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TensorEncoding.h"

// The tablegen-generated `.h.inc` files below require the MLIR headers above
// (for `mlir::Attribute`, `mlir::AttributeInterface`, `mlir::FieldParser`,
// `llvm::DenseMapInfo`, etc.). This comment also acts as a non-include barrier
// so clang-format does not regroup these includes above the MLIR headers.

#include "tensor_ir/Dialect/TensorAttrInterfaces.h.inc"
#include "tensor_ir/Dialect/TensorEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "tensor_ir/Dialect/TensorAttrs.h.inc"

#endif // TENSOR_IR_DIALECT_TENSORIRATTRS_H_
