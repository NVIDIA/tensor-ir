// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- TensorAttrInterfaces.td - Tensor IR Attr Interfaces -*- tablegen -*-===//
//
// This file defines the attribute interfaces for the Tensor IR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_IR_DIALECT_TENSOR_ATTR_INTERFACES_TD
#define TENSOR_IR_DIALECT_TENSOR_ATTR_INTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// LayoutSourceAttrInterface
//===----------------------------------------------------------------------===//

def LayoutSourceAttrInterface : AttrInterface<"LayoutSourceAttrInterface"> {
  let cppNamespace = "::mlir::nv_tensor_ir";
  let description = [{
    Interface for layout source attributes used in layout propagation.

    Layout source attributes represent the memory layout of tensors as they
    flow through a computation graph. This interface provides a common
    abstraction for different types of layout annotations.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Get the shape of the tensor represented by this layout source.
      }],
      /*retTy=*/"::llvm::SmallVector<int64_t>",
      /*methodName=*/"getShape",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Apply a last-dimension-fastest reshape to the layout.
      }],
      /*retTy=*/"::mlir::nv_tensor_ir::LayoutSourceAttrInterface",
      /*methodName=*/"reshape",
      /*args=*/(ins "::llvm::ArrayRef<int64_t>":$newShape)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Apply broadcast operation to the layout.
      }],
      /*retTy=*/"::mlir::nv_tensor_ir::LayoutSourceAttrInterface",
      /*methodName=*/"broadcast",
      /*args=*/(ins "::llvm::ArrayRef<int64_t>":$newShape)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Apply transpose operation to the layout.
      }],
      /*retTy=*/"::mlir::nv_tensor_ir::LayoutSourceAttrInterface",
      /*methodName=*/"transpose",
      /*args=*/(ins "::llvm::ArrayRef<int64_t>":$permutation)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Apply slice operation to the layout.
      }],
      /*retTy=*/"::mlir::nv_tensor_ir::LayoutSourceAttrInterface",
      /*methodName=*/"slice",
      /*args=*/(ins "::llvm::ArrayRef<int64_t>":$starts,
                    "::llvm::ArrayRef<int64_t>":$limits,
                    "::llvm::ArrayRef<int64_t>":$strides)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Normalize the layout (convert to iteration space).
      }],
      /*retTy=*/"::mlir::nv_tensor_ir::LayoutSourceAttrInterface",
      /*methodName=*/"normalize",
      /*args=*/(ins)
    >,
  ];
}

#endif // TENSOR_IR_DIALECT_TENSOR_ATTR_INTERFACES_TD
