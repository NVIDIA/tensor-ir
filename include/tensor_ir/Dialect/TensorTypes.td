// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSORIR_TYPES
#define TENSORIR_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"

include "tensor_ir/Dialect/TensorDialect.td"
include "tensor_ir/Dialect/TensorEnums.td"
include "tensor_ir/Dialect/TensorAttrs.td"

// Floating-point Type
def TensorIR_Float
    : Type<CPred<"::llvm::isa<::mlir::FloatType>($_self)">, "floating-point">;

class TensorIR_TensorTypeOf<list<Type> dtypes>
    : RankedTensorOf<
          dtypes, [CPred<"::mlir::nv_tensor_ir::isTensorType($_self)">],
          "TensorIR-compatible ranked tensor"> {
  string cppClassName = "::mlir::RankedTensorType";
}

def TensorIR_FloatTensor : TensorIR_TensorTypeOf<[TensorIR_Float]>;
def TensorIR_FloatTensorLike : AnyTypeOf<[TensorIR_Float, TensorIR_FloatTensor]>;


// Boolean Type (I1)
def TensorIR_BoolTensor : TensorIR_TensorTypeOf<[I1]>;
def TensorIR_BoolTensorLike : AnyTypeOf<[I1, TensorIR_BoolTensor]>;

// Integer Type
def TensorIR_Int : AnyTypeOf<[SI8, SI16, SI32, SI64, UI8, UI16, UI32, UI64], "integer">;
def TensorIR_IntTensor : TensorIR_TensorTypeOf<[TensorIR_Int]>;
def TensorIR_IntTensorLike : AnyTypeOf<[TensorIR_Int, TensorIR_IntTensor]>;

def TensorIR_TensorElementType : AnyTypeOf<[TensorIR_Float, I1, TensorIR_Int]>;
def TensorIR_TensorType : TensorIR_TensorTypeOf<[TensorIR_TensorElementType]>;
def TensorIR_TensorLike : AnyTypeOf<[TensorIR_FloatTensorLike, TensorIR_BoolTensorLike, TensorIR_IntTensorLike]>;

// 32b Integer Type
def TensorIR_SignedInt32Tensor  : TensorIR_TensorTypeOf<[SI32]>;
def TensorIR_UnsignedInt32Tensor  : TensorIR_TensorTypeOf<[UI32]>;

// 64b Integer Type
def TensorIR_SignedInt64Tensor  : TensorIR_TensorTypeOf<[SI64]>;
def TensorIR_UnsignedInt64Tensor  : TensorIR_TensorTypeOf<[UI64]>;
#endif  // TENSORIR_TYPES
