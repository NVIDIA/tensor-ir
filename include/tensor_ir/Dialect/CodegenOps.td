// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSORIR_DIALECT_CODEGENOPS_TD
#define TENSORIR_DIALECT_CODEGENOPS_TD

// This file is included from TensorOps.td after the TensorIR operation base,
// types, and attributes are defined.
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Tiled-memory operations
//===----------------------------------------------------------------------===//

// The memref describes the complete logical view, independently of tile shape.
class TensorIR_TiledMemoryOp<string mnemonic, list<Trait> traits = []>
    : TensorIR_Op<mnemonic, traits> {
  let extraClassDeclaration = [{
    SmallVector<OpFoldResult> getMixedIndices();
  }];
  let hasVerifier = 1;
}

def TensorIR_LoadOp : TensorIR_TiledMemoryOp<"load"> {
  let summary = "Load a tensor tile from a strided memref view";
  let description = [{
    Loads a tile from `view`. The view and tile have the same rank and element
    type. `tile_indices` specifies tile coordinates. For tile shape T, tile
    coordinates I, and local element j, the logical
    coordinate is x[d] = I[d] * T[d] + j[d]. The address is the memref's aligned
    base plus its offset plus sum(x[d] * stride[d]), measured in elements.

    Static indices are nonnegative; dynamic entries in `static_indices` consume
    index operands in order. View sizes, offset and coordinates must be
    nonnegative, and strides must be strictly positive. For dynamic strides,
    positivity is a runtime precondition. An element is in bounds when
    x[d] < size[d] in every dimension. Out-of-bounds elements do not read memory
    and use `padding`, or unspecified values when padding is absent. Nonzero
    padding modes require floating-point elements.

    `alignment` guarantees power-of-two byte alignment of the view origin.
    `allow_tma` is a performance hint; false forbids TMA for this access.
  }];
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$view,
                       Variadic<Index>:$indices,
                       DenseI64ArrayAttr:$static_indices,
                       IntValidAlignment<I64Attr>:$alignment,
                       OptionalAttr<TilePaddingAttr>:$padding,
                       DefaultValuedAttr<BoolAttr, "true">:$allow_tma);
  let results = (outs TensorIR_TensorType:$tile);
  let assemblyFormat = [{
    $view `tile_indices` custom<DynamicIndexList>($indices, $static_indices)
    `,` `alignment` $alignment (`,` `padding` $padding^)? attr-dict
    `:` type($view) `to` type($tile)
  }];
  let builders = [OpBuilder<(ins "Type":$tile, "Value":$view,
      "ArrayRef<OpFoldResult>":$indices, "IntegerAttr":$alignment,
      "TilePaddingAttr":$padding, "bool":$allowTma)>];
}

def TensorIR_StoreOp : TensorIR_TiledMemoryOp<"store"> {
  let summary = "Store a tensor tile to a strided memref view";
  let description = [{
    Stores a tile using the same coordinate, address, alignment, and bounds
    semantics as `nv_tensor_ir.load`. The view and tile have the same rank and
    element type. Out-of-bounds elements do not write memory. No ordering
    is specified between writes to overlapping addresses. `allow_tma = false`
    forbids TMA without changing access semantics.
  }];
  let arguments = (ins TensorIR_TensorType:$tile,
                       Arg<AnyMemRef, "", [MemWrite]>:$view,
                       Variadic<Index>:$indices,
                       DenseI64ArrayAttr:$static_indices,
                       IntValidAlignment<I64Attr>:$alignment,
                       DefaultValuedAttr<BoolAttr, "true">:$allow_tma);
  let assemblyFormat = [{
    $tile `,` $view `tile_indices` custom<DynamicIndexList>($indices, $static_indices)
    `,` `alignment` $alignment attr-dict
    `:` type($tile) `,` type($view)
  }];
  let builders = [OpBuilder<(ins "Value":$tile, "Value":$view,
      "ArrayRef<OpFoldResult>":$indices, "IntegerAttr":$alignment,
      "bool":$allowTma)>];
}

#endif // TENSORIR_DIALECT_CODEGENOPS_TD
