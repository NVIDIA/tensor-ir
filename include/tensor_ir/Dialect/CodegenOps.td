// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TENSOR_IR_CODEGEN_OPS
#define TENSOR_IR_CODEGEN_OPS

// This file is included from TensorOps.td after the TensorIR operation base,
// types, and attributes are defined.
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Dialect/Ptr/IR/PtrDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Tiled-memory operations
//===----------------------------------------------------------------------===//

def TensorIR_LoadOp : TensorIR_Op<"load", [
    AttrSizedOperandSegments,
    OffsetSizeAndStrideOpInterface
  ]> {
  let summary = "Load a tensor tile from an opaque pointer";
  let description = [{
    Loads a tensor tile from the memory described by `base` and `view`.
    The result type supplies the tile shape and element type.

    The view has rank `r`, equal to the rank of `tile`. Its metadata consists
    of one scalar offset `o`, `r` sizes `S[d]`, and `r` strides `s[d]`.
    Offset and strides are measured in elements of the tile's element type.
    Sizes, strides, offset, and tile coordinates must be nonnegative.

    Each `static_offsets`, `static_sizes`, or `static_strides` entry is either
    a constant or `ShapedType::kDynamic`. Dynamic entries take their values
    from the corresponding `offsets`, `sizes`, or `strides` operands, in
    order. In particular, `static_offsets` always has exactly one entry:
    `offsets` is empty for a constant offset and contains one `index` operand
    for a dynamic offset. There are no per-dimension offsets. In assembly,
    `view offset: [7]` and `view offset: [%off]` illustrate these two cases.

    Let `T[d]` be the tile shape and `I[d]` the mixed tile coordinates,
    written in brackets after `base`. The `static_indices` array has one
    entry per dimension; each `ShapedType::kDynamic` entry consumes one
    `index` operand from `indices`, in order. For example, `[2, %column]`
    combines a static row coordinate with a dynamic column coordinate.
    For a local tile element `j`, where `0 <= j[d] < T[d]`, define:

    ```text
    x[d] = I[d] * T[d] + j[d]
    a(j) = o + sum(d = 0 .. r-1, x[d] * s[d])
    p(j) = element_ptr(base, a(j))
    ```

    Here `element_ptr(base, a)` advances `base` by `a` elements of the tile's
    element type. For an element type occupying a whole number `E` of bytes,
    this is the byte address `base + E * a`. Thus the scalar offset shifts
    the view origin once, while each tile coordinate is scaled by its extent
    and then by its dimension's stride. For example, with `T = [16, 32]`,
    `I = [2, 3]`, `o = 7`, and `s = [128, 1]`, local element `[u, v]` has
    element offset `7 + (32 + u) * 128 + 96 + v` from `base`.

    A dimension whose stride is statically zero is a broadcast dimension:
    its coordinates do not affect `p(j)`, and its size and coordinates do
    not participate in the memory bounds check. An element is in bounds
    when `x[d] < S[d]` for every other dimension. In particular, a zero size
    in a non-broadcast dimension makes every element out of bounds.

    For each in-bounds element, `tile[j]` is the value read at `p(j)`.
    Loads broadcast that value along statically zero-stride dimensions.
    Out-of-bounds elements do not read memory. Their values are selected by
    `padding`: `zero`, `neg_zero`, `nan`, `pos_inf`, or `neg_inf`; all but
    `zero` require a floating-point element type. Without `padding`, the
    values of out-of-bounds elements are unspecified.

    `alignment` is a positive power-of-two byte-alignment guarantee for the
    view origin `element_ptr(base, o)`. It does not change the address
    calculation or guarantee that every element address has that alignment.
  }];
  let arguments = (ins Arg<Ptr_PtrType, "", [MemRead]>:$base,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       Variadic<Index>:$indices,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_sizes,
                       DenseI64ArrayAttr:$static_strides,
                       DenseI64ArrayAttr:$static_indices,
                       IntValidAlignment<I64Attr>:$alignment,
                       OptionalAttr<TilePaddingAttr>:$padding);
  let results = (outs TensorIR_TensorType:$tile);
  let assemblyFormat = [{
    $base `` custom<DynamicIndexList>($indices, $static_indices) `view`
    `offset` `` `:` custom<DynamicIndexList>($offsets, $static_offsets) `,`
    `sizes` `` `:` custom<DynamicIndexList>($sizes, $static_sizes) `,`
    `strides` `` `:` custom<DynamicIndexList>($strides, $static_strides) `,`
    `alignment` `` `:` $alignment
    (`padding` `=` $padding^)? attr-dict
    `:` qualified(type($base)) `to` type($tile)
  }];
  let builders = [OpBuilder<(ins
      "Type":$tile, "Value":$base, "OpFoldResult":$offset,
      "ArrayRef<OpFoldResult>":$sizes, "ArrayRef<OpFoldResult>":$strides,
      "ArrayRef<OpFoldResult>":$indices, "IntegerAttr":$alignment,
      "TilePaddingAttr":$padding)>];
  let extraClassDeclaration = [{
    /// Sizes and strides are available through the interface's getMixedSizes
    /// and getMixedStrides methods. The view has a single mixed offset.
    OpFoldResult getMixedOffset() { return getMixedOffsets().front(); }

    /// Return all tile coordinates, including static entries.
    SmallVector<OpFoldResult> getMixedIndices();

    /// Return the expected ranks of the offset, size, and stride metadata.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getTile().getType().getRank();
      return {1, rank, rank};
    }

    /// Return the number of operands preceding the view metadata.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }
  }];
  let hasVerifier = 1;
}

def TensorIR_StoreOp : TensorIR_Op<"store", [
    AttrSizedOperandSegments,
    OffsetSizeAndStrideOpInterface
  ]> {
  let summary = "Store a tensor tile through an opaque pointer";
  let description = [{
    Stores `tile` to the memory described by `base` and `view`.
    The type of `tile` supplies the tile shape and element type.

    The view has rank `r`, equal to the rank of `tile`. Its metadata consists
    of one scalar offset `o`, `r` sizes `S[d]`, and `r` strides `s[d]`.
    Offset and strides are measured in elements of the tile's element type.
    Sizes, strides, offset, and tile coordinates must be nonnegative.

    Each `static_offsets`, `static_sizes`, or `static_strides` entry is either
    a constant or `ShapedType::kDynamic`. Dynamic entries take their values
    from the corresponding `offsets`, `sizes`, or `strides` operands, in
    order. In particular, `static_offsets` always has exactly one entry:
    `offsets` is empty for a constant offset and contains one `index` operand
    for a dynamic offset. There are no per-dimension offsets. In assembly,
    `view offset: [7]` and `view offset: [%off]` illustrate these two cases.

    Let `T[d]` be the tile shape and `I[d]` the mixed tile coordinates,
    written in brackets after `base`. The `static_indices` array has one
    entry per dimension; each `ShapedType::kDynamic` entry consumes one
    `index` operand from `indices`, in order. For example, `[2, %column]`
    combines a static row coordinate with a dynamic column coordinate.
    For a local tile element `j`, where `0 <= j[d] < T[d]`, define:

    ```text
    x[d] = I[d] * T[d] + j[d]
    a(j) = o + sum(d = 0 .. r-1, x[d] * s[d])
    p(j) = element_ptr(base, a(j))
    ```

    Here `element_ptr(base, a)` advances `base` by `a` elements of the tile's
    element type. For an element type occupying a whole number `E` of bytes,
    this is the byte address `base + E * a`. Thus the scalar offset shifts
    the view origin once, while each tile coordinate is scaled by its extent
    and then by its dimension's stride. For example, with `T = [16, 32]`,
    `I = [2, 3]`, `o = 7`, and `s = [128, 1]`, local element `[u, v]` has
    element offset `7 + (32 + u) * 128 + 96 + v` from `base`.

    A dimension whose stride is statically zero is a broadcast dimension:
    its coordinates do not affect `p(j)`, and its size and coordinates do
    not participate in the memory bounds check. An element is in bounds
    when `x[d] < S[d]` for every other dimension. In particular, a zero size
    in a non-broadcast dimension makes every element out of bounds.

    For each in-bounds element, the operation writes `tile[j]` to `p(j)`.
    In a statically zero-stride dimension, only the slice with `j[d] = 0`
    is stored; the other slices are ignored. Out-of-bounds elements do not
    write memory. No ordering is specified between writes to overlapping
    addresses.

    `alignment` is a positive power-of-two byte-alignment guarantee for the
    view origin `element_ptr(base, o)`. It does not change the address
    calculation or guarantee that every element address has that alignment.
  }];
  let arguments = (ins TensorIR_TensorType:$tile,
                       Arg<Ptr_PtrType, "", [MemWrite]>:$base,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       Variadic<Index>:$indices,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_sizes,
                       DenseI64ArrayAttr:$static_strides,
                       DenseI64ArrayAttr:$static_indices,
                       IntValidAlignment<I64Attr>:$alignment);
  let assemblyFormat = [{
    $tile `,` $base `` custom<DynamicIndexList>($indices, $static_indices) `view`
    `offset` `` `:` custom<DynamicIndexList>($offsets, $static_offsets) `,`
    `sizes` `` `:` custom<DynamicIndexList>($sizes, $static_sizes) `,`
    `strides` `` `:` custom<DynamicIndexList>($strides, $static_strides) `,`
    `alignment` `` `:` $alignment
    attr-dict
    `:` type($tile) `,` qualified(type($base))
  }];
  let builders = [OpBuilder<(ins
      "Value":$tile, "Value":$base, "OpFoldResult":$offset,
      "ArrayRef<OpFoldResult>":$sizes, "ArrayRef<OpFoldResult>":$strides,
      "ArrayRef<OpFoldResult>":$indices, "IntegerAttr":$alignment)>];
  let extraClassDeclaration = [{
    /// Sizes and strides are available through the interface's getMixedSizes
    /// and getMixedStrides methods. The view has a single mixed offset.
    OpFoldResult getMixedOffset() { return getMixedOffsets().front(); }

    /// Return all tile coordinates, including static entries.
    SmallVector<OpFoldResult> getMixedIndices();

    /// Return the expected ranks of the offset, size, and stride metadata.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getTile().getType().getRank();
      return {1, rank, rank};
    }

    /// Return the number of operands preceding the view metadata.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 2; }
  }];
  let hasVerifier = 1;
}

#endif // TENSOR_IR_CODEGEN_OPS
