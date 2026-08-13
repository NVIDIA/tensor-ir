// RUN: tensor_ir-opt -materialize-default-strides -layout-propagation-annotation -split-input-file %s | FileCheck %s

// Tests for the layout propagation annotation pass.

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

// Join dimensions: (4,16) -> (64) = (64):(1)
// CHECK-LABEL: @reshape_join_dims
// CHECK: reshape %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(64):(1)">}
nv_tensor_ir.graph @reshape_join_dims(%in: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) -> (tensor<64xf32>) {
    %transposed = transpose %in permutation = [1, 0] : tensor<4x16xf32> -> tensor<16x4xf32>
    %out = reshape %transposed : tensor<16x4xf32> -> tensor<64xf32>
    results %out : tensor<64xf32>
}

// -----

// Split dimensions: (64) -> (8,8) = (8,8):(1,8)
// CHECK-LABEL: @reshape_split_1d_to_2d
// CHECK: transpose %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(8,8):(1,8)">}
nv_tensor_ir.graph @reshape_split_1d_to_2d(%in: tensor<64xf32>) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"}) {
    %reshaped = reshape %in : tensor<64xf32> -> tensor<8x8xf32>
    %out = transpose %reshaped permutation = [1, 0] : tensor<8x8xf32> -> tensor<8x8xf32>
    results %out : tensor<8x8xf32>
}

// -----

// Split last dimension: (4,16) -> (4,8,2) = (4,8,2):(1,4,32)
// CHECK-LABEL: @reshape_split_last_dim
// CHECK: transpose %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(4,8,2):(1,4,32)">}
nv_tensor_ir.graph @reshape_split_last_dim(%in: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) -> (tensor<4x8x2xf32> {nv_tensor_ir.stride = "(1,4,32)"}) {
    %transposed = transpose %in permutation = [1, 0] : tensor<4x16xf32> -> tensor<16x4xf32>
    %reshaped = reshape %transposed : tensor<16x4xf32> -> tensor<2x8x4xf32>
    %out = transpose %reshaped permutation = [2, 1, 0] : tensor<2x8x4xf32> -> tensor<4x8x2xf32>
    results %out : tensor<4x8x2xf32>
}

// -----

// Simple reshape: (2,3) -> (3,2) = (3,2):(1,3)
// CHECK-LABEL: @reshape_simple
// CHECK: transpose %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(3,2):(1,3)">}
nv_tensor_ir.graph @reshape_simple(%in: tensor<2x3xf32> {nv_tensor_ir.stride = "(1,2)"}) -> (tensor<3x2xf32> {nv_tensor_ir.stride = "(1,3)"}) {
    %transposed = transpose %in permutation = [1, 0] : tensor<2x3xf32> -> tensor<3x2xf32>
    %reshaped = reshape %transposed : tensor<3x2xf32> -> tensor<2x3xf32>
    %out = transpose %reshaped permutation = [1, 0] : tensor<2x3xf32> -> tensor<3x2xf32>
    results %out : tensor<3x2xf32>
}

// -----

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

// Broadcast first dimension: (1,16) -> (8,16) = (8,16):(0,1)
// CHECK-LABEL: @broadcast_first_dim
// CHECK: broadcast %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(8,16):(0,1)">}
nv_tensor_ir.graph @broadcast_first_dim(%in: tensor<1x16xf32> {nv_tensor_ir.stride = "(0,1)"}) -> (tensor<8x16xf32> {nv_tensor_ir.stride = "(1,8)"}) {
    %out = broadcast %in : tensor<1x16xf32> -> tensor<8x16xf32>
    results %out : tensor<8x16xf32>
}

// -----

// Broadcast last dimension: (16,1) -> (16,8) = (16,8):(1,0)
// CHECK-LABEL: @broadcast_last_dim
// CHECK: broadcast %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(16,8):(1,0)">}
nv_tensor_ir.graph @broadcast_last_dim(%in: tensor<16x1xf32> {nv_tensor_ir.stride = "(1,0)"}) -> (tensor<16x8xf32> {nv_tensor_ir.stride = "(1,16)"}) {
    %out = broadcast %in : tensor<16x1xf32> -> tensor<16x8xf32>
    results %out : tensor<16x8xf32>
}

// -----

// Broadcast multiple dimensions: (1,1,16) -> (4,8,16) = (4,8,16):(0,0,1)
// CHECK-LABEL: @broadcast_multiple_dims
// CHECK: broadcast %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(4,8,16):(0,0,1)">}
nv_tensor_ir.graph @broadcast_multiple_dims(%in: tensor<1x1x16xf32> {nv_tensor_ir.stride = "(0,0,1)"}) -> (tensor<4x8x16xf32> {nv_tensor_ir.stride = "(1,4,32)"}) {
    %out = broadcast %in : tensor<1x1x16xf32> -> tensor<4x8x16xf32>
    results %out : tensor<4x8x16xf32>
}

// -----

// Broadcast middle dimension: (4,1,8) -> (4,16,8) = (4,16,8):(1,0,4)
// CHECK-LABEL: @broadcast_middle_dim
// CHECK: broadcast %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(4,16,8):(1,0,4)">}
nv_tensor_ir.graph @broadcast_middle_dim(%in: tensor<4x1x8xf32> {nv_tensor_ir.stride = "(1,0,4)"}) -> (tensor<4x16x8xf32> {nv_tensor_ir.stride = "(1,4,64)"}) {
    %out = broadcast %in : tensor<4x1x8xf32> -> tensor<4x16x8xf32>
    results %out : tensor<4x16x8xf32>
}

// -----

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

// 2D transpose: (4,16) -> (16,4) with permutation [1,0]
// CHECK-LABEL: @transpose_2d
// CHECK: transpose %{{.*}} permutation = [1, 0] {layout = #nv_tensor_ir.tensor_source<0, 0, "(16,4):(4,1)">}
nv_tensor_ir.graph @transpose_2d(%in: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) -> (tensor<16x4xf32> {nv_tensor_ir.stride = "(1,16)"}) {
    %out = transpose %in permutation = [1, 0] : tensor<4x16xf32> -> tensor<16x4xf32>
    results %out : tensor<16x4xf32>
}

// -----

// 3D transpose: (2,4,8) -> (8,2,4) with permutation [2,0,1]
// C++ test: TensorSource::create("(2,4,(2,8))"_lay)->transpose({2, 0, 1})
// CHECK-LABEL: @transpose_3d
// CHECK: transpose %{{.*}} permutation = [2, 0, 1] {layout = #nv_tensor_ir.tensor_source<0, 0, "(8,2,4):(8,1,2)">}
nv_tensor_ir.graph @transpose_3d(%in: tensor<2x4x8xf32> {nv_tensor_ir.stride = "(1,2,8)"}) -> (tensor<8x2x4xf32> {nv_tensor_ir.stride = "(1,8,16)"}) {
    %out = transpose %in permutation = [2, 0, 1] : tensor<2x4x8xf32> -> tensor<8x2x4xf32>
    results %out : tensor<8x2x4xf32>
}

// -----

// 3D transpose reverse: (2,4,8) -> (8,4,2) with permutation [2,1,0]
// CHECK-LABEL: @transpose_3d_reverse
// CHECK: transpose %{{.*}} permutation = [2, 1, 0] {layout = #nv_tensor_ir.tensor_source<0, 0, "(8,4,2):(8,2,1)">}
nv_tensor_ir.graph @transpose_3d_reverse(%in: tensor<2x4x8xf32> {nv_tensor_ir.stride = "(1,2,8)"}) -> (tensor<8x4x2xf32> {nv_tensor_ir.stride = "(1,8,32)"}) {
    %out = transpose %in permutation = [2, 1, 0] : tensor<2x4x8xf32> -> tensor<8x4x2xf32>
    results %out : tensor<8x4x2xf32>
}

// -----

// Identity transpose (permutation [0,1]) - layout unchanged
// CHECK-LABEL: @transpose_identity
// CHECK: transpose %{{.*}} permutation = [0, 1] {layout = #nv_tensor_ir.tensor_source<0, 0, "(4,16):(1,4)">}
nv_tensor_ir.graph @transpose_identity(%in: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) -> (tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) {
    %out = transpose %in permutation = [0, 1] : tensor<4x16xf32> -> tensor<4x16xf32>
    results %out : tensor<4x16xf32>
}

// -----

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

// Basic contiguous slice: (16) -> (4) from start
// CHECK-LABEL: @slice_contiguous
// CHECK: slice %{{.*}} starts = [0] limits = [4] strides = [1] {layout = #nv_tensor_ir.tensor_source<0, 0, "(4):(1)">}
nv_tensor_ir.graph @slice_contiguous(%in: tensor<16xf32>) -> (tensor<4xf32>) {
    %out = slice %in starts = [0] limits = [4] strides = [1] : tensor<16xf32> -> tensor<4xf32>
    results %out : tensor<4xf32>
}

// -----

// Slice with offset and stride: (16) -> (3) starting at 4, stride 2
// CHECK-LABEL: @slice_offset_and_stride
// CHECK: slice %{{.*}} starts = [4] limits = [10] strides = [2] {layout = #nv_tensor_ir.tensor_source<0, 4, "(3):(2)">}
nv_tensor_ir.graph @slice_offset_and_stride(%in: tensor<16xf32>) -> (tensor<3xf32>) {
    %out = slice %in starts = [4] limits = [10] strides = [2] : tensor<16xf32> -> tensor<3xf32>
    results %out : tensor<3xf32>
}

// -----

// Multi-dimensional slice with offset
// CHECK-LABEL: @slice_multidim_offset
// CHECK: slice %{{.*}} starts = [1, 2] limits = [9, 7] strides = [2, 1] {layout = #nv_tensor_ir.tensor_source<0, 33, "(4,5):(2,16)">}
nv_tensor_ir.graph @slice_multidim_offset(%in: tensor<16x8xf32> {nv_tensor_ir.stride = "(1,16)"}) -> (tensor<4x5xf32> {nv_tensor_ir.stride = "(1,4)"}) {
    %out = slice %in starts = [1, 2] limits = [9, 7] strides = [2, 1] : tensor<16x8xf32> -> tensor<4x5xf32>
    results %out : tensor<4x5xf32>
}

// -----

// Unit slice: result in single element (stride becomes 0)
// CHECK-LABEL: @slice_unit_result
// CHECK: slice %{{.*}} starts = [2] limits = [3] strides = [1] {layout = #nv_tensor_ir.tensor_source<0, 2, "(1):(0)">}
nv_tensor_ir.graph @slice_unit_result(%in: tensor<8xf32>) -> (tensor<1xf32> {nv_tensor_ir.stride = "(0)"}) {
    %out = slice %in starts = [2] limits = [3] strides = [1] : tensor<8xf32> -> tensor<1xf32>
    results %out : tensor<1xf32>
}

// -----

// 2D slice with strides in both dimensions
// CHECK-LABEL: @slice_2d_with_strides
// CHECK: slice %{{.*}} starts = [0, 0] limits = [8, 8] strides = [2, 2] {layout = #nv_tensor_ir.tensor_source<0, 0, "(4,4):(2,16)">}
nv_tensor_ir.graph @slice_2d_with_strides(%in: tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"}) -> (tensor<4x4xf32> {nv_tensor_ir.stride = "(1,4)"}) {
    %out = slice %in starts = [0, 0] limits = [8, 8] strides = [2, 2] : tensor<8x8xf32> -> tensor<4x4xf32>
    results %out : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @slice_reshape_slice
// CHECK: slice %{{.*}} starts = [8, 8] limits = [40, 40] strides = [1, 1] {layout = #nv_tensor_ir.tensor_source<0, 520, "(32,32):(1,64)">}
// CHECK: slice %{{.*}} starts = [32] limits = [544] strides = [2] {layout = #nv_tensor_ir.tensor_source<0, 584, "((16,16)):((64,2))">}
nv_tensor_ir.graph @slice_reshape_slice(%arg0: tensor<64x64xf32> {nv_tensor_ir.stride = "(1,64)"}) -> (tensor<256xf32>) {
    %slice1 = slice %arg0 starts = [8, 8] limits = [40, 40] strides = [1, 1] : tensor<64x64xf32> -> tensor<32x32xf32>
    %transposed = transpose %slice1 permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    %reshape = reshape %transposed : tensor<32x32xf32> -> tensor<1024xf32>
    %slice2 = slice %reshape starts = [32] limits = [544] strides = [2] : tensor<1024xf32> -> tensor<256xf32>
    results %slice2 : tensor<256xf32>
}

// -----

//===----------------------------------------------------------------------===//
// SplatOp / ConstantOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @splat_1d
// CHECK: splat %{{.*}} {layout = #nv_tensor_ir.tensor_source<-1, 0, "(16):(0)">}
nv_tensor_ir.graph @splat_1d(%val: f32) -> (tensor<16xf32>) {
    %out = splat %val : tensor<16xf32>
    results %out : tensor<16xf32>
}

// -----

// CHECK-LABEL: @splat_2d
// CHECK: splat %{{.*}} {layout = #nv_tensor_ir.tensor_source<-1, 0, "(4,16):(0,0)">}
nv_tensor_ir.graph @splat_2d(%val: f32) -> (tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) {
    %out = splat %val : tensor<4x16xf32>
    results %out : tensor<4x16xf32>
}

// -----

// CHECK-LABEL: @constant_op
// CHECK: constant {{.*}} {layout = #nv_tensor_ir.tensor_source<-1, 0, "(16):(0)">}
nv_tensor_ir.graph @constant_op() -> (tensor<16xf32>) {
    %out = constant dense<0.0> : tensor<16xf32>
    results %out : tensor<16xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Elementwise ops
//===----------------------------------------------------------------------===//

// Unary chain: layout flows through unchanged
// CHECK-LABEL: @unary_chain_preserves_layout
// CHECK: abs %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(4,16):(1,4)">}
// CHECK: neg %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(4,16):(1,4)">}
// CHECK: exp %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(4,16):(1,4)">}
nv_tensor_ir.graph @unary_chain_preserves_layout(%in: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) -> (tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) {
    %0 = abs %in : tensor<4x16xf32>
    %1 = neg %0 : tensor<4x16xf32>
    %2 = exp %1 : tensor<4x16xf32>
    results %2 : tensor<4x16xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Binary ops
//===----------------------------------------------------------------------===//

// Same input used twice - single TensorSourceAttr (not composite)
// CHECK-LABEL: @binary_same_input
// CHECK: add %in, %in {layout = #nv_tensor_ir.tensor_source<0, 0, "(4,16):(1,4)">}
nv_tensor_ir.graph @binary_same_input(%in: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) -> (tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) {
    %out = add %in, %in : tensor<4x16xf32>
    results %out : tensor<4x16xf32>
}

// -----

// Different inputs - creates CompositeSourceAttr
// CHECK-LABEL: @binary_different_inputs
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(4,16):(1,4)">, #nv_tensor_ir.tensor_source<1, 0, "(4,16):(1,4)">>}
nv_tensor_ir.graph @binary_different_inputs(%a: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"},
                                             %b: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) -> (tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) {
    %out = add %a, %b : tensor<4x16xf32>
    results %out : tensor<4x16xf32>
}


// -----

// Transpose + original creates composite with different strides (same tensor_id)
// CHECK-LABEL: @transpose_and_original_composite
// CHECK: transpose %{{.*}} permutation = [1, 0] {layout = #nv_tensor_ir.tensor_source<0, 0, "(8,8):(8,1)">}
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8,8):(1,8)">, #nv_tensor_ir.tensor_source<0, 0, "(8,8):(8,1)">>}
nv_tensor_ir.graph @transpose_and_original_composite(%in: tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"}) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"}) {
    %transposed = transpose %in permutation = [1, 0] : tensor<8x8xf32> -> tensor<8x8xf32>
    %out = add %in, %transposed : tensor<8x8xf32>
    results %out : tensor<8x8xf32>
}

// -----

// Different slices of same input - composite with different offsets
// CHECK-LABEL: @slice_creates_composite_offsets
// CHECK: slice %{{.*}} starts = [0] limits = [8] strides = [1] {layout = #nv_tensor_ir.tensor_source<0, 0, "(8):(1)">}
// CHECK: slice %{{.*}} starts = [8] limits = [16] strides = [1] {layout = #nv_tensor_ir.tensor_source<0, 8, "(8):(1)">}
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8):(1)">, #nv_tensor_ir.tensor_source<0, 8, "(8):(1)">>}
nv_tensor_ir.graph @slice_creates_composite_offsets(%in: tensor<16xf32>) -> (tensor<8xf32>) {
    %first = slice %in starts = [0] limits = [8] strides = [1] : tensor<16xf32> -> tensor<8xf32>
    %second = slice %in starts = [8] limits = [16] strides = [1] : tensor<16xf32> -> tensor<8xf32>
    %out = add %first, %second : tensor<8xf32>
    results %out : tensor<8xf32>
}

// -----

// Splat combined with regular input
// CHECK-LABEL: @splat_with_input
// CHECK: splat %{{.*}} {layout = #nv_tensor_ir.tensor_source<-1, 0, "(4,16):(0,0)">}
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(4,16):(1,4)">, #nv_tensor_ir.tensor_source<-1, 0, "(4,16):(0,0)">>}
nv_tensor_ir.graph @splat_with_input(%in: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}, %val: f32) -> (tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) {
    %splatted = splat %val : tensor<4x16xf32>
    %out = add %in, %splatted : tensor<4x16xf32>
    results %out : tensor<4x16xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Composite source propagation through transformations
//===----------------------------------------------------------------------===//

// Composite through reshape - all sources reshaped
// CHECK-LABEL: @composite_through_reshape
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<
// CHECK: reshape %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(64):(1)">, #nv_tensor_ir.tensor_source<1, 0, "(64):(1)">>}
nv_tensor_ir.graph @composite_through_reshape(%a: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"},
                                               %b: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) -> (tensor<64xf32>) {
    %sum = add %a, %b : tensor<4x16xf32>
    %transposed = transpose %sum permutation = [1, 0] : tensor<4x16xf32> -> tensor<16x4xf32>
    %out = reshape %transposed : tensor<16x4xf32> -> tensor<64xf32>
    results %out : tensor<64xf32>
}

// -----

// Composite through transpose
// CHECK-LABEL: @composite_through_transpose
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<
// CHECK: transpose %{{.*}} permutation = [1, 0] {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(16,4):(4,1)">, #nv_tensor_ir.tensor_source<1, 0, "(16,4):(4,1)">>}
nv_tensor_ir.graph @composite_through_transpose(%a: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"},
                                                 %b: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) -> (tensor<16x4xf32> {nv_tensor_ir.stride = "(1,16)"}) {
    %sum = add %a, %b : tensor<4x16xf32>
    %out = transpose %sum permutation = [1, 0] : tensor<4x16xf32> -> tensor<16x4xf32>
    results %out : tensor<16x4xf32>
}

// -----

// Composite through broadcast
// CHECK-LABEL: @composite_through_broadcast
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<
// CHECK: broadcast %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8,16):(0,1)">, #nv_tensor_ir.tensor_source<1, 0, "(8,16):(0,1)">>}
nv_tensor_ir.graph @composite_through_broadcast(%a: tensor<1x16xf32> {nv_tensor_ir.stride = "(0,1)"},
                                                 %b: tensor<1x16xf32> {nv_tensor_ir.stride = "(0,1)"}) -> (tensor<8x16xf32> {nv_tensor_ir.stride = "(1,8)"}) {
    %sum = add %a, %b : tensor<1x16xf32>
    %out = broadcast %sum : tensor<1x16xf32> -> tensor<8x16xf32>
    results %out : tensor<8x16xf32>
}

// -----

// Composite through slice
// C++ test: TestComposite()->slice(...)
// CHECK-LABEL: @composite_through_slice
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<
// CHECK: slice %{{.*}} starts = [4] limits = [10] strides = [2] {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 4, "(3):(2)">, #nv_tensor_ir.tensor_source<1, 4, "(3):(2)">>}
nv_tensor_ir.graph @composite_through_slice(%a: tensor<16xf32>,
                                             %b: tensor<16xf32>) -> (tensor<3xf32>) {
    %sum = add %a, %b : tensor<16xf32>
    %out = slice %sum starts = [4] limits = [10] strides = [2] : tensor<16xf32> -> tensor<3xf32>
    results %out : tensor<3xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Multi-operation transformation chains
//===----------------------------------------------------------------------===//

// Chain: reshape -> transpose -> slice
// CHECK-LABEL: @chain_reshape_transpose_slice
// CHECK: reshape %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(8,8):(8,1)">}
// CHECK: slice %{{.*}} starts = [0, 0] limits = [8, 8] strides = [2, 2] {layout = #nv_tensor_ir.tensor_source<0, 0, "(4,4):(16,2)">}
nv_tensor_ir.graph @chain_reshape_transpose_slice(%in: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) -> (tensor<4x4xf32> {nv_tensor_ir.stride = "(1,4)"}) {
    %transposed = transpose %in permutation = [1, 0] : tensor<4x16xf32> -> tensor<16x4xf32>
    %1 = reshape %transposed : tensor<16x4xf32> -> tensor<8x8xf32>
    %2 = slice %1 starts = [0, 0] limits = [8, 8] strides = [2, 2] : tensor<8x8xf32> -> tensor<4x4xf32>
    results %2 : tensor<4x4xf32>
}

// -----

// Chain: multiple reshapes
// CHECK-LABEL: @chain_multiple_reshapes
// CHECK: reshape %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(8,8):(8,1)">}
// CHECK: reshape %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(64):(1)">}
// CHECK: transpose %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(2,32):(1,2)">}
nv_tensor_ir.graph @chain_multiple_reshapes(%in: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) -> (tensor<2x32xf32> {nv_tensor_ir.stride = "(1,2)"}) {
    %transposed = transpose %in permutation = [1, 0] : tensor<4x16xf32> -> tensor<16x4xf32>
    %0 = reshape %transposed : tensor<16x4xf32> -> tensor<8x8xf32>
    %1 = reshape %0 : tensor<8x8xf32> -> tensor<64xf32>
    %reshape2 = reshape %1 : tensor<64xf32> -> tensor<32x2xf32>
    %2 = transpose %reshape2 permutation = [1, 0] : tensor<32x2xf32> -> tensor<2x32xf32>
    results %2 : tensor<2x32xf32>
}

// -----

// Composite through multiple transformations
// CHECK-LABEL: @composite_through_chain
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<
// CHECK: transpose %{{.*}} {layout = #nv_tensor_ir.composite_source<
// CHECK: reshape %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8,8):(8,1)">, #nv_tensor_ir.tensor_source<1, 0, "(8,8):(8,1)">>}
nv_tensor_ir.graph @composite_through_chain(%a: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"},
                                             %b: tensor<4x16xf32> {nv_tensor_ir.stride = "(1,4)"}) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"}) {
    %sum = add %a, %b : tensor<4x16xf32>
    %transposed = transpose %sum permutation = [1, 0] : tensor<4x16xf32> -> tensor<16x4xf32>
    %out = reshape %transposed : tensor<16x4xf32> -> tensor<8x8xf32>
    results %out : tensor<8x8xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Diamond patterns
//===----------------------------------------------------------------------===//

// Double transpose back to original - same layout so single TensorSourceAttr
// CHECK-LABEL: @diamond_double_transpose
// CHECK: transpose %{{.*}} permutation = [1, 0] {layout = #nv_tensor_ir.tensor_source<0, 0, "(16,8):(8,1)">}
// CHECK: transpose %{{.*}} permutation = [1, 0] {layout = #nv_tensor_ir.tensor_source<0, 0, "(8,16):(1,8)">}
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(8,16):(1,8)">}
nv_tensor_ir.graph @diamond_double_transpose(%in: tensor<8x16xf32> {nv_tensor_ir.stride = "(1,8)"}) -> (tensor<8x16xf32> {nv_tensor_ir.stride = "(1,8)"}) {
    %transposed = transpose %in permutation = [1, 0] : tensor<8x16xf32> -> tensor<16x8xf32>
    %back = transpose %transposed permutation = [1, 0] : tensor<16x8xf32> -> tensor<8x16xf32>
    %out = add %in, %back : tensor<8x16xf32>
    results %out : tensor<8x16xf32>
}

// -----

// Two paths with same reshape - same layout so single TensorSourceAttr
// CHECK-LABEL: @diamond_two_reshape_paths
// CHECK: reshape %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(64):(1)">}
// CHECK: reshape %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(64):(1)">}
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(64):(1)">}
nv_tensor_ir.graph @diamond_two_reshape_paths(%in: tensor<8x8xf32> {nv_tensor_ir.stride = "(1,8)"}) -> (tensor<64xf32>) {
    %transposed = transpose %in permutation = [1, 0] : tensor<8x8xf32> -> tensor<8x8xf32>
    %path1 = reshape %transposed : tensor<8x8xf32> -> tensor<64xf32>
    %path2 = reshape %transposed : tensor<8x8xf32> -> tensor<64xf32>
    %out = add %path1, %path2 : tensor<64xf32>
    results %out : tensor<64xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Multiple inputs tracking
//===----------------------------------------------------------------------===//

// Three-input chain - composite grows
// CHECK-LABEL: @three_input_chain
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(16):(1)">, #nv_tensor_ir.tensor_source<1, 0, "(16):(1)">>}
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(16):(1)">, #nv_tensor_ir.tensor_source<1, 0, "(16):(1)">, #nv_tensor_ir.tensor_source<2, 0, "(16):(1)">>}
nv_tensor_ir.graph @three_input_chain(%a: tensor<16xf32>,
                                       %b: tensor<16xf32>,
                                       %c: tensor<16xf32>) -> (tensor<16xf32>) {
    %ab = add %a, %b : tensor<16xf32>
    %out = add %ab, %c : tensor<16xf32>
    results %out : tensor<16xf32>
}

// -----

// Four-input chain - all sources tracked
// CHECK-LABEL: @four_input_chain
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(16):(1)">, #nv_tensor_ir.tensor_source<1, 0, "(16):(1)">>}
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<2, 0, "(16):(1)">, #nv_tensor_ir.tensor_source<3, 0, "(16):(1)">>}
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(16):(1)">, #nv_tensor_ir.tensor_source<1, 0, "(16):(1)">, #nv_tensor_ir.tensor_source<2, 0, "(16):(1)">, #nv_tensor_ir.tensor_source<3, 0, "(16):(1)">>}
nv_tensor_ir.graph @four_input_chain(%a: tensor<16xf32>,
                                      %b: tensor<16xf32>,
                                      %c: tensor<16xf32>,
                                      %d: tensor<16xf32>) -> (tensor<16xf32>) {
    %ab = add %a, %b : tensor<16xf32>
    %cd = add %c, %d : tensor<16xf32>
    %out = add %ab, %cd : tensor<16xf32>
    results %out : tensor<16xf32>
}

// -----

// Transformed inputs combined - different transformations tracked
// CHECK-LABEL: @transformed_inputs_combined
// CHECK: transpose %{{.*}} permutation = [1, 0] {layout = #nv_tensor_ir.tensor_source<0, 0, "(8,4):(4,1)">}
// CHECK: transpose %{{.*}} {layout = #nv_tensor_ir.tensor_source<1, 0, "(8,4):(1,8)">}
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(8,4):(4,1)">, #nv_tensor_ir.tensor_source<1, 0, "(8,4):(1,8)">>}
nv_tensor_ir.graph @transformed_inputs_combined(%a: tensor<4x8xf32> {nv_tensor_ir.stride = "(1,4)"},
                                                  %b: tensor<32xf32>) -> (tensor<8x4xf32> {nv_tensor_ir.stride = "(1,8)"}) {
    %a_t = transpose %a permutation = [1, 0] : tensor<4x8xf32> -> tensor<8x4xf32>
    %b_reshape = reshape %b : tensor<32xf32> -> tensor<4x8xf32>
    %b_r = transpose %b_reshape permutation = [1, 0] : tensor<4x8xf32> -> tensor<8x4xf32>
    %out = add %a_t, %b_r : tensor<8x4xf32>
    results %out : tensor<8x4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// IotaOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @iota_1d
// CHECK: iota dimension = 0 {layout = #nv_tensor_ir.tensor_source<-1, 0, "(128):(1)">}
nv_tensor_ir.graph @iota_1d() -> tensor<128xf32> {
    %out = iota dimension = 0 : tensor<128xf32>
    results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: @iota_2d_dim0
// CHECK: iota_0{{.*}} = iota dimension = 0 {layout = #nv_tensor_ir.tensor_source<-1, 0, "(4,16):(1,0)">}
nv_tensor_ir.graph @iota_2d_dim0() -> tensor<4x16xf32> {
    %out = iota dimension = 0 : tensor<4x16xf32>
    results %out : tensor<4x16xf32>
}

// -----

// CHECK-LABEL: @iota_2d_dim1
// CHECK: iota_1{{.*}} = iota dimension = 1 {layout = #nv_tensor_ir.tensor_source<-1, 0, "(4,16):(0,1)">}
nv_tensor_ir.graph @iota_2d_dim1() -> tensor<4x16xf32> {
    %out = iota dimension = 1 : tensor<4x16xf32>
    results %out : tensor<4x16xf32>
}

// -----

// CHECK-LABEL: @iota_with_input
// CHECK: iota dimension = 0 {layout = #nv_tensor_ir.tensor_source<-1, 0, "(4,16):(1,0)">}
// CHECK: add %{{.*}}, %{{.*}} {layout = #nv_tensor_ir.composite_source<#nv_tensor_ir.tensor_source<0, 0, "(4,16):(16,1)">, #nv_tensor_ir.tensor_source<-1, 0, "(4,16):(1,0)">>}
nv_tensor_ir.graph @iota_with_input(%in: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %idx = iota dimension = 0 : tensor<4x16xf32>
    %out = add %in, %idx : tensor<4x16xf32>
    results %out : tensor<4x16xf32>
}

// -----

// iota followed by reshape, broadcast, and transpose.
// CHECK-LABEL: @iota_layout_modifying_ops
// CHECK: iota dimension = 0 {layout = #nv_tensor_ir.tensor_source<-1, 0, "(128):(1)">}
// CHECK: transpose %{{.*}} {layout = #nv_tensor_ir.tensor_source<-1, 0, "(16,1,8):(1,0,16)">}
// CHECK: broadcast %{{.*}} {layout = #nv_tensor_ir.tensor_source<-1, 0, "(16,4,8):(1,0,16)">}
// CHECK: transpose %{{.*}} {layout = #nv_tensor_ir.tensor_source<-1, 0, "(8,4,16):(16,0,1)">}
nv_tensor_ir.graph @iota_layout_modifying_ops() -> tensor<8x4x16xf32> {
    %iota_0 = iota dimension = 0 : tensor<128xf32>
    %reshaped = reshape %iota_0 : tensor<128xf32> -> tensor<8x1x16xf32>
    %0 = transpose %reshaped permutation = [2, 1, 0] : tensor<8x1x16xf32> -> tensor<16x1x8xf32>
    %broadcast = broadcast %0 : tensor<16x1x8xf32> -> tensor<16x4x8xf32>
    %1 = transpose %broadcast permutation = [2, 1, 0] : tensor<16x4x8xf32> -> tensor<8x4x16xf32>
    results %1 : tensor<8x4x16xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Dynamic shapes
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @dynamic_shape_default_stride
// CHECK: abs %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(?,32):(32,1)", [0]>}
nv_tensor_ir.graph @dynamic_shape_default_stride(
    %in: tensor<?x32xf32>
) -> tensor<?x32xf32> {
    %out = abs %in : tensor<?x32xf32>
    results %out : tensor<?x32xf32>
}

// -----

// CHECK-LABEL: @dynamic_shape_explicit_stride
// CHECK: abs %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(32,?):(1,32)", [0]>}
nv_tensor_ir.graph @dynamic_shape_explicit_stride(
    %in: tensor<32x?xf32> {nv_tensor_ir.stride = "(1,32)"}
) -> tensor<32x?xf32> {
    %out = abs %in : tensor<32x?xf32>
    results %out : tensor<32x?xf32>
}

// -----

// CHECK-LABEL: @dynamic_strides
// CHECK: abs %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(16,32):(?,?)", [0, 1]>}
nv_tensor_ir.graph @dynamic_strides(
    %in: tensor<16x32xf32> {nv_tensor_ir.stride = "(?,?)"}
) -> tensor<16x32xf32> {
    %out = abs %in : tensor<16x32xf32>
    results %out : tensor<16x32xf32>
}

// -----

// CHECK-LABEL: @dynamic_shape_and_stride
// CHECK: abs %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(?,32):(1,?)", [0, 1]>}
nv_tensor_ir.graph @dynamic_shape_and_stride(
    %in: tensor<?x32xf32> {nv_tensor_ir.stride = "(1,?)"}
) -> tensor<?x32xf32> {
    %out = abs %in : tensor<?x32xf32>
    results %out : tensor<?x32xf32>
}

// -----

// CHECK-LABEL: @reshape_dynamic_with_static_split
// CHECK: transpose %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(8,8,?):(1,8,?)", [0, 1]>}
nv_tensor_ir.graph @reshape_dynamic_with_static_split(
    %in: tensor<64x?xf32> {nv_tensor_ir.stride = "(1,?)"}
) -> tensor<8x8x?xf32> {
    %transposed = transpose %in permutation = [1, 0] : tensor<64x?xf32> -> tensor<?x64xf32>
    %reshaped = reshape %transposed : tensor<?x64xf32> -> tensor<?x8x8xf32>
    %out = transpose %reshaped permutation = [2, 1, 0] : tensor<?x8x8xf32> -> tensor<8x8x?xf32>
    results %out : tensor<8x8x?xf32>
}

// -----

// CHECK-LABEL: @reshape_dynamic_with_static_join
// CHECK: transpose %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(64,?):(1,?)", [0, 1]>}
nv_tensor_ir.graph @reshape_dynamic_with_static_join(
    %in: tensor<8x8x?xf32> {nv_tensor_ir.stride = "(1,8,?)"}
) -> tensor<64x?xf32> {
    %transposed = transpose %in permutation = [2, 1, 0] : tensor<8x8x?xf32> -> tensor<?x8x8xf32>
    %reshaped = reshape %transposed : tensor<?x8x8xf32> -> tensor<?x64xf32>
    %out = transpose %reshaped permutation = [1, 0] : tensor<?x64xf32> -> tensor<64x?xf32>
    results %out : tensor<64x?xf32>
}

// -----

// CHECK-LABEL: @transpose_dynamic_shape_only
// CHECK: transpose %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(?,?):(32,1)", [1, 0]>}
nv_tensor_ir.graph @transpose_dynamic_shape_only(
    %in: tensor<?x?xf32> {nv_tensor_ir.stride = "(1,32)"}
) -> tensor<?x?xf32> {
    %out = transpose %in permutation = [1, 0] : tensor<?x?xf32> -> tensor<?x?xf32>
    results %out : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @transpose_dynamic_stride_only
// CHECK: transpose %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(64,32):(?,?)", [1, 0]>}
nv_tensor_ir.graph @transpose_dynamic_stride_only(
    %in: tensor<32x64xf32> {nv_tensor_ir.stride = "(?,?)"}
) -> tensor<64x32xf32> {
    %out = transpose %in permutation = [1, 0] : tensor<32x64xf32> -> tensor<64x32xf32>
    results %out : tensor<64x32xf32>
}

// -----

// CHECK-LABEL: @transpose_dynamic_shape_and_strides
// CHECK: transpose %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(?,?):(?,?)", [1, 0, 3, 2]>}
nv_tensor_ir.graph @transpose_dynamic_shape_and_strides(
    %in: tensor<?x?xf32> {nv_tensor_ir.stride = "(?,?)"}
) -> tensor<?x?xf32> {
    %out = transpose %in permutation = [1, 0] : tensor<?x?xf32> -> tensor<?x?xf32>
    results %out : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @transpose_dynamic_and_static_mixed
// CHECK: transpose %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "(?,8,?,4,?,2):(?,8,?,2,?,1)", [2, 1, 0, 5, 4, 3]>}
nv_tensor_ir.graph @transpose_dynamic_and_static_mixed(
    %in: tensor<2x?x4x?x8x?xf32> {nv_tensor_ir.stride = "(1,?,2,?,8,?)"}
) -> tensor<?x8x?x4x?x2xf32> {
    %out = transpose %in permutation = [5, 4, 3, 2, 1, 0] : tensor<2x?x4x?x8x?xf32> -> tensor<?x8x?x4x?x2xf32>
    results %out : tensor<?x8x?x4x?x2xf32>
}
