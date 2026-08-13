// RUN: tensor_ir-opt -materialize-default-strides -layout-propagation-annotation -split-input-file %s | FileCheck %s

// CHECK-LABEL: @layout_unary_elementwise
// CHECK: abs %
// CHECK-SAME: layout = #nv_tensor_ir.tensor_source<0, 0, "(4,16):(16,1)">
nv_tensor_ir.graph @layout_unary_elementwise(%arg0: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %result = abs %arg0 : tensor<4x16xf32>
    results %result : tensor<4x16xf32>
}

// -----

// CHECK-LABEL: @layout_binary_elementwise
// CHECK: add %
// CHECK-SAME: layout = #nv_tensor_ir.composite_source
// CHECK-SAME: #nv_tensor_ir.tensor_source<0, 0, "(4,16):(16,1)">
// CHECK-SAME: #nv_tensor_ir.tensor_source<1, 0, "(4,16):(16,1)">
nv_tensor_ir.graph @layout_binary_elementwise(%arg0: tensor<4x16xf32>,
                                              %arg1: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %result = add %arg0, %arg1 : tensor<4x16xf32>
    results %result : tensor<4x16xf32>
}

// -----

// CHECK-LABEL: @layout_reshape
// CHECK: transpose %{{.*}} {layout = #nv_tensor_ir.tensor_source<0, 0, "((4,4),4):((1,16),4)">}
nv_tensor_ir.graph @layout_reshape(%arg0: tensor<4x16xf32>) -> tensor<16x4xf32> {
    %transposed = transpose %arg0 permutation = [1, 0] : tensor<4x16xf32> -> tensor<16x4xf32>
    %reshaped = reshape %transposed : tensor<16x4xf32> -> tensor<4x16xf32>
    %result = transpose %reshaped permutation = [1, 0] : tensor<4x16xf32> -> tensor<16x4xf32>
    results %result : tensor<16x4xf32>
}

// -----

// CHECK-LABEL: @layout_broadcast
// CHECK: broadcast %
// CHECK-SAME: layout = #nv_tensor_ir.tensor_source<0, 0, "(8,16):(0,1)">
nv_tensor_ir.graph @layout_broadcast(%arg0: tensor<1x16xf32>) -> tensor<8x16xf32> {
    %result = broadcast %arg0 : tensor<1x16xf32> -> tensor<8x16xf32>
    results %result : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: @layout_transpose
// CHECK: transpose %
// CHECK-SAME: layout = #nv_tensor_ir.tensor_source<0, 0, "(16,4):(1,16)">
nv_tensor_ir.graph @layout_transpose(%arg0: tensor<4x16xf32>) -> tensor<16x4xf32> {
    %result = transpose %arg0 permutation = [1, 0] : tensor<4x16xf32> -> tensor<16x4xf32>
    results %result : tensor<16x4xf32>
}

// -----

// CHECK-LABEL: @layout_slice
// CHECK: slice %
// CHECK-SAME: layout = #nv_tensor_ir.tensor_source<0, 8, "(10):(4)">
nv_tensor_ir.graph @layout_slice(%arg0: tensor<64xf32>) -> tensor<10xf32> {
    %result = slice %arg0 starts = [8] limits = [48] strides = [4] : tensor<64xf32> -> tensor<10xf32>
    results %result : tensor<10xf32>
}

// -----

// CHECK-LABEL: @layout_transpose_add
// CHECK: add %
// CHECK-SAME: layout = #nv_tensor_ir.composite_source
// CHECK-SAME: #nv_tensor_ir.tensor_source<0, 0, "(8,8):(8,1)">
// CHECK-SAME: #nv_tensor_ir.tensor_source<0, 0, "(8,8):(1,8)">
nv_tensor_ir.graph @layout_transpose_add(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %arg0_trans = transpose %arg0 permutation = [1, 0] : tensor<8x8xf32> -> tensor<8x8xf32>
    %result = add %arg0, %arg0_trans : tensor<8x8xf32>
    results %result : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: @layout_a_plus_transpose_b_row_major
// CHECK: add %
// CHECK-SAME: layout = #nv_tensor_ir.composite_source
// CHECK-SAME: #nv_tensor_ir.tensor_source<0, 0, "(64,32):(32,1)">
// CHECK-SAME: #nv_tensor_ir.tensor_source<1, 0, "(64,32):(1,64)">
nv_tensor_ir.graph @layout_a_plus_transpose_b_row_major(
    %arg0: tensor<64x32xf32> {nv_tensor_ir.stride = "(32,1)"},
    %arg1: tensor<32x64xf32> {nv_tensor_ir.stride = "(64,1)"}) ->
    (tensor<64x32xf32> {nv_tensor_ir.stride = "(32,1)"}) {
  %t = transpose %arg1 permutation = [1, 0] : tensor<32x64xf32> -> tensor<64x32xf32>
  %add = add %arg0, %t : tensor<64x32xf32>
  results %add : tensor<64x32xf32>
}
