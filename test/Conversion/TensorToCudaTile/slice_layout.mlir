// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// CHECK-LABEL: entry @slice_reshape_row_major_output
// CHECK-SAME: (%[[ARG0:.+]]: tile<ptr<f32>>, %[[OUT:.+]]: tile<ptr<f32>>)
nv_tensor_ir.graph @slice_reshape_row_major_output(
    %arg0: tensor<1024xf32>)
    -> (tensor<16x16xf32>)
    attributes {tile_size = array<i32: 2, 16>} {
  // CHECK: %[[C768:.*]] = constant <i64: 768> : tile<i64>
  // CHECK: %[[PTR:.+]] = offset %[[ARG0]], %[[C768]]
  // CHECK: make_tensor_view %[[PTR]], shape = [16, 16], strides = [1, 16]
  // CHECK: %[[TILE:.+]], %{{.*}} = load_view_tko
  %slice = slice %arg0 starts = [768] limits = [1024] strides = [1] : tensor<1024xf32> -> tensor<256xf32>
  %reshape_reversed = reshape %slice : tensor<256xf32> -> tensor<16x16xf32>
  %reshape = transpose %reshape_reversed permutation = [1, 0] : tensor<16x16xf32> -> tensor<16x16xf32>
  // CHECK: make_tensor_view %[[OUT]], shape = [16, 16], strides = [16, 1]
  // CHECK: store_view_tko weak %[[TILE]]
  results %reshape : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: entry @slice_reshape_col_major_output
// CHECK-SAME: (%[[ARG0:.+]]: tile<ptr<f32>>, %[[OUT:.+]]: tile<ptr<f32>>)
nv_tensor_ir.graph @slice_reshape_col_major_output(
    %arg0: tensor<1024xf32>)
    -> (tensor<16x16xf32> {nv_tensor_ir.stride = "(1,16)"})
    attributes {tile_size = array<i32: 16, 2>} {
  // CHECK: %[[C768:.*]] = constant <i64: 768> : tile<i64>
  // CHECK: %[[PTR:.+]] = offset %[[ARG0]], %[[C768]]
  // CHECK: make_tensor_view %[[PTR]], shape = [16, 16], strides = [1, 16]
  // CHECK: %[[TILE:.+]], %{{.*}} = load_view_tko
  %slice = slice %arg0 starts = [768] limits = [1024] strides = [1] : tensor<1024xf32> -> tensor<256xf32>
  %reshape_reversed = reshape %slice : tensor<256xf32> -> tensor<16x16xf32>
  %reshape = transpose %reshape_reversed permutation = [1, 0] : tensor<16x16xf32> -> tensor<16x16xf32>
  // CHECK: make_tensor_view %[[OUT]], shape = [16, 16], strides = [1, 16]
  // CHECK: store_view_tko weak %[[TILE]]
  results %reshape : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: entry @reshape_slice
// CHECK-SAME: (%[[ARG0:.+]]: tile<ptr<f32>>, %[[OUT:.+]]: tile<ptr<f32>>)
// CHECK: %[[PTR:.+]] = offset %[[ARG0]], %cst_528_i64
// CHECK: make_tensor_view %[[PTR]], shape = [16, 16], strides = [1, 32]
// CHECK: %[[TILE:.+]], %{{.*}} = load_view_tko
// CHECK: store_view_tko weak %[[TILE]]

module {
  nv_tensor_ir.graph @reshape_slice(
      %arg0: tensor<1024xf32>)
      -> (tensor<16x16xf32>)
      attributes {tile_size = array<i32: 16, 4>} {
    %reshape_reversed = reshape %arg0 : tensor<1024xf32> -> tensor<32x32xf32>
    %reshape = transpose %reshape_reversed permutation = [1, 0] : tensor<32x32xf32> -> tensor<32x32xf32>
    %slice = slice %reshape starts = [16, 16] limits = [32, 32] strides = [1, 1] : tensor<32x32xf32> -> tensor<16x16xf32>
    results %slice : tensor<16x16xf32>
  }
}

// -----

// CHECK-LABEL: entry @slice_reshape_slice
// CHECK-SAME: (%[[ARG0:.+]]: tile<ptr<f32>>, %[[OUT:.+]]: tile<ptr<f32>>)
// CHECK: make_tensor_view %[[ARG0]], shape = [8, 8], strides = [1, 16]
// CHECK: %[[TILE:.+]], %{{.*}} = load_view_tko
// CHECK: store_view_tko weak %[[TILE]]

module {
  nv_tensor_ir.graph @slice_reshape_slice(
      %arg0: tensor<1024xf32>)
      -> (tensor<8x8xf32>)
      attributes {tile_size = array<i32: 8, 8>} {
    %slice1 = slice %arg0 starts = [0] limits = [256] strides = [1] : tensor<1024xf32> -> tensor<256xf32>
    %reshape_reversed = reshape %slice1 : tensor<256xf32> -> tensor<16x16xf32>
    %reshape = transpose %reshape_reversed permutation = [1, 0] : tensor<16x16xf32> -> tensor<16x16xf32>
    %slice2 = slice %reshape starts = [0, 0] limits = [8, 8] strides = [1, 1] : tensor<16x16xf32> -> tensor<8x8xf32>
    results %slice2 : tensor<8x8xf32>
  }
}

// -----

// CHECK-LABEL: entry @slice_transpose_slice
// CHECK-SAME: (%[[ARG0:.+]]: tile<ptr<f32>>, %[[OUT:.+]]: tile<ptr<f32>>)
// CHECK: %[[PTR:.+]] = offset %[[ARG0]], %cst_256_i64
// CHECK: make_tensor_view %[[PTR]], shape = [8, 8], strides = [1, 32]
// CHECK: %[[TILE:.+]], %{{.*}} = load_view_tko
// CHECK: store_view_tko weak %[[TILE]]

module {
  nv_tensor_ir.graph @slice_transpose_slice(
      %arg0: tensor<32x32xf32>)
      -> (tensor<8x8xf32>)
      attributes {tile_size = array<i32: 8, 8>} {
    %slice1 = slice %arg0 starts = [0, 0] limits = [16, 16] strides = [1, 1] : tensor<32x32xf32> -> tensor<16x16xf32>
    %trans = transpose %slice1 permutation = [1, 0] : tensor<16x16xf32> -> tensor<16x16xf32>
    %slice2 = slice %trans starts = [0, 8] limits = [8, 16] strides = [1, 1] : tensor<16x16xf32> -> tensor<8x8xf32>
    results %slice2 : tensor<8x8xf32>
  }
}

// -----

// CHECK-LABEL: entry @pw_slice_pw_transpose_reshape_pw_slice
// CHECK-SAME: (%[[ARG0:.+]]: tile<ptr<f32>>, %[[OUT:.+]]: tile<ptr<f32>>)
// CHECK: make_tensor_view %[[ARG0]], shape = [8, 8], strides = [32, 2]
// CHECK: %[[TILE:.+]], %{{.*}} = load_view_tko
// CHECK: %[[ADD:.+]] = addf %[[TILE]], %[[TILE]]
// CHECK: %[[MUL:.+]] = mulf %[[ADD]], %[[ADD]]
// CHECK: %[[ABS:.+]] = absf %[[MUL]]
// CHECK: store_view_tko weak %[[ABS]]

module {
  nv_tensor_ir.graph @pw_slice_pw_transpose_reshape_pw_slice(
      %arg0: tensor<32x32xf32>)
      -> (tensor<64xf32>)
      attributes {tile_size = array<i32: 8, 8>} {
    %add = add %arg0, %arg0 : tensor<32x32xf32>
    %slice1 = slice %add starts = [0, 0] limits = [16, 16] strides = [1, 1] : tensor<32x32xf32> -> tensor<16x16xf32>
    %mul = mul %slice1, %slice1 : tensor<16x16xf32>
    %reshape = reshape %mul : tensor<16x16xf32> -> tensor<256xf32>
    %abs = abs %reshape : tensor<256xf32>
    %slice2 = slice %abs starts = [0] limits = [128] strides = [2] : tensor<256xf32> -> tensor<64xf32>
    results %slice2 : tensor<64xf32>
  }
}
