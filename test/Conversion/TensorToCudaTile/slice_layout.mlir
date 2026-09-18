// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// CHECK-LABEL: entry @slice_reshape_row_major_output
// CHECK-SAME: (%[[ARG0:.+]]: tile<ptr<f32>>, %[[OUT:.+]]: tile<ptr<f32>>)
nv_tensor_ir.graph @slice_reshape_row_major_output(
    %arg0: tensor<1024xf32>)
    -> (tensor<16x16xf32>)
    attributes {tile_size = array<i32: 2, 16>} {
  // Slice offsets use the pipeline's i32 index type.
  // CHECK: %[[C768:.+]] = constant <i32: 768> : tile<i32>
  // CHECK: %[[C8:.+]] = constant <i32: 8> : tile<i32>
  // CHECK: %[[BID_X:[^,]+]], {{.*}} = get_tile_block_id : tile<i32>
  // CHECK: %[[COORD0:.+]] = remi %[[BID_X]], %[[C8]] unsigned : tile<i32>
  // CHECK: %[[COORD1:.+]] = divi %[[BID_X]], %[[C8]] unsigned : tile<i32>
  // CHECK: %[[PTR:.+]] = offset %[[ARG0]], %[[C768]] : tile<ptr<f32>>, tile<i32> -> tile<ptr<f32>>
  // CHECK: %[[VIN:.+]] = make_tensor_view %[[PTR]], shape = [16, 16], strides = [1, 16] : tensor_view<16x16xf32, strides=[1,16]>
  // CHECK: %[[PIN:.+]] = make_partition_view %[[VIN]] : partition_view<tile=(2x16), tensor_view<16x16xf32, strides=[1,16]>>
  // CHECK: %[[TILE:.+]], %{{.*}} = load_view_tko weak %[[PIN]][%[[COORD0]], %[[COORD1]]] : partition_view<tile=(2x16), tensor_view<16x16xf32, strides=[1,16]>>, tile<i32> -> tile<2x16xf32>, token
  %slice = slice %arg0 starts = [768] limits = [1024] strides = [1] : tensor<1024xf32> -> tensor<256xf32>
  %reshape_reversed = reshape %slice : tensor<256xf32> -> tensor<16x16xf32>
  %reshape = transpose %reshape_reversed permutation = [1, 0] : tensor<16x16xf32> -> tensor<16x16xf32>
  // CHECK: %[[VOUT:.+]] = make_tensor_view %[[OUT]], shape = [16, 16], strides = [16, 1] : tensor_view<16x16xf32, strides=[16,1]>
  // CHECK: %[[POUT:.+]] = make_partition_view %[[VOUT]] : partition_view<tile=(2x16), tensor_view<16x16xf32, strides=[16,1]>>
  // CHECK: store_view_tko weak %[[TILE]], %[[POUT]][%[[COORD0]], %[[COORD1]]] : tile<2x16xf32>, partition_view<tile=(2x16), tensor_view<16x16xf32, strides=[16,1]>>, tile<i32> -> token
  results %reshape : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: entry @slice_reshape_col_major_output
// CHECK-SAME: (%[[ARG0:.+]]: tile<ptr<f32>>, %[[OUT:.+]]: tile<ptr<f32>>)
nv_tensor_ir.graph @slice_reshape_col_major_output(
    %arg0: tensor<1024xf32>)
    -> (tensor<16x16xf32> {nv_tensor_ir.stride = "(1,16)"})
    attributes {tile_size = array<i32: 16, 2>} {
  // Slice offsets use the pipeline's i32 index type.
  // CHECK: %[[C768:.+]] = constant <i32: 768> : tile<i32>
  // CHECK: %[[C0:.+]] = constant <i32: 0> : tile<i32>
  // CHECK: %[[BID_X:[^,]+]], {{.*}} = get_tile_block_id : tile<i32>
  // CHECK: %[[PTR:.+]] = offset %[[ARG0]], %[[C768]] : tile<ptr<f32>>, tile<i32> -> tile<ptr<f32>>
  // CHECK: %[[VIN:.+]] = make_tensor_view %[[PTR]], shape = [16, 16], strides = [1, 16] : tensor_view<16x16xf32, strides=[1,16]>
  // CHECK: %[[PIN:.+]] = make_partition_view %[[VIN]] : partition_view<tile=(16x2), tensor_view<16x16xf32, strides=[1,16]>>
  // CHECK: %[[TILE:.+]], %{{.*}} = load_view_tko weak %[[PIN]][%[[C0]], %[[BID_X]]] : partition_view<tile=(16x2), tensor_view<16x16xf32, strides=[1,16]>>, tile<i32> -> tile<16x2xf32>, token
  %slice = slice %arg0 starts = [768] limits = [1024] strides = [1] : tensor<1024xf32> -> tensor<256xf32>
  %reshape_reversed = reshape %slice : tensor<256xf32> -> tensor<16x16xf32>
  %reshape = transpose %reshape_reversed permutation = [1, 0] : tensor<16x16xf32> -> tensor<16x16xf32>
  // CHECK: %[[VOUT:.+]] = make_tensor_view %[[OUT]], shape = [16, 16], strides = [1, 16] : tensor_view<16x16xf32, strides=[1,16]>
  // CHECK: %[[POUT:.+]] = make_partition_view %[[VOUT]] : partition_view<tile=(16x2), tensor_view<16x16xf32, strides=[1,16]>>
  // CHECK: store_view_tko weak %[[TILE]], %[[POUT]][%[[C0]], %[[BID_X]]] : tile<16x2xf32>, partition_view<tile=(16x2), tensor_view<16x16xf32, strides=[1,16]>>, tile<i32> -> token
  results %reshape : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: entry @reshape_slice
// CHECK-SAME: (%[[ARG0:.+]]: tile<ptr<f32>>, %[[OUT:.+]]: tile<ptr<f32>>)
// CHECK: %[[C528:.+]] = constant <i32: 528> : tile<i32>
// CHECK: %[[C0:.+]] = constant <i32: 0> : tile<i32>
// CHECK: %[[BID_X:[^,]+]], {{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[PTR:.+]] = offset %[[ARG0]], %[[C528]] : tile<ptr<f32>>, tile<i32> -> tile<ptr<f32>>
// CHECK: %[[VIN:.+]] = make_tensor_view %[[PTR]], shape = [16, 16], strides = [1, 32] : tensor_view<16x16xf32, strides=[1,32]>
// CHECK: %[[PIN:.+]] = make_partition_view %[[VIN]] : partition_view<tile=(16x4), tensor_view<16x16xf32, strides=[1,32]>>
// CHECK: %[[TILE:.+]], %{{.*}} = load_view_tko weak %[[PIN]][%[[C0]], %[[BID_X]]] : partition_view<tile=(16x4), tensor_view<16x16xf32, strides=[1,32]>>, tile<i32> -> tile<16x4xf32>, token
// CHECK: %[[VOUT:.+]] = make_tensor_view %[[OUT]], shape = [16, 16], strides = [16, 1] : tensor_view<16x16xf32, strides=[16,1]>
// CHECK: %[[POUT:.+]] = make_partition_view %[[VOUT]] : partition_view<tile=(16x4), tensor_view<16x16xf32, strides=[16,1]>>
// CHECK: store_view_tko weak %[[TILE]], %[[POUT]][%[[C0]], %[[BID_X]]] : tile<16x4xf32>, partition_view<tile=(16x4), tensor_view<16x16xf32, strides=[16,1]>>, tile<i32> -> token

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
// CHECK: %[[C0:.+]] = constant <i32: 0> : tile<i32>
// CHECK: %[[BID_X:[^,]+]], {{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[VIN:.+]] = make_tensor_view %[[ARG0]], shape = [8, 8], strides = [1, 16] : tensor_view<8x8xf32, strides=[1,16]>
// CHECK: %[[PIN:.+]] = make_partition_view %[[VIN]] : partition_view<tile=(8x8), tensor_view<8x8xf32, strides=[1,16]>>
// CHECK: %[[TILE:.+]], %{{.*}} = load_view_tko weak %[[PIN]][%[[C0]], %[[BID_X]]] : partition_view<tile=(8x8), tensor_view<8x8xf32, strides=[1,16]>>, tile<i32> -> tile<8x8xf32>, token
// CHECK: %[[VOUT:.+]] = make_tensor_view %[[OUT]], shape = [8, 8], strides = [8, 1] : tensor_view<8x8xf32, strides=[8,1]>
// CHECK: %[[POUT:.+]] = make_partition_view %[[VOUT]] : partition_view<tile=(8x8), tensor_view<8x8xf32, strides=[8,1]>>
// CHECK: store_view_tko weak %[[TILE]], %[[POUT]][%[[C0]], %[[BID_X]]] : tile<8x8xf32>, partition_view<tile=(8x8), tensor_view<8x8xf32, strides=[8,1]>>, tile<i32> -> token

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
// CHECK: %[[C256:.+]] = constant <i32: 256> : tile<i32>
// CHECK: %[[C0:.+]] = constant <i32: 0> : tile<i32>
// CHECK: %[[BID_X:[^,]+]], {{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[PTR:.+]] = offset %[[ARG0]], %[[C256]] : tile<ptr<f32>>, tile<i32> -> tile<ptr<f32>>
// CHECK: %[[VIN:.+]] = make_tensor_view %[[PTR]], shape = [8, 8], strides = [1, 32] : tensor_view<8x8xf32, strides=[1,32]>
// CHECK: %[[PIN:.+]] = make_partition_view %[[VIN]] : partition_view<tile=(8x8), tensor_view<8x8xf32, strides=[1,32]>>
// CHECK: %[[TILE:.+]], %{{.*}} = load_view_tko weak %[[PIN]][%[[C0]], %[[BID_X]]] : partition_view<tile=(8x8), tensor_view<8x8xf32, strides=[1,32]>>, tile<i32> -> tile<8x8xf32>, token
// CHECK: %[[VOUT:.+]] = make_tensor_view %[[OUT]], shape = [8, 8], strides = [8, 1] : tensor_view<8x8xf32, strides=[8,1]>
// CHECK: %[[POUT:.+]] = make_partition_view %[[VOUT]] : partition_view<tile=(8x8), tensor_view<8x8xf32, strides=[8,1]>>
// CHECK: store_view_tko weak %[[TILE]], %[[POUT]][%[[C0]], %[[BID_X]]] : tile<8x8xf32>, partition_view<tile=(8x8), tensor_view<8x8xf32, strides=[8,1]>>, tile<i32> -> token

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
// CHECK: %[[C0:.+]] = constant <i32: 0> : tile<i32>
// CHECK: %[[BID_X:[^,]+]], {{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[VIN:.+]] = make_tensor_view %[[ARG0]], shape = [8, 8], strides = [32, 2] : tensor_view<8x8xf32, strides=[32,2]>
// CHECK: %[[PIN:.+]] = make_partition_view %[[VIN]] : partition_view<tile=(8x8), tensor_view<8x8xf32, strides=[32,2]>>
// CHECK: %[[TILE:.+]], %{{.*}} = load_view_tko weak %[[PIN]][%[[C0]], %[[BID_X]]] : partition_view<tile=(8x8), tensor_view<8x8xf32, strides=[32,2]>>, tile<i32> -> tile<8x8xf32>, token
// CHECK: %[[ADD:.+]] = addf %[[TILE]], %[[TILE]] : tile<8x8xf32>
// CHECK: %[[MUL:.+]] = mulf %[[ADD]], %[[ADD]] : tile<8x8xf32>
// CHECK: %[[ABS:.+]] = absf %[[MUL]] : tile<8x8xf32>
// CHECK: %[[VOUT:.+]] = make_tensor_view %[[OUT]], shape = [8, 8], strides = [8, 1] : tensor_view<8x8xf32, strides=[8,1]>
// CHECK: %[[POUT:.+]] = make_partition_view %[[VOUT]] : partition_view<tile=(8x8), tensor_view<8x8xf32, strides=[8,1]>>
// CHECK: store_view_tko weak %[[ABS]], %[[POUT]][%[[C0]], %[[BID_X]]] : tile<8x8xf32>, partition_view<tile=(8x8), tensor_view<8x8xf32, strides=[8,1]>>, tile<i32> -> token

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
