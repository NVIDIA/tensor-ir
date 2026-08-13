// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// ============================================================================
// TEST 1: reshape (2048) -> (64,32), then transpose -> (32,64), self-add
// ============================================================================
// CHECK-LABEL: entry @reshape_then_transpose
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[TV_IN:.+]] = make_tensor_view %[[IN_PTR]], shape = [2048], strides = [1] : tensor_view<2048xf32, strides=[1]>
//       CHECK:   %[[TILE:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}xf32>, token
//       CHECK:   %[[ADD:.+]] = addf %[[TILE]], %[[TILE]]
//       CHECK:   %[[TV_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [2048], strides = [1] : tensor_view<2048xf32, strides=[1]>
//       CHECK:   store_view_tko weak %[[ADD]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @reshape_then_transpose(%arg0: tensor<2048xf32>) ->
                          (tensor<32x64xf32>)
      attributes {tile_size = array<i32: 256>} {
    %r = reshape %arg0 : tensor<2048xf32> -> tensor<32x64xf32>
    %add = add %r, %r : tensor<32x64xf32>
    results %add : tensor<32x64xf32>
  }
}

// -----

// ============================================================================
// TEST 2: reshape -> transpose as a pure layout transformation (no compute)
//   Same views as TEST 1 but tile is passed straight through to store.
// ============================================================================
// CHECK-LABEL: entry @reshape_transpose_passthrough
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[TV_IN:.+]] = make_tensor_view %[[IN_PTR]], shape = [2048], strides = [1] : tensor_view<2048xf32, strides=[1]>
//       CHECK:   %[[TILE:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}xf32>, token
//       CHECK:   %[[TV_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [2048], strides = [1] : tensor_view<2048xf32, strides=[1]>
//       CHECK:   store_view_tko weak %[[TILE]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @reshape_transpose_passthrough(%arg0: tensor<2048xf32>) ->
                          (tensor<32x64xf32>)
      attributes {tile_size = array<i32: 256>} {
    %r = reshape %arg0 : tensor<2048xf32> -> tensor<32x64xf32>
    results %r : tensor<32x64xf32>
  }
}

// -----

// ============================================================================
// TEST 3: 3D reshape then transpose last two dims, self-add
// ============================================================================
// CHECK-LABEL: entry @reshape_3d_then_transpose
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[TV_IN:.+]] = make_tensor_view %[[IN_PTR]], shape = [8, 256], strides = [1, 8] : tensor_view<8x256xf32, strides=[1,8]>
//       CHECK:   %[[TILE:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}x{{[0-9]+}}xf32>, token
//       CHECK:   %[[ADD:.+]] = addf %[[TILE]], %[[TILE]]
//       CHECK:   %[[TV_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [8, 256], strides = [256, 1] : tensor_view<8x256xf32, strides=[256,1]>
//       CHECK:   store_view_tko weak %[[ADD]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @reshape_3d_then_transpose(%arg0: tensor<2048xf32>) ->
                          (tensor<8x32x8xf32>)
      attributes {tile_size = array<i32: 8, 8>} {
    %r = reshape %arg0 : tensor<2048xf32> -> tensor<32x8x8xf32>
    %t = transpose %r permutation = [2, 0, 1] : tensor<32x8x8xf32> -> tensor<8x32x8xf32>
    %add = add %t, %t : tensor<8x32x8xf32>
    results %add : tensor<8x32x8xf32>
  }
}

// -----

// ============================================================================
// TEST 4: 3D reshape then 3-cycle transpose [2,0,1], self-add
// ============================================================================
// CHECK-LABEL: entry @reshape_3d_full_cycle
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[TV_IN:.+]] = make_tensor_view %[[IN_PTR]], shape = [32, 4, 16], strides = [64, 1, 4] : tensor_view<32x4x16xf32, strides=[64,1,4]>
//       CHECK:   %[[TILE:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}x{{[0-9]+}}x{{[0-9]+}}xf32>, token
//       CHECK:   %[[ADD:.+]] = addf %[[TILE]], %[[TILE]]
//       CHECK:   %[[TV_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [32, 4, 16], strides = [64, 16, 1] : tensor_view<32x4x16xf32, strides=[64,16,1]>
//       CHECK:   store_view_tko weak %[[ADD]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @reshape_3d_full_cycle(%arg0: tensor<2048xf32>) ->
                          (tensor<32x4x16xf32>)
      attributes {tile_size = array<i32: 4, 4, 4>} {
    %r = reshape %arg0 : tensor<2048xf32> -> tensor<32x16x4xf32>
    %t = transpose %r permutation = [0, 2, 1] : tensor<32x16x4xf32> -> tensor<32x4x16xf32>
    %add = add %t, %t : tensor<32x4x16xf32>
    results %add : tensor<32x4x16xf32>
  }
}

// -----

// ============================================================================
// TEST 5: transpose then reshape to 1D, self-add
// ============================================================================
// CHECK-LABEL: entry @transpose_then_reshape
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[TV_IN:.+]] = make_tensor_view %[[IN_PTR]], shape = [2048], strides = [1] : tensor_view<2048xf32, strides=[1]>
//       CHECK:   %[[TILE:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}xf32>, token
//       CHECK:   %[[ADD:.+]] = addf %[[TILE]], %[[TILE]]
//       CHECK:   %[[TV_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [2048], strides = [1] : tensor_view<2048xf32, strides=[1]>
//       CHECK:   store_view_tko weak %[[ADD]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @transpose_then_reshape(%arg0: tensor<32x64xf32>) ->
                          (tensor<2048xf32>)
      attributes {tile_size = array<i32: 32>} {
    %r = reshape %arg0 : tensor<32x64xf32> -> tensor<2048xf32>
    %add = add %r, %r : tensor<2048xf32>
    results %add : tensor<2048xf32>
  }
}

// -----

// ============================================================================
// TEST 6: One input transposed+reshaped, the other just reshaped, then add.
// ============================================================================
// CHECK-LABEL: entry @mixed_transpose_reshape
//  CHECK-SAME: (%[[IN0_PTR:.+]]: tile<ptr<f32>>, %[[IN1_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   make_tensor_view %[[IN0_PTR]], shape = [32, 64], strides = [64, 1]
//       CHECK:   %[[T0:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}x{{[0-9]+}}xf32>, token
//       CHECK:   make_tensor_view %[[IN1_PTR]], shape = [32, 64], strides = [1, 32]
//       CHECK:   %[[T1:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}x{{[0-9]+}}xf32>, token
//       CHECK:   %[[ADD:.+]] = addf %[[T0]], %[[T1]] : tile<{{[0-9]+}}x{{[0-9]+}}xf32>
//       CHECK:   make_tensor_view %[[OUT_PTR]], shape = [32, 64], strides = [64, 1]
//       CHECK:   store_view_tko weak %[[ADD]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @mixed_transpose_reshape(%arg0: tensor<32x64xf32>,
                                               %arg1: tensor<64x32xf32>) ->
                          (tensor<2048xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %r0 = reshape %arg0 : tensor<32x64xf32> -> tensor<2048xf32>
    %r1_input_reversed = transpose %arg1 permutation = [1, 0] : tensor<64x32xf32> -> tensor<32x64xf32>
    %r1 = reshape %r1_input_reversed : tensor<32x64xf32> -> tensor<2048xf32>
    %add = add %r0, %r1 : tensor<2048xf32>
    results %add : tensor<2048xf32>
  }
}

// -----

// ============================================================================
// TEST 7: reshape -> 3D transpose -> add second input -> reshape -> 2D transpose
// ============================================================================
// CHECK-LABEL: entry @reshape_transpose_chain
//  CHECK-SAME: (%[[IN0_PTR:.+]]: tile<ptr<f32>>, %[[IN1_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   make_tensor_view %[[IN0_PTR]], shape = [32, 8, 8], strides = [1, 32, 256]
//       CHECK:   %[[T0:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}x{{[0-9]+}}x{{[0-9]+}}xf32>, token
//       CHECK:   make_tensor_view %[[IN1_PTR]], shape = [32, 8, 8], strides = [1, 32, 256]
//       CHECK:   %[[T1:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}x{{[0-9]+}}x{{[0-9]+}}xf32>, token
//       CHECK:   %[[ADD:.+]] = addf %[[T0]], %[[T1]] : tile<{{[0-9]+}}x{{[0-9]+}}x{{[0-9]+}}xf32>
//       CHECK:   make_tensor_view %[[OUT_PTR]], shape = [32, 8, 8], strides = [64, 8, 1]
//       CHECK:   store_view_tko weak %[[ADD]]
//       CHECK:   return

module {
  nv_tensor_ir.graph @reshape_transpose_chain(
      %arg0: tensor<64x32xf32>,
      %arg1: tensor<8x8x32xf32>) ->
                          (tensor<32x64xf32>)
      attributes {tile_size = array<i32: 4, 4, 4>} {
    %r0_input_reversed = transpose %arg0 permutation = [1, 0] : tensor<64x32xf32> -> tensor<32x64xf32>
    %r0 = reshape %r0_input_reversed : tensor<32x64xf32> -> tensor<32x8x8xf32>
    %t0 = transpose %r0 permutation = [1, 2, 0] : tensor<32x8x8xf32> -> tensor<8x8x32xf32>
    %add = add %t0, %arg1 : tensor<8x8x32xf32>
    %r1_input_reversed = transpose %add permutation = [2, 1, 0] : tensor<8x8x32xf32> -> tensor<32x8x8xf32>
    %r1 = reshape %r1_input_reversed : tensor<32x8x8xf32> -> tensor<32x64xf32>
    results %r1 : tensor<32x64xf32>
  }
}
