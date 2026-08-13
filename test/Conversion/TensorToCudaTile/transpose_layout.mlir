// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// ============================================================================
// TEST 1: Single input, transpose then self-add -> (64,32):(32,1)
// ============================================================================
// CHECK-LABEL: entry @transpose_self_add
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[TVIEW_IN:.+]] = make_tensor_view %[[IN_PTR]], shape = [64, 32], strides = [1, 64] : tensor_view<64x32xf32, strides=[1,64]>
//       CHECK:   %[[TILE0:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}x{{[0-9]+}}xf32>, token
//       CHECK:   %[[ADD:.+]] = addf %[[TILE0]], %[[TILE0]]
//       CHECK:   %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [64, 32], strides = [32, 1] : tensor_view<64x32xf32, strides=[32,1]>
//       CHECK:   store_view_tko weak %[[ADD]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @transpose_self_add(%arg0: tensor<32x64xf32>) ->
                          (tensor<64x32xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %t = transpose %arg0 permutation = [1, 0] : tensor<32x64xf32> -> tensor<64x32xf32>
    %add = add %t, %t : tensor<64x32xf32>
    results %add : tensor<64x32xf32>
  }
}

// -----

// ============================================================================
// TEST 2: Two inputs, both transposed, then added
// ============================================================================
// CHECK-LABEL: entry @transpose_two_inputs
//  CHECK-SAME: (%[[IN0_PTR:.+]]: tile<ptr<f32>>, %[[IN1_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[TVIEW_IN0:.+]] = make_tensor_view %[[IN0_PTR]], shape = [64, 32], strides = [1, 64] : tensor_view<64x32xf32, strides=[1,64]>
//       CHECK:   %[[TILE0:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}x{{[0-9]+}}xf32>, token
//       CHECK:   %[[TVIEW_IN1:.+]] = make_tensor_view %[[IN1_PTR]], shape = [64, 32], strides = [1, 64] : tensor_view<64x32xf32, strides=[1,64]>
//       CHECK:   %[[TILE1:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}x{{[0-9]+}}xf32>, token
//       CHECK:   %[[ADD:.+]] = addf %[[TILE0]], %[[TILE1]]
//       CHECK:   %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [64, 32], strides = [32, 1] : tensor_view<64x32xf32, strides=[32,1]>
//       CHECK:   store_view_tko weak %[[ADD]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @transpose_two_inputs(%arg0: tensor<32x64xf32>,
                                           %arg1: tensor<32x64xf32>) ->
                          (tensor<64x32xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %t0 = transpose %arg0 permutation = [1, 0] : tensor<32x64xf32> -> tensor<64x32xf32>
    %t1 = transpose %arg1 permutation = [1, 0] : tensor<32x64xf32> -> tensor<64x32xf32>
    %add = add %t0, %t1 : tensor<64x32xf32>
    results %add : tensor<64x32xf32>
  }
}

// -----

// ============================================================================
// TEST 3: transpose(A+B) + C -> mixed transposed/non-transposed inputs
//   arg0,arg1: (64,32):(32,1) -> transposed
//   arg2: (64,32):(1,64) -> column-major
// ============================================================================
// CHECK-LABEL: entry @transpose_add_chain
//  CHECK-SAME: (%[[A_PTR:.+]]: tile<ptr<f32>>, %[[B_PTR:.+]]: tile<ptr<f32>>, %[[C_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[TV_A:.+]] = make_tensor_view %[[A_PTR]], shape = [64, 32], strides = [1, 64] : tensor_view<64x32xf32, strides=[1,64]>
//       CHECK:   %[[TILE_A:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}x{{[0-9]+}}xf32>, token
//       CHECK:   %[[TV_B:.+]] = make_tensor_view %[[B_PTR]], shape = [64, 32], strides = [1, 64] : tensor_view<64x32xf32, strides=[1,64]>
//       CHECK:   %[[TILE_B:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}x{{[0-9]+}}xf32>, token
//       CHECK:   %[[TV_C:.+]] = make_tensor_view %[[C_PTR]], shape = [64, 32], strides = [32, 1] : tensor_view<64x32xf32, strides=[32,1]>
//       CHECK:   %[[TILE_C:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}x{{[0-9]+}}xf32>, token
//       CHECK:   %[[ADD0:.+]] = addf %[[TILE_A]], %[[TILE_B]] : tile<{{[0-9]+}}x{{[0-9]+}}xf32>
//       CHECK:   %[[ADD1:.+]] = addf %[[ADD0]], %[[TILE_C]] : tile<{{[0-9]+}}x{{[0-9]+}}xf32>
//       CHECK:   %[[TV_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [64, 32], strides = [32, 1] : tensor_view<64x32xf32, strides=[32,1]>
//       CHECK:   store_view_tko weak %[[ADD1]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @transpose_add_chain(%arg0: tensor<32x64xf32>,
                                          %arg1: tensor<32x64xf32>,
                                          %arg2: tensor<64x32xf32>) ->
                          (tensor<64x32xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %add0 = add %arg0, %arg1 : tensor<32x64xf32>
    %t = transpose %add0 permutation = [1, 0] : tensor<32x64xf32> -> tensor<64x32xf32>
    %add1 = add %t, %arg2 : tensor<64x32xf32>
    results %add1 : tensor<64x32xf32>
  }
}

// -----

// ============================================================================
// TEST 4: Row-major explicit strides -> add with nv_tensor_ir.stride = "(32,1)"
// ============================================================================
// CHECK-LABEL: entry @add_row_major
//       CHECK:   make_tensor_view {{.*}}, shape = [2048], strides = [1]
//       CHECK:   %[[T0:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}xf32>, token
//       CHECK:   make_tensor_view {{.*}}, shape = [2048], strides = [1]
//       CHECK:   %[[T1:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}xf32>, token
//       CHECK:   addf %[[T0]], %[[T1]] : tile<{{[0-9]+}}xf32>
//       CHECK:   make_tensor_view {{.*}}, shape = [2048], strides = [1]
//       CHECK:   store_view_tko
//       CHECK:   return
module {
  nv_tensor_ir.graph @add_row_major(
      %arg0: tensor<64x32xf32> {nv_tensor_ir.stride = "(32,1)"},
      %arg1: tensor<64x32xf32> {nv_tensor_ir.stride = "(32,1)"}) ->
                          (tensor<64x32xf32> {nv_tensor_ir.stride = "(32,1)"})
      attributes {tile_size = array<i32: 256>} {
    %add = add %arg0, %arg1 : tensor<64x32xf32>
    results %add : tensor<64x32xf32>
  }
}

// -----

// ============================================================================
// TEST 5: A + transpose(B) with non-power-of-two dimensions (30, 50)
//   Layout-based tiling rounds to power-of-two: budget 256 -> tile (16, 16).
// ============================================================================
// CHECK-LABEL: entry @a_plus_transpose_b_non_pow2
//  CHECK-SAME: (%[[A_PTR:.+]]: tile<ptr<f32>>, %[[B_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[TV_A:.+]] = make_tensor_view %[[A_PTR]], shape = [30, 50], strides = [50, 1]
//       CHECK:   %[[TILE_A:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}x{{[0-9]+}}xf32>, token
//       CHECK:   %[[TV_B:.+]] = make_tensor_view %[[B_PTR]], shape = [30, 50], strides = [1, 30]
//       CHECK:   %[[TILE_B:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}x{{[0-9]+}}xf32>, token
//       CHECK:   %[[ADD:.+]] = addf %[[TILE_A]], %[[TILE_B]] : tile<{{[0-9]+}}x{{[0-9]+}}xf32>
//       CHECK:   %[[TV_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [30, 50], strides = [50, 1]
//       CHECK:   store_view_tko weak %[[ADD]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @a_plus_transpose_b_non_pow2(
      %arg0: tensor<30x50xf32>,
      %arg1: tensor<50x30xf32>) ->
      (tensor<30x50xf32>)
      attributes {tile_size = array<i32: 16, 16>} {
    %t = transpose %arg1 permutation = [1, 0] : tensor<50x30xf32> -> tensor<30x50xf32>
    %add = add %arg0, %t : tensor<30x50xf32>
    results %add : tensor<30x50xf32>
  }
}

// -----

// ============================================================================
// TEST 6: Transpose with explicit row-major output strides
//   Inputs: (32,64):(1,32) -> transposed -> (64,32):(32,1).
//   Output: has explicit nv_tensor_ir.stride = "(32,1)" (row-major).
// ============================================================================
// CHECK-LABEL: entry @transpose_to_row_major
//  CHECK-SAME: (%[[IN0:.+]]: tile<ptr<f32>>, %[[IN1:.+]]: tile<ptr<f32>>, %[[OUT:.+]]: tile<ptr<f32>>)
//       CHECK:   make_tensor_view %[[IN0]], shape = [64, 32], strides = [1, 64]
//       CHECK:   %[[T0:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}x{{[0-9]+}}xf32>
//       CHECK:   make_tensor_view %[[IN1]], shape = [64, 32], strides = [1, 64]
//       CHECK:   %[[T1:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}x{{[0-9]+}}xf32>
//       CHECK:   addf %[[T0]], %[[T1]] : tile<{{[0-9]+}}x{{[0-9]+}}xf32>
//       CHECK:   make_tensor_view %[[OUT]], shape = [64, 32], strides = [32, 1]
//       CHECK:   store_view_tko weak {{.*}} partition_view<tile=({{[0-9]+}}x{{[0-9]+}}), tensor_view<64x32xf32, strides=[32,1]>>
//       CHECK:   return
module {
  nv_tensor_ir.graph @transpose_to_row_major(
      %arg0: tensor<32x64xf32>,
      %arg1: tensor<32x64xf32>) ->
                          (tensor<64x32xf32> {nv_tensor_ir.stride = "(32,1)"})
      attributes {tile_size = array<i32: 16, 16>} {
    %t0 = transpose %arg0 permutation = [1, 0] : tensor<32x64xf32> -> tensor<64x32xf32>
    %t1 = transpose %arg1 permutation = [1, 0] : tensor<32x64xf32> -> tensor<64x32xf32>
    %add = add %t0, %t1 : tensor<64x32xf32>
    results %add : tensor<64x32xf32>
  }
}

// -----

// ============================================================================
// TEST 7: A + transpose(B) with explicit row-major strides
//   arg0: (64,32):(32,1) -- row-major
//   arg1: (32,64):(64,1) -- row-major, transposed to (64,32):(1,64)
//   out:  (64,32):(32,1) -- row-major
// ============================================================================
// CHECK-LABEL: entry @a_plus_transpose_b_row_major
//  CHECK-SAME: (%[[A_PTR:.+]]: tile<ptr<f32>>, %[[B_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[TV_A:.+]] = make_tensor_view %[[A_PTR]], shape = [64, 32], strides = [32, 1]
//       CHECK:   %[[TILE_A:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}x{{[0-9]+}}xf32>, token
//       CHECK:   %[[TV_B:.+]] = make_tensor_view %[[B_PTR]], shape = [64, 32], strides = [1, 64]
//       CHECK:   %[[TILE_B:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}x{{[0-9]+}}xf32>, token
//       CHECK:   %[[ADD:.+]] = addf %[[TILE_A]], %[[TILE_B]] : tile<{{[0-9]+}}x{{[0-9]+}}xf32>
//       CHECK:   %[[TV_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [64, 32], strides = [32, 1]
//       CHECK:   store_view_tko weak %[[ADD]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @a_plus_transpose_b_row_major(
      %arg0: tensor<64x32xf32> {nv_tensor_ir.stride = "(32,1)"},
      %arg1: tensor<32x64xf32> {nv_tensor_ir.stride = "(64,1)"}) ->
      (tensor<64x32xf32> {nv_tensor_ir.stride = "(32,1)"})
      attributes {tile_size = array<i32: 16, 16>} {
    %t = transpose %arg1 permutation = [1, 0] : tensor<32x64xf32> -> tensor<64x32xf32>
    %add = add %arg0, %t : tensor<64x32xf32>
    results %add : tensor<64x32xf32>
  }
}
