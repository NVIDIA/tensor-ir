// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// ============================================================================
// TEST 1: Join dims (64,32) -> (2048), then self-add
// ============================================================================
// CHECK-LABEL: entry @reshape_join_self_add
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[TVIEW_IN:.+]] = make_tensor_view %[[IN_PTR]], shape = [2048], strides = [1] : tensor_view<2048xf32, strides=[1]>
//       CHECK:   %[[TILE0:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}xf32>, token
//       CHECK:   %[[ADD:.+]] = addf %[[TILE0]], %[[TILE0]]
//       CHECK:   %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [2048], strides = [1] : tensor_view<2048xf32, strides=[1]>
//       CHECK:   store_view_tko weak %[[ADD]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @reshape_join_self_add(
      %arg0: tensor<64x32xf32> {nv_tensor_ir.stride = "(1,64)"}) ->
      (tensor<2048xf32>)
      attributes {tile_size = array<i32: 32>} {
    %input_reversed = transpose %arg0 permutation = [1, 0] : tensor<64x32xf32> -> tensor<32x64xf32>
    %r = reshape %input_reversed : tensor<32x64xf32> -> tensor<2048xf32>
    %add = add %r, %r : tensor<2048xf32>
    results %add : tensor<2048xf32>
  }
}

// -----

// ============================================================================
// TEST 2: Split dims (2048) -> (64,32), then self-add
// ============================================================================
// CHECK-LABEL: entry @reshape_split_self_add
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[TVIEW_IN:.+]] = make_tensor_view %[[IN_PTR]], shape = [2048], strides = [1] : tensor_view<2048xf32, strides=[1]>
//       CHECK:   %[[TILE0:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}xf32>, token
//       CHECK:   %[[ADD:.+]] = addf %[[TILE0]], %[[TILE0]]
//       CHECK:   %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [2048], strides = [1] : tensor_view<2048xf32, strides=[1]>
//       CHECK:   store_view_tko weak %[[ADD]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @reshape_split_self_add(
      %arg0: tensor<2048xf32>) ->
      (tensor<64x32xf32>)
      attributes {tile_size = array<i32: 32>} {
    %r = reshape %arg0 : tensor<2048xf32> -> tensor<64x32xf32>
    %add = add %r, %r : tensor<64x32xf32>
    results %add : tensor<64x32xf32>
  }
}

// -----

// ============================================================================
// TEST 3: reshape(A+B) + C -> mixed shapes through reshape
// ============================================================================
// CHECK-LABEL: entry @reshape_add_chain
//  CHECK-SAME: (%[[A_PTR:.+]]: tile<ptr<f32>>, %[[B_PTR:.+]]: tile<ptr<f32>>, %[[C_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[TV_A:.+]] = make_tensor_view %[[A_PTR]], shape = [2048], strides = [1]
//       CHECK:   %[[TILE_A:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}xf32>, token
//       CHECK:   %[[TV_B:.+]] = make_tensor_view %[[B_PTR]], shape = [2048], strides = [1]
//       CHECK:   %[[TILE_B:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}xf32>, token
//       CHECK:   %[[TV_C:.+]] = make_tensor_view %[[C_PTR]], shape = [2048], strides = [1]
//       CHECK:   %[[TILE_C:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}xf32>, token
//       CHECK:   %[[ADD0:.+]] = addf %[[TILE_A]], %[[TILE_B]] : tile<{{[0-9]+}}xf32>
//       CHECK:   %[[ADD1:.+]] = addf %[[ADD0]], %[[TILE_C]] : tile<{{[0-9]+}}xf32>
//       CHECK:   %[[TV_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [2048], strides = [1]
//       CHECK:   store_view_tko weak %[[ADD1]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @reshape_add_chain(
      %arg0: tensor<64x32xf32> {nv_tensor_ir.stride = "(1,64)"},
      %arg1: tensor<64x32xf32> {nv_tensor_ir.stride = "(1,64)"},
      %arg2: tensor<2048xf32>) ->
      (tensor<2048xf32>)
      attributes {tile_size = array<i32: 32>} {
    %sum = add %arg0, %arg1 : tensor<64x32xf32>
    %sum_reversed = transpose %sum permutation = [1, 0] : tensor<64x32xf32> -> tensor<32x64xf32>
    %r = reshape %sum_reversed : tensor<32x64xf32> -> tensor<2048xf32>
    %add = add %r, %arg2 : tensor<2048xf32>
    results %add : tensor<2048xf32>
  }
}

// -----

// ============================================================================
// TEST 4: Two inputs with different original shapes reshaped to same target
// ============================================================================
// CHECK-LABEL: entry @reshape_two_inputs
//  CHECK-SAME: (%[[IN0_PTR:.+]]: tile<ptr<f32>>, %[[IN1_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[TV0:.+]] = make_tensor_view %[[IN0_PTR]], shape = [2048], strides = [1]
//       CHECK:   %[[TILE0:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}xf32>, token
//       CHECK:   %[[TV1:.+]] = make_tensor_view %[[IN1_PTR]], shape = [2048], strides = [1]
//       CHECK:   %[[TILE1:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}xf32>, token
//       CHECK:   %[[ADD:.+]] = addf %[[TILE0]], %[[TILE1]] : tile<{{[0-9]+}}xf32>
//       CHECK:   %[[TV_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [2048], strides = [1]
//       CHECK:   store_view_tko weak %[[ADD]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @reshape_two_inputs(
      %arg0: tensor<64x32xf32> {nv_tensor_ir.stride = "(1,64)"},
      %arg1: tensor<32x64xf32> {nv_tensor_ir.stride = "(1,32)"}) ->
      (tensor<2048xf32>)
      attributes {tile_size = array<i32: 32>} {
    %arg0_reversed = transpose %arg0 permutation = [1, 0] : tensor<64x32xf32> -> tensor<32x64xf32>
    %r0 = reshape %arg0_reversed : tensor<32x64xf32> -> tensor<2048xf32>
    %arg1_reversed = transpose %arg1 permutation = [1, 0] : tensor<32x64xf32> -> tensor<64x32xf32>
    %r1 = reshape %arg1_reversed : tensor<64x32xf32> -> tensor<2048xf32>
    %add = add %r0, %r1 : tensor<2048xf32>
    results %add : tensor<2048xf32>
  }
}

// -----

// ============================================================================
// TEST 5: Reshape to 3D: (64,32) -> (8,8,32), then self-add
// ============================================================================
// CHECK-LABEL: entry @reshape_to_3d
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[TVIEW_IN:.+]] = make_tensor_view %[[IN_PTR]], shape = [2048], strides = [1] : tensor_view<2048xf32, strides=[1]>
//       CHECK:   %[[TILE0:.+]], {{.*}} = load_view_tko weak {{.*}} :{{.*}}-> tile<{{[0-9]+}}xf32>, token
//       CHECK:   %[[ADD:.+]] = addf %[[TILE0]], %[[TILE0]]
//       CHECK:   %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [2048], strides = [1] : tensor_view<2048xf32, strides=[1]>
//       CHECK:   store_view_tko weak %[[ADD]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @reshape_to_3d(
      %arg0: tensor<64x32xf32>) ->
      (tensor<8x8x32xf32>)
      attributes {tile_size = array<i32: 32>} {
    %r = reshape %arg0 : tensor<64x32xf32> -> tensor<8x8x32xf32>
    %add = add %r, %r : tensor<8x8x32xf32>
    results %add : tensor<8x8x32xf32>
  }
}

// -----


// ============================================================================
// TEST 7: Reshape (4,8,16,32) -> (32,512) with explicit strides (col major).
// ============================================================================
// CHECK-LABEL: entry @reshape_4d_to_2d_rowmajor
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[TVIEW_IN:.+]] = make_tensor_view %[[IN_PTR]], shape = [32, 512], strides = [1, 32]
//       CHECK:   %[[TILE0:.+]], {{.*}} = load_view_tko weak
//       CHECK:   %[[ADD:.+]] = addf %[[TILE0]], %[[TILE0]]
//       CHECK:   %[[TVIEW_OUT:.+]] = make_tensor_view %[[OUT_PTR]], shape = [32, 512], strides = [1, 32]
//       CHECK:   store_view_tko weak %[[ADD]]
//       CHECK:   return
nv_tensor_ir.graph @reshape_4d_to_2d_rowmajor(
    %arg0: tensor<4x8x16x32xf32> {nv_tensor_ir.stride = "(8,1,1024,32)"}
) -> (tensor<32x512xf32> {nv_tensor_ir.stride = "(1,32)"}) {
  %r = reshape %arg0 : tensor<4x8x16x32xf32> -> tensor<32x512xf32>
  %add = add %r, %r : tensor<32x512xf32>
  results %add : tensor<32x512xf32>
}

// -----

// ============================================================================
// TEST 8: Reshape (4,8) -> (32) with explicit row-major strides.
//   Dimension 1 (size 8, stride 1) should be iterated first.
// ============================================================================
// CHECK-LABEL: entry @reshape_2d_to_1d_rowmajor
//  CHECK-SAME: (%[[IN_PTR:.+]]: tile<ptr<f32>>, %[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   make_tensor_view %[[IN_PTR]], shape = [32], strides = [1]
//       CHECK:   make_tensor_view %[[OUT_PTR]], shape = [32], strides = [1]
module {
  nv_tensor_ir.graph @reshape_2d_to_1d_rowmajor(
      %arg0: tensor<4x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<32xf32>) {
    %r = reshape %arg0 : tensor<4x8xf32> -> tensor<32xf32>
    %add = add %r, %r : tensor<32xf32>
    results %add : tensor<32xf32>
  }
}

// -----

// ============================================================================
// TEST 9: A compile-time constant feeding a layout-changing op (reshape)
// (regression, was LayoutPropInputValidation.ConstantFeedingLayoutChangingOpLowersCleanly).
// The constant must receive a layout during layout propagation; before the fix it
// reached the reshape with no layout data and lowering aborted with "missing
// layout data". It must now lower cleanly: the constant is materialized as a tile
// and stored (the trivial reshape folds away).
// ============================================================================
// CHECK-LABEL: entry @constant_feeding_reshape
//  CHECK-SAME: (%[[OUT_PTR:.+]]: tile<ptr<f32>>)
//       CHECK:   %[[CST:.+]] = constant <f32: 1.000000e+00> : tile<1xf32>
//       CHECK:   make_tensor_view %[[OUT_PTR]], shape = [1], strides = [1]
//       CHECK:   store_view_tko weak %[[CST]]
//       CHECK:   return
module {
  nv_tensor_ir.graph @constant_feeding_reshape()
      -> (tensor<1x1xf32> {nv_tensor_ir.stride = "(1,1)"})
      attributes {tile_size = array<i32: 1>} {
    %c = nv_tensor_ir.constant dense<1.0> : tensor<1x1xf32>
    %r = reshape %c {lexicographic = true} : tensor<1x1xf32> -> tensor<1x1xf32>
    results %r : tensor<1x1xf32>
  }
}
