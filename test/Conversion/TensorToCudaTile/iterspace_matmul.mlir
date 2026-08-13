// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// ============================================================================
// TEST 1: Simple 2D matmul (no batch)
// ============================================================================
// CHECK-LABEL: @test_matmul_2d_simple
// CHECK: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : tile<32x16xf32>
// CHECK: %[[LHS:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<32x128xf32>
// CHECK: %[[RHS:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<128x16xf32>
// CHECK: %[[RESULT:.*]] = mmaf %[[LHS]], %[[RHS]], %[[ACCUM]]
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_matmul_2d_simple(
    %arg0: tensor<32x128xf32>,
    %arg1: tensor<128x16xf32>
    ) -> (tensor<32x16xf32>)
    attributes {tile_size = array<i32: 32, 16>} {
  %out = matmul(%arg0, %arg1)
    : (tensor<32x128xf32>, tensor<128x16xf32>) -> tensor<32x16xf32>
  results %out : tensor<32x16xf32>
}

// -----

// ============================================================================
// TEST 2: Simple 3D matmul (one batch per block)
// ============================================================================
// CHECK-LABEL: @test_matmul_3d_simple_1
// CHECK: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : tile<32x16xf32>
// CHECK: %[[LHS:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE_LHS:tile<1x32x128xf32>]]
// CHECK: %[[RHS:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE_RHS:tile<1x128x16xf32>]]
// CHECK: %[[LHS2:.*]] = reshape %[[LHS]] : [[TILE_LHS]] -> tile<32x128xf32>
// CHECK: %[[RHS2:.*]] = reshape %[[RHS]] : [[TILE_RHS]] -> tile<128x16xf32>
// CHECK: %[[MMA:.*]] = mmaf %[[LHS2]], %[[RHS2]], %[[ACCUM]]
// CHECK: %[[RESULT:.*]] = reshape %[[MMA]] : tile<32x16xf32> -> tile<1x32x16xf32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_matmul_3d_simple_1(
    %arg0: tensor<8x32x128xf32>,
    %arg1: tensor<8x128x16xf32>
    ) -> (tensor<8x32x16xf32>)
    attributes {tile_size = array<i32: 1, 32, 16>} {
  %out = matmul(%arg0, %arg1)
    : (tensor<8x32x128xf32>, tensor<8x128x16xf32>) -> tensor<8x32x16xf32>
  results %out : tensor<8x32x16xf32>
}

// -----

// ============================================================================
// TEST 3: Simple 3D matmul (multiple batches per block)
// ============================================================================
// CHECK-LABEL: @test_matmul_3d_simple_2
// CHECK: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : tile<2x32x16xf32>
// CHECK: %[[LHS:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<2x32x128xf32>
// CHECK: %[[RHS:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<2x128x16xf32>
// CHECK: %[[RESULT:.*]] = mmaf %[[LHS]], %[[RHS]], %[[ACCUM]]
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_matmul_3d_simple_2(
    %arg0: tensor<8x32x128xf32>,
    %arg1: tensor<8x128x16xf32>
    ) -> (tensor<8x32x16xf32>)
    attributes {tile_size = array<i32: 2, 32, 16>} {
  %out = matmul(%arg0, %arg1)
    : (tensor<8x32x128xf32>, tensor<8x128x16xf32>) -> tensor<8x32x16xf32>
  results %out : tensor<8x32x16xf32>
}

// -----

// ============================================================================
// TEST 4: Small matmul (no loop) with transpose
// ============================================================================
// CHECK-LABEL: @test_matmul_small_transpose
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0> : tile<i32>
// CHECK-DAG: %[[CST3:.*]] = constant <i32: 3> : tile<i32>
// CHECK-DAG: %[[CST4:.*]] = constant <i32: 4> : tile<i32>
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : tile<2x32x16xf32>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[IDX_B:.*]] = remi %[[BLOCK]], %[[CST4]]
// CHECK: %[[TEMP:.*]] = divi %[[BLOCK]], %[[CST4]]
// CHECK: %[[IDX_N:.*]] = remi %[[TEMP]], %[[CST3]]
// CHECK: %[[IDX_M:.*]] = divi %[[TEMP]], %[[CST3]]
// CHECK: %[[LHS:.*]], %{{.*}} = load_view_tko weak {{.*}}[%[[IDX_B]], %[[IDX_M]], %[[ZERO]]] : {{.*}} -> tile<2x32x128xf32>
// CHECK: %[[RHS:.*]], %{{.*}} = load_view_tko weak {{.*}}[%[[IDX_B]], %[[ZERO]], %[[IDX_N]]] : {{.*}} -> tile<2x128x16xf32>
// CHECK: %[[MMA:.*]] = mmaf %[[LHS]], %[[RHS]], %[[ACCUM]]
// CHECK: %[[RESULT:.*]] = permute %[[MMA]] [0, 2, 1] : tile<2x32x16xf32> -> tile<2x16x32xf32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_matmul_small_transpose(
    %arg0: tensor<8x64x128xf32>,
    %arg1: tensor<8x128x48xf32>
    ) -> (tensor<8x48x64xf32>)
    attributes {tile_size = array<i32: 2, 16, 32>} {
  %mma = matmul(%arg0, %arg1)
    : (tensor<8x64x128xf32>, tensor<8x128x48xf32>) -> tensor<8x64x48xf32>
  %out = transpose %mma permutation = [0, 2, 1] : tensor<8x64x48xf32> -> tensor<8x48x64xf32>
  results %out : tensor<8x48x64xf32>
}

// -----

// ============================================================================
// TEST 5: Large matmul (one loop)
// ============================================================================
// CHECK-LABEL: @test_matmul_large_one_loop
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0> : tile<i32>
// CHECK-DAG: %[[CST1:.*]] = constant <i32: 1> : tile<i32>
// CHECK-DAG: %[[CST2:.*]] = constant <i32: 2> : tile<i32>
// CHECK-DAG: %[[CST4:.*]] = constant <i32: 4> : tile<i32>
// CHECK-DAG: %[[CST8:.*]] = constant <i32: 8> : tile<i32>
// CHECK-DAG: %[[INIT:.*]] = constant <f32: 0.000000e+00> : tile<2x32x16xf32>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[IDX_B:.*]] = remi %[[BLOCK]], %[[CST4]]
// CHECK: %[[TEMP:.*]] = divi %[[BLOCK]], %[[CST4]]
// CHECK: %[[IDX_M:.*]] = remi %[[TEMP]], %[[CST2]]
// CHECK: %[[IDX_N:.*]] = divi %[[TEMP]], %[[CST2]]
// CHECK: %[[RESULT:.*]] = for %[[IDX_K:.*]] in (%[[ZERO]] to %[[CST8]], step %[[CST1]])
// CHECK-SAME: iter_values(%[[ACCUM:.*]] = %[[INIT]])
// CHECK: %[[LHS:.*]], %{{.*}} = load_view_tko weak {{.*}}[%[[IDX_B]], %[[IDX_M]], %[[IDX_K]]] : {{.*}} -> tile<2x32x128xf32>
// CHECK: %[[RHS:.*]], %{{.*}} = load_view_tko weak {{.*}}[%[[IDX_B]], %[[IDX_K]], %[[IDX_N]]] : {{.*}} -> tile<2x128x16xf32>
// CHECK: %[[MMA:.*]] = mmaf %[[LHS]], %[[RHS]], %[[ACCUM]]
// CHECK: continue %[[MMA]] : tile<2x32x16xf32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_matmul_large_one_loop(
    %arg0: tensor<8x64x1024xf32>,
    %arg1: tensor<8x1024x48xf32>
    ) -> (tensor<8x64x48xf32>)
    attributes {tile_size = array<i32: 2, 32, 16>} {
  %out = matmul(%arg0, %arg1)
    : (tensor<8x64x1024xf32>, tensor<8x1024x48xf32>) -> tensor<8x64x48xf32>
  results %out : tensor<8x64x48xf32>
}

// -----

// ============================================================================
// TEST 6: Large matmul (two loops)
// ============================================================================
// CHECK-LABEL: @test_matmul_large_two_loops
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0> : tile<i32>
// CHECK-DAG: %[[CST1:.*]] = constant <i32: 1> : tile<i32>
// CHECK-DAG: %[[CST2:.*]] = constant <i32: 2> : tile<i32>
// CHECK-DAG: %[[CST4:.*]] = constant <i32: 4> : tile<i32>
// CHECK-DAG: %[[INIT:.*]] = constant <f32: 0.000000e+00> : tile<32x16xf32>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[IDX_M:.*]] = remi %[[BLOCK]], %[[CST2]]
// CHECK: %[[IDX_N:.*]] = divi %[[BLOCK]], %[[CST2]]
// CHECK: %[[RESULT:.*]] = for %[[IVAR1:.*]] in (%[[ZERO]] to %[[CST2]], step %[[CST1]])
// CHECK-SAME: iter_values(%[[IARG1:.*]] = %[[INIT]])
// CHECK: %[[INNER:.*]] = for %[[IVAR2:.*]] in (%[[ZERO]] to %[[CST4]], step %[[CST1]])
// CHECK-SAME: iter_values(%[[IARG2:.*]] = %[[IARG1]])
// CHECK:   %[[LHS:.*]], %{{.*}} = load_view_tko weak {{.*}}[%[[IDX_M]], %[[IVAR1]], %[[IVAR2]]] : {{.*}} -> tile<32x16x8xf32>
// CHECK:   %[[RHS:.*]], %{{.*}} = load_view_tko weak {{.*}}[%[[IVAR1]], %[[IVAR2]], %[[IDX_N]]] : {{.*}} -> tile<16x8x16xf32>
// CHECK:   %[[LHS2:.*]] = reshape %[[LHS]] : tile<32x16x8xf32> -> tile<32x128xf32>
// CHECK:   %[[RHS2:.*]] = reshape %[[RHS]] : tile<16x8x16xf32> -> tile<128x16xf32>
// CHECK:   %[[MMA:.*]] = mmaf %[[LHS2]], %[[RHS2]], %[[IARG2]]
// CHECK:   continue %[[MMA]] : tile<32x16xf32>
// CHECK: continue %[[INNER]] : tile<32x16xf32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_matmul_large_two_loops(
    %arg0: tensor<64x32x32xf32> {nv_tensor_ir.stride = "(1024,32,1)"},
    %arg1: tensor<1024x48xf32>
    ) -> (tensor<64x48xf32>)
    attributes {tile_size = array<i32: 32, 16>} {
  %arg0_reversed = transpose %arg0 permutation = [2, 1, 0] : tensor<64x32x32xf32> -> tensor<32x32x64xf32>
  %reshape_reversed = reshape %arg0_reversed : tensor<32x32x64xf32> -> tensor<1024x64xf32>
  %reshape = transpose %reshape_reversed permutation = [1, 0] : tensor<1024x64xf32> -> tensor<64x1024xf32>
  %out = matmul(%reshape, %arg1)
    : (tensor<64x1024xf32>, tensor<1024x48xf32>) -> tensor<64x48xf32>
  results %out : tensor<64x48xf32>
}

// -----

// ============================================================================
// TEST 7: Matmul with broadcast
// ============================================================================
// CHECK-LABEL: @test_matmul_broadcast
// CHECK: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : tile<32x16xf32>
// CHECK: %[[LHS:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<32x128xf32>
// CHECK: %[[RHS:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<128x16xf32>
// CHECK: %[[MMA:.*]] = mmaf %[[LHS]], %[[RHS]], %[[ACCUM]]
// CHECK: %[[RESHAPE:.*]] = reshape %[[MMA]] : tile<32x16xf32> -> tile<1x32x16xf32>
// CHECK: %[[RESULT:.*]] = broadcast %[[RESHAPE]] : tile<1x32x16xf32> -> tile<2x32x16xf32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_matmul_broadcast(
    %arg0: tensor<32x128xf32>,
    %arg1: tensor<128x16xf32>
    ) -> (tensor<8x32x16xf32>)
    attributes {tile_size = array<i32: 2, 32, 16>} {
  %mma = matmul(%arg0, %arg1)
    : (tensor<32x128xf32>, tensor<128x16xf32>) -> tensor<32x16xf32>
  %reshape = reshape %mma : tensor<32x16xf32> -> tensor<1x32x16xf32>
  %out = broadcast %reshape : tensor<1x32x16xf32> -> tensor<8x32x16xf32>
  results %out : tensor<8x32x16xf32>
}

// -----

// ============================================================================
// TEST 8: Matmul with broadcasts that split dimensions
// ============================================================================
// CHECK-LABEL: @test_matmul_dimension_split
// CHECK: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : tile<2x64x16xf32>
// CHECK: %[[LHS:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE_LHS:tile<2x8x8x128xf32>]]
// CHECK: %[[RHS:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE_RHS:tile<2x128x4x4xf32>]]
// CHECK: %[[LHS2:.*]] = reshape %[[LHS]] : [[TILE_LHS]] -> tile<2x64x128xf32>
// CHECK: %[[RHS2:.*]] = reshape %[[RHS]] : [[TILE_RHS]] -> tile<2x128x16xf32>
// CHECK: %[[MMA:.*]] = mmaf %[[LHS2]], %[[RHS2]], %[[ACCUM]]
// CHECK: %[[RESHAPE:.*]] = reshape %[[MMA]] : tile<2x64x16xf32> -> [[TILE_RS:tile<1x1x2x8x8x4x4xf32>]]
// CHECK: %[[TRANS:.*]] = permute %[[RESHAPE]] [2, 4, 0, 3, 6, 1, 5] : [[TILE_RS]] -> [[TILE_TR:tile<2x8x1x8x4x1x4xf32>]]
// CHECK: %[[RESULT:.*]] = broadcast %[[TRANS]] : [[TILE_TR]] -> tile<2x8x1x8x4x2x4xf32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_matmul_dimension_split(
    %arg0: tensor<8x64x128xf32>,
    %arg1: tensor<8x128x16xf32>
    ) -> (tensor<64x8x32x8x4xf32>)
    attributes {tile_size = array<i32: 2, 8, 1, 8, 4, 2, 4>} {
    // Resolved matmul tile sizes: B=8, M=16, N=16, broadcast=2.
  %mma = matmul(%arg0, %arg1)
    : (tensor<8x64x128xf32>, tensor<8x128x16xf32>) -> tensor<8x64x16xf32>
  %reshape = reshape %mma : tensor<8x64x16xf32> -> tensor<64x1x32x1x4xf32>
  %out = broadcast %reshape : tensor<64x1x32x1x4xf32> -> tensor<64x8x32x8x4xf32>
  results %out : tensor<64x8x32x8x4xf32>
}

// -----

// ============================================================================
// TEST 9: Matmul with a block-argument LHS (regression, was
// LayoutPropInputValidation.MatmulWithBlockArgLhsLowersCleanly). The matmul owns
// two iteration spaces (LHS, RHS); the operand-to-iteration-space relation must
// keep LHS on space 0 and RHS on space 1 so the block-argument LHS tile is loaded
// from the LHS tensor. Data flow: the (M,K) tile comes from the [128,64] LHS
// tensor and the (K,N) tile from the [64,128] RHS tensor, and they feed mmaf in
// that operand order.
// ============================================================================
// CHECK-LABEL: @matmul_blockarg_lhs
// CHECK:       %[[ACC:.*]] = constant <f32: 0.000000e+00> : tile<32x16xf32>
// CHECK:       make_tensor_view %{{.*}}, shape = [128, 64]
// CHECK:       %[[LHS:.*]], %{{.*}} = load_view_tko weak {{.*}} -> tile<32x64xf32>, token
// CHECK:       make_tensor_view %{{.*}}, shape = [64, 128]
// CHECK:       %[[RHS:.*]], %{{.*}} = load_view_tko weak {{.*}} -> tile<64x16xf32>, token
// CHECK:       %[[MMA:.*]] = mmaf %[[LHS]], %[[RHS]], %[[ACC]]
// CHECK:       store_view_tko weak %[[MMA]]
nv_tensor_ir.graph @matmul_blockarg_lhs(
    %arg0: tensor<128x64xf32> {nv_tensor_ir.stride = "(64,1)"},
    %arg1: tensor<64x128xf32> {nv_tensor_ir.stride = "(128,1)"}
    ) -> (tensor<128x128xf32> {nv_tensor_ir.stride = "(128,1)"})
    attributes {tile_size = array<i32: 32, 16>} {
  %c = "nv_tensor_ir.matmul"(%arg0, %arg1)
    : (tensor<128x64xf32>, tensor<64x128xf32>) -> tensor<128x128xf32>
  results %c : tensor<128x128xf32>
}
