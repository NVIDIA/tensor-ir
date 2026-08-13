// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// ============================================================================
// TEST 1: Reduce with single small dimension (no blocks)
// ============================================================================
// CHECK-LABEL: @test_reduce_1dim_small
// CHECK: %[[INPUT:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<32x16xf32>
// CHECK: %[[RESULT:.*]] = reduce %[[INPUT]] dim=1
// CHECK:   (%[[VAL:.*]]: tile<f32>, %[[ACC:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[ACC]], %[[VAL]] : tile<f32>
// CHECK:   yield %[[RES]] : tile<f32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_reduce_1dim_small(
    %arg0: tensor<64x16xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <add>>
    : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// ============================================================================
// TEST 2: Reduce with single large dimension
// ============================================================================
// CHECK-LABEL: @test_reduce_1dim_large
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[ONE:.*]] = constant <i32: 1>
// CHECK-DAG: %[[C128:.*]] = constant <i32: 128>
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : [[TILE:tile<32x128xf32>]]
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[LOOP:.*]] = for %[[IVAR:.*]] in (%[[ZERO]] to %[[C128]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG:.*]] = %[[ACCUM]]) -> ([[TILE]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[BLOCK]], %[[IVAR]]]
// CHECK:   %[[INNER:.*]] = addf %[[IARG]], %[[ARG0]] : [[TILE]]
// CHECK:   continue %[[INNER]] : [[TILE]]
// CHECK: %[[RESULT:.*]] = reduce %[[LOOP]] dim=1
// CHECK:   (%[[VAL:.*]]: tile<f32>, %[[ACC:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[ACC]], %[[VAL]] : tile<f32>
// CHECK:   yield %[[RES]] : tile<f32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_reduce_1dim_large(
    %arg0: tensor<64x16384xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <add>>
    : tensor<64x16384xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// ============================================================================
// TEST 3: Reduce with two small dimensions
// ============================================================================
// CHECK-LABEL: @test_reduce_2dim_small
// CHECK: %[[INPUT:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<32x8x8xf32>
// CHECK: %[[MERGED:.*]] = reshape %[[INPUT]] : tile<32x8x8xf32> -> tile<32x64xf32>
// CHECK: %[[RESULT:.*]] = reduce %[[MERGED]] dim=1
// CHECK:   (%[[VAL:.*]]: tile<f32>, %[[ACC:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[ACC]], %[[VAL]] : tile<f32>
// CHECK:   yield %[[RES]] : tile<f32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_reduce_2dim_small(
    %arg0: tensor<8x64x8xf32>
    ) -> (tensor<1x64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [0, 2], reduction_mode = <add>>
    : tensor<8x64x8xf32> -> tensor<1x64x1xf32>
  results %out : tensor<1x64x1xf32>
}

// -----

// ============================================================================
// TEST 4: Reduce with two mixed dimensions (small and large)
// ============================================================================
// CHECK-LABEL: @test_reduce_2dim_mixed
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[ONE:.*]] = constant <i32: 1>
// CHECK-DAG: %[[C512:.*]] = constant <i32: 512>
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : [[TILE:tile<32x8x16xf32>]]
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[LOOP:.*]] = for %[[IVAR:.*]] in (%[[ZERO]] to %[[C512]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG:.*]] = %[[ACCUM]]) -> ([[TILE]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[BLOCK]], %[[ZERO]], %[[IVAR]]]
// CHECK:   %[[INNER:.*]] = addf %[[IARG]], %[[ARG0]] : [[TILE]]
// CHECK:   continue %[[INNER]] : [[TILE]]
// CHECK: %[[MERGED:.*]] = reshape %[[LOOP]] : [[TILE]] -> tile<32x128xf32>
// CHECK: %[[RESULT:.*]] = reduce %[[MERGED]] dim=1
// CHECK:   (%[[VAL:.*]]: tile<f32>, %[[ACC:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[ACC]], %[[VAL]] : tile<f32>
// CHECK:   yield %[[RES]] : tile<f32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_reduce_2dim_mixed(
    %arg0: tensor<8x64x8192xf32>
    ) -> (tensor<1x64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [0, 2], reduction_mode = <add>>
    : tensor<8x64x8192xf32> -> tensor<1x64x1xf32>
  results %out : tensor<1x64x1xf32>
}

// -----

// ============================================================================
// TEST 5: Reduce with two large dimensions
// ============================================================================
// CHECK-LABEL: @test_reduce_2dim_large
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[ONE:.*]] = constant <i32: 1>
// CHECK-DAG: %[[C16:.*]] = constant <i32: 16>
// CHECK-DAG: %[[C128:.*]] = constant <i32: 128>
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : [[TILE:tile<32x16x8xf32>]]
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[LOOP1:.*]] = for %[[IVAR1:.*]] in (%[[ZERO]] to %[[C16]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG1:.*]] = %[[ACCUM]]) -> ([[TILE]])
// CHECK:   %[[LOOP2:.*]] = for %[[IVAR2:.*]] in (%[[ZERO]] to %[[C128]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG2:.*]] = %[[IARG1]]) -> ([[TILE]])
// CHECK:     %[[ARG0:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[BLOCK]], %[[IVAR1]], %[[IVAR2]]]
// CHECK:     %[[INNER:.*]] = addf %[[IARG2]], %[[ARG0]] : [[TILE]]
// CHECK:     continue %[[INNER]] : [[TILE]]
// CHECK:   continue %[[LOOP2]] : [[TILE]]
// CHECK: %[[MERGED:.*]] = reshape %[[LOOP1]] : [[TILE]] -> tile<32x128xf32>
// CHECK: %[[RESULT:.*]] = reduce %[[MERGED]] dim=1
// CHECK:   (%[[VAL:.*]]: tile<f32>, %[[ACC:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[ACC]], %[[VAL]] : tile<f32>
// CHECK:   yield %[[RES]] : tile<f32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_reduce_2dim_large(
    %arg0: tensor<256x64x1024xf32>
    ) -> (tensor<1x64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [0, 2], reduction_mode = <add>>
    : tensor<256x64x1024xf32> -> tensor<1x64x1xf32>
  results %out : tensor<1x64x1xf32>
}

// -----

// ============================================================================
// TEST 6: Nested reduce with small dimensions
// ============================================================================
// CHECK-LABEL: @test_reduce_nested_small
// CHECK: %[[INPUT:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<32x16x16xf32>
// CHECK: %[[REDUCE1:.*]] = reduce %[[INPUT]] dim=2 {{.*}} : tile<32x16x16xf32> -> tile<32x16xf32>
// CHECK:   (%[[VAL1:.*]]: tile<f32>, %[[ACC1:.*]]: tile<f32>)
// CHECK:   %[[RES1:.*]] = addf %[[ACC1]], %[[VAL1]] : tile<f32>
// CHECK:   yield %[[RES1]] : tile<f32>
// CHECK: %[[REDUCE2:.*]] = reduce %[[REDUCE1]] dim=1 {{.*}} : tile<32x16xf32> -> tile<32xf32>
// CHECK:   (%[[VAL2:.*]]: tile<f32>, %[[ACC2:.*]]: tile<f32>)
// CHECK:   %[[RES2:.*]] = mulf %[[ACC2]], %[[VAL2]] : tile<f32>
// CHECK:   yield %[[RES2]] : tile<f32>
// CHECK: store_view_tko weak %[[REDUCE2]]

nv_tensor_ir.graph @test_reduce_nested_small(
    %arg0: tensor<64x16x16xf32>
    ) -> (tensor<64x1x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %reduce = reduce(%arg0) <dimensions = [2], reduction_mode = <add>>
    : tensor<64x16x16xf32> -> tensor<64x16x1xf32>
  %out = reduce(%reduce) <dimensions = [1], reduction_mode = <mul>>
    : tensor<64x16x1xf32> -> tensor<64x1x1xf32>
  results %out : tensor<64x1x1xf32>
}

// -----

// ============================================================================
// TEST 7: Nested reduce with large dimensions
// ============================================================================
// CHECK-LABEL: @test_reduce_nested_large
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[ONE:.*]] = constant <i32: 1>
// CHECK-DAG: %[[C8:.*]] = constant <i32: 8>
// CHECK-DAG: %[[ACCUM_SUM:.*]] = constant <f32: 0.000000e+00> : [[TILE1:tile<1x128x128xf32>]]
// CHECK-DAG: %[[ACCUM_MUL:.*]] = constant <f32: 1.000000e+00> : [[TILE2:tile<1x128xf32>]]
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[LOOP1:.*]] = for %[[IVAR1:.*]] in (%[[ZERO]] to %[[C8]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG1:.*]] = %[[ACCUM_MUL]]) -> ([[TILE2]])
// CHECK:   %[[LOOP2:.*]] = for %[[IVAR2:.*]] in (%[[ZERO]] to %[[C8]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG2:.*]] = %[[ACCUM_SUM]]) -> ([[TILE1]])
// CHECK:     %[[ARG0:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[BLOCK]], %[[IVAR1]], %[[IVAR2]]]
// CHECK:     %[[TADD:.*]] = addf %[[IARG2]], %[[ARG0]] : [[TILE1]]
// CHECK:     continue %[[TADD]] : [[TILE1]]
// CHECK:   %[[REDUCE1:.*]] = reduce %[[LOOP2]] dim=2 {{.*}} : [[TILE1]] -> [[TILE2]]
// CHECK:     (%[[VAL1:.*]]: tile<f32>, %[[ACC1:.*]]: tile<f32>)
// CHECK:     %[[RES1:.*]] = addf %[[ACC1]], %[[VAL1]] : tile<f32>
// CHECK:     yield %[[RES1]] : tile<f32>
// CHECK:   %[[TMUL:.*]] = mulf %[[IARG1]], %[[REDUCE1]] : [[TILE2]]
// CHECK:   continue %[[TMUL]] : [[TILE2]]
// CHECK: %[[REDUCE2:.*]] = reduce %[[LOOP1]] dim=1 {{.*}} : [[TILE2]] -> tile<1xf32>
// CHECK:   (%[[VAL2:.*]]: tile<f32>, %[[ACC2:.*]]: tile<f32>)
// CHECK:   %[[RES2:.*]] = mulf %[[ACC2]], %[[VAL2]] : tile<f32>
// CHECK:   yield %[[RES2]] : tile<f32>
// CHECK: store_view_tko weak %[[REDUCE2]]

nv_tensor_ir.graph @test_reduce_nested_large(
    %arg0: tensor<64x1024x1024xf32>
    ) -> (tensor<64x1x1xf32>)
    attributes {tile_size = array<i32: 1>} {
  %reduce = reduce(%arg0) <dimensions = [2], reduction_mode = <add>>
    : tensor<64x1024x1024xf32> -> tensor<64x1024x1xf32>
  %out = reduce(%reduce) <dimensions = [1], reduction_mode = <mul>>
    : tensor<64x1024x1xf32> -> tensor<64x1x1xf32>
  results %out : tensor<64x1x1xf32>
}

// -----

// ============================================================================
// TEST 8: Softmax reduction [exp(X) / sum(exp(X))]
// ============================================================================
// CHECK-LABEL: @test_reduce_softmax
// CHECK: %[[REDUCE:.*]] = reduce %{{.*}} dim=1
// CHECK: %[[RESHAPE:.*]] = reshape %[[REDUCE]] : tile<32xf32> -> tile<32x1xf32>
// CHECK: %[[BCAST:.*]] = broadcast %[[RESHAPE]] : tile<32x1xf32> -> tile<32x128xf32>
// CHECK: %[[RESULT:.*]] = divf %{{.*}}, %[[BCAST]]
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_reduce_softmax(
    %arg0: tensor<64x128xf32>
    ) -> (tensor<64x128xf32>)
    attributes {tile_size = array<i32: 32, 128>} {
  %exp = exp %arg0 : tensor<64x128xf32>
  %reduce = reduce(%exp) <dimensions = [1], reduction_mode = <add>>
    : tensor<64x128xf32> -> tensor<64x1xf32>
  %sum = broadcast %reduce : tensor<64x1xf32> -> tensor<64x128xf32>
  %out = div %exp, %sum : tensor<64x128xf32>
  results %out : tensor<64x128xf32>
}

// -----

// ============================================================================
// TEST 9: Full reduction (edge case)
// ============================================================================
// CHECK-LABEL: @test_reduce_full
// CHECK: %[[INPUT:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<1x128xf32>
// CHECK: %[[RESULT:.*]] = reduce %[[INPUT]] dim=1 {{.*}} : tile<1x128xf32> -> tile<1xf32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_reduce_full(
    %arg0: tensor<128xf32>
    ) -> (tensor<1xf32>)
    attributes {tile_size = array<i32: 1>} {
  %out = reduce(%arg0) <dimensions = [0], reduction_mode = <add>>
    : tensor<128xf32> -> tensor<1xf32>
  results %out : tensor<1xf32>
}

// -----

// ============================================================================
// TEST 10: Single-source reduction (regression, was
// LayoutPropInputValidation.ReduceLowersCleanly). No-regression check that the
// operand-to-iteration-space relation maps the sole reduce operand to space 0, so
// its tile is loaded from the reduction's iteration space and feeds the reduce.
// ============================================================================
// CHECK-LABEL: @reduce_add
// CHECK: %[[INPUT:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<32x16xf32>
// CHECK: %[[RESULT:.*]] = reduce %[[INPUT]] dim=1
// CHECK:   (%[[VAL:.*]]: tile<f32>, %[[ACC:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[ACC]], %[[VAL]] : tile<f32>
// CHECK:   yield %[[RES]] : tile<f32>
// CHECK: store_view_tko weak %[[RESULT]]
nv_tensor_ir.graph @reduce_add(
    %arg0: tensor<64x16xf32> {nv_tensor_ir.stride = "(16,1)"}
    ) -> (tensor<64x1xf32> {nv_tensor_ir.stride = "(1,1)"})
    attributes {tile_size = array<i32: 32>} {
  %out = reduce(%arg0) <dimensions = [1], reduction_mode = <add>> : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}
