// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// ============================================================================
// TEST 1: Reduce with no loops, single input/output
// ============================================================================
// CHECK-LABEL: @test_reduce_noloop_single
// CHECK: %[[INPUT:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<32x16xf32>
// CHECK: %[[RESULT:.*]] = reduce %[[INPUT]] dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[VAL:.*]]: tile<f32>, %[[ACC:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[ACC]], %[[VAL]] : tile<f32>
// CHECK:   yield %[[RES]] : tile<f32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_reduce_noloop_single(
    %arg0: tensor<64x16xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0.0 : f32]>
  (%accum: f32, %value: f32) {
    %0 = arith.addf %accum, %value : f32
    nv_tensor_ir.yield %0 : f32
  } : tensor<64x16xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// ============================================================================
// TEST 2: Reduce with no loops, multiple inputs/outputs
// ============================================================================
// CHECK-LABEL: @test_reduce_noloop_multiple
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<32x16xf32>
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<32x16xf16>
// CHECK: %[[RESULT:.*]]:2 = reduce %[[ARG0]], %[[ARG1]] dim=1 identities=[1.000000e+00 : f32, 0.000000e+00 : f16]
// CHECK:   (%[[VAL0:.*]]: tile<f32>, %[[ACC0:.*]]: tile<f32>, %[[VAL1:.*]]: tile<f16>, %[[ACC1:.*]]: tile<f16>)
// CHECK:   %[[RES0:.*]] = mulf %[[ACC0]], %[[VAL0]] : tile<f32>
// CHECK:   %[[RES1:.*]] = addf %[[ACC1]], %[[VAL1]] : tile<f16>
// CHECK:   yield %[[RES0]], %[[RES1]] : tile<f32>, tile<f16>
// CHECK: store_view_tko weak %[[RESULT]]#0

nv_tensor_ir.graph @test_reduce_noloop_multiple(
    %arg0: tensor<64x16xf32>,
    %arg1: tensor<64x16xf16>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out0, %out1 = reduce_ud(%arg0, %arg1) <dimensions = [1], identity = [1.0 : f32, 0.0 : f16]>
  (%acc0: f32, %acc1: f16, %val0: f32, %val1: f16) {
    %0 = arith.mulf %acc0, %val0 : f32
    %1 = arith.addf %acc1, %val1 : f16
    nv_tensor_ir.yield %0, %1 : f32, f16
  } : tensor<64x16xf32>, tensor<64x16xf16>
    -> tensor<64x1xf32>, tensor<64x1xf16>
  results %out0 : tensor<64x1xf32>
}

// -----

// ============================================================================
// TEST 3: Reduce with one loop, single input/output
// ============================================================================
// CHECK-LABEL: @test_reduce_1loop_single
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[ONE:.*]] = constant <i32: 1>
// CHECK-DAG: %[[C8:.*]] = constant <i32: 8>
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : [[TILE:tile<32x128xf32>]]
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[LOOP:.*]] = for %[[IVAR:.*]] in (%[[ZERO]] to %[[C8]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG:.*]] = %[[ACCUM]]) -> ([[TILE]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[BLOCK]], %[[IVAR]]]
// CHECK:   %[[INNER:.*]] = addf %[[IARG]], %[[ARG0]] : [[TILE]]
// CHECK:   continue %[[INNER]] : [[TILE]]
// CHECK: %[[RESULT:.*]] = reduce %[[LOOP]] dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[VAL:.*]]: tile<f32>, %[[ACC:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[ACC]], %[[VAL]] : tile<f32>
// CHECK:   yield %[[RES]] : tile<f32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_reduce_1loop_single(
    %arg0: tensor<64x1024xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0.0 : f32]>
  (%accum: f32, %value: f32) {
    %0 = arith.addf %accum, %value : f32
    nv_tensor_ir.yield %0 : f32
  } : tensor<64x1024xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// ============================================================================
// TEST 4: Reduce with one loop, multiple inputs/outputs
// ============================================================================
// CHECK-LABEL: @test_reduce_1loop_multiple
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[ONE:.*]] = constant <i32: 1>
// CHECK-DAG: %[[C8:.*]] = constant <i32: 8>
// CHECK-DAG: %[[ACCUM1:.*]] = constant <f32: 1.000000e+00> : [[TILE1:tile<32x128xf32>]]
// CHECK-DAG: %[[ACCUM2:.*]] = constant <f16: 0.000000e+00> : [[TILE2:tile<32x128xf16>]]
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[LOOP:.*]]:2 = for %[[IVAR:.*]] in (%[[ZERO]] to %[[C8]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG1:.*]] = %[[ACCUM1]], %[[IARG2:.*]] = %[[ACCUM2]]) -> ([[TILE1]], [[TILE2]])
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[BLOCK]], %[[IVAR]]]
// CHECK:   %[[ARG1:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[BLOCK]], %[[IVAR]]]
// CHECK:   %[[MUL:.*]] = mulf %[[IARG1]], %[[ARG0]] : [[TILE1]]
// CHECK:   %[[ADD:.*]] = addf %[[IARG2]], %[[ARG1]] : [[TILE2]]
// CHECK:   continue %[[MUL]], %[[ADD]] : [[TILE1]], [[TILE2]]
// CHECK: %[[RESULT:.*]]:2 = reduce %[[LOOP]]#0, %[[LOOP]]#1 dim=1 identities=[1.000000e+00 : f32, 0.000000e+00 : f16]
// CHECK:   (%[[VAL0:.*]]: tile<f32>, %[[ACC0:.*]]: tile<f32>, %[[VAL1:.*]]: tile<f16>, %[[ACC1:.*]]: tile<f16>)
// CHECK:   %[[RES0:.*]] = mulf %[[ACC0]], %[[VAL0]] : tile<f32>
// CHECK:   %[[RES1:.*]] = addf %[[ACC1]], %[[VAL1]] : tile<f16>
// CHECK:   yield %[[RES0]], %[[RES1]] : tile<f32>, tile<f16>
// CHECK: store_view_tko weak %[[RESULT]]#0

nv_tensor_ir.graph @test_reduce_1loop_multiple(
    %arg0: tensor<64x1024xf32>,
    %arg1: tensor<64x1024xf16>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out0, %out1 = reduce_ud(%arg0, %arg1) <dimensions = [1], identity = [1.0 : f32, 0.0 : f16]>
  (%acc0: f32, %acc1: f16, %val0: f32, %val1: f16) {
    %0 = arith.mulf %acc0, %val0 : f32
    %1 = arith.addf %acc1, %val1 : f16
    nv_tensor_ir.yield %0, %1 : f32, f16
  } : tensor<64x1024xf32>, tensor<64x1024xf16>
    -> tensor<64x1xf32>, tensor<64x1xf16>
  results %out0 : tensor<64x1xf32>
}

// -----

// ============================================================================
// TEST 5: Reduce with two loops, single input/output
// ============================================================================
// CHECK-LABEL: @test_reduce_2loops_single
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[ONE:.*]] = constant <i32: 1>
// CHECK-DAG: %[[C16:.*]] = constant <i32: 16>
// CHECK-DAG: %[[C32:.*]] = constant <i32: 32>
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : [[TILE:tile<32x16x8xf32>]]
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[LOOP1:.*]] = for %[[IVAR:.*]] in (%[[ZERO]] to %[[C16]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG:.*]] = %[[ACCUM]]) -> ([[TILE]])
// CHECK:   %[[LOOP2:.*]] = for %[[JVAR:.*]] in (%[[ZERO]] to %[[C32]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[JARG:.*]] = %[[IARG]]) -> ([[TILE]])
// CHECK:     %[[ARG0:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[BLOCK]], %[[IVAR]], %[[JVAR]]]
// CHECK:     %[[INNER:.*]] = addf %[[JARG]], %[[ARG0]] : [[TILE]]
// CHECK:     continue %[[INNER]] : [[TILE]]
// CHECK:   continue %[[LOOP2]] : [[TILE]]
// CHECK: %[[MERGED:.*]] = reshape %[[LOOP1]] : [[TILE]] -> tile<32x128xf32>
// CHECK: %[[RESULT:.*]] = reduce %[[MERGED]] dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[VAL:.*]]: tile<f32>, %[[ACC:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[ACC]], %[[VAL]] : tile<f32>
// CHECK:   yield %[[RES]] : tile<f32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_reduce_2loops_single(
    %arg0: tensor<256x64x256xf32>
    ) -> (tensor<1x64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [0, 2], identity = [0.0 : f32]>
  (%accum: f32, %value: f32) {
    %0 = arith.addf %accum, %value : f32
    nv_tensor_ir.yield %0 : f32
  } : tensor<256x64x256xf32> -> tensor<1x64x1xf32>
  results %out : tensor<1x64x1xf32>
}

// -----

// ============================================================================
// TEST 6: Reduce with two loops, multiple inputs/outputs
// ============================================================================
// CHECK-LABEL: @test_reduce_2loops_multiple
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[ONE:.*]] = constant <i32: 1>
// CHECK-DAG: %[[C16:.*]] = constant <i32: 16>
// CHECK-DAG: %[[C32:.*]] = constant <i32: 32>
// CHECK-DAG: %[[ACCUM1:.*]] = constant <f32: 1.000000e+00> : [[TILE1:tile<32x16x8xf32>]]
// CHECK-DAG: %[[ACCUM2:.*]] = constant <f16: 0.000000e+00> : [[TILE2:tile<32x16x8xf16>]]
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[LOOP1:.*]]:2 = for %[[IVAR:.*]] in (%[[ZERO]] to %[[C16]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG1:.*]] = %[[ACCUM1]], %[[IARG2:.*]] = %[[ACCUM2]]) -> ([[TILE1]], [[TILE2]])
// CHECK:   %[[LOOP2:.*]]:2 = for %[[JVAR:.*]] in (%[[ZERO]] to %[[C32]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[JARG1:.*]] = %[[IARG1]], %[[JARG2:.*]] = %[[IARG2]]) -> ([[TILE1]], [[TILE2]])
// CHECK:     %[[ARG0:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[BLOCK]], %[[IVAR]], %[[JVAR]]]
// CHECK:     %[[ARG1:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[BLOCK]], %[[IVAR]], %[[JVAR]]]
// CHECK:     %[[MUL:.*]] = mulf %[[JARG1]], %[[ARG0]] : [[TILE1]]
// CHECK:     %[[ADD:.*]] = addf %[[JARG2]], %[[ARG1]] : [[TILE2]]
// CHECK:     continue %[[MUL]], %[[ADD]] : [[TILE1]], [[TILE2]]
// CHECK:   continue %[[LOOP2]]#0, %[[LOOP2]]#1 : [[TILE1]], [[TILE2]]
// CHECK: %[[MERGED1:.*]] = reshape %[[LOOP1]]#0 : [[TILE1]] -> tile<32x128xf32>
// CHECK: %[[MERGED2:.*]] = reshape %[[LOOP1]]#1 : [[TILE2]] -> tile<32x128xf16>
// CHECK: %[[RESULT:.*]]:2 = reduce %[[MERGED1]], %[[MERGED2]] dim=1 identities=[1.000000e+00 : f32, 0.000000e+00 : f16]
// CHECK:   (%[[VAL0:.*]]: tile<f32>, %[[ACC0:.*]]: tile<f32>, %[[VAL1:.*]]: tile<f16>, %[[ACC1:.*]]: tile<f16>)
// CHECK:   %[[RES0:.*]] = mulf %[[ACC0]], %[[VAL0]] : tile<f32>
// CHECK:   %[[RES1:.*]] = addf %[[ACC1]], %[[VAL1]] : tile<f16>
// CHECK:   yield %[[RES0]], %[[RES1]] : tile<f32>, tile<f16>
// CHECK: store_view_tko weak %[[RESULT]]#0

nv_tensor_ir.graph @test_reduce_2loops_multiple(
    %arg0: tensor<256x64x256xf32>,
    %arg1: tensor<256x64x256xf16>
    ) -> (tensor<1x64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out0, %out1 = reduce_ud(%arg0, %arg1) <dimensions = [0, 2], identity = [1.0 : f32, 0.0 : f16]>
  (%acc0: f32, %acc1: f16, %val0: f32, %val1: f16) {
    %0 = arith.mulf %acc0, %val0 : f32
    %1 = arith.addf %acc1, %val1 : f16
    nv_tensor_ir.yield %0, %1 : f32, f16
  } : tensor<256x64x256xf32>, tensor<256x64x256xf16>
    -> tensor<1x64x1xf32>, tensor<1x64x1xf16>
  results %out0 : tensor<1x64x1xf32>
}

// -----

// ============================================================================
// TEST 7: Reduce with broadcasted outputs
// ============================================================================
// CHECK-LABEL: @test_reduce_broadcast
// CHECK: %[[REDUCE:.*]]:2 = reduce %{{.*}} dim=1 identities=[1.000000e+00 : f32, 0.000000e+00 : f16]
// CHECK-DAG: %[[RESHAPE1:.*]] = reshape %[[REDUCE]]#0 : tile<32xf32> -> tile<32x1xf32>
// CHECK-DAG: %[[RESHAPE2:.*]] = reshape %[[REDUCE]]#1 : tile<32xf16> -> tile<32x1xf16>
// CHECK: %[[BCAST1:.*]] = broadcast %[[RESHAPE1]] : tile<32x1xf32> -> tile<32x8xf32>
// CHECK: %[[BCAST2:.*]] = broadcast %[[RESHAPE2]] : tile<32x1xf16> -> tile<32x8xf16>
// CHECK: %[[CONVERT:.*]] = ftof %[[BCAST2]] : tile<32x8xf16> -> tile<32x8xf32>
// CHECK: %[[RESULT:.*]] = divf %[[BCAST1]], %[[CONVERT]]
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_reduce_broadcast(
    %arg0: tensor<64x16xf32>,
    %arg1: tensor<64x16xf16>
    ) -> (tensor<64x8xf32>)
    attributes {tile_size = array<i32: 32, 8>} {
  %out0, %out1 = reduce_ud(%arg0, %arg1) <dimensions = [1], identity = [1.0 : f32, 0.0 : f16]>
  (%acc0: f32, %acc1: f16, %val0: f32, %val1: f16) {
    %0 = arith.mulf %acc0, %val0 : f32
    %1 = arith.addf %acc1, %val1 : f16
    nv_tensor_ir.yield %0, %1 : f32, f16
  } : tensor<64x16xf32>, tensor<64x16xf16>
    -> tensor<64x1xf32>, tensor<64x1xf16>
  %conv = convert %out1 : tensor<64x1xf16> -> tensor<64x1xf32>
  %div = div %out0, %conv : tensor<64x1xf32>
  %bcast = broadcast %div : tensor<64x1xf32> -> tensor<64x8xf32>
  results %bcast : tensor<64x8xf32>
}

// -----

// ============================================================================
// TEST 8: Welford reduction
// ============================================================================
// CHECK-LABEL: @test_reduce_welford
// CHECK-DAG: %[[MEAN_INPUT:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<32x64xf32>
// CHECK-DAG: %[[M2_INPUT:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<32x64xf32>
// CHECK-DAG: %[[WEIGHT_INPUT:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<32x64xf32>
// CHECK: %[[REDUCE:.*]]:3 = reduce %[[MEAN_INPUT]], %[[M2_INPUT]], %[[WEIGHT_INPUT]] dim=1
// CHECK-SAME: identities=[0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32] :
// CHECK-SAME: tile<32x64xf32>, tile<32x64xf32>, tile<32x64xf32> -> tile<32xf32>, tile<32xf32>, tile<32xf32>
// CHECK: %[[VAL_IN:[^:]*]]: tile<f32>, %[[ACC_IN:[^:]*]]: tile<f32>,
// CHECK-SAME: %[[VAL_M2:[^:]*]]: tile<f32>, %[[ACC_M2:[^:]*]]: tile<f32>,
// CHECK-SAME: %[[VAL_W:[^:]*]]: tile<f32>, %[[ACC_W:[^:]*]]: tile<f32>
// CHECK-DAG:   %[[ACC_EMPTY:.*]] = cmpf equal ordered %[[ACC_W]], %[[ZERO:[^ ]+]] : tile<f32> -> tile<i1>
// CHECK-DAG:   %[[VAL_EMPTY:.*]] = cmpf equal ordered %[[VAL_W]], %[[ZERO]] : tile<f32> -> tile<i1>
// CHECK-DAG:   %[[SAFE_ACC_MEAN:.*]] = select %[[ACC_EMPTY]], %[[VAL_IN]], %[[ACC_IN]] : tile<i1>, tile<f32>
// CHECK-DAG:   %[[SAFE_VAL_MEAN:.*]] = select %[[VAL_EMPTY]], %[[ACC_IN]], %[[VAL_IN]] : tile<i1>, tile<f32>
// CHECK-DAG:   %[[DELTA:.*]] = subf %[[SAFE_VAL_MEAN]], %[[SAFE_ACC_MEAN]] : tile<f32>
// CHECK-DAG:   %[[WEIGHT_SUM:.*]] = addf %[[ACC_W]], %[[VAL_W]] : tile<f32>
// CHECK:       %[[WEIGHT_SUM_ZERO:.*]] = cmpf equal ordered %[[WEIGHT_SUM]], %[[ZERO]] : tile<f32> -> tile<i1>
// CHECK:       %[[SAFE_WEIGHT_SUM:.*]] = select %[[WEIGHT_SUM_ZERO]], %[[ONE:[^,]+]], %[[WEIGHT_SUM]] : tile<i1>, tile<f32>
// CHECK:       %[[RHS_FRACTION:.*]] = divf %[[VAL_W]], %[[SAFE_WEIGHT_SUM]] rounding<full> : tile<f32>
// CHECK-DAG:   %[[MEAN_UPDATE:.*]] = mulf %[[DELTA]], %[[RHS_FRACTION]] : tile<f32>
// CHECK-DAG:   %[[M2_SUM:.*]] = addf %[[ACC_M2]], %[[VAL_M2]] : tile<f32>
// CHECK-DAG:   %[[DELTA_SQUARED:.*]] = mulf %[[DELTA]], %[[DELTA]] : tile<f32>
// CHECK-DAG:   %[[MEAN_CANDIDATE:.*]] = addf %[[MEAN_UPDATE]], %[[SAFE_ACC_MEAN]] : tile<f32>
// CHECK-DAG:   %[[WEIGHTED_DELTA:.*]] = mulf %[[DELTA_SQUARED]], %[[ACC_W]] : tile<f32>
// CHECK:       %[[M2_UPDATE:.*]] = mulf %[[WEIGHTED_DELTA]], %[[RHS_FRACTION]] : tile<f32>
// CHECK:       %[[M2_CANDIDATE:.*]] = addf %[[M2_UPDATE]], %[[M2_SUM]] : tile<f32>
// CHECK-DAG:   %[[MEAN_OR_ACC:.*]] = select %[[VAL_EMPTY]], %[[ACC_IN]], %[[MEAN_CANDIDATE]] : tile<i1>, tile<f32>
// CHECK-DAG:   %[[COMBINED_MEAN:.*]] = select %[[ACC_EMPTY]], %[[VAL_IN]], %[[MEAN_OR_ACC]] : tile<i1>, tile<f32>
// CHECK-DAG:   %[[M2_OR_ACC:.*]] = select %[[VAL_EMPTY]], %[[ACC_M2]], %[[M2_CANDIDATE]] : tile<i1>, tile<f32>
// CHECK-DAG:   %[[COMBINED_M2:.*]] = select %[[ACC_EMPTY]], %[[VAL_M2]], %[[M2_OR_ACC]] : tile<i1>, tile<f32>
// CHECK:       yield %[[COMBINED_MEAN]], %[[COMBINED_M2]], %[[WEIGHT_SUM]] : tile<f32>, tile<f32>, tile<f32>
// CHECK: store_view_tko weak %[[REDUCE]]#0

nv_tensor_ir.graph @test_reduce_welford(
    %arg0: tensor<128x64xf32>,
    %arg1: tensor<128x64xf32>,
    %arg2: tensor<128x64xf32>
    ) -> (tensor<128x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %reduce0, %reduce1, %reduce2 = reduce_ud(%arg0, %arg1, %arg2)
  <dimensions = [1], identity = [0.0 : f32, 0.0 : f32, 0.0 : f32]>
  (%mean_lhs: f32, %m2_lhs: f32, %weight_lhs: f32,
   %mean_rhs: f32, %m2_rhs: f32, %weight_rhs: f32) {
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %lhs_is_empty = arith.cmpf oeq, %weight_lhs, %zero : f32
    %rhs_is_empty = arith.cmpf oeq, %weight_rhs, %zero : f32
    %safe_mean_lhs = arith.select %lhs_is_empty, %mean_rhs, %mean_lhs : f32
    %safe_mean_rhs = arith.select %rhs_is_empty, %mean_lhs, %mean_rhs : f32
    %delta = arith.subf %safe_mean_rhs, %safe_mean_lhs : f32
    %weight_sum = arith.addf %weight_lhs, %weight_rhs : f32
    %weight_sum_is_zero = arith.cmpf oeq, %weight_sum, %zero : f32
    %safe_weight_sum = arith.select %weight_sum_is_zero, %one, %weight_sum : f32
    %rhs_fraction = arith.divf %weight_rhs, %safe_weight_sum : f32
    %mean_update = arith.mulf %delta, %rhs_fraction : f32
    %mean_candidate = arith.addf %safe_mean_lhs, %mean_update : f32
    %m2_sum = arith.addf %m2_lhs, %m2_rhs : f32
    %delta_squared = arith.mulf %delta, %delta : f32
    %weighted_delta = arith.mulf %delta_squared, %weight_lhs : f32
    %m2_update = arith.mulf %weighted_delta, %rhs_fraction : f32
    %m2_candidate = arith.addf %m2_sum, %m2_update : f32
    %mean_or_lhs = arith.select %rhs_is_empty, %mean_lhs, %mean_candidate : f32
    %combined_mean = arith.select %lhs_is_empty, %mean_rhs, %mean_or_lhs : f32
    %m2_or_lhs = arith.select %rhs_is_empty, %m2_lhs, %m2_candidate : f32
    %combined_m2 = arith.select %lhs_is_empty, %m2_rhs, %m2_or_lhs : f32
    nv_tensor_ir.yield %combined_mean, %combined_m2, %weight_sum : f32, f32, f32
  } : tensor<128x64xf32>, tensor<128x64xf32>,  tensor<128x64xf32>
    -> tensor<128x1xf32>, tensor<128x1xf32>, tensor<128x1xf32>
  results %reduce0 : tensor<128x1xf32>
}

// -----

// ============================================================================
// TEST 9: Multi-input user-defined reduction (regression, was
// LayoutPropInputValidation.MultiInputReduceUDLowersCleanly). All operands of a
// user-defined reduction share the single reduction iteration space; the relation
// maps every operand to space 0 directly. Data flow: operand 0 (f32) and operand
// 1 (f16) each load from that shared space and feed the fused reduce.
// ============================================================================
// CHECK-LABEL: @reduce_ud_multi
// CHECK: %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<32x16xf32>
// CHECK: %[[ARG1:.*]], %{{.*}} = load_view_tko {{.*}} -> tile<32x16xf16>
// CHECK: %[[RESULT:.*]]:2 = reduce %[[ARG0]], %[[ARG1]] dim=1 identities=[1.000000e+00 : f32, 0.000000e+00 : f16]
// CHECK:   (%[[VAL0:.*]]: tile<f32>, %[[ACC0:.*]]: tile<f32>, %[[VAL1:.*]]: tile<f16>, %[[ACC1:.*]]: tile<f16>)
// CHECK:   %[[RES0:.*]] = mulf %[[ACC0]], %[[VAL0]] : tile<f32>
// CHECK:   %[[RES1:.*]] = addf %[[ACC1]], %[[VAL1]] : tile<f16>
// CHECK:   yield %[[RES0]], %[[RES1]] : tile<f32>, tile<f16>
// CHECK: store_view_tko weak %[[RESULT]]#0
nv_tensor_ir.graph @reduce_ud_multi(
    %arg0: tensor<64x16xf32> {nv_tensor_ir.stride = "(16,1)"},
    %arg1: tensor<64x16xf16> {nv_tensor_ir.stride = "(16,1)"}
    ) -> (tensor<64x1xf32> {nv_tensor_ir.stride = "(1,1)"})
    attributes {tile_size = array<i32: 32>} {
  %out0, %out1 = reduce_ud(%arg0, %arg1) <dimensions = [1], identity = [1.0 : f32, 0.0 : f16]>
  (%acc0: f32, %acc1: f16, %val0: f32, %val1: f16) {
    %0 = arith.mulf %acc0, %val0 : f32
    %1 = arith.addf %acc1, %val1 : f16
    nv_tensor_ir.yield %0, %1 : f32, f16
  } : tensor<64x16xf32>, tensor<64x16xf16>
    -> tensor<64x1xf32>, tensor<64x1xf16>
  results %out0 : tensor<64x1xf32>
}
