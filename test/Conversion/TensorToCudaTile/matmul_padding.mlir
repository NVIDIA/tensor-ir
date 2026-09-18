// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// ============================================================================
// Padding applied at tile load
// ============================================================================

// CHECK-LABEL: entry @test_pad_load_no_loop(
// CHECK-SAME: %[[IN0:[^,]+]]: tile<ptr<f32>>, %[[IN1:[^,]+]]: tile<ptr<f32>>, %[[OUT:[^)]+]]: tile<ptr<f32>>)
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : tile<8x4xf32>
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[C8:.*]] = constant <i32: 8>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[ROW:.*]] = remi %[[BLOCK]], %[[C8]] unsigned
// CHECK: %[[COL:.*]] = divi %[[BLOCK]], %[[C8]] unsigned
// CHECK: %[[LHS_VIEW:.*]] = make_tensor_view %[[IN0]], shape = [64, 50], strides = [50, 1]
// CHECK: %[[PVIEW_LHS:.*]] = make_partition_view %[[LHS_VIEW]] : partition_view<tile=(8x64), padding_value = zero, tensor_view<64x50xf32, strides=[50,1]>>
// CHECK: %[[LHS:.*]], %{{.*}} = load_view_tko weak %[[PVIEW_LHS]][%[[ROW]], %[[ZERO]]]{{.*}} -> tile<8x64xf32>
// CHECK: %[[RHS_VIEW:.*]] = make_tensor_view %[[IN1]], shape = [50, 32], strides = [32, 1]
// CHECK: %[[PVIEW_RHS:.*]] = make_partition_view %[[RHS_VIEW]] : partition_view<tile=(64x4), padding_value = zero, tensor_view<50x32xf32, strides=[32,1]>>
// CHECK: %[[RHS:.*]], %{{.*}} = load_view_tko weak %[[PVIEW_RHS]][%[[ZERO]], %[[COL]]]{{.*}} -> tile<64x4xf32>
// CHECK: %[[RESULT:.*]] = mmaf %[[LHS]], %[[RHS]], %[[ACCUM]]
// CHECK: %[[OUT_VIEW:.*]] = make_tensor_view %[[OUT]], shape = [64, 32], strides = [32, 1]
// CHECK: %[[OUT_PVIEW:.*]] = make_partition_view %[[OUT_VIEW]] : partition_view<tile=(8x4), tensor_view<64x32xf32, strides=[32,1]>>
// CHECK: store_view_tko weak %[[RESULT]], %[[OUT_PVIEW]][%[[ROW]], %[[COL]]]

nv_tensor_ir.graph @test_pad_load_no_loop(
    %arg0: tensor<64x50xf32>, %arg1: tensor<50x32xf32>
    ) -> (tensor<64x32xf32>)
    attributes {tile_size = array<i32: 8, 4>} {
  %out = matmul(%arg0, %arg1)
    : (tensor<64x50xf32>, tensor<50x32xf32>) -> tensor<64x32xf32>
  results %out : tensor<64x32xf32>
}

// -----

// CHECK-LABEL: entry @test_pad_load_one_loop(
// CHECK-SAME: %[[IN0:[^,]+]]: tile<ptr<f32>>, %[[IN1:[^,]+]]: tile<ptr<f32>>, %[[OUT:[^)]+]]: tile<ptr<f32>>)
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[ONE:.*]] = constant <i32: 1>
// CHECK-DAG: %[[C4:.*]] = constant <i32: 4>
// CHECK-DAG: %[[C8:.*]] = constant <i32: 8>
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : [[TILE:tile<8x4xf32>]]
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[IDX0:.*]] = remi %[[BLOCK]], %[[C8]] unsigned
// CHECK: %[[IDX1:.*]] = divi %[[BLOCK]], %[[C8]] unsigned
// CHECK: %[[RESULT:.*]] = for %[[IVAR:.*]] in (%[[ZERO]] to %[[C4]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG:.*]] = %[[ACCUM]]) -> ([[TILE]])
// CHECK:   %[[LHS_VIEW:.*]] = make_tensor_view %[[IN0]], shape = [64, 500], strides = [500, 1]
// CHECK:   %[[PVIEW_LHS:.*]] = make_partition_view %[[LHS_VIEW]] : partition_view<tile=(8x128), padding_value = zero, tensor_view<64x500xf32, strides=[500,1]>>
// CHECK:   %[[LHS:.*]], %{{.*}} = load_view_tko weak %[[PVIEW_LHS]][%[[IDX0]], %[[IVAR]]]
// CHECK:   %[[RHS_VIEW:.*]] = make_tensor_view %[[IN1]], shape = [500, 32], strides = [32, 1]
// CHECK:   %[[PVIEW_RHS:.*]] = make_partition_view %[[RHS_VIEW]] : partition_view<tile=(128x4), padding_value = zero, tensor_view<500x32xf32, strides=[32,1]>>
// CHECK:   %[[RHS:.*]], %{{.*}} = load_view_tko weak %[[PVIEW_RHS]][%[[IVAR]], %[[IDX1]]]
// CHECK:   %[[INNER:.*]] = mmaf %[[LHS]], %[[RHS]], %[[IARG]]
// CHECK:   continue %[[INNER]] : [[TILE]]
// CHECK: %[[OUT_VIEW:.*]] = make_tensor_view %[[OUT]], shape = [64, 32], strides = [32, 1]
// CHECK: %[[OUT_PVIEW:.*]] = make_partition_view %[[OUT_VIEW]] : partition_view<tile=(8x4), tensor_view<64x32xf32, strides=[32,1]>>
// CHECK: store_view_tko weak %[[RESULT]], %[[OUT_PVIEW]][%[[IDX0]], %[[IDX1]]]

nv_tensor_ir.graph @test_pad_load_one_loop(
    %arg0: tensor<64x500xf32>, %arg1: tensor<500x32xf32>
    ) -> (tensor<64x32xf32>)
    attributes {tile_size = array<i32: 8, 4>} {
  %out = matmul(%arg0, %arg1)
    : (tensor<64x500xf32>, tensor<500x32xf32>) -> tensor<64x32xf32>
  results %out : tensor<64x32xf32>
}

// -----

// CHECK-LABEL: entry @test_pad_load_two_loops(
// CHECK-SAME: %[[IN0:[^,]+]]: tile<ptr<f32>>, %[[IN1:[^,]+]]: tile<ptr<f32>>, %[[OUT:[^)]+]]: tile<ptr<f32>>)
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[ONE:.*]] = constant <i32: 1>
// CHECK-DAG: %[[C2:.*]] = constant <i32: 2>
// CHECK-DAG: %[[C8:.*]] = constant <i32: 8>
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : [[TILE:tile<8x4xf32>]]
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[IDX0:.*]] = remi %[[BLOCK]], %[[C8]] unsigned
// CHECK: %[[IDX1:.*]] = divi %[[BLOCK]], %[[C8]] unsigned
// CHECK: %[[RESULT:.*]] = for %[[IVAR1:.*]] in (%[[ZERO]] to %[[C2]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG1:.*]] = %[[ACCUM]]) -> ([[TILE]])
// CHECK:   %[[LOOP:.*]] = for %[[IVAR2:.*]] in (%[[ZERO]] to %[[C8]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG2:.*]] = %[[IARG1]]) -> ([[TILE]])
// CHECK:     %[[LHS_VIEW:.*]] = make_tensor_view %[[IN0]], shape = [64, 30, 60], strides = [2048, 64, 1]
// CHECK:     %[[PVIEW_LHS:.*]] = make_partition_view %[[LHS_VIEW]] : partition_view<tile=(8x16x8), padding_value = zero, tensor_view<64x30x60xf32, strides=[2048,64,1]>>
// CHECK:     %[[LHS:.*]], %{{.*}} = load_view_tko weak %[[PVIEW_LHS]][%[[IDX0]], %[[IVAR1]], %[[IVAR2]]]
// CHECK:     %[[RHS_VIEW:.*]] = make_tensor_view %[[IN1]], shape = [30, 60, 32], strides = [2048, 32, 1]
// CHECK:     %[[PVIEW_RHS:.*]] = make_partition_view %[[RHS_VIEW]] : partition_view<tile=(16x8x4), padding_value = zero, tensor_view<30x60x32xf32, strides=[2048,32,1]>>
// CHECK:     %[[RHS:.*]], %{{.*}} = load_view_tko weak %[[PVIEW_RHS]][%[[IVAR1]], %[[IVAR2]], %[[IDX1]]]
// CHECK:     %[[LHS_NORM:.*]] = reshape %[[LHS]] : tile<8x16x8xf32> -> tile<8x128xf32>
// CHECK:     %[[RHS_NORM:.*]] = reshape %[[RHS]] : tile<16x8x4xf32> -> tile<128x4xf32>
// CHECK:     %[[INNER:.*]] = mmaf %[[LHS_NORM]], %[[RHS_NORM]], %[[IARG2]]
// CHECK:     continue %[[INNER]] : [[TILE]]
// CHECK:   continue %[[LOOP]] : [[TILE]]
// CHECK: %[[OUT_VIEW:.*]] = make_tensor_view %[[OUT]], shape = [64, 32], strides = [32, 1]
// CHECK: %[[OUT_PVIEW:.*]] = make_partition_view %[[OUT_VIEW]] : partition_view<tile=(8x4), tensor_view<64x32xf32, strides=[32,1]>>
// CHECK: store_view_tko weak %[[RESULT]], %[[OUT_PVIEW]][%[[IDX0]], %[[IDX1]]]

nv_tensor_ir.graph @test_pad_load_two_loops(
    %arg0: tensor<64x30x60xf32> {nv_tensor_ir.stride = "(2048,64,1)"},
    %arg1: tensor<30x60x32xf32> {nv_tensor_ir.stride = "(2048,32,1)"}
    ) -> (tensor<64x32xf32>)
    attributes {tile_size = array<i32: 8, 4>} {
  %lhs = reshape %arg0 : tensor<64x30x60xf32> -> tensor<64x1800xf32>
  %rhs = reshape %arg1 : tensor<30x60x32xf32> -> tensor<1800x32xf32>
  %out = matmul(%lhs, %rhs)
    : (tensor<64x1800xf32>, tensor<1800x32xf32>) -> tensor<64x32xf32>
  results %out : tensor<64x32xf32>
}

// -----

// CHECK-LABEL: entry @test_pad_load_chain(
// CHECK-SAME: %[[IN0:[^,]+]]: tile<ptr<f32>>, %[[IN1:[^,]+]]: tile<ptr<f32>>, %[[OUT:[^)]+]]: tile<ptr<f32>>)
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : tile<8x4xf32>
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[C8:.*]] = constant <i32: 8>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[ROW:.*]] = remi %[[BLOCK]], %[[C8]] unsigned
// CHECK: %[[COL:.*]] = divi %[[BLOCK]], %[[C8]] unsigned
// CHECK: %[[LHS_VIEW:.*]] = make_tensor_view %[[IN0]], shape = [64, 50], strides = [64, 1]
// CHECK: %[[PVIEW_LHS:.*]] = make_partition_view %[[LHS_VIEW]] : partition_view<tile=(8x64), padding_value = zero, tensor_view<64x50xf32, strides=[64,1]>>
// CHECK: %[[LHS:.*]], %{{.*}} = load_view_tko weak %[[PVIEW_LHS]][%[[ROW]], %[[ZERO]]]{{.*}} -> tile<8x64xf32>
// CHECK: %[[RHS_VIEW:.*]] = make_tensor_view %[[IN1]], shape = [50, 32], strides = [1, 50]
// CHECK: %[[PVIEW_RHS:.*]] = make_partition_view %[[RHS_VIEW]] : partition_view<tile=(64x4), padding_value = zero, tensor_view<50x32xf32, strides=[1,50]>>
// CHECK: %[[RHS:.*]], %{{.*}} = load_view_tko weak %[[PVIEW_RHS]][%[[ZERO]], %[[COL]]]{{.*}} -> tile<64x4xf32>
// CHECK: %[[RESULT:.*]] = mmaf %[[LHS]], %[[RHS]], %[[ACCUM]]
// CHECK: %[[OUT_VIEW:.*]] = make_tensor_view %[[OUT]], shape = [64, 32], strides = [32, 1]
// CHECK: %[[OUT_PVIEW:.*]] = make_partition_view %[[OUT_VIEW]] : partition_view<tile=(8x4), tensor_view<64x32xf32, strides=[32,1]>>
// CHECK: store_view_tko weak %[[RESULT]], %[[OUT_PVIEW]][%[[ROW]], %[[COL]]]

nv_tensor_ir.graph @test_pad_load_chain(
    %arg0: tensor<64x64xf32>, %arg1: tensor<32x50xf32>
    ) -> (tensor<64x32xf32>)
    attributes {tile_size = array<i32: 8, 4>} {
  %lhs = slice %arg0 starts = [0, 0] limits = [64, 50] strides = [1, 1]
    : tensor<64x64xf32> -> tensor<64x50xf32>
  %rhs = transpose %arg1 permutation = [1, 0]
    : tensor<32x50xf32> -> tensor<50x32xf32>
  %out = matmul(%lhs, %rhs)
    : (tensor<64x50xf32>, tensor<50x32xf32>) -> tensor<64x32xf32>
  results %out : tensor<64x32xf32>
}

// -----

// ============================================================================
// Padding applied before the matmul
// ============================================================================

// CHECK-LABEL: entry @test_pad_mask_no_loop(
// CHECK-SAME: %[[IN0:[^,]+]]: tile<ptr<f32>>, %[[IN1:[^,]+]]: tile<ptr<f32>>, %[[OUT:[^)]+]]: tile<ptr<f32>>)
// CHECK-DAG: %[[C50:.*]] = constant <i32: 50>
// CHECK-DAG: %[[LHS_NEUTRAL:.*]] = constant <f32: 0.000000e+00> : [[LHS_TILE:tile<8x64xf32>]]
// CHECK-DAG: %[[RHS_NEUTRAL:.*]] = constant <f32: 0.000000e+00> : [[RHS_TILE:tile<64x4xf32>]]
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : tile<8x4xf32>
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[C8:.*]] = constant <i32: 8>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[IDX0:.*]] = remi %[[BLOCK]], %[[C8]] unsigned
// CHECK: %[[IDX1:.*]] = divi %[[BLOCK]], %[[C8]] unsigned
// CHECK: %[[IOTA:.*]] = iota : tile<64xi32>
// CHECK: %[[CMP:.*]] = cmpi less_than %[[IOTA]], %[[C50]], unsigned : tile<64xi32> -> tile<64xi1>
// CHECK: %[[CMPL:.*]] = reshape %[[CMP]] : tile<64xi1> -> tile<1x64xi1>
// CHECK: %[[CMPR:.*]] = reshape %[[CMP]] : tile<64xi1> -> tile<64x1xi1>
// CHECK-DAG: %[[LHS_VIEW:.*]] = make_tensor_view %[[IN0]], shape = [64, 50], strides = [50, 1]
// CHECK-DAG: %[[LHS_PVIEW:.*]] = make_partition_view %[[LHS_VIEW]] : partition_view<tile=(8x64), tensor_view<64x50xf32, strides=[50,1]>>
// CHECK-DAG: %[[LHS:.*]], %{{.*}} = load_view_tko weak %[[LHS_PVIEW]][%[[IDX0]], %[[ZERO]]]{{.*}} -> [[LHS_TILE]]
// CHECK-DAG: %[[RHS_VIEW:.*]] = make_tensor_view %[[IN1]], shape = [50, 32], strides = [32, 1]
// CHECK-DAG: %[[RHS_PVIEW:.*]] = make_partition_view %[[RHS_VIEW]] : partition_view<tile=(64x4), tensor_view<50x32xf32, strides=[32,1]>>
// CHECK-DAG: %[[RHS:.*]], %{{.*}} = load_view_tko weak %[[RHS_PVIEW]][%[[ZERO]], %[[IDX1]]]{{.*}} -> [[RHS_TILE]]
// CHECK-DAG: %[[LHS_EXP:.*]] = exp %[[LHS]] : [[LHS_TILE]]
// CHECK-DAG: %[[RHS_EXP:.*]] = exp %[[RHS]] : [[RHS_TILE]]
// CHECK: %[[CMPLB:.*]] = broadcast %[[CMPL]] : tile<1x64xi1> -> tile<8x64xi1>
// CHECK: %[[LHS_MASKED:.*]] = select %[[CMPLB]], %[[LHS_EXP]], %[[LHS_NEUTRAL]]
// CHECK: %[[CMPRB:.*]] = broadcast %[[CMPR]] : tile<64x1xi1> -> tile<64x4xi1>
// CHECK: %[[RHS_MASKED:.*]] = select %[[CMPRB]], %[[RHS_EXP]], %[[RHS_NEUTRAL]]
// CHECK: %[[RESULT:.*]] = mmaf %[[LHS_MASKED]], %[[RHS_MASKED]], %[[ACCUM]]
// CHECK: %[[OUT_VIEW:.*]] = make_tensor_view %[[OUT]], shape = [64, 32], strides = [32, 1]
// CHECK: %[[OUT_PVIEW:.*]] = make_partition_view %[[OUT_VIEW]] : partition_view<tile=(8x4), tensor_view<64x32xf32, strides=[32,1]>>
// CHECK: store_view_tko weak %[[RESULT]], %[[OUT_PVIEW]][%[[IDX0]], %[[IDX1]]]

nv_tensor_ir.graph @test_pad_mask_no_loop(
    %arg0: tensor<64x50xf32>, %arg1: tensor<50x32xf32>
    ) -> (tensor<64x32xf32>)
    attributes {tile_size = array<i32: 8, 4>} {
  %lhs = exp %arg0 : tensor<64x50xf32>
  %rhs = exp %arg1 : tensor<50x32xf32>
  %out = matmul(%lhs, %rhs)
    : (tensor<64x50xf32>, tensor<50x32xf32>) -> tensor<64x32xf32>
  results %out : tensor<64x32xf32>
}

// -----

// CHECK-LABEL: entry @test_pad_mask_one_loop(
// CHECK-SAME: %[[IN0:[^,]+]]: tile<ptr<f32>>, %[[IN1:[^,]+]]: tile<ptr<f32>>, %[[OUT:[^)]+]]: tile<ptr<f32>>)
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[ONE:.*]] = constant <i32: 1>
// CHECK-DAG: %[[C4:.*]] = constant <i32: 4>
// CHECK-DAG: %[[C8:.*]] = constant <i32: 8>
// CHECK-DAG: %[[C128:.*]] = constant <i32: 128>
// CHECK-DAG: %[[C500:.*]] = constant <i32: 500>
// CHECK-DAG: %[[LHS_NEUTRAL:.*]] = constant <f32: 0.000000e+00> : [[LHS_TILE:tile<8x128xf32>]]
// CHECK-DAG: %[[RHS_NEUTRAL:.*]] = constant <f32: 0.000000e+00> : [[RHS_TILE:tile<128x4xf32>]]
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : [[TILE:tile<8x4xf32>]]
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[IDX0:.*]] = remi %[[BLOCK]], %[[C8]] unsigned
// CHECK: %[[IDX1:.*]] = divi %[[BLOCK]], %[[C8]] unsigned
// CHECK: %[[RESULT:.*]] = for %[[IVAR:.*]] in (%[[ZERO]] to %[[C4]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG:.*]] = %[[ACCUM]]) -> ([[TILE]])
// CHECK:   %[[MUL:.*]] = muli %[[IVAR]], %[[C128]] : tile<i32>
// CHECK:   %[[MUL1:.*]] = reshape %[[MUL]] : tile<i32> -> tile<1xi32>
// CHECK:   %[[MUL2:.*]] = broadcast %[[MUL1]] : tile<1xi32> -> tile<128xi32>
// CHECK:   %[[IOTA:.*]] = iota : tile<128xi32>
// CHECK:   %[[ADD:.*]] = addi %[[MUL2]], %[[IOTA]] : tile<128xi32>
// CHECK:   %[[CMP:.*]] = cmpi less_than %[[ADD]], %[[C500]], unsigned : tile<128xi32> -> tile<128xi1>
// CHECK:   %[[CMPL:.*]] = reshape %[[CMP]] : tile<128xi1> -> tile<1x128xi1>
// CHECK:   %[[CMPR:.*]] = reshape %[[CMP]] : tile<128xi1> -> tile<128x1xi1>
// CHECK-DAG:   %[[LHS_VIEW:.*]] = make_tensor_view %[[IN0]], shape = [64, 500], strides = [500, 1]
// CHECK-DAG:   %[[LHS_PVIEW:.*]] = make_partition_view %[[LHS_VIEW]] : partition_view<tile=(8x128), tensor_view<64x500xf32, strides=[500,1]>>
// CHECK-DAG:   %[[LHS:.*]], %{{.*}} = load_view_tko weak %[[LHS_PVIEW]][%[[IDX0]], %[[IVAR]]]
// CHECK-DAG:   %[[RHS_VIEW:.*]] = make_tensor_view %[[IN1]], shape = [500, 32], strides = [32, 1]
// CHECK-DAG:   %[[RHS_PVIEW:.*]] = make_partition_view %[[RHS_VIEW]] : partition_view<tile=(128x4), tensor_view<500x32xf32, strides=[32,1]>>
// CHECK-DAG:   %[[RHS:.*]], %{{.*}} = load_view_tko weak %[[RHS_PVIEW]][%[[IVAR]], %[[IDX1]]]
// CHECK-DAG:   %[[LHS_EXP:.*]] = exp %[[LHS]] : [[LHS_TILE]]
// CHECK-DAG:   %[[RHS_EXP:.*]] = exp %[[RHS]] : [[RHS_TILE]]
// CHECK:   %[[CMPLB:.*]] = broadcast %[[CMPL]] : tile<1x128xi1> -> tile<8x128xi1>
// CHECK:   %[[LHS_MASKED:.*]] = select %[[CMPLB]], %[[LHS_EXP]], %[[LHS_NEUTRAL]]
// CHECK:   %[[CMPRB:.*]] = broadcast %[[CMPR]] : tile<128x1xi1> -> tile<128x4xi1>
// CHECK:   %[[RHS_MASKED:.*]] = select %[[CMPRB]], %[[RHS_EXP]], %[[RHS_NEUTRAL]]
// CHECK:   %[[INNER:.*]] = mmaf %[[LHS_MASKED]], %[[RHS_MASKED]], %[[IARG]]
// CHECK:   continue %[[INNER]] : [[TILE]]
// CHECK: %[[OUT_VIEW:.*]] = make_tensor_view %[[OUT]], shape = [64, 32], strides = [32, 1]
// CHECK: %[[OUT_PVIEW:.*]] = make_partition_view %[[OUT_VIEW]] : partition_view<tile=(8x4), tensor_view<64x32xf32, strides=[32,1]>>
// CHECK: store_view_tko weak %[[RESULT]], %[[OUT_PVIEW]][%[[IDX0]], %[[IDX1]]]

nv_tensor_ir.graph @test_pad_mask_one_loop(
    %arg0: tensor<64x500xf32>, %arg1: tensor<500x32xf32>
    ) -> (tensor<64x32xf32>)
    attributes {tile_size = array<i32: 8, 4>} {
  %lhs = exp %arg0 : tensor<64x500xf32>
  %rhs = exp %arg1 : tensor<500x32xf32>
  %out = matmul(%lhs, %rhs)
    : (tensor<64x500xf32>, tensor<500x32xf32>) -> tensor<64x32xf32>
  results %out : tensor<64x32xf32>
}

// -----

// CHECK-LABEL: entry @test_pad_mask_two_loops(
// CHECK-SAME: %[[IN0:[^,]+]]: tile<ptr<f32>>, %[[IN1:[^,]+]]: tile<ptr<f32>>, %[[OUT:[^)]+]]: tile<ptr<f32>>)
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[ONE:.*]] = constant <i32: 1>
// CHECK-DAG: %[[C2:.*]] = constant <i32: 2>
// CHECK-DAG: %[[C8:.*]] = constant <i32: 8>
// CHECK-DAG: %[[C16:.*]] = constant <i32: 16>
// CHECK-DAG: %[[C30:.*]] = constant <i32: 30>
// CHECK-DAG: %[[C60:.*]] = constant <i32: 60>
// CHECK-DAG: %[[LHS_NEUTRAL:.*]] = constant <f32: 0.000000e+00> : [[LHS_TILE:tile<8x16x8xf32>]]
// CHECK-DAG: %[[RHS_NEUTRAL:.*]] = constant <f32: 0.000000e+00> : [[RHS_TILE:tile<16x8x4xf32>]]
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : [[TILE:tile<8x4xf32>]]
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[IDX0:.*]] = remi %[[BLOCK]], %[[C8]] unsigned
// CHECK: %[[IDX1:.*]] = divi %[[BLOCK]], %[[C8]] unsigned
// CHECK: %[[RESULT:.*]] = for %[[IVAR1:.*]] in (%[[ZERO]] to %[[C2]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG1:.*]] = %[[ACCUM]]) -> ([[TILE]])
// CHECK:   %[[A_MUL:.*]] = muli %[[IVAR1]], %[[C16]] : tile<i32>
// CHECK:   %[[A_MUL1:.*]] = reshape %[[A_MUL]] : tile<i32> -> tile<1xi32>
// CHECK:   %[[A_MUL2:.*]] = broadcast %[[A_MUL1]] : tile<1xi32> -> tile<16xi32>
// CHECK:   %[[A_IOTA:.*]] = iota : tile<16xi32>
// CHECK:   %[[A_ADD:.*]] = addi %[[A_MUL2]], %[[A_IOTA]] : tile<16xi32>
// CHECK:   %[[A_CMP:.*]] = cmpi less_than %[[A_ADD]], %[[C30]], unsigned : tile<16xi32> -> tile<16xi1>
// CHECK:   %[[A_CMP1:.*]] = reshape %[[A_CMP]] : tile<16xi1> -> tile<16x1xi1>
// CHECK:   %[[A_CMP2:.*]] = broadcast %[[A_CMP1]] : tile<16x1xi1> -> tile<16x8xi1>
// CHECK:   %[[LOOP:.*]] = for %[[IVAR2:.*]] in (%[[ZERO]] to %[[C8]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG2:.*]] = %[[IARG1]]) -> ([[TILE]])
// CHECK:     %[[B_MUL:.*]] = muli %[[IVAR2]], %[[C8]] : tile<i32>
// CHECK:     %[[B_MUL1:.*]] = reshape %[[B_MUL]] : tile<i32> -> tile<1xi32>
// CHECK:     %[[B_MUL2:.*]] = broadcast %[[B_MUL1]] : tile<1xi32> -> tile<8xi32>
// CHECK:     %[[B_IOTA:.*]] = iota : tile<8xi32>
// CHECK:     %[[B_ADD:.*]] = addi %[[B_MUL2]], %[[B_IOTA]] : tile<8xi32>
// CHECK:     %[[B_CMP:.*]] = cmpi less_than %[[B_ADD]], %[[C60]], unsigned : tile<8xi32> -> tile<8xi1>
// CHECK:     %[[B_CMP1:.*]] = reshape %[[B_CMP]] : tile<8xi1> -> tile<1x8xi1>
// CHECK:     %[[B_CMP2:.*]] = broadcast %[[B_CMP1]] : tile<1x8xi1> -> tile<16x8xi1>
// CHECK:     %[[MASK:.*]] = andi %[[A_CMP2]], %[[B_CMP2]] : tile<16x8xi1>
// CHECK:     %[[MASKL:.*]] = reshape %[[MASK]] : tile<16x8xi1> -> tile<1x16x8xi1>
// CHECK:     %[[MASKR:.*]] = reshape %[[MASK]] : tile<16x8xi1> -> tile<16x8x1xi1>
// CHECK-DAG:     %[[LHS_VIEW:.*]] = make_tensor_view %[[IN0]], shape = [64, 30, 60], strides = [2048, 64, 1]
// CHECK-DAG:     %[[LHS_PVIEW:.*]] = make_partition_view %[[LHS_VIEW]] : partition_view<tile=(8x16x8), tensor_view<64x30x60xf32, strides=[2048,64,1]>>
// CHECK-DAG:     %[[LHS:.*]], %{{.*}} = load_view_tko weak %[[LHS_PVIEW]][%[[IDX0]], %[[IVAR1]], %[[IVAR2]]]
// CHECK-DAG:     %[[RHS_VIEW:.*]] = make_tensor_view %[[IN1]], shape = [30, 60, 32], strides = [2048, 32, 1]
// CHECK-DAG:     %[[RHS_PVIEW:.*]] = make_partition_view %[[RHS_VIEW]] : partition_view<tile=(16x8x4), tensor_view<30x60x32xf32, strides=[2048,32,1]>>
// CHECK-DAG:     %[[RHS:.*]], %{{.*}} = load_view_tko weak %[[RHS_PVIEW]][%[[IVAR1]], %[[IVAR2]], %[[IDX1]]]
// CHECK-DAG:     %[[LHS_EXP:.*]] = exp %[[LHS]] : [[LHS_TILE]]
// CHECK-DAG:     %[[RHS_EXP:.*]] = exp %[[RHS]] : [[RHS_TILE]]
// CHECK-DAG:     %[[CMPLB:.*]] = broadcast %[[MASKL]] : tile<1x16x8xi1> -> tile<8x16x8xi1>
// CHECK-DAG:     %[[LHS_MASKED:.*]] = select %[[CMPLB]], %[[LHS_EXP]], %[[LHS_NEUTRAL]]
// CHECK-DAG:     %[[LHS_NORM:.*]] = reshape %[[LHS_MASKED]] : [[LHS_TILE]] -> tile<8x128xf32>
// CHECK-DAG:     %[[CMPRB:.*]] = broadcast %[[MASKR]] : tile<16x8x1xi1> -> tile<16x8x4xi1>
// CHECK-DAG:     %[[RHS_MASKED:.*]] = select %[[CMPRB]], %[[RHS_EXP]], %[[RHS_NEUTRAL]]
// CHECK-DAG:     %[[RHS_NORM:.*]] = reshape %[[RHS_MASKED]] : [[RHS_TILE]] -> tile<128x4xf32>
// CHECK:     %[[INNER:.*]] = mmaf %[[LHS_NORM]], %[[RHS_NORM]], %[[IARG2]]
// CHECK:     continue %[[INNER]] : [[TILE]]
// CHECK:   continue %[[LOOP]] : [[TILE]]
// CHECK: %[[OUT_VIEW:.*]] = make_tensor_view %[[OUT]], shape = [64, 32], strides = [32, 1]
// CHECK: %[[OUT_PVIEW:.*]] = make_partition_view %[[OUT_VIEW]] : partition_view<tile=(8x4), tensor_view<64x32xf32, strides=[32,1]>>
// CHECK: store_view_tko weak %[[RESULT]], %[[OUT_PVIEW]][%[[IDX0]], %[[IDX1]]]

nv_tensor_ir.graph @test_pad_mask_two_loops(
    %arg0: tensor<64x30x60xf32> {nv_tensor_ir.stride = "(2048,64,1)"},
    %arg1: tensor<30x60x32xf32> {nv_tensor_ir.stride = "(2048,32,1)"}
    ) -> (tensor<64x32xf32>)
    attributes {tile_size = array<i32: 8, 4>} {
  %lhs = exp %arg0 : tensor<64x30x60xf32>
  %rhs = exp %arg1 : tensor<30x60x32xf32>
  %lhs_norm = reshape %lhs : tensor<64x30x60xf32> -> tensor<64x1800xf32>
  %rhs_norm = reshape %rhs : tensor<30x60x32xf32> -> tensor<1800x32xf32>
  %out = matmul(%lhs_norm, %rhs_norm)
    : (tensor<64x1800xf32>, tensor<1800x32xf32>) -> tensor<64x32xf32>
  results %out : tensor<64x32xf32>
}

// -----

// CHECK-LABEL: entry @test_pad_mask_lhs_only(
// CHECK-SAME: %[[IN0:[^,]+]]: tile<ptr<f32>>, %[[IN1:[^,]+]]: tile<ptr<f32>>, %[[OUT:[^)]+]]: tile<ptr<f32>>)
// CHECK-DAG: %[[C50:.*]] = constant <i32: 50>
// CHECK-DAG: %[[LHS_NEUTRAL:.*]] = constant <f32: 0.000000e+00> : [[LHS_TILE:tile<8x64xf32>]]
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : tile<8x4xf32>
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[C8:.*]] = constant <i32: 8>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[IDX0:.*]] = remi %[[BLOCK]], %[[C8]] unsigned
// CHECK: %[[IDX1:.*]] = divi %[[BLOCK]], %[[C8]] unsigned
// CHECK: %[[IOTA:.*]] = iota : tile<64xi32>
// CHECK: %[[CMP:.*]] = cmpi less_than %[[IOTA]], %[[C50]], unsigned : tile<64xi32> -> tile<64xi1>
// CHECK: %[[CMPL:.*]] = reshape %[[CMP]] : tile<64xi1> -> tile<1x64xi1>
// CHECK: %[[LHS_VIEW:.*]] = make_tensor_view %[[IN0]], shape = [64, 50], strides = [50, 1]
// CHECK: %[[PVIEW_LHS:.*]] = make_partition_view %[[LHS_VIEW]] : partition_view<tile=(8x64), tensor_view<64x50xf32, strides=[50,1]>>
// CHECK: %[[LHS:.*]], %{{.*}} = load_view_tko weak %[[PVIEW_LHS]][%[[IDX0]], %[[ZERO]]]{{.*}} -> [[LHS_TILE]]
// CHECK-DAG: %[[RHS_VIEW:.*]] = make_tensor_view %[[IN1]], shape = [50, 32], strides = [32, 1]
// CHECK-DAG: %[[PVIEW_RHS:.*]] = make_partition_view %[[RHS_VIEW]] : partition_view<tile=(64x4), padding_value = zero, tensor_view<50x32xf32, strides=[32,1]>>
// CHECK-DAG: %[[RHS:.*]], %{{.*}} = load_view_tko weak %[[PVIEW_RHS]][%[[ZERO]], %[[IDX1]]]{{.*}} -> tile<64x4xf32>
// CHECK-DAG: %[[LHS_EXP:.*]] = exp %[[LHS]] : [[LHS_TILE]]
// CHECK: %[[CMPLB:.*]] = broadcast %[[CMPL]] : tile<1x64xi1> -> tile<8x64xi1>
// CHECK: %[[LHS_MASKED:.*]] = select %[[CMPLB]], %[[LHS_EXP]], %[[LHS_NEUTRAL]]
// CHECK: %[[RESULT:.*]] = mmaf %[[LHS_MASKED]], %[[RHS]], %[[ACCUM]]
// CHECK: %[[OUT_VIEW:.*]] = make_tensor_view %[[OUT]], shape = [64, 32], strides = [32, 1]
// CHECK: %[[OUT_PVIEW:.*]] = make_partition_view %[[OUT_VIEW]]
// CHECK: store_view_tko weak %[[RESULT]], %[[OUT_PVIEW]][%[[IDX0]], %[[IDX1]]]

nv_tensor_ir.graph @test_pad_mask_lhs_only(
    %arg0: tensor<64x50xf32>, %arg1: tensor<50x32xf32>
    ) -> (tensor<64x32xf32>)
    attributes {tile_size = array<i32: 8, 4>} {
  %lhs = exp %arg0 : tensor<64x50xf32>
  %out = matmul(%lhs, %arg1)
    : (tensor<64x50xf32>, tensor<50x32xf32>) -> tensor<64x32xf32>
  results %out : tensor<64x32xf32>
}

// -----

// CHECK-LABEL: entry @test_pad_mask_rhs_only(
// CHECK-SAME: %[[IN0:[^,]+]]: tile<ptr<f32>>, %[[IN1:[^,]+]]: tile<ptr<f32>>, %[[OUT:[^)]+]]: tile<ptr<f32>>)
// CHECK-DAG: %[[C50:.*]] = constant <i32: 50>
// CHECK-DAG: %[[RHS_NEUTRAL:.*]] = constant <f32: 0.000000e+00> : [[RHS_TILE:tile<64x4xf32>]]
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : tile<8x4xf32>
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[C8:.*]] = constant <i32: 8>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[IDX0:.*]] = remi %[[BLOCK]], %[[C8]] unsigned
// CHECK: %[[IDX1:.*]] = divi %[[BLOCK]], %[[C8]] unsigned
// CHECK: %[[IOTA:.*]] = iota : tile<64xi32>
// CHECK: %[[CMP:.*]] = cmpi less_than %[[IOTA]], %[[C50]], unsigned : tile<64xi32> -> tile<64xi1>
// CHECK: %[[CMPR:.*]] = reshape %[[CMP]] : tile<64xi1> -> tile<64x1xi1>
// CHECK: %[[LHS_VIEW:.*]] = make_tensor_view %[[IN0]], shape = [64, 50], strides = [50, 1]
// CHECK: %[[PVIEW_LHS:.*]] = make_partition_view %[[LHS_VIEW]] : partition_view<tile=(8x64), padding_value = zero, tensor_view<64x50xf32, strides=[50,1]>>
// CHECK: %[[LHS:.*]], %{{.*}} = load_view_tko weak %[[PVIEW_LHS]][%[[IDX0]], %[[ZERO]]]{{.*}} -> tile<8x64xf32>
// CHECK: %[[RHS_VIEW:.*]] = make_tensor_view %[[IN1]], shape = [50, 32], strides = [32, 1]
// CHECK: %[[PVIEW_RHS:.*]] = make_partition_view %[[RHS_VIEW]] : partition_view<tile=(64x4), tensor_view<50x32xf32, strides=[32,1]>>
// CHECK: %[[RHS:.*]], %{{.*}} = load_view_tko weak %[[PVIEW_RHS]][%[[ZERO]], %[[IDX1]]]{{.*}} -> [[RHS_TILE]]
// CHECK: %[[RHS_EXP:.*]] = exp %[[RHS]] : [[RHS_TILE]]
// CHECK: %[[CMPRB:.*]] = broadcast %[[CMPR]] : tile<64x1xi1> -> tile<64x4xi1>
// CHECK: %[[RHS_MASKED:.*]] = select %[[CMPRB]], %[[RHS_EXP]], %[[RHS_NEUTRAL]]
// CHECK: %[[RESULT:.*]] = mmaf %[[LHS]], %[[RHS_MASKED]], %[[ACCUM]]
// CHECK: %[[OUT_VIEW:.*]] = make_tensor_view %[[OUT]], shape = [64, 32], strides = [32, 1]
// CHECK: %[[OUT_PVIEW:.*]] = make_partition_view %[[OUT_VIEW]]
// CHECK: store_view_tko weak %[[RESULT]], %[[OUT_PVIEW]][%[[IDX0]], %[[IDX1]]]

nv_tensor_ir.graph @test_pad_mask_rhs_only(
    %arg0: tensor<64x50xf32>, %arg1: tensor<50x32xf32>
    ) -> (tensor<64x32xf32>)
    attributes {tile_size = array<i32: 8, 4>} {
  %rhs = exp %arg1 : tensor<50x32xf32>
  %out = matmul(%arg0, %rhs)
    : (tensor<64x50xf32>, tensor<50x32xf32>) -> tensor<64x32xf32>
  results %out : tensor<64x32xf32>
}
