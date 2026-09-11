// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// ============================================================================
// Padding applied at tile load
// ============================================================================

// CHECK-LABEL: @test_pad_load_no_loop
// CHECK: %[[PVIEW:.*]] = make_partition_view %{{.*}} : partition_view<tile=(32x64), padding_value = zero
// CHECK: %[[INPUT:.*]], %{{.*}} = load_view_tko weak %[[PVIEW]]{{.*}} -> tile<32x64xf32>
// CHECK: %[[RESULT:.*]] = reduce %[[INPUT]] dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[VAL:.*]]: tile<f32>, %[[ACC:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[ACC]], %[[VAL]] : tile<f32>
// CHECK:   yield %[[RES]] : tile<f32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_pad_load_no_loop(
    %arg0: tensor<64x50xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0.0 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.addf %acc, %val : f32
    nv_tensor_ir.yield %0 : f32
  } : tensor<64x50xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_pad_load_one_loop
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[ONE:.*]] = constant <i32: 1>
// CHECK-DAG: %[[C4:.*]] = constant <i32: 4>
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : [[TILE:tile<32x128xf32>]]
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[LOOP:.*]] = for %[[IVAR:.*]] in (%[[ZERO]] to %[[C4]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG:.*]] = %[[ACCUM]]) -> ([[TILE]])
// CHECK:   %[[PVIEW:.*]] = make_partition_view %{{.*}} : partition_view<tile=(32x128), padding_value = zero
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko weak %[[PVIEW]][%[[BLOCK]], %[[IVAR]]]
// CHECK:   %[[INNER:.*]] = addf %[[IARG]], %[[ARG0]] : [[TILE]]
// CHECK:   continue %[[INNER]] : [[TILE]]
// CHECK: %[[RESULT:.*]] = reduce %[[LOOP]] dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[VAL:.*]]: tile<f32>, %[[ACC:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[ACC]], %[[VAL]] : tile<f32>
// CHECK:   yield %[[RES]] : tile<f32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_pad_load_one_loop(
    %arg0: tensor<64x500xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0.0 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.addf %acc, %val : f32
    nv_tensor_ir.yield %0 : f32
  } : tensor<64x500xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_pad_load_two_loops
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[ONE:.*]] = constant <i32: 1>
// CHECK-DAG: %[[C2:.*]] = constant <i32: 2>
// CHECK-DAG: %[[C8:.*]] = constant <i32: 8>
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 0.000000e+00> : [[TILE:tile<32x16x8xf32>]]
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[LOOP1:.*]] = for %[[IVAR1:.*]] in (%[[ZERO]] to %[[C2]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG1:.*]] = %[[ACCUM]]) -> ([[TILE]])
// CHECK:   %[[LOOP2:.*]] = for %[[IVAR2:.*]] in (%[[ZERO]] to %[[C8]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG2:.*]] = %[[IARG1]]) -> ([[TILE]])
// CHECK:     %[[PVIEW:.*]] = make_partition_view %{{.*}} : partition_view<tile=(32x16x8), padding_value = zero
// CHECK:     %[[ARG0:.*]], %{{.*}} = load_view_tko weak %[[PVIEW]][%[[BLOCK]], %[[IVAR1]], %[[IVAR2]]]
// CHECK:     %[[INNER:.*]] = addf %[[IARG2]], %[[ARG0]] : [[TILE]]
// CHECK:     continue %[[INNER]] : [[TILE]]
// CHECK:   continue %[[LOOP2]] : [[TILE]]
// CHECK: %[[MERGED:.*]] = reshape %[[LOOP1]] : [[TILE]] -> tile<32x128xf32>
// CHECK: %[[RESULT:.*]] = reduce %[[MERGED]] dim=1 identities=[0.000000e+00 : f32]
// CHECK:   (%[[VAL:.*]]: tile<f32>, %[[ACC:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[ACC]], %[[VAL]] : tile<f32>
// CHECK:   yield %[[RES]] : tile<f32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_pad_load_two_loops(
    %arg0: tensor<64x30x60xf32> {nv_tensor_ir.stride = "(2048,64,1)"}
    ) -> (tensor<64x1x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1, 2], identity = [0.0 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.addf %acc, %val : f32
    nv_tensor_ir.yield %0 : f32
  } : tensor<64x30x60xf32> -> tensor<64x1x1xf32>
  results %out : tensor<64x1x1xf32>
}

// -----

// CHECK-LABEL: @test_pad_load_max
// CHECK: %[[PVIEW:.*]] = make_partition_view %{{.*}} : partition_view<tile=(32x64), padding_value = neg_inf
// CHECK: %[[INPUT:.*]], %{{.*}} = load_view_tko weak %[[PVIEW]]{{.*}} -> tile<32x64xf32>
// CHECK: %[[RESULT:.*]] = reduce %[[INPUT]] dim=1 identities=[0xFF800000 : f32]
// CHECK:   (%[[VAL:.*]]: tile<f32>, %[[ACC:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = maxf %[[ACC]], %[[VAL]] : tile<f32>
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_pad_load_max(
    %arg0: tensor<64x50xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0xFF800000 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.maximumf %acc, %val : f32
    nv_tensor_ir.yield %0 : f32
  } : tensor<64x50xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_pad_load_min
// CHECK: %[[PVIEW:.*]] = make_partition_view %{{.*}} : partition_view<tile=(32x64), padding_value = pos_inf
// CHECK: %[[INPUT:.*]], %{{.*}} = load_view_tko weak %[[PVIEW]]{{.*}} -> tile<32x64xf32>
// CHECK: %[[RESULT:.*]] = reduce %[[INPUT]] dim=1 identities=[0x7F800000 : f32]
// CHECK:   (%[[VAL:.*]]: tile<f32>, %[[ACC:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = minf %[[ACC]], %[[VAL]] : tile<f32>
// CHECK:   yield %[[RES]]
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_pad_load_min(
    %arg0: tensor<64x50xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [0x7F800000 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.minimumf %acc, %val : f32
    nv_tensor_ir.yield %0 : f32
  } : tensor<64x50xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_pad_load_chain
// CHECK: %[[PVIEW:.*]] = make_partition_view %{{.*}} : partition_view<tile=(1x8x64), padding_value = zero
// CHECK: %[[INPUT:.*]], %{{.*}} = load_view_tko weak %[[PVIEW]]{{.*}} -> tile<1x8x64xbf16>
// CHECK: %[[BCAST:.*]] = broadcast %[[INPUT]] : tile<1x8x64xbf16> -> tile<4x8x64xbf16>
// CHECK: %[[CONVERT:.*]] = ftof %[[BCAST]] : tile<4x8x64xbf16> -> tile<4x8x64xf32>
// CHECK: %[[RESULT:.*]] = reduce %[[CONVERT]] dim=2 identities=[0.000000e+00 : f32]
// CHECK:   (%[[VAL:.*]]: tile<f32>, %[[ACC:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = addf %[[ACC]], %[[VAL]] : tile<f32>
// CHECK:   yield %[[RES]] : tile<f32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_pad_load_chain(
    %arg0: tensor<64x1x8xbf16>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 4, 8>} {
  %0 = convert %arg0 : tensor<64x1x8xbf16> -> tensor<64x1x8xf32>
  %1 = broadcast %0 : tensor<64x1x8xf32> -> tensor<64x8x8xf32>
  %2 = reshape %1 : tensor<64x8x8xf32> -> tensor<64x64xf32>
  %3 = slice %2 starts = [0, 0] limits = [50, 64] strides = [1, 1] : tensor<64x64xf32> -> tensor<50x64xf32>
  %4 = transpose %3 permutation = [1, 0] : tensor<50x64xf32> -> tensor<64x50xf32>
  %out = reduce_ud(%4) <dimensions = [1], identity = [0.0 : f32]> (%acc: f32, %val: f32) {
    %res = arith.addf %acc, %val : f32
    nv_tensor_ir.yield %res : f32
  } : tensor<64x50xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// ============================================================================
// Padding applied before the reduction
// ============================================================================

// CHECK-LABEL: @test_pad_mask_no_loop
// CHECK-NOT: padding_value
// CHECK-DAG: %[[C50:.*]] = constant <i32: 50>
// CHECK-DAG: %[[NEUTRAL:.*]] = constant <f32: 1.000000e+00> : [[TILE:tile<32x64xf32>]]
// CHECK: %[[IOTA:.*]] = iota : tile<64xi32>
// CHECK: %[[CMP:.*]] = cmpi less_than %[[IOTA]], %[[C50]], unsigned : tile<64xi32> -> tile<64xi1>
// CHECK: %[[CMP1:.*]] = reshape %[[CMP]] : tile<64xi1> -> tile<1x64xi1>
// CHECK: %[[INPUT:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[CMP2:.*]] = broadcast %[[CMP1]] : tile<1x64xi1> -> tile<32x64xi1>
// CHECK: %[[MASKED:.*]] = select %[[CMP2]], %[[INPUT]], %[[NEUTRAL]] : tile<32x64xi1>, [[TILE]]
// CHECK: %[[RESULT:.*]] = reduce %[[MASKED]] dim=1 identities=[1.000000e+00 : f32]
// CHECK:   (%[[VAL:.*]]: tile<f32>, %[[ACC:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = mulf %[[ACC]], %[[VAL]] : tile<f32>
// CHECK:   yield %[[RES]] : tile<f32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_pad_mask_no_loop(
    %arg0: tensor<64x50xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [1.0 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.mulf %acc, %val : f32
    nv_tensor_ir.yield %0 : f32
  } : tensor<64x50xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_pad_mask_one_loop
// CHECK-NOT: padding_value
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[ONE:.*]] = constant <i32: 1>
// CHECK-DAG: %[[C4:.*]] = constant <i32: 4>
// CHECK-DAG: %[[C128:.*]] = constant <i32: 128>
// CHECK-DAG: %[[C500:.*]] = constant <i32: 500>
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 1.000000e+00> : [[TILE:tile<32x128xf32>]]
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[LOOP:.*]] = for %[[IVAR:.*]] in (%[[ZERO]] to %[[C4]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG:.*]] = %[[ACCUM]]) -> ([[TILE]])
// CHECK:   %[[MUL:.*]] = muli %[[IVAR]], %[[C128]] : tile<i32>
// CHECK:   %[[MUL1:.*]] = reshape %[[MUL]] : tile<i32> -> tile<1xi32>
// CHECK:   %[[MUL2:.*]] = broadcast %[[MUL1]] : tile<1xi32> -> tile<128xi32>
// CHECK:   %[[IOTA:.*]] = iota : tile<128xi32>
// CHECK:   %[[ADD:.*]] = addi %[[MUL2]], %[[IOTA]] : tile<128xi32>
// CHECK:   %[[CMP:.*]] = cmpi less_than %[[ADD]], %[[C500]], unsigned : tile<128xi32> -> tile<128xi1>
// CHECK:   %[[CMP1:.*]] = reshape %[[CMP]] : tile<128xi1> -> tile<1x128xi1>
// CHECK:   %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}}[%[[BLOCK]], %[[IVAR]]]
// CHECK:   %[[CMP2:.*]] = broadcast %[[CMP1]] : tile<1x128xi1> -> tile<32x128xi1>
// CHECK:   %[[MASKED:.*]] = select %[[CMP2]], %[[ARG0]], %[[ACCUM]] : tile<32x128xi1>, [[TILE]]
// CHECK:   %[[INNER:.*]] = mulf %[[IARG]], %[[MASKED]] : [[TILE]]
// CHECK:   continue %[[INNER]] : [[TILE]]
// CHECK: %[[RESULT:.*]] = reduce %[[LOOP]] dim=1 identities=[1.000000e+00 : f32]
// CHECK:   (%[[VAL:.*]]: tile<f32>, %[[ACC:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = mulf %[[ACC]], %[[VAL]] : tile<f32>
// CHECK:   yield %[[RES]] : tile<f32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_pad_mask_one_loop(
    %arg0: tensor<64x500xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1], identity = [1.0 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.mulf %acc, %val : f32
    nv_tensor_ir.yield %0 : f32
  } : tensor<64x500xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @test_pad_mask_two_loops
// CHECK-NOT: padding_value
// CHECK-DAG: %[[ZERO:.*]] = constant <i32: 0>
// CHECK-DAG: %[[ONE:.*]] = constant <i32: 1>
// CHECK-DAG: %[[C2:.*]] = constant <i32: 2>
// CHECK-DAG: %[[C8:.*]] = constant <i32: 8>
// CHECK-DAG: %[[C16:.*]] = constant <i32: 16>
// CHECK-DAG: %[[C30:.*]] = constant <i32: 30>
// CHECK-DAG: %[[C60:.*]] = constant <i32: 60>
// CHECK-DAG: %[[ACCUM:.*]] = constant <f32: 1.000000e+00> : [[TILE:tile<32x16x8xf32>]]
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
// CHECK: %[[LOOP1:.*]] = for %[[IVAR1:.*]] in (%[[ZERO]] to %[[C2]], step %[[ONE]])
// CHECK-SAME: iter_values(%[[IARG1:.*]] = %[[ACCUM]]) -> ([[TILE]])
// CHECK:   %[[A_MUL:.*]] = muli %[[IVAR1]], %[[C16]] : tile<i32>
// CHECK:   %[[A_MUL1:.*]] = reshape %[[A_MUL]] : tile<i32> -> tile<1xi32>
// CHECK:   %[[A_MUL2:.*]] = broadcast %[[A_MUL1]] : tile<1xi32> -> tile<16xi32>
// CHECK:   %[[A_IOTA:.*]] = iota : tile<16xi32>
// CHECK:   %[[A_ADD:.*]] = addi %[[A_MUL2]], %[[A_IOTA]] : tile<16xi32>
// CHECK:   %[[A_CMP:.*]] = cmpi less_than %[[A_ADD]], %[[C30]], unsigned : tile<16xi32> -> tile<16xi1>
// CHECK:   %[[A_CMP1:.*]] = reshape %[[A_CMP]] : tile<16xi1> -> tile<16x1xi1>
// CHECK:   %[[A_CMP2:.*]] = broadcast %[[A_CMP1]] : tile<16x1xi1> -> tile<16x8xi1>
// CHECK:   %[[LOOP2:.*]] = for %[[IVAR2:.*]] in (%[[ZERO]] to %[[C8]], step %[[ONE]])
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
// CHECK:     %[[MASK1:.*]] = reshape %[[MASK]] : tile<16x8xi1> -> tile<1x16x8xi1>
// CHECK:     %[[ARG0:.*]], %{{.*}} = load_view_tko {{.*}}[%[[BLOCK]], %[[IVAR1]], %[[IVAR2]]]
// CHECK:     %[[MASK2:.*]] = broadcast %[[MASK1]] : tile<1x16x8xi1> -> tile<32x16x8xi1>
// CHECK:     %[[MASKED:.*]] = select %[[MASK2]], %[[ARG0]], %[[ACCUM]] : tile<32x16x8xi1>, [[TILE]]
// CHECK:     %[[INNER:.*]] = mulf %[[IARG2]], %[[MASKED]] : [[TILE]]
// CHECK:     continue %[[INNER]] : [[TILE]]
// CHECK:   continue %[[LOOP2]] : [[TILE]]
// CHECK: %[[MERGED:.*]] = reshape %[[LOOP1]] : [[TILE]] -> tile<32x128xf32>
// CHECK: %[[RESULT:.*]] = reduce %[[MERGED]] dim=1 identities=[1.000000e+00 : f32]
// CHECK:   (%[[VAL:.*]]: tile<f32>, %[[ACC:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = mulf %[[ACC]], %[[VAL]] : tile<f32>
// CHECK:   yield %[[RES]] : tile<f32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_pad_mask_two_loops(
    %arg0: tensor<64x30x60xf32> {nv_tensor_ir.stride = "(2048,64,1)"}
    ) -> (tensor<64x1x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %out = reduce_ud(%arg0) <dimensions = [1, 2], identity = [1.0 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.mulf %acc, %val : f32
    nv_tensor_ir.yield %0 : f32
  } : tensor<64x30x60xf32> -> tensor<64x1x1xf32>
  results %out : tensor<64x1x1xf32>
}

// -----

// CHECK-LABEL: @test_pad_mask_chain
// CHECK-NOT: padding_value
// CHECK-DAG: %[[C50:.*]] = constant <i32: 50>
// CHECK-DAG: %[[NEUTRAL:.*]] = constant <f32: 0xFF800000> : [[TILE:tile<32x64xf32>]]
// CHECK: %[[IOTA:.*]] = iota : tile<64xi32>
// CHECK: %[[CMP:.*]] = cmpi less_than %[[IOTA]], %[[C50]], unsigned : tile<64xi32> -> tile<64xi1>
// CHECK: %[[CMP1:.*]] = reshape %[[CMP]] : tile<64xi1> -> tile<1x64xi1>
// CHECK: %[[INPUT:.*]], %{{.*}} = load_view_tko {{.*}} -> [[TILE]]
// CHECK: %[[ABS:.*]] = absf %[[INPUT]] : [[TILE]]
// CHECK: %[[CMP2:.*]] = broadcast %[[CMP1]] : tile<1x64xi1> -> tile<32x64xi1>
// CHECK: %[[MASKED:.*]] = select %[[CMP2]], %[[ABS]], %[[NEUTRAL]] : tile<32x64xi1>, [[TILE]]
// CHECK: %[[RESULT:.*]] = reduce %[[MASKED]] dim=1 identities=[0xFF800000 : f32]
// CHECK:   (%[[VAL:.*]]: tile<f32>, %[[ACC:.*]]: tile<f32>)
// CHECK:   %[[RES:.*]] = maxf %[[ACC]], %[[VAL]] : tile<f32>
// CHECK:   yield %[[RES]] : tile<f32>
// CHECK: store_view_tko weak %[[RESULT]]

nv_tensor_ir.graph @test_pad_mask_chain(
    %arg0: tensor<64x50xf32>
    ) -> (tensor<64x1xf32>)
    attributes {tile_size = array<i32: 32>} {
  %abs = abs %arg0 : tensor<64x50xf32>
  %out = reduce_ud(%abs) <dimensions = [1], identity = [0xFF800000 : f32]> (%acc: f32, %val: f32) {
    %0 = arith.maximumf %acc, %val : f32
    nv_tensor_ir.yield %0 : f32
  } : tensor<64x50xf32> -> tensor<64x1xf32>
  results %out : tensor<64x1xf32>
}
