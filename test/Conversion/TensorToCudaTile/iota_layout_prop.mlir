// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// CHECK-LABEL: entry @test_iota_1d
// CHECK-SAME: (%[[OUT:.*]]: tile<ptr<f32>>
// CHECK: %[[C64:.*]] = constant <i32: 64> : tile<i32>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[IOTA:.*]] = iota : tile<64xi32>
// CHECK: %[[OFF:.*]] = muli %[[BLOCK]], %[[C64]] : tile<i32>
// CHECK: %[[OFF_SHAPE:.*]] = reshape %[[OFF]] : tile<i32> -> tile<1xi32>
// CHECK: %[[BCAST:.*]] = broadcast %[[OFF_SHAPE]] : tile<1xi32> -> tile<64xi32>
// CHECK: %[[IDX:.*]] = addi %[[IOTA]], %[[BCAST]] : tile<64xi32>
// CHECK: %[[F32:.*]] = itof %[[IDX]] unsigned : tile<64xi32> -> tile<64xf32>
// CHECK: %[[TVIEW:.*]] = make_tensor_view %[[OUT]], shape = [128], strides = [1]
// CHECK: %[[PVIEW:.*]] = make_partition_view %[[TVIEW]]
// CHECK: store_view_tko weak %[[F32]], %[[PVIEW]][%[[BLOCK]]]
nv_tensor_ir.graph @test_iota_1d() -> tensor<128xf32>
    attributes {tile_size = array<i32: 64>} {
  %out = iota dimension = 0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: entry @test_iota_2d_dim0
// CHECK-SAME: (%[[OUT:.*]]: tile<ptr<f32>>
// CHECK: %[[C2:.*]] = constant <i32: 2> : tile<i32>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[ROW:.*]] = remi %[[BLOCK]], %[[C2]] unsigned : tile<i32>
// CHECK: %[[COL:.*]] = divi %[[BLOCK]], %[[C2]] unsigned : tile<i32>
// CHECK: %[[IOTA:.*]] = iota : tile<2xi32>
// CHECK: %[[R0:.*]] = reshape %[[IOTA]] : tile<2xi32> -> tile<2x1xi32>
// CHECK: %[[B0:.*]] = broadcast %[[R0]] : tile<2x1xi32> -> tile<2x16xi32>
// CHECK: %[[OFF:.*]] = muli %[[ROW]], %[[C2]] : tile<i32>
// CHECK: %[[OFF_SHAPE:.*]] = reshape %[[OFF]] : tile<i32> -> tile<1x1xi32>
// CHECK: %[[BASE:.*]] = broadcast %[[OFF_SHAPE]] : tile<1x1xi32> -> tile<2x16xi32>
// CHECK: %[[IDX:.*]] = addi %[[B0]], %[[BASE]] : tile<2x16xi32>
// CHECK: %[[F32:.*]] = itof %[[IDX]] unsigned : tile<2x16xi32> -> tile<2x16xf32>
// CHECK: %[[TVIEW:.*]] = make_tensor_view %[[OUT]], shape = [4, 16], strides = [16, 1]
// CHECK: %[[PVIEW:.*]] = make_partition_view %[[TVIEW]]
// CHECK: store_view_tko weak %[[F32]], %[[PVIEW]][%[[ROW]], %[[COL]]]
nv_tensor_ir.graph @test_iota_2d_dim0() -> tensor<4x16xf32>
    attributes {tile_size = array<i32: 2, 16>} {
  %out = iota dimension = 0 : tensor<4x16xf32>
  results %out : tensor<4x16xf32>
}

// -----

// CHECK-LABEL: entry @test_iota_2d_dim1
// CHECK-SAME: (%[[OUT:.*]]: tile<ptr<f32>>
// CHECK: %[[C16:.*]] = constant <i32: 16> : tile<i32>
// CHECK: %[[ZERO:.*]] = constant <i32: 0> : tile<i32>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[IOTA:.*]] = iota : tile<16xi32>
// CHECK: %[[R0:.*]] = reshape %[[IOTA]] : tile<16xi32> -> tile<1x16xi32>
// CHECK: %[[B0:.*]] = broadcast %[[R0]] : tile<1x16xi32> -> tile<4x16xi32>
// CHECK: %[[OFF:.*]] = muli %[[BLOCK]], %[[C16]] : tile<i32>
// CHECK: %[[OFF_SHAPE:.*]] = reshape %[[OFF]] : tile<i32> -> tile<1x1xi32>
// CHECK: %[[BASE:.*]] = broadcast %[[OFF_SHAPE]] : tile<1x1xi32> -> tile<4x16xi32>
// CHECK: %[[IDX:.*]] = addi %[[B0]], %[[BASE]] : tile<4x16xi32>
// CHECK: %[[F32:.*]] = itof %[[IDX]] unsigned : tile<4x16xi32> -> tile<4x16xf32>
// CHECK: %[[TVIEW:.*]] = make_tensor_view %[[OUT]], shape = [4, 16], strides = [16, 1]
// CHECK: %[[PVIEW:.*]] = make_partition_view %[[TVIEW]]
// CHECK: store_view_tko weak %[[F32]], %[[PVIEW]][%[[ZERO]], %[[BLOCK]]]
nv_tensor_ir.graph @test_iota_2d_dim1() -> tensor<4x16xf32>
    attributes {tile_size = array<i32: 4, 16>} {
  %out = iota dimension = 1 : tensor<4x16xf32>
  results %out : tensor<4x16xf32>
}

// -----

// CHECK-LABEL: entry @test_iota_add_input
// CHECK-SAME: (%[[IN:.*]]: tile<ptr<f32>>, %[[OUT:.*]]: tile<ptr<f32>>
// CHECK: %[[C64:.*]] = constant <i32: 64> : tile<i32>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[IN_VIEW:.*]] = make_tensor_view %[[IN]], shape = [128], strides = [1]
// CHECK: %[[IN_PVIEW:.*]] = make_partition_view %[[IN_VIEW]]
// CHECK: %[[LOAD:.*]], %{{.*}} = load_view_tko weak %[[IN_PVIEW]][%[[BLOCK]]]
// CHECK: %[[IOTA:.*]] = iota : tile<64xi32>
// CHECK: %[[OFF:.*]] = muli %[[BLOCK]], %[[C64]] : tile<i32>
// CHECK: %[[OFF_SHAPE:.*]] = reshape %[[OFF]] : tile<i32> -> tile<1xi32>
// CHECK: %[[BCAST:.*]] = broadcast %[[OFF_SHAPE]] : tile<1xi32> -> tile<64xi32>
// CHECK: %[[IDX:.*]] = addi %[[IOTA]], %[[BCAST]] : tile<64xi32>
// CHECK: %[[F32:.*]] = itof %[[IDX]] unsigned : tile<64xi32> -> tile<64xf32>
// CHECK: %[[SUM:.*]] = addf %[[LOAD]], %[[F32]]
// CHECK: %[[OUT_VIEW:.*]] = make_tensor_view %[[OUT]], shape = [128], strides = [1]
// CHECK: %[[OUT_PVIEW:.*]] = make_partition_view %[[OUT_VIEW]]
// CHECK: store_view_tko weak %[[SUM]], %[[OUT_PVIEW]][%[[BLOCK]]]
nv_tensor_ir.graph @test_iota_add_input(%arg0: tensor<128xf32>)
    -> tensor<128xf32>
    attributes {tile_size = array<i32: 64>} {
  %idx = iota dimension = 0 : tensor<128xf32>
  %out = add %arg0, %idx : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: entry @test_iota_i32
// CHECK-SAME: (%[[OUT:.*]]: tile<ptr<i32>>
// CHECK: %[[C32:.*]] = constant <i32: 32> : tile<i32>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[IOTA:.*]] = iota : tile<32xi32>
// CHECK: %[[OFF:.*]] = muli %[[BLOCK]], %[[C32]] : tile<i32>
// CHECK: %[[OFF_SHAPE:.*]] = reshape %[[OFF]] : tile<i32> -> tile<1xi32>
// CHECK: %[[BCAST:.*]] = broadcast %[[OFF_SHAPE]] : tile<1xi32> -> tile<32xi32>
// CHECK: %[[IDX:.*]] = addi %[[IOTA]], %[[BCAST]] : tile<32xi32>
// CHECK-NOT: itof
// CHECK: %[[TVIEW:.*]] = make_tensor_view %[[OUT]], shape = [64], strides = [1]
// CHECK: %[[PVIEW:.*]] = make_partition_view %[[TVIEW]]
// CHECK: store_view_tko weak %[[IDX]], %[[PVIEW]][%[[BLOCK]]]
nv_tensor_ir.graph @test_iota_i32() -> (tensor<64xsi32>)
    attributes {tile_size = array<i32: 32>} {
  %out = iota dimension = 0 : tensor<64xsi32>
  results %out : tensor<64xsi32>
}

// -----

// iota + reshape + broadcast: iteration-space layout strides (16,0,1) are
// lowered from one local iota tile and two ranked broadcasts.
// CHECK-LABEL: entry @test_iota_propagation
// CHECK-SAME: (%[[OUT:.*]]: tile<ptr<f32>>
// CHECK: %[[TILE_SIZE:.*]] = constant <i32: 4> : tile<i32>
// CHECK: %[[ROW_STRIDE:.*]] = constant <i32: 64> : tile<i32>
// CHECK: %[[IOTA_STRIDE:.*]] = constant <i32: 16> : tile<4x4x4xi32>
// CHECK: %[[ZERO:.*]] = constant <i32: 0> : tile<i32>
// CHECK: %[[C2:.*]] = constant <i32: 2> : tile<i32>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[ROW:.*]] = remi %[[BLOCK]], %[[C2]] unsigned : tile<i32>
// CHECK: %[[COL:.*]] = divi %[[BLOCK]], %[[C2]] unsigned : tile<i32>
// CHECK: %[[IOTA:.*]] = iota : tile<4xi32>
// CHECK: %[[R0:.*]] = reshape %[[IOTA]] : tile<4xi32> -> tile<4x1x1xi32>
// CHECK: %[[B0:.*]] = broadcast %[[R0]] : tile<4x1x1xi32> -> tile<4x4x4xi32>
// CHECK: %[[SCALED0:.*]] = muli %[[B0]], %[[IOTA_STRIDE]] : tile<4x4x4xi32>
// CHECK: %[[ROW_OFF:.*]] = muli %[[ROW]], %[[ROW_STRIDE]] : tile<i32>
// CHECK: %[[R1:.*]] = reshape %[[IOTA]] : tile<4xi32> -> tile<1x1x4xi32>
// CHECK: %[[B1:.*]] = broadcast %[[R1]] : tile<1x1x4xi32> -> tile<4x4x4xi32>
// CHECK: %[[IDX_BASE:.*]] = addi %[[SCALED0]], %[[B1]] : tile<4x4x4xi32>
// CHECK: %[[COL_OFF:.*]] = muli %[[COL]], %[[TILE_SIZE]] : tile<i32>
// CHECK: %[[OFF:.*]] = addi %[[ROW_OFF]], %[[COL_OFF]] : tile<i32>
// CHECK: %[[R2:.*]] = reshape %[[OFF]] : tile<i32> -> tile<1x1x1xi32>
// CHECK: %[[B2:.*]] = broadcast %[[R2]] : tile<1x1x1xi32> -> tile<4x4x4xi32>
// CHECK: %[[IDX:.*]] = addi %[[IDX_BASE]], %[[B2]] : tile<4x4x4xi32>
// CHECK: %[[F32:.*]] = itof %[[IDX]] unsigned : tile<4x4x4xi32> -> tile<4x4x4xf32>
// CHECK: %[[TVIEW:.*]] = make_tensor_view %[[OUT]], shape = [8, 4, 16], strides = [64, 16, 1]
// CHECK: %[[PVIEW:.*]] = make_partition_view %[[TVIEW]]
// CHECK: store_view_tko weak %[[F32]], %[[PVIEW]][%[[ROW]], %[[ZERO]], %[[COL]]]
nv_tensor_ir.graph @test_iota_propagation() -> (tensor<8x4x16xf32>)
    attributes {tile_size = array<i32: 4, 4, 4>} {
  %out = iota dimension = 0 : tensor<128xf32>
  %rs = reshape %out : tensor<128xf32> -> tensor<8x1x16xf32>
  %bc = broadcast %rs : tensor<8x1x16xf32> -> tensor<8x4x16xf32>
  results %bc : tensor<8x4x16xf32>
}
