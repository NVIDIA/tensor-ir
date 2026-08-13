// RUN: tensor_ir-opt -layout-propagation-pipeline -split-input-file %s | FileCheck %s

// CHECK-LABEL: entry @test_iota_1d
// CHECK-SAME: (%[[OUT:.*]]: tile<ptr<f32>>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[OFF:.*]] = muli %[[BLOCK]]
// CHECK: %[[BCAST:.*]] = broadcast %{{.*}} : tile<1xi32> -> tile<64xi32>
// CHECK: %[[IOTA:.*]] = iota : tile<64xi32>
// CHECK: %[[IDX:.*]] = addi %[[BCAST]], %[[IOTA]] : tile<64xi32>
// CHECK: %[[F32:.*]] = itof %[[IDX]]
// CHECK: store_view_tko weak %[[F32]], %{{.*}}[%[[BLOCK]]]
nv_tensor_ir.graph @test_iota_1d() -> tensor<128xf32>
    attributes {tile_size = array<i32: 64>} {
  %out = iota dimension = 0 : tensor<128xf32>
  results %out : tensor<128xf32>
}

// -----

// CHECK-LABEL: entry @test_iota_2d_dim0
// CHECK-SAME: (%[[OUT:.*]]: tile<ptr<f32>>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[IOTA:.*]] = iota : tile<2xi32>
// CHECK: %[[R0:.*]] = reshape %[[IOTA]] : tile<2xi32> -> tile<2x1xi32>
// CHECK: %[[B0:.*]] = broadcast %[[R0]] : tile<2x1xi32> -> tile<2x16xi32>
// CHECK: %[[IDX:.*]] = addi %{{.*}}, %[[B0]] : tile<2x16xi32>
// CHECK: %[[F32:.*]] = itof %[[IDX]]
// CHECK: store_view_tko weak %[[F32]], %{{.*}}[%[[BLOCK]]
nv_tensor_ir.graph @test_iota_2d_dim0() -> tensor<4x16xf32>
    attributes {tile_size = array<i32: 2, 16>} {
  %out = iota dimension = 0 : tensor<4x16xf32>
  results %out : tensor<4x16xf32>
}

// -----

// CHECK-LABEL: entry @test_iota_2d_dim1
// CHECK-SAME: (%[[OUT:.*]]: tile<ptr<f32>>
// CHECK: %[[IOTA:.*]] = iota : tile<16xi32>
// CHECK: %[[R0:.*]] = reshape %[[IOTA]] : tile<16xi32> -> tile<1x16xi32>
// CHECK: %[[B0:.*]] = broadcast %[[R0]] : tile<1x16xi32> -> tile<4x16xi32>
// CHECK: %[[IDX:.*]] = addi %{{.*}}, %[[B0]] : tile<4x16xi32>
// CHECK: %[[F32:.*]] = itof %[[IDX]]
// CHECK: store_view_tko weak %[[F32]]
nv_tensor_ir.graph @test_iota_2d_dim1() -> tensor<4x16xf32>
    attributes {tile_size = array<i32: 4, 16>} {
  %out = iota dimension = 1 : tensor<4x16xf32>
  results %out : tensor<4x16xf32>
}

// -----

// CHECK-LABEL: entry @test_iota_add_input
// CHECK-SAME: (%[[IN:.*]]: tile<ptr<f32>>, %[[OUT:.*]]: tile<ptr<f32>>
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[LOAD:.*]], %{{.*}} = load_view_tko weak %{{.*}}[%[[BLOCK]]]
// CHECK: %[[IOTA:.*]] = iota : tile<64xi32>
// CHECK: %[[IDX:.*]] = addi %{{.*}}, %{{.*}} : tile<64xi32>
// CHECK: %[[F32:.*]] = itof %[[IDX]]
// CHECK: %[[SUM:.*]] = addf %[[LOAD]], %[[F32]]
// CHECK: store_view_tko weak %[[SUM]], %{{.*}}[%[[BLOCK]]]
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
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[IOTA:.*]] = iota : tile<32xi32>
// CHECK: %[[IDX:.*]] = addi %{{.*}}, %{{.*}} : tile<32xi32>
// CHECK-NOT: itof
// CHECK: store_view_tko weak %[[IDX]], %{{.*}}[%[[BLOCK]]]
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
// CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// CHECK: %[[IOTA:.*]] = iota : tile<4xi32>
// CHECK-DAG: %[[SCALED0:.*]] = muli %[[IOTA]]
// CHECK-DAG: %[[R0:.*]] = reshape %[[SCALED0]] : tile<4xi32> -> tile<4x1x1xi32>
// CHECK-DAG: %[[B0:.*]] = broadcast %[[R0]] : tile<4x1x1xi32> -> tile<4x4x4xi32>
// CHECK-DAG: %[[R1:.*]] = reshape %[[IOTA]] : tile<4xi32> -> tile<1x1x4xi32>
// CHECK-DAG: %[[B1:.*]] = broadcast %[[R1]] : tile<1x1x4xi32> -> tile<4x4x4xi32>
// CHECK: %[[IDX:.*]] = addi %{{.*}}, %[[B1]] : tile<4x4x4xi32>
// CHECK: %[[F32:.*]] = itof %[[IDX]]
// CHECK: store_view_tko weak %[[F32]]
nv_tensor_ir.graph @test_iota_propagation() -> (tensor<8x4x16xf32>)
    attributes {tile_size = array<i32: 4, 4, 4>} {
  %out = iota dimension = 0 : tensor<128xf32>
  %rs = reshape %out : tensor<128xf32> -> tensor<8x1x16xf32>
  %bc = broadcast %rs : tensor<8x1x16xf32> -> tensor<8x4x16xf32>
  results %bc : tensor<8x4x16xf32>
}
