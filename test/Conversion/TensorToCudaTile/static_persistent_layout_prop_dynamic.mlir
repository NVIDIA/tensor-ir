// RUN: tensor_ir-opt -layout-propagation-pipeline="tile-size=1,8,64 persistence=static sm-count=148 occupancy=2" %s | FileCheck %s

module {
  // CHECK-LABEL: entry @dynamic_shape_static_persistence(
  // CHECK-SAME: %[[IN0:[^,]*]]: tile<ptr<f32>>, %[[M0:[^,]*]]: tile<i32>,
  // CHECK-SAME: %[[IN1:[^,]*]]: tile<ptr<f32>>, %[[M1:[^,]*]]: tile<i32>,
  // CHECK-SAME: %[[OUT:[^,]*]]: tile<ptr<f32>>, %[[MR:[^)]*]]: tile<i32>)
  // CHECK-SAME: optimization_hints=<default = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 2}>
  // CHECK-DAG: %[[FOUR:.*]] = constant <i32: 4> : tile<i32>
  // CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
  // CHECK: %[[GRID:.*]], %{{.*}}, %{{.*}} = get_num_tile_blocks : tile<i32>
  // CHECK: %[[TOTAL:.*]] = muli %[[MR]], %[[FOUR]] : tile<i32>
  // CHECK: for %[[TILE:.*]] in (%[[BLOCK]] to %[[TOTAL]], step %[[GRID]]) : tile<i32>
  // CHECK: %[[IDX0:.*]] = remi %[[TILE]], %[[MR]] unsigned : tile<i32>
  // CHECK: %[[REST:.*]] = divi %[[TILE]], %[[MR]] unsigned : tile<i32>
  // CHECK: %[[IDX1:.*]] = remi %[[REST]], %[[FOUR]] unsigned : tile<i32>
  // CHECK: %[[IDX2:.*]] = divi %[[REST]], %[[FOUR]] unsigned : tile<i32>
  // CHECK: %[[IN0_VIEW:.*]] = make_tensor_view %[[IN0]], shape = [%[[M0]], 32, 64], strides = [8192, 128, 1]
  // CHECK: %[[IN0_PVIEW:.*]] = make_partition_view %[[IN0_VIEW]]
  // CHECK: %[[TILE0:.*]], %{{.*}} = load_view_tko weak %[[IN0_PVIEW]][%[[IDX0]], %[[IDX1]], %[[IDX2]]]
  // CHECK: %[[IN1_VIEW:.*]] = make_tensor_view %[[IN1]], shape = [%[[M1]], 32, 64], strides = [8192, 128, 1]
  // CHECK: %[[IN1_PVIEW:.*]] = make_partition_view %[[IN1_VIEW]]
  // CHECK: %[[TILE1:.*]], %{{.*}} = load_view_tko weak %[[IN1_PVIEW]][%[[IDX0]], %[[IDX1]], %[[IDX2]]]
  // CHECK: %[[SUM:.*]] = addf %[[TILE0]], %[[TILE1]]
  // CHECK: %[[OUT_VIEW:.*]] = make_tensor_view %[[OUT]], shape = [%[[MR]], 32, 64], strides = [8192, 128, 1]
  // CHECK: %[[OUT_PVIEW:.*]] = make_partition_view %[[OUT_VIEW]]
  // CHECK: store_view_tko weak %[[SUM]], %[[OUT_PVIEW]][%[[IDX0]], %[[IDX1]], %[[IDX2]]]
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  nv_tensor_ir.graph @dynamic_shape_static_persistence(
      %arg0: tensor<?x32x64xf32> {nv_tensor_ir.stride = "(8192,128,1)"},
      %arg1: tensor<?x32x64xf32> {nv_tensor_ir.stride = "(8192,128,1)"})
      -> (tensor<?x32x64xf32> {nv_tensor_ir.stride = "(8192,128,1)"}) {
    %result = add %arg0, %arg1 : tensor<?x32x64xf32>
    results %result : tensor<?x32x64xf32>
  }
}
