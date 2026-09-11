// RUN: tensor_ir-opt -layout-propagation-pipeline="tile-size=1,8,64 persistence=static sm-count=148 occupancy=2" %s | FileCheck %s

module {
  // CHECK-LABEL: entry @dynamic_shape_static_persistence(
  // CHECK-SAME: %[[IN0:.*]]: tile<ptr<f32>>, %[[M0:.*]]: tile<i32>, %[[IN1:.*]]: tile<ptr<f32>>, %[[M1:.*]]: tile<i32>, %[[OUT:.*]]: tile<ptr<f32>>, %[[MR:.*]]: tile<i32>)
  // CHECK-DAG: %[[ONE:.*]] = constant <i64: 1> : tile<i64>
  // CHECK-DAG: %[[GRID_LIMIT:.*]] = constant <i64: 296> : tile<i64>
  // CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
  // CHECK: %[[BLOCK64:.*]] = exti %[[BLOCK]] unsigned : tile<i32> -> tile<i64>
  // CHECK: %[[OUT_VIEW:.*]] = make_tensor_view %[[OUT]], shape = [%[[MR]], 32, 64], strides = [8192, 128, 1]
  // CHECK: %[[OUT_PVIEW:.*]] = make_partition_view %[[OUT_VIEW]]
  // CHECK: %[[SPACE:.*]]:3 = get_index_space_shape %[[OUT_PVIEW]]
  // CHECK: %[[MUL0:.*]] = muli %[[ONE]], %[[SPACE]]#0 : tile<i64>
  // CHECK: %[[MUL1:.*]] = muli %[[MUL0]], %[[SPACE]]#1 : tile<i64>
  // CHECK: %[[TOTAL:.*]] = muli %[[MUL1]], %[[SPACE]]#2 : tile<i64>
  // CHECK: %[[GRID:.*]] = mini %[[TOTAL]], %[[GRID_LIMIT]] unsigned
  // CHECK: %[[STEP:.*]] = maxi %[[GRID]], %[[ONE]] unsigned
  // CHECK: for unsigned %[[TILE:.*]] in (%[[BLOCK64]] to %[[TOTAL]], step %[[STEP]]) : tile<i64>
  // CHECK: %[[IDX0:.*]] = remi %[[TILE]], %[[SPACE]]#0 unsigned : tile<i64>
  // CHECK: %[[TMP:.*]] = divi %[[TILE]], %[[SPACE]]#0 unsigned : tile<i64>
  // CHECK: %[[IDX1:.*]] = remi %[[TMP]], %[[SPACE]]#1 unsigned : tile<i64>
  // CHECK: %[[IDX2:.*]] = divi %[[TMP]], %[[SPACE]]#1 unsigned : tile<i64>
  // CHECK: %[[IN0_VIEW:.*]] = make_tensor_view %[[IN0]], shape = [%[[M0]], 32, 64], strides = [8192, 128, 1]
  // CHECK: %[[IN0_PVIEW:.*]] = make_partition_view %[[IN0_VIEW]]
  // CHECK: %[[TILE0:.*]], %{{.*}} = load_view_tko weak %[[IN0_PVIEW]][%[[IDX0]], %[[IDX1]], %[[IDX2]]] : {{.*}} -> tile<1x8x64xf32>, token
  // CHECK: %[[IN1_VIEW:.*]] = make_tensor_view %[[IN1]], shape = [%[[M1]], 32, 64], strides = [8192, 128, 1]
  // CHECK: %[[IN1_PVIEW:.*]] = make_partition_view %[[IN1_VIEW]]
  // CHECK: %[[TILE1:.*]], %{{.*}} = load_view_tko weak %[[IN1_PVIEW]][%[[IDX0]], %[[IDX1]], %[[IDX2]]] : {{.*}} -> tile<1x8x64xf32>, token
  // CHECK: %[[SUM:.*]] = addf %[[TILE0]], %[[TILE1]] : tile<1x8x64xf32>
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
