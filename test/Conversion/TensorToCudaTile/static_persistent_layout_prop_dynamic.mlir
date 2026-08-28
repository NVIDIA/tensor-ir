// RUN: tensor_ir-opt -layout-propagation-pipeline="tile-size=1,8,64 persistence=static sm-count=148 occupancy=2" %s | FileCheck %s

module {
  // CHECK-LABEL: entry @dynamic_shape_static_persistence(
  // CHECK-DAG: %[[ONE:.*]] = constant <i64: 1> : tile<i64>
  // CHECK-DAG: %[[GRID_LIMIT:.*]] = constant <i64: 296> : tile<i64>
  // CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
  // CHECK: %[[BLOCK64:.*]] = exti %[[BLOCK]] unsigned : tile<i32> -> tile<i64>
  // CHECK: get_index_space_shape
  // CHECK: muli
  // CHECK: muli
  // CHECK: %[[TOTAL:.*]] = muli
  // CHECK: %[[GRID:.*]] = mini %[[TOTAL]], %[[GRID_LIMIT]] unsigned
  // CHECK: %[[STEP:.*]] = maxi %[[GRID]], %[[ONE]] unsigned
  // CHECK: for unsigned %[[TILE:.*]] in (%[[BLOCK64]] to %[[TOTAL]], step %[[STEP]]) : tile<i64>
  // CHECK: load_view_tko
  // CHECK: load_view_tko
  // CHECK: addf
  // CHECK: store_view_tko
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
