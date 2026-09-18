// RUN: tensor_ir-opt -layout-propagation-pipeline="tile-size=1,8,64 persistence=static sm-count=148 occupancy=2" -split-input-file %s | FileCheck %s

// Tensor (3200,32,64) with tile size (1,8,64) has
// totalTiles = 3200 * 4 * 1 = 12800 and grid = 148 * 2 = 296.
// Use explicit non-coalesced strides so layout propagation keeps the 3D
// iteration space instead of normalizing contiguous row-major dims to 1D.
module {
  // CHECK-LABEL: entry @add_layout_prop_static_persistent(
  // CHECK-SAME: optimization_hints=<default = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 2}>
  // CHECK-DAG: %[[TOTAL:.*]] = constant <i32: 12800> : tile<i32>
  // CHECK-DAG: %[[GRID:.*]] = constant <i32: 296> : tile<i32>
  // CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
  // CHECK: for %[[TILE:.*]] in (%[[BLOCK]] to %[[TOTAL]], step %[[GRID]]) : tile<i32>
  // CHECK: load_view_tko
  // CHECK: load_view_tko
  // CHECK: addf
  // CHECK: store_view_tko
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  nv_tensor_ir.graph @add_layout_prop_static_persistent(
      %arg0: tensor<3200x32x64xf32> {nv_tensor_ir.stride = "(8192,128,1)"},
      %arg1: tensor<3200x32x64xf32> {nv_tensor_ir.stride = "(8192,128,1)"})
      -> (tensor<3200x32x64xf32> {nv_tensor_ir.stride = "(8192,128,1)"}) {
    %result = add %arg0, %arg1 : tensor<3200x32x64xf32>
    results %result : tensor<3200x32x64xf32>
  }
}

// -----

// Static persistence is intentionally not emitted when all tiles fit in the
// persistent launch grid.
module {
  // CHECK-LABEL: entry @add_layout_prop_static_persistent_small_total(
  // CHECK: %[[BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
  // CHECK-NOT: for %
  // CHECK: load_view_tko
  // CHECK: load_view_tko
  // CHECK: addf
  // CHECK: store_view_tko
  // CHECK: return
  nv_tensor_ir.graph @add_layout_prop_static_persistent_small_total(
      %arg0: tensor<4x8x64xf32> {nv_tensor_ir.stride = "(8192,128,1)"},
      %arg1: tensor<4x8x64xf32> {nv_tensor_ir.stride = "(8192,128,1)"})
      -> (tensor<4x8x64xf32> {nv_tensor_ir.stride = "(8192,128,1)"}) {
    %result = add %arg0, %arg1 : tensor<4x8x64xf32>
    results %result : tensor<4x8x64xf32>
  }
}
