// RUN: tensor_ir-opt -discover-iteration-space-info -convert-tensor-to-cuda-tile="codegen-strategy=affine_map persistence=static sm-count=148 occupancy=2" %s | FileCheck %s
// Testing static persistent kernels in TensorToCudaTile conversion
// Static persistence generates a cuda_tile.for loop when gridSize < totalTiles
// With smCount=148 and occupancy=2, gridSize = 148 * 2 = 296

module {

// Test: Large tensor that triggers persistence
// Tensor (3200,32,64) with tile size (1,8,64) -> totalTiles = 3200*4*1 = 12800
// Since gridSize=296 < totalTiles=12800, persistence loop should be generated
//
// CHECK-LABEL: cuda_tile.module @cuda_tile_module
// CHECK:         entry @add_static_persistent
// CHECK-SAME:      optimization_hints=<default = {occupancy = 2}>
// Check for get_tile_block_id
// CHECK:           %[[BLOCK_X:.*]], %{{.*}}, %{{.*}} = get_tile_block_id
// Check for constants: totalTiles=12800 and gridSize=296
// CHECK-DAG:       %[[TOTAL:.*]] = constant <i32: 12800>
// CHECK-DAG:       %[[GRID:.*]] = constant <i32: 296>
// Check for cuda_tile.for loop (static persistence)
// CHECK:           for %[[LOOP_IDX:.*]] in (%[[BLOCK_X]] to %[[TOTAL]], step %[[GRID]]) : tile<i32>
// CHECK:             load_view_tko
// CHECK:             load_view_tko
// CHECK:             addf
// CHECK:             store_view_tko
// CHECK:           return
  nv_tensor_ir.graph @add_static_persistent(
      %arg0: tensor<3200x32x64xf32>{nv_tensor_ir.stride = "(2048,64,1)"},
      %arg1: tensor<3200x32x64xf32>{nv_tensor_ir.stride = "(2048,64,1)"}
  ) -> (tensor<3200x32x64xf32> {nv_tensor_ir.stride = "(2048,64,1)"}) {
      %result = add %arg0, %arg1 : tensor<3200x32x64xf32>
      results %result : tensor<3200x32x64xf32>
  }
}
