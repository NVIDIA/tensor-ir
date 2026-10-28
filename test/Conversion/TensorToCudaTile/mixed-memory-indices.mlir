// RUN: tensor_ir-opt %s -convert-tensor-to-cuda-tile="codegen-strategy=layout_propagation" | FileCheck %s

module @mixed_indices_module {
  gpu.module @kernels {
    // CHECK-LABEL: entry @mixed_indices(
    func.func @mixed_indices(%buffer: memref<64x128xf32>) attributes {gpu.kernel} {
      // CHECK: %[[BLOCK:[^,]+]], {{.*}} = get_tile_block_id
      %block = gpu.block_id x
      // CHECK: %[[TWO:.*]] = constant <i32: 2>
      // CHECK: %[[TILE:[^,]+]], {{.*}} = load_view_tko {{.*}}[%[[TWO]], %[[BLOCK]]]
      %tile = nv_tensor_ir.load %buffer tile_indices [2, %block], alignment 4
          : memref<64x128xf32> to tensor<8x8xf32>
      // CHECK: %[[THREE:.*]] = constant <i32: 3>
      // CHECK: store_view_tko weak %[[TILE]], {{.*}}[%[[BLOCK]], %[[THREE]]]
      nv_tensor_ir.store %tile, %buffer tile_indices [%block, 3], alignment 4
          : tensor<8x8xf32>, memref<64x128xf32>
      return
    }
  }
}
