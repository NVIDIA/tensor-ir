// RUN: tensor_ir-opt %s -convert-tensor-to-cuda-tile="codegen-strategy=layout_propagation" | FileCheck %s

module @mixed_indices_module {
  gpu.module @kernels {
    // CHECK-LABEL: entry @mixed_indices(
    func.func @mixed_indices(%buffer: memref<64x128xf32, #ptr.generic_space>) attributes {gpu.kernel} {
      // CHECK: %[[BLOCK:[^,]+]], {{.*}} = get_tile_block_id
      %block = gpu.block_id x
      %ptr = ptr.to_ptr %buffer : memref<64x128xf32, #ptr.generic_space>
          -> !ptr.ptr<#ptr.generic_space>
      // CHECK: %[[TWO:.*]] = constant <i32: 2>
      // CHECK: %[[TILE:[^,]+]], {{.*}} = load_view_tko {{.*}}[%[[TWO]], %[[BLOCK]]]
      %tile = nv_tensor_ir.load %ptr[2, %block]
          view offset: [0], sizes: [64, 128], strides: [128, 1], alignment: 4
          : !ptr.ptr<#ptr.generic_space> to tensor<8x8xf32>
      // CHECK: %[[THREE:.*]] = constant <i32: 3>
      // CHECK: store_view_tko weak %[[TILE]], {{.*}}[%[[BLOCK]], %[[THREE]]]
      nv_tensor_ir.store %tile, %ptr[%block, 3]
          view offset: [0], sizes: [64, 128], strides: [128, 1], alignment: 4
          : tensor<8x8xf32>, !ptr.ptr<#ptr.generic_space>
      return
    }
  }
}
