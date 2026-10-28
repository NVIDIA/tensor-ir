// RUN: tensor_ir-opt %s --test-side-effects --verify-diagnostics

func.func @memory_effects(%view: memref<8xf32>, %tile: tensor<8xf32>,
                         %index: index) -> tensor<8xf32> {
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  %loaded = nv_tensor_ir.load %view tile_indices [%index], alignment 4
      : memref<8xf32> to tensor<8xf32>
  // expected-remark @below {{found an instance of 'write' on op operand 1, on resource '<Default>'}}
  nv_tensor_ir.store %tile, %view tile_indices [%index], alignment 4
      : tensor<8xf32>, memref<8xf32>
  return %loaded : tensor<8xf32>
}
