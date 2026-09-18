// RUN: tensor_ir-opt %s --test-side-effects --verify-diagnostics

func.func @memory_effects(
    %base: !ptr.ptr<#ptr.generic_space>, %tile: tensor<8xf32>,
    %index: index) -> tensor<8xf32> {
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  %loaded = nv_tensor_ir.load %base[%index]
      view offset: [0], sizes: [8], strides: [1], alignment: 4
      : !ptr.ptr<#ptr.generic_space> to tensor<8xf32>
  // expected-remark @below {{found an instance of 'write' on op operand 1, on resource '<Default>'}}
  nv_tensor_ir.store %tile, %base[%index]
      view offset: [0], sizes: [8], strides: [1], alignment: 4
      : tensor<8xf32>, !ptr.ptr<#ptr.generic_space>
  return %loaded : tensor<8xf32>
}
