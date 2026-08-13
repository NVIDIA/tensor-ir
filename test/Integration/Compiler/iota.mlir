// RUN: tensor_ir-compiler %s --codegen-strategy=layout-propagation --verbose --launch --verify

module {
  nv_tensor_ir.graph @iota_op_simple()
      -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %iota = nv_tensor_ir.iota dimension = 1 : tensor<8x8xf32>
    results %iota : tensor<8x8xf32>
  }
}
