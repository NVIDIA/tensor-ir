// RUN: tensor_ir-compiler %s --codegen-strategy=layout-propagation --verbose --launch --verify

module {
  nv_tensor_ir.graph @swish_fwd_op_simple(
    %input: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %swish = swish_fwd <beta = 1.500000e+00 : f32> %input : tensor<8x8xf32>
    results %swish : tensor<8x8xf32>
  }
}
