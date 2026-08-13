// RUN: tensor_ir-compiler %s --codegen-strategy=layout-propagation --verbose --launch --verify

module {
  nv_tensor_ir.graph @elu_fwd_op_simple(
    %input: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %elu = elu_fwd <beta = 2.500000e-01 : f32> %input : tensor<8x8xf32>
    results %elu : tensor<8x8xf32>
  }
}
