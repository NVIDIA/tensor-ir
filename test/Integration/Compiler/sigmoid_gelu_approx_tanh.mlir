// RUN: tensor_ir-compiler %s --codegen-strategy=layout-propagation --verbose --launch --verify

module {
  nv_tensor_ir.graph @sigmoid_gelu_approx_tanh_op_simple(
    %input: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %sigmoid = sigmoid_fwd %input : tensor<8x8xf32>
    %gelu = gelu_approx_tanh_fwd %sigmoid : tensor<8x8xf32>
    results %gelu : tensor<8x8xf32>
  }
}
