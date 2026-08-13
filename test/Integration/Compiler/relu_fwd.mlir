// RUN: tensor_ir-compiler %s --codegen-strategy=layout-propagation --verbose --launch --verify

module {
  nv_tensor_ir.graph @relu_fwd_op_simple(
    %input: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %relu_fwd = relu_fwd %input : tensor<8x8xf32>
    results %relu_fwd : tensor<8x8xf32>
  }
}
