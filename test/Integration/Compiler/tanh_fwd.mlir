// RUN: tensor_ir-compiler %s --codegen-strategy=layout-propagation --verbose --launch --verify

module {
  nv_tensor_ir.graph @tanh_fwd_op_simple(
    %input: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %tanh_fwd = tanh_fwd %input : tensor<8x8xf32>
    results %tanh_fwd : tensor<8x8xf32>
  }
}
