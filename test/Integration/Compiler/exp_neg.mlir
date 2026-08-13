// RUN: tensor_ir-compiler %s --verbose --launch --verify

module {
  nv_tensor_ir.graph @exp_neg_op_simple(
    %input: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %neg = neg %input : tensor<8x8xf32>
    %exp = exp %neg : tensor<8x8xf32>
    results %exp : tensor<8x8xf32>
  }
}
