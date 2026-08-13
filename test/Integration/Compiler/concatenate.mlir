// RUN: tensor_ir-compiler %s --verbose --launch --verify

module {
  nv_tensor_ir.graph @concatenate_op_simple(
    %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"},
    %b: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x16xf32> {nv_tensor_ir.stride = "(16,1)"}) {
    %concat = concatenate %a, %b dimension = 1
      : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x16xf32>
    results %concat : tensor<8x16xf32>
  }
}
