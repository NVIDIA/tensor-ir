// RUN: tensor_ir-compiler %s --verbose --launch --verify

module {
  nv_tensor_ir.graph @pow_op_simple(
    %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"},
    %b: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %pow = pow %a, %b : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    results %pow : tensor<8x8xf32>
  }
}
