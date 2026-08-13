// RUN: tensor_ir-compiler %s --codegen-strategy=layout-propagation --verbose --launch --verify

module {
  nv_tensor_ir.graph @erf_atan2_op_simple(
    %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"},
    %b: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %erf = erf %a : tensor<8x8xf32>
    %atan2 = atan2 %erf, %b : tensor<8x8xf32>
    results %atan2 : tensor<8x8xf32>
  }
}
