// RUN: tensor_ir-compiler %s --codegen-strategy=layout-propagation --verbose --launch --verify

module {
  nv_tensor_ir.graph @logical_i32_static(
    %lhs: tensor<8x8xsi32> {nv_tensor_ir.stride = "(8,1)"},
    %rhs: tensor<8x8xsi32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xi1> {nv_tensor_ir.stride = "(8,1)"}) {
    %gt = cmp %lhs gt %rhs : tensor<8x8xsi32>
    %lt = cmp %lhs lt %rhs : tensor<8x8xsi32>
    %and = and %gt, %lt : tensor<8x8xi1>
    %or = or %gt, %lt : tensor<8x8xi1>
    %out = or %and, %or : tensor<8x8xi1>
    results %out : tensor<8x8xi1>
  }
}
