// RUN: tensor_ir-compiler %s --verbose --launch --verify

module {
  nv_tensor_ir.graph @binary_select_i32_static(
    %selector: tensor<8x8xi1> {nv_tensor_ir.stride = "(8,1)"},
    %lhs: tensor<8x8xsi32> {nv_tensor_ir.stride = "(8,1)"},
    %rhs: tensor<8x8xsi32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xsi32> {nv_tensor_ir.stride = "(8,1)"}) {
    %out = binary_select %selector, %lhs, %rhs : tensor<8x8xsi32>
    results %out : tensor<8x8xsi32>
  }
}
