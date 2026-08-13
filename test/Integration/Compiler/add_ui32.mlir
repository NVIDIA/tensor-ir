// RUN: tensor_ir-compiler %s --verbose --launch --verify

module {
  nv_tensor_ir.graph @add_ui32(
    %lhs: tensor<8x8xui32> {nv_tensor_ir.stride = "(8,1)"},
    %rhs: tensor<8x8xui32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xui32> {nv_tensor_ir.stride = "(8,1)"}) {
    %add = add %lhs, %rhs : tensor<8x8xui32>
    %big = constant dense<4294967295> : tensor<8x8xui32>
    %two = constant dense<2> : tensor<8x8xui32>
    %half = div %big, %two : tensor<8x8xui32>
    %result = add %add, %half : tensor<8x8xui32>
    results %result : tensor<8x8xui32>
  }
}
