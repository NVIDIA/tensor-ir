// RUN: tensor_ir-compiler %s --verbose --launch --verify

module {
  nv_tensor_ir.graph @pointwise_i32_static(
    %a: tensor<8x8xsi32> {nv_tensor_ir.stride = "(8,1)"},
    %b: tensor<8x8xsi32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xsi32> {nv_tensor_ir.stride = "(8,1)"}) {
    %add = add %a, %b : tensor<8x8xsi32>
    %sub = sub %add, %a : tensor<8x8xsi32>
    %mul = mul %sub, %b : tensor<8x8xsi32>
    %max = max %mul, %a : tensor<8x8xsi32>
    %min = min %max, %add : tensor<8x8xsi32>
    %add_square = add_square %min, %sub : tensor<8x8xsi32>
    results %add_square : tensor<8x8xsi32>
  }
}
