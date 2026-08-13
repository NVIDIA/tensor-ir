// RUN: tensor_ir-compiler %s --verbose --launch --verify

module {
  nv_tensor_ir.graph @pointwise_bf16_static(
    %a: tensor<8x8xbf16> {nv_tensor_ir.stride = "(8,1)"},
    %b: tensor<8x8xbf16> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xbf16> {nv_tensor_ir.stride = "(8,1)"}) {
    %max = max %a, %b : tensor<8x8xbf16>
    %min = min %max, %a : tensor<8x8xbf16>
    %add = add %min, %min : tensor<8x8xbf16>
    %sub = sub %add, %min : tensor<8x8xbf16>
    %div = div %add, %add : tensor<8x8xbf16>
    %mul = mul %sub, %div : tensor<8x8xbf16>
    results %mul : tensor<8x8xbf16>
  }
}
