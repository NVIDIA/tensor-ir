// RUN: tensor_ir-compiler %s --verbose --launch --verify

module {
  nv_tensor_ir.graph @rem_bf16_static(
    %a: tensor<8x8xbf16> {nv_tensor_ir.stride = "(8,1)"},
    %b: tensor<8x8xbf16> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xbf16> {nv_tensor_ir.stride = "(8,1)"}) {
    %rem = rem %a, %b : tensor<8x8xbf16>
    results %rem : tensor<8x8xbf16>
  }
}
