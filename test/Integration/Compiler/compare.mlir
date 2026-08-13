// RUN: tensor_ir-compiler %s --verbose --launch --verify

module {
  nv_tensor_ir.graph @compareSelectF32Static(
    %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"},
    %b: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %selector = cmp %a ogt %b : tensor<8x8xf32>
    %out = binary_select %selector, %a, %b : tensor<8x8xf32>
    results %out : tensor<8x8xf32>
  }
}
